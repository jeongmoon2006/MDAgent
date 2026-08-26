"""Mechanical campaign loop: simulate → diagnose → persist → decide → repeat.

This is the state machine described in `docs/architecture.md`. The LLM is
called exactly once per round (in `scientist.decide`); everything else is
deterministic Python.

Engine independence (M3): the loop talks to MD engines exclusively through
the `MDAdapter` Protocol. Swap the adapter to change engines; nothing in
this file or in `scientist.py` needs to know.

Persistence (M2): per-campaign SQLite at `<work_dir>/state.db` is the
source of truth for completed rounds; the adapter's checkpoint captures
the state needed to continue. Commit order is `save_checkpoint` THEN
`store.append_round` — a crash between leaves the checkpoint dangling
(harmless) and the round absent, so restart re-runs it. The per-round
JSON file is kept alongside SQLite for human inspection.

Metadynamics pivot (M4): when `decide()` returns `switch_to_metad`, the
campaign does not end — it pivots in place. The proposed CV is resolved
against the topology, a bias is sized deterministically from the just-run
(vanilla) trajectory, a `plumed.dat` is written, and a biased adapter is
constructed from the same `SystemSpec`. Subsequent rounds run under that
bias and record its `plumed_dat_path`. The metaD phase starts from the
cached minimized state (the vanilla checkpoint is not portable across the
added `PlumedForce`); SIGMA is still sized from the vanilla basin sampling,
which is the relevant width. Warm-starting metaD from the vanilla endpoint
is a possible follow-up.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast

import mdtraj as md
import numpy as np

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import PlumedInput, enable_restart
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.diagnostics.free_energy import metad_report
from mdpilot.diagnostics.report import make_report
from mdpilot import preflight
from mdpilot.observables import ObservableSpec, campaign_observable
from mdpilot.memory import store
from mdpilot.orchestrator.scientist import Decision, MetadProposal, decide
from mdpilot.sampling.bias_designer import design_bias, design_upper_wall
from mdpilot.sampling.cv_designer import CVProposal, design_cv

_FS_PER_NS = 1_000_000.0
_PLUMED_DAT_NAME = "plumed.dat"  # canonical bias artifact (the OpenMM adapter also writes here)
# PLUMED's own outputs, alongside plumed.dat. These names must match the
# defaults `_build_plumed_input` leaves on MetadynamicsBias.hills_file and
# PlumedInput.colvar_file — they are what the rendered FILE= directives point at.
_HILLS_NAME = "HILLS"
_COLVAR_NAME = "COLVAR"

StopReason = Literal[
    "scientist_said_stop",
    "max_rounds_reached",
    "switch_to_metad_requested",
    "biased_budget_exhausted",
]

# Given a rendered plumed.dat string, build an adapter that runs biased MD.
BiasedAdapterFactory = Callable[[str], MDAdapter]

# Optional observer, called as the campaign progresses. Purely for watching —
# a progress bar, a log file, an HPC monitor. It cannot influence the campaign,
# and `_emit` swallows anything it raises: a long biased run must not die
# because something that was only watching it threw.
CampaignEvent = Callable[[str, dict[str, Any]], None]


def _emit(on_event: CampaignEvent | None, name: str, **payload: Any) -> None:
    if on_event is None:
        return
    try:
        on_event(name, payload)
    except Exception:  # noqa: BLE001 - an observer must never break the run
        pass


def steps_per_ns_for(adapter: MDAdapter) -> int:
    """Steps in one nanosecond at this engine's timestep (500_000 at 2 fs).

    Read from the adapter rather than hardcoded. `extra_ns`, `max_extra_ns`
    and `max_biased_ns` are all stated in nanoseconds and converted here, so a
    constant that merely happened to match both adapters would have made every
    round the wrong length on an engine at a different timestep — with
    `extra_ns` quietly no longer meaning nanoseconds anywhere in the campaign
    record.

    Public because `benchmarks/run_cln025.py` states its budget in nanoseconds
    too, and a second copy of the conversion is exactly the drift this exists
    to remove.
    """
    return max(int(round(_FS_PER_NS / adapter.timestep_fs)), 1)


@dataclass(frozen=True)
class RoundResult:
    index: int
    n_steps: int
    dcd_path: Path
    summary_path: Path
    report: dict[str, Any]
    decision: Decision
    plumed_dat_path: Path | None = None


@dataclass(frozen=True)
class CampaignResult:
    work_dir: Path
    rounds: tuple[RoundResult, ...]
    stop_reason: StopReason


def run_campaign(
    work_dir: Path,
    *,
    adapter: MDAdapter | None = None,
    initial_steps: int = 25_000,         # 50 ps default at 2 fs
    max_rounds: int = 10,
    max_extra_ns: float = 2.0,
    max_biased_ns: float | None = None,
    min_recrossings: int = 1,
    state_thresholds: tuple[float, float] | None = None,
    max_cv_switches: int = 1,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
    observable: ObservableSpec | None = None,
    description: str | None = None,
    on_event: CampaignEvent | None = None,
    seed: int = 42,
    report_interval_steps: int = 500,    # 1 ps/frame at 2 fs
    equilibration_steps: int = 0,
    task_expectation: str | None = None,
    biased_adapter_factory: BiasedAdapterFactory | None = None,
) -> CampaignResult:
    """Run the closed loop, resuming from `work_dir/state.db` if it exists.

    `adapter` defaults to `OpenMMAdapter(work_dir=work_dir, seed=seed)`.
    Pass a different `MDAdapter` to run through another engine.

    `state_thresholds` is `(low, high)` on the campaign observable — the two
    states the *task* defines, as positions on that coordinate rather than as
    roles. For a folding campaign that is (native, extended) on CA-RMSD; for a
    binding campaign it is (bound, unbound) on a distance, and nothing here
    changes. When given, biased-round recrossings are counted there rather than
    between the two deepest basins of the current free-energy surface. The
    surface-derived band moves as the bias fills, vanishes when fewer than two
    basins resolve, and collapses once the barrier is filled, so a count taken
    against it means something different every round (F7, F9). A fixed band on
    the coordinate the task defines its states on is comparable across rounds
    and across a change of biased CV.

    `max_cv_switches` is how many times the scientist may replace the biased
    CV within one campaign. While switches remain, `switch_cv` is offered in
    the biased action space; once they are spent it is dropped from the tool
    schema, so a further revision is unrepresentable rather than emitted and
    then refused. Counted across resumes from the persisted rounds.

    `max_rounds`, `max_extra_ns`, `max_biased_ns` and `max_cv_switches` are
    loop-control bounds and may differ between invocations. Everything that
    defines what the campaign *is* — seed, initial_steps,
    report_interval_steps, equilibration_steps, the system spec, the engine,
    task_expectation, cv_upper_wall_nm, state_thresholds, min_recrossings and
    the bias shape — is locked at first init and a mismatch on resume raises.
    The thermostat temperature and timestep live on `spec.ensemble` and so are
    covered by the system-spec half of that lock; they used to be adapter class
    constants covered only by the engine-name lock, which would have stopped
    covering them the moment they became settable.

    `max_biased_ns` caps *cumulative* simulation time in the metadynamics
    phase, counting across rounds and across resumes (it is recomputed from
    the persisted biased rounds, so a restart cannot reset the meter). The
    round that would exceed it is shortened to land exactly on the budget and
    the campaign then ends with `biased_budget_exhausted`. The vanilla phase
    is not counted. Left at None the biased phase is bounded only by
    `max_rounds`. A compute budget stated only in `task_expectation` is
    advisory — the model can read it and still ask for more — so anything
    running unattended wants this set.

    `task_expectation` is free-form campaign-level guidance threaded into
    every `decide()` call — what the trajectory must accomplish, the
    characteristic timescale, the compute budget. Required for campaigns
    where the scientist may need to choose `switch_to_metad`; otherwise the
    LLM has no basis to judge "the budget cannot reach the transition."

    `observable` is the coordinate every round is judged on — the vanilla
    convergence bundle summarizes it and `state_thresholds` are positions on
    it. Left at None it is CA-RMSD to the campaign topology in Angstrom, the
    M1 observable. It defines what the campaign measures, so it locks with the
    rest of the config.

    `bias_pace` and `bias_factor` override the METAD deposition stride and the
    well-tempered gamma that `sampling.bias_designer` would otherwise pick.
    Left at None the designer's own defaults apply, so there is one source of
    truth for them rather than a second copy in this signature. Both are
    biased-phase physics and lock with the rest of the campaign config.
    gamma=10 flattens barriers up to ~gamma*kT; a surface deeper than that
    wants a larger one.

    `description` is prose about the system, used only by the pre-flight
    checks — it is compared against the structure that was actually fetched.
    It is deliberately *not* part of the locked config: editing prose must not
    stop a campaign resuming.

    `on_event` is an optional observer called with `(name, payload)` as the
    campaign progresses — `campaign_start`, `round_start`, `simulated`,
    `report`, `decision`, `override`, `pivot`, `campaign_end`. It cannot change
    what happens and anything it raises is swallowed, because a multi-hour
    biased run must not die because something watching it threw.

    `biased_adapter_factory` builds the adapter used after a `switch_to_metad`
    pivot from a rendered plumed.dat string. Defaults to an `OpenMMAdapter`
    over the same `SystemSpec` with `plumed_input` set. Inject to run the
    biased phase through a different engine (or a fake, in tests).
    """
    # A biased phase without task states would fall back to counting
    # recrossings between the two deepest basins of the current surface, which
    # F7 and F9 showed is not comparable between rounds — it migrates, vanishes
    # when fewer than two basins resolve, and inflates when the barrier fills.
    # `task_expectation` is the sole input gating `switch_to_metad`, so it is
    # exactly the predicate for "this campaign can reach a biased phase".
    # Refused here, before any MD is paid for, rather than at the pivot.
    if task_expectation is not None and state_thresholds is None:
        raise ValueError(
            "run_campaign: task_expectation is set, so this campaign can pivot "
            "to metadynamics, but state_thresholds is None. A biased phase "
            "needs the task's own state definitions to count recrossings "
            "against; without them the count is taken between whichever two "
            "basins are currently deepest, which means something different "
            "every round. Pass state_thresholds=(low, high) on the campaign "
            "observable."
        )
    if state_thresholds is not None:
        low, high = state_thresholds
        if not high > low:
            raise ValueError(
                f"run_campaign: state_thresholds must be (low, high) on the "
                f"campaign observable with high > low; got {state_thresholds!r}. "
                f"`count_recrossings` silently returns 0 for an inverted band, "
                f"so a swapped pair would read as a run that never crossed."
            )

    work_dir = Path(work_dir)
    rounds_dir = work_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    plumed_dat_path = work_dir / _PLUMED_DAT_NAME

    if adapter is None:
        adapter = OpenMMAdapter(work_dir=work_dir, seed=seed)

    base_spec = adapter.spec
    # Captured once, from the vanilla adapter. The biased adapter is built by
    # `_default_biased_factory` over the same engine (and an injected factory
    # is documented as engine-matching, since the CV's atom indices are
    # resolved against this adapter's topology), so one reading holds for the
    # whole campaign.
    steps_per_ns = steps_per_ns_for(adapter)
    temperature_k = adapter.temperature_k

    if biased_adapter_factory is None:
        biased_adapter_factory = _default_biased_factory(
            adapter, work_dir=work_dir, seed=seed, spec=base_spec
        )

    config = {
        "seed": seed,
        "initial_steps": initial_steps,
        "report_interval_steps": report_interval_steps,
        "equilibration_steps": equilibration_steps,
        "system_spec": base_spec.to_dict(),
        # Engine identity is physics-bound: checkpoints are engine-specific
        # binary formats, and a swapped adapter would feed an OpenMM .chk to
        # `grompp -t` (or the reverse) instead of failing the config guard.
        "engine": type(adapter).__name__,
        # task_expectation is the only input gating `switch_to_metad`. Resuming
        # without it silently reverts an enhanced-sampling campaign to a plain
        # convergence task, so it locks with the rest.
        "task_expectation": task_expectation,
        # The wall bounds the CV the bias acts on, so it is biased-phase physics
        # rather than a loop-control bound: changing it on resume would join two
        # halves of a campaign sampled under different Hamiltonians.
        "cv_upper_wall_nm": cv_upper_wall_nm,
        # The band every biased-round recrossing count is taken against.
        # Resuming with a different one would splice two definitions of
        # "a transition" into a single campaign's history. Listed as a list
        # rather than a tuple so the JSON round-trip is stable.
        "state_thresholds": (
            list(state_thresholds) if state_thresholds is not None else None
        ),
        # The other half of that same definition: `state_thresholds` says where
        # the states are, `min_recrossings` says how many transitions between
        # them count as done. It is the threshold `fes_converged` compares the
        # count against, and `_refuse_premature_stop` reads that verdict to
        # decide whether the scientist may stop — so resuming with a different
        # one changes when the campaign is allowed to end, retroactively, for
        # rounds already judged under the old value.
        "min_recrossings": min_recrossings,
    }
    # Bias shape is biased-phase physics: resuming under a different gamma or
    # deposition stride would join two differently-filled surfaces into one
    # `sum_hills` integration. Added to the lock only when set, for the same
    # reason `SystemSpec.to_dict` omits a default ensemble — an unconditional
    # key would make every campaign recorded before this existed unresumable.
    # Omitted when default for the same reason `SystemSpec.to_dict` omits a
    # default ensemble: an unconditional key would strand campaigns recorded
    # before observables were declarable.
    if observable is not None and observable != ObservableSpec.ca_rmsd_angstrom():
        config["observable"] = observable.to_dict()
    if bias_pace is not None:
        config["bias_pace"] = bias_pace
    if bias_factor is not None:
        config["bias_factor"] = bias_factor
    store.init_campaign(work_dir, config)
    prior_rows = store.list_rounds(work_dir)
    rounds: list[RoundResult] = [_row_to_result(r) for r in prior_rows]

    if rounds and rounds[-1].decision.decision == "stop":
        return _finish(on_event, work_dir, rounds, "scientist_said_stop")

    if (
        rounds
        and rounds[-1].decision.decision == "switch_to_metad"
        and rounds[-1].plumed_dat_path is not None
    ):
        # Second pivot from an already-biased round: the in-process loop below
        # refuses this and ends cleanly for human inspection. Re-invoking must
        # not quietly perform the pivot the live run declined — the resume
        # branches would otherwise rebuild a bias from a *biased* trajectory's
        # spread, which is not the width anything was measured against.
        return _finish(on_event, work_dir, rounds, "switch_to_metad_requested")

    # Build the vanilla engine state up front. This is idempotent and cheap on
    # resume (rebuilds from cache), and it guarantees the cached System +
    # minimized state + topology exist for a biased adapter to reuse.
    adapter.prepare()
    adapter.start()

    # Pre-flight, on the starting structure, before a single step is
    # integrated. These compare across the one seam no schema can check: what
    # the task file *says* against what was actually built, and the
    # observable's own magnitude against the bands it will be judged in. A
    # campaign that cannot reach its own done criterion should cost seconds,
    # not the forty minutes it once did.
    preflight.check_residue_count(adapter.topology_path, description)
    reference = md.load(str(adapter.topology_path))
    first_value, observable_name = campaign_observable(
        reference, adapter.topology_path, observable
    )
    preflight.check_observable_scale(
        float(first_value[0]), state_thresholds, observable_name
    )
    _emit(
        on_event, "preflight_ok",
        residues=sum(1 for r in reference.topology.residues if r.is_protein),
        observable=observable_name,
        first_value=float(first_value[0]),
        state_thresholds=list(state_thresholds) if state_thresholds else None,
    )

    in_metad = False
    current_plumed_dat: Path | None = None
    last = prior_rows[-1] if prior_rows else None

    if last is None:
        if equilibration_steps > 0:
            adapter.run_steps(equilibration_steps)
        start_round = 1
        n_steps = initial_steps
    elif last.decision == "switch_to_metad":
        # Pivot-resume: the metaD phase has produced no round yet. Rebuild the
        # bias from the switch round's proposal + its (vanilla) trajectory and
        # start a fresh biased simulation. The vanilla checkpoint is not loaded
        # (not portable across the added PlumedForce).
        if last.metad_proposal is None:
            raise RuntimeError(
                f"round {last.round_index} decided switch_to_metad but stored "
                f"no metad_proposal; cannot build the bias"
            )
        adapter = _pivot_to_metad(
            MetadProposal.from_dict(last.metad_proposal),
            source_trajectory=last.dcd_path,
            topology_path=adapter.topology_path,
            factory=biased_adapter_factory,
            plumed_dat_path=plumed_dat_path,
            temperature_k=temperature_k,
            cv_upper_wall_nm=cv_upper_wall_nm,
            bias_pace=bias_pace,
            bias_factor=bias_factor,
        )
        in_metad = True
        current_plumed_dat = plumed_dat_path
        start_round = last.round_index + 1
        n_steps = initial_steps
    elif last.decision == "switch_cv":
        # CV-revision resume. Ordered *before* the generic biased branch below:
        # a switch_cv round is itself biased, so `plumed_dat_path is not None`
        # would match first and resume the campaign on the CV the scientist
        # just rejected, with RESTART reading its hills back.
        if last.metad_proposal is None:
            raise RuntimeError(
                f"round {last.round_index} decided switch_cv but stored no "
                f"metad_proposal; cannot build the replacement bias"
            )
        _clear_bias_state(plumed_dat_path.parent)
        adapter = _pivot_to_metad(
            MetadProposal.from_dict(last.metad_proposal),
            source_trajectory=last.dcd_path,
            topology_path=adapter.topology_path,
            factory=biased_adapter_factory,
            plumed_dat_path=plumed_dat_path,
            temperature_k=temperature_k,
            cv_upper_wall_nm=cv_upper_wall_nm,
            bias_pace=bias_pace,
            bias_factor=bias_factor,
        )
        in_metad = True
        current_plumed_dat = plumed_dat_path
        start_round = last.round_index + 1
        n_steps = initial_steps
    elif last.plumed_dat_path is not None:
        # Mid-metaD-phase resume: rebuild the biased adapter from the persisted
        # plumed.dat and continue from that round's (biased) checkpoint. The
        # deposited bias is the other half of that resume point — restore the
        # snapshot paired with the checkpoint, then turn RESTART on so METAD
        # reads it back instead of backing it up and refilling from zero.
        _restore_bias_state(
            last.plumed_dat_path.parent, rounds_dir, last.round_index
        )
        adapter = biased_adapter_factory(
            enable_restart(last.plumed_dat_path.read_text())
        )
        adapter.prepare()
        adapter.start()
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        in_metad = True
        current_plumed_dat = last.plumed_dat_path
        start_round = last.round_index + 1
        n_steps = _extend_steps(last.extra_ns, max_extra_ns, steps_per_ns)
    else:
        # Vanilla resume.
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        start_round = last.round_index + 1
        n_steps = _extend_steps(last.extra_ns, max_extra_ns, steps_per_ns)

    # Cumulative biased steps already spent, so a resume continues the meter
    # rather than restarting it.
    biased_steps_run = sum(
        r.n_steps for r in prior_rows if r.plumed_dat_path is not None
    )
    # Same reasoning as the biased-step meter: recomputed from disk so a
    # restart cannot buy the scientist a second allowance of CV switches.
    cv_switches_used = sum(1 for r in prior_rows if r.decision == "switch_cv")
    biased_step_budget = (
        int(max_biased_ns * steps_per_ns) if max_biased_ns is not None else None
    )

    ledger_notes: list[store.LedgerNote] = list(store.list_ledger_notes(work_dir))

    _emit(
        on_event,
        "campaign_start",
        work_dir=str(work_dir),
        engine=type(adapter).__name__,
        forcefield=base_spec.forcefield,
        temperature_k=temperature_k,
        padding_nm=base_spec.padding_nm,
        timestep_fs=adapter.timestep_fs,
        observable=(observable or ObservableSpec.ca_rmsd_angstrom()).name,
        resuming_from_round=len(prior_rows),
        start_round=start_round,
        max_rounds=max_rounds,
        phase="metad" if in_metad else "vanilla",
    )

    for round_idx in range(start_round, max_rounds + 1):
        if in_metad and biased_step_budget is not None:
            remaining = biased_step_budget - biased_steps_run
            if remaining <= 0:
                return _finish(on_event, work_dir, rounds, "biased_budget_exhausted")
            # Shorten the round that would overshoot rather than skipping it —
            # a partial round still deposits hills and still gets diagnosed.
            n_steps = min(n_steps, remaining)

        dcd = rounds_dir / f"round_{round_idx:03d}{adapter.trajectory_extension}"
        _emit(
            on_event, "round_start",
            round_index=round_idx, n_steps=n_steps,
            ns=n_steps / steps_per_ns, phase="metad" if in_metad else "vanilla",
        )
        started = time.monotonic()
        adapter.run_steps(
            n_steps,
            trajectory_path=dcd,
            report_interval_steps=report_interval_steps,
        )
        _emit(
            on_event, "simulated",
            round_index=round_idx, seconds=time.monotonic() - started,
            trajectory=str(dcd),
        )
        if in_metad:
            biased_steps_run += n_steps
        # Checkpoint first: everything after this line can fail on something
        # external (a `plumed sum_hills` that is not on PATH, a transient
        # Anthropic outage) and the MD is already paid for. Without a
        # checkpoint, restart has no resume point and re-runs the round. A
        # checkpoint with no matching row is the documented harmless case (D4).
        ckpt = adapter.save_checkpoint(rounds_dir / f"round_{round_idx:03d}.chk")
        if current_plumed_dat is not None:
            _snapshot_bias_state(
                current_plumed_dat.parent, rounds_dir, round_idx
            )
        report = _round_report(
            dcd,
            adapter.topology_path,
            plumed_dat_path=current_plumed_dat,
            temperature_k=temperature_k,
            fes_dir=rounds_dir / f"round_{round_idx:03d}_fes",
            min_recrossings=min_recrossings,
            rounds_dir=rounds_dir,
            round_index=round_idx,
            state_thresholds=state_thresholds,
            observable=observable,
        )
        _emit(on_event, "report", round_index=round_idx, report=report)
        prior_summaries = [_compact_prior(r) for r in rounds]
        decision = decide(
            report,
            prior_round_summaries=prior_summaries,
            hypothesis_ledger=[f"R{n.round_index}: {n.text}" for n in ledger_notes],
            task_expectation=task_expectation,
            phase="metad" if in_metad else "vanilla",
            allow_cv_switch=in_metad and cv_switches_used < max_cv_switches,
        )

        override_note: str | None = None
        if in_metad:
            remaining = (
                biased_step_budget - biased_steps_run
                if biased_step_budget is not None
                else None
            )
            decision, override_note = _refuse_premature_stop(
                decision, report, remaining
            )
            if override_note:
                _emit(on_event, "override", round_index=round_idx, note=override_note)

        _emit(
            on_event, "decision",
            round_index=round_idx, decision=decision.decision,
            reason=decision.reason, extra_ns=decision.extra_ns,
            metad_proposal=(
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
        )
        summary_path = rounds_dir / f"round_{round_idx:03d}.json"
        _persist_round_json(
            summary_path, round_idx, n_steps, dcd, report, decision, current_plumed_dat
        )
        store.append_round(
            work_dir,
            round_index=round_idx,
            n_steps=n_steps,
            dcd_path=dcd,
            checkpoint_path=ckpt,
            report=report,
            decision=decision.decision,
            reason=decision.reason,
            extra_ns=decision.extra_ns,
            metad_proposal=(
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
            plumed_dat_path=current_plumed_dat,
        )
        for note in (decision.ledger_note, override_note):
            if not note:
                continue
            store.append_ledger_note(work_dir, round_index=round_idx, text=note)
            ledger_notes.append(
                store.LedgerNote(round_index=round_idx, text=note)
            )
        rounds.append(
            RoundResult(
                round_idx, n_steps, dcd, summary_path, report, decision, current_plumed_dat
            )
        )

        if decision.decision == "stop":
            return _finish(on_event, work_dir, rounds, "scientist_said_stop")

        if decision.decision == "switch_to_metad":
            if in_metad:
                # Already biased: a second switch is out of scope. End cleanly
                # so a human can inspect rather than rebuild the bias in a loop.
                return _finish(
                    on_event, work_dir, rounds, "switch_to_metad_requested"
                )
            assert decision.metad_proposal is not None  # guaranteed by the parser
            adapter = _pivot_to_metad(
                decision.metad_proposal,
                source_trajectory=dcd,
                topology_path=adapter.topology_path,
                factory=biased_adapter_factory,
                plumed_dat_path=plumed_dat_path,
                temperature_k=temperature_k,
                cv_upper_wall_nm=cv_upper_wall_nm,
                bias_pace=bias_pace,
                bias_factor=bias_factor,
            )
            in_metad = True
            current_plumed_dat = plumed_dat_path
            n_steps = initial_steps
            _emit(
                on_event, "pivot",
                round_index=round_idx, kind="switch_to_metad",
                plumed_dat=str(plumed_dat_path),
                cv=decision.metad_proposal.to_dict(),
            )
            continue

        if decision.decision == "switch_cv":
            assert decision.metad_proposal is not None  # guaranteed by the parser
            # The replacement CV is sized on *this* round's trajectory, which
            # was run under the outgoing bias. That spread is inflated by the
            # bias that drove the walker across the old coordinate, which is
            # what `_SIGMA_CEILINGS` in bias_designer exists to catch.
            _clear_bias_state(plumed_dat_path.parent)
            adapter = _pivot_to_metad(
                decision.metad_proposal,
                source_trajectory=dcd,
                topology_path=adapter.topology_path,
                factory=biased_adapter_factory,
                plumed_dat_path=plumed_dat_path,
                temperature_k=temperature_k,
                cv_upper_wall_nm=cv_upper_wall_nm,
                bias_pace=bias_pace,
                bias_factor=bias_factor,
            )
            cv_switches_used += 1
            current_plumed_dat = plumed_dat_path
            n_steps = initial_steps
            _emit(
                on_event, "pivot",
                round_index=round_idx, kind="switch_cv",
                plumed_dat=str(plumed_dat_path),
                cv=decision.metad_proposal.to_dict(),
            )
            continue

        n_steps = _extend_steps(decision.extra_ns, max_extra_ns, steps_per_ns)

    return _finish(on_event, work_dir, rounds, "max_rounds_reached")


def _finish(
    on_event: CampaignEvent | None,
    work_dir: Path,
    rounds: list[RoundResult],
    stop_reason: StopReason,
) -> CampaignResult:
    """Single exit point, so every `return` reports the outcome."""
    _emit(
        on_event, "campaign_end",
        stop_reason=stop_reason, n_rounds=len(rounds),
        biased_rounds=sum(1 for r in rounds if r.plumed_dat_path is not None),
    )
    return CampaignResult(work_dir, tuple(rounds), stop_reason)


def _default_biased_factory(
    adapter: MDAdapter,
    *,
    work_dir: Path,
    seed: int,
    spec: SystemSpec,
) -> BiasedAdapterFactory:
    """Biased adapter over the *same engine* that ran the vanilla phase.

    Engine-matching is a correctness requirement, not a preference. The CV's
    atom indices are resolved against the vanilla adapter's topology, and each
    engine builds a different system from the same `SystemSpec`: `gmx solvate`
    and OpenMM's Modeller place different numbers of waters, and pdb2gmx and
    Modeller name and order hydrogens differently. Handing those indices to
    another engine's system biases whichever atoms happen to sit at those
    positions — wrong physics, no error anywhere. (This is not an off-by-one:
    the 0-based to PLUMED 1-based conversion in `plumed_writer` is correct and
    no offset can reconcile two different atom orderings.)

    Only OpenMM has a bias path today. Any other engine gets a refusal naming
    the injection point rather than a silent engine swap.
    """
    if isinstance(adapter, OpenMMAdapter):

        def openmm_factory(plumed_input: str) -> MDAdapter:
            return OpenMMAdapter(
                work_dir=work_dir,
                seed=seed,
                spec=spec,
                plumed_input=plumed_input,
            )

        return openmm_factory

    engine = type(adapter).__name__

    def unsupported(plumed_input: str) -> MDAdapter:
        raise NotImplementedError(
            f"{engine} has no metadynamics path, and the CV's atom indices were "
            f"resolved against its topology, so they are not transferable to "
            f"another engine. Pass `biased_adapter_factory=` to run the biased "
            f"phase through a {engine}-compatible adapter."
        )

    return unsupported


# HILLS and COLVAR are to a biased round what the checkpoint is to a vanilla
# one: the state needed to continue. They are snapshotted together and restored
# together.
_BIAS_STATE_FILES = (_HILLS_NAME, _COLVAR_NAME)


def _snapshot_bias_state(bias_dir: Path, rounds_dir: Path, round_index: int) -> None:
    """Copy the deposited bias alongside the round's checkpoint.

    Turning RESTART on makes the live HILLS load-bearing, which turns the
    existing mid-round crash window (D4: a crash between `save_checkpoint` and
    `append_round` leaves the round absent, so restart re-runs it) from
    "wasted time" into "wrong physics" — the re-run would deposit that round's
    hills a second time on top of the ones already on disk. Snapshotting at
    the same moment as the checkpoint means resume can restore the bias to
    exactly the point the positions correspond to.
    """
    for name in _BIAS_STATE_FILES:
        src = bias_dir / name
        if src.exists():
            shutil.copy2(src, _bias_snapshot_path(rounds_dir, round_index, name))


def _clear_bias_state(bias_dir: Path) -> None:
    """Drop the live HILLS/COLVAR so a new CV starts from zero bias.

    Hills deposited on the previous coordinate must not carry over: they
    describe a different CV and PLUMED would read them back as if they did
    not. Deleting rather than archiving is deliberate — the outgoing bias is
    already preserved at `rounds/round_NNN.hills` by `_snapshot_bias_state`,
    and delete is idempotent, so a resume that re-enters this path cannot
    accumulate half-written archives of an interrupted round.
    """
    for name in _BIAS_STATE_FILES:
        (bias_dir / name).unlink(missing_ok=True)


def _restore_bias_state(bias_dir: Path, rounds_dir: Path, round_index: int) -> None:
    """Put the bias back to its state at the end of `round_index`.

    A missing snapshot is not an error: campaigns that pivoted before
    snapshotting existed have none, and leaving the live files untouched is
    the best available behaviour there.
    """
    bias_dir.mkdir(parents=True, exist_ok=True)
    for name in _BIAS_STATE_FILES:
        snapshot = _bias_snapshot_path(rounds_dir, round_index, name)
        if snapshot.exists():
            shutil.copy2(snapshot, bias_dir / name)


def _bias_snapshot_path(rounds_dir: Path, round_index: int, name: str) -> Path:
    return rounds_dir / f"round_{round_index:03d}.{name.lower()}"


def _round_report(
    trajectory_path: Path,
    topology_path: Path,
    *,
    plumed_dat_path: Path | None,
    temperature_k: float,
    fes_dir: Path,
    min_recrossings: int = 1,
    rounds_dir: Path | None = None,
    round_index: int | None = None,
    state_thresholds: tuple[float, float] | None = None,
    observable: ObservableSpec | None = None,
) -> dict[str, Any]:
    """Diagnostic bundle for one round, chosen by phase.

    A vanilla round gets the equilibrium convergence bundle. A biased round
    gets the free-energy bundle instead — *instead*, not alongside. The
    equilibrium statistics describe an equilibrium ensemble, and a biased
    trajectory is not one: the bias drives the observable, so a long
    autocorrelation means the bias is still filling and a bimodal marginal
    means the bias worked. Emitting them on a biased round would invite the
    scientist to read them as convergence evidence, which is the category
    error `diagnostics.free_energy` exists to remove.

    HILLS and COLVAR accumulate across the whole biased phase in a single
    file, so the surface integrated here is cumulative — which is what the
    well-tempered convergence test wants — not per-round.
    """
    if plumed_dat_path is None:
        report = make_report(trajectory_path, topology_path, observable)
        report["phase"] = "vanilla"
        return report

    # Only computed when there are task states to count against: it costs a
    # trajectory load, and without thresholds nothing consumes the result.
    series = observable_name = None
    if (
        state_thresholds is not None
        and rounds_dir is not None
        and round_index is not None
    ):
        series, observable_name = _accumulated_observable(
            rounds_dir, round_index, trajectory_path, topology_path, observable
        )

    bias_dir = plumed_dat_path.parent
    report = metad_report(
        bias_dir / _HILLS_NAME,
        bias_dir / _COLVAR_NAME,
        fes_dir,
        temperature_k=temperature_k,
        min_recrossings=min_recrossings,
        observable=series,
        observable_name=observable_name,
        state_thresholds=state_thresholds,
    )
    report["phase"] = "metad"
    report["trajectory_path"] = str(trajectory_path)
    report["plumed_dat_path"] = str(plumed_dat_path)
    return report


def _accumulated_observable(
    rounds_dir: Path,
    round_index: int,
    trajectory_path: Path,
    topology_path: Path,
    observable: ObservableSpec | None = None,
) -> tuple[np.ndarray, str]:
    """The campaign observable over every biased round so far, in order.

    Cumulative, because the surface-derived count it replaces was cumulative:
    HILLS and COLVAR accumulate across the whole biased phase, so a per-round
    count would silently change what the number means.

    Each round's series is persisted next to its checkpoint rather than
    recomputed from the trajectories. The series is a few thousand floats
    (~16 KB) against a ~117 MB DCD, so concatenating the saved ones costs
    nothing while re-deriving them every round would mean re-reading the entire
    biased phase — over a gigabyte by the end of a 20 ns campaign.
    """
    traj = md.load(str(trajectory_path), top=str(topology_path))
    series, name = campaign_observable(traj, topology_path, observable)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    np.save(rounds_dir / f"round_{round_index:03d}.obs.npy", series)

    chunks: list[np.ndarray] = []
    for index in range(1, round_index + 1):
        path = rounds_dir / f"round_{index:03d}.obs.npy"
        if path.exists():
            chunks.append(np.load(path))
    return (np.concatenate(chunks) if chunks else series), name


def _refuse_premature_stop(
    decision: Decision, report: dict[str, Any], remaining_steps: int | None
) -> tuple[Decision, str | None]:
    """Convert a biased-phase `stop` into an extend while the surface is unconverged.

    The system prompt already states the rule — `fes_converged=true` → stop,
    otherwise extend — but a rule the model can reason its way around is not a
    rule. On the first CLN025 campaign the scientist read `fes_converged=false`,
    wrote a paragraph rationalising the constituent numbers, and stopped with 16
    of 20 ns unspent and the done criterion one recrossing short.

    This does not take judgement away from the scientist: it still chooses the
    CV, sizes each extension, and decides when to stop once the diagnostic
    actually reports convergence. It removes only the ability to declare victory
    against the diagnostic. The refusal is written to the hypothesis ledger
    rather than swallowed, so the next round sees that it happened.
    """
    if decision.decision != "stop":
        return decision, None
    if report.get("fes_converged") is True:
        return decision, None
    if remaining_steps is not None and remaining_steps <= 0:
        return decision, None

    note = (
        f"stop refused: the scientist chose stop but fes_converged="
        f"{report.get('fes_converged')!r} "
        f"(drift={report.get('fes_drift_kj_per_mol')}, "
        f"recrossings={report.get('recrossings')}, "
        f"required>={report.get('min_recrossings')}). Budget remains, so the "
        f"round was converted to an extend. Reason given was: {decision.reason}"
    )
    return replace(decision, decision="extend", extra_ns=decision.extra_ns or 0.5), note


def _extend_steps(
    extra_ns: float | None, max_extra_ns: float, steps_per_ns: int
) -> int:
    """Steps for an extend round, clamped to the caller's `max_extra_ns`.

    What SQLite stores is the model's raw request, so the clamp has to be
    re-applied on every read. Applying it only in the live loop made a resumed
    campaign run a longer round than the uninterrupted one would have.
    """
    return max(int(min(extra_ns or 0.5, max_extra_ns) * steps_per_ns), 1)


def _pivot_to_metad(
    proposal: MetadProposal,
    *,
    source_trajectory: Path,
    topology_path: Path,
    factory: BiasedAdapterFactory,
    plumed_dat_path: Path,
    temperature_k: float,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
) -> MDAdapter:
    """Resolve a CV proposal into a biased, started adapter.

    Renders plumed.dat from the proposal + the CV's fluctuation on
    `source_trajectory`, writes it to `plumed_dat_path` (the authoritative
    audit artifact), then builds and starts the biased adapter. PLUMED's own
    outputs (HILLS, COLVAR) are directed alongside plumed.dat.
    """
    plumed_dat_path.parent.mkdir(parents=True, exist_ok=True)
    plumed_input = _build_plumed_input(
        proposal,
        source_trajectory,
        topology_path,
        plumed_dat_path.parent,
        temperature_k=temperature_k,
        cv_upper_wall_nm=cv_upper_wall_nm,
        bias_pace=bias_pace,
        bias_factor=bias_factor,
    )
    plumed_dat_path.write_text(plumed_input)
    biased = factory(plumed_input)
    biased.prepare()
    biased.start()
    return biased


def _build_plumed_input(
    proposal: MetadProposal,
    trajectory_path: Path,
    topology_path: Path,
    output_dir: Path,
    *,
    temperature_k: float,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
) -> str:
    """Proposal → resolved CV → sized bias → rendered plumed.dat text.

    `output_dir` is where PLUMED writes HILLS and COLVAR. It has to be
    absolute and campaign-local: PLUMED resolves relative FILE= paths against
    the process working directory, so the deposited bias would otherwise land
    outside the campaign entirely.
    """
    # Loaded with coordinates, not just connectivity: an `rmsd` CV measures
    # against a reference structure, and the campaign topology is the same
    # reference the vanilla observable uses, so both phases score against one
    # fixed structure rather than two different ones.
    reference = md.load(str(topology_path))
    cv = design_cv(
        CVProposal(
            cv_type=proposal.cv_type,
            selections=tuple(proposal.selections),
            label=proposal.label,
        ),
        reference.topology,
        reference=reference,
        output_dir=output_dir,
    )
    # None means "let bias_designer decide", so its defaults stay the single
    # definition of PACE and BIASFACTOR rather than being restated here.
    overrides = {
        k: v
        for k, v in (("pace", bias_pace), ("bias_factor", bias_factor))
        if v is not None
    }
    bias = design_bias(
        cv, trajectory_path, topology_path, temperature_k=temperature_k, **overrides
    )
    wall = design_upper_wall(cv, cv_upper_wall_nm)
    return PlumedInput(
        cvs=(cv,),
        bias=bias,
        walls=(wall,) if wall is not None else (),
        output_dir=Path(output_dir).resolve(),
    ).render()


def _require_checkpoint(row: store.RoundRow) -> None:
    if row.checkpoint_path is None or not row.checkpoint_path.exists():
        raise FileNotFoundError(
            f"round {row.round_index} has no readable checkpoint; cannot resume"
        )


def _row_to_result(row: store.RoundRow) -> RoundResult:
    metad = (
        MetadProposal.from_dict(row.metad_proposal) if row.metad_proposal else None
    )
    return RoundResult(
        index=row.round_index,
        n_steps=row.n_steps,
        dcd_path=row.dcd_path,
        summary_path=row.dcd_path.with_suffix(".json"),
        report=row.report,
        decision=Decision(
            decision=cast(
                Literal["extend", "stop", "switch_to_metad"], row.decision
            ),
            reason=row.reason,
            extra_ns=row.extra_ns,
            metad_proposal=metad,
        ),
        plumed_dat_path=row.plumed_dat_path,
    )


def _compact_prior(r: RoundResult) -> dict[str, Any]:
    """Lean view of a prior round for the scientist's context — no raw report.

    Phase-keyed for the same reason the per-round report is: carrying `ess`
    and `plateau_reached` forward from a biased round would re-introduce the
    equilibrium statistics the biased report deliberately omits.
    """
    base = {
        "round_index": r.index,
        "n_steps": r.n_steps,
        "phase": r.report.get("phase", "vanilla"),
        "decision": r.decision.decision,
        "reason": r.decision.reason,
    }
    if r.plumed_dat_path is not None:
        base.update(
            # Which coordinate this round was judged on. Across a switch_cv the
            # history holds counts from two different CVs, and a bare list of
            # recrossings would invite exactly the comparison F7 was about.
            cv_label=r.report.get("cv_label"),
            fes_drift_kj_per_mol=r.report.get("fes_drift_kj_per_mol"),
            recrossings=r.report.get("recrossings"),
            # The boundaries move as the surface fills, so a count carried
            # forward without them is not comparable across rounds.
            recrossing_low=r.report.get("recrossing_low"),
            recrossing_high=r.report.get("recrossing_high"),
            fes_converged=r.report.get("fes_converged"),
        )
    else:
        base.update(
            trajectory_length_ns=r.report.get("trajectory_length_ns"),
            ess=r.report.get("ess"),
            plateau_reached=r.report.get("plateau_reached"),
        )
    return base


def _persist_round_json(
    path: Path,
    round_idx: int,
    n_steps: int,
    dcd: Path,
    report: dict[str, Any],
    decision: Decision,
    plumed_dat_path: Path | None,
) -> None:
    payload = {
        "round_index": round_idx,
        "n_steps": n_steps,
        "dcd_path": str(dcd),
        "plumed_dat_path": str(plumed_dat_path) if plumed_dat_path else None,
        "report": report,
        "decision": {
            "decision": decision.decision,
            "reason": decision.reason,
            "extra_ns": decision.extra_ns,
            "metad_proposal": (
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
