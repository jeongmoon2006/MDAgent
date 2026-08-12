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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

import mdtraj as md
import numpy as np

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import PlumedInput
from mdpilot.diagnostics.report import make_report
from mdpilot.memory import store
from mdpilot.orchestrator.scientist import Decision, MetadProposal, decide
from mdpilot.sampling.bias_designer import design_bias
from mdpilot.sampling.cv_designer import CVProposal, design_cv

_TIMESTEP_FS = 2.0  # matches the OpenMM adapter's integrator timestep
_STEPS_PER_NS = int(1_000_000.0 / _TIMESTEP_FS)  # 500_000 steps/ns at 2 fs
_METAD_TEMPERATURE_K = 300.0  # matches the adapters' thermostat; follows SystemSpec when temperature does
_PLUMED_DAT_NAME = "plumed.dat"  # canonical bias artifact (the OpenMM adapter also writes here)

StopReason = Literal[
    "scientist_said_stop", "max_rounds_reached", "switch_to_metad_requested"
]

# Given a rendered plumed.dat string, build an adapter that runs biased MD.
BiasedAdapterFactory = Callable[[str], MDAdapter]


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
    seed: int = 42,
    report_interval_steps: int = 500,    # 1 ps/frame at 2 fs
    equilibration_steps: int = 0,
    task_expectation: str | None = None,
    biased_adapter_factory: BiasedAdapterFactory | None = None,
) -> CampaignResult:
    """Run the closed loop, resuming from `work_dir/state.db` if it exists.

    `adapter` defaults to `OpenMMAdapter(work_dir=work_dir, seed=seed)`.
    Pass a different `MDAdapter` to run through another engine.

    `max_rounds` and `max_extra_ns` are loop-control bounds and may differ
    between invocations; the physics-bound params (seed, initial_steps,
    report_interval_steps, equilibration_steps) are locked at first init
    and a mismatch on resume raises.

    `task_expectation` is free-form campaign-level guidance threaded into
    every `decide()` call — what the trajectory must accomplish, the
    characteristic timescale, the compute budget. Required for campaigns
    where the scientist may need to choose `switch_to_metad`; otherwise the
    LLM has no basis to judge "the budget cannot reach the transition."

    `biased_adapter_factory` builds the adapter used after a `switch_to_metad`
    pivot from a rendered plumed.dat string. Defaults to an `OpenMMAdapter`
    over the same `SystemSpec` with `plumed_input` set. Inject to run the
    biased phase through a different engine (or a fake, in tests).
    """
    work_dir = Path(work_dir)
    rounds_dir = work_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    plumed_dat_path = work_dir / _PLUMED_DAT_NAME

    if adapter is None:
        adapter = OpenMMAdapter(work_dir=work_dir, seed=seed)

    base_spec = adapter.spec

    if biased_adapter_factory is None:
        def biased_adapter_factory(plumed_input: str) -> MDAdapter:
            return OpenMMAdapter(
                work_dir=work_dir,
                seed=seed,
                spec=base_spec,
                plumed_input=plumed_input,
            )

    config = {
        "seed": seed,
        "initial_steps": initial_steps,
        "report_interval_steps": report_interval_steps,
        "equilibration_steps": equilibration_steps,
        "system_spec": base_spec.to_dict(),
    }
    store.init_campaign(work_dir, config)
    prior_rows = store.list_rounds(work_dir)
    rounds: list[RoundResult] = [_row_to_result(r) for r in prior_rows]

    if rounds and rounds[-1].decision.decision == "stop":
        return CampaignResult(work_dir, tuple(rounds), "scientist_said_stop")

    # Build the vanilla engine state up front. This is idempotent and cheap on
    # resume (rebuilds from cache), and it guarantees the cached System +
    # minimized state + topology exist for a biased adapter to reuse.
    adapter.prepare()
    adapter.start()

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
        )
        in_metad = True
        current_plumed_dat = plumed_dat_path
        start_round = last.round_index + 1
        n_steps = initial_steps
    elif last.plumed_dat_path is not None:
        # Mid-metaD-phase resume: rebuild the biased adapter from the persisted
        # plumed.dat and continue from that round's (biased) checkpoint.
        adapter = biased_adapter_factory(last.plumed_dat_path.read_text())
        adapter.prepare()
        adapter.start()
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        in_metad = True
        current_plumed_dat = last.plumed_dat_path
        start_round = last.round_index + 1
        n_steps = max(int((last.extra_ns or 0.5) * _STEPS_PER_NS), 1)
    else:
        # Vanilla resume.
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        start_round = last.round_index + 1
        n_steps = max(int((last.extra_ns or 0.5) * _STEPS_PER_NS), 1)

    ledger_notes: list[store.LedgerNote] = list(store.list_ledger_notes(work_dir))

    for round_idx in range(start_round, max_rounds + 1):
        dcd = rounds_dir / f"round_{round_idx:03d}{adapter.trajectory_extension}"
        adapter.run_steps(
            n_steps,
            trajectory_path=dcd,
            report_interval_steps=report_interval_steps,
        )
        report = make_report(dcd, adapter.topology_path)
        prior_summaries = [_compact_prior(r) for r in rounds]
        decision = decide(
            report,
            prior_round_summaries=prior_summaries,
            hypothesis_ledger=[f"R{n.round_index}: {n.text}" for n in ledger_notes],
            task_expectation=task_expectation,
        )

        ckpt = adapter.save_checkpoint(rounds_dir / f"round_{round_idx:03d}.chk")
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
        if decision.ledger_note:
            store.append_ledger_note(
                work_dir, round_index=round_idx, text=decision.ledger_note
            )
            ledger_notes.append(
                store.LedgerNote(round_index=round_idx, text=decision.ledger_note)
            )
        rounds.append(
            RoundResult(
                round_idx, n_steps, dcd, summary_path, report, decision, current_plumed_dat
            )
        )

        if decision.decision == "stop":
            return CampaignResult(work_dir, tuple(rounds), "scientist_said_stop")

        if decision.decision == "switch_to_metad":
            if in_metad:
                # Already biased: a second switch is out of scope. End cleanly
                # so a human can inspect rather than rebuild the bias in a loop.
                return CampaignResult(
                    work_dir, tuple(rounds), "switch_to_metad_requested"
                )
            assert decision.metad_proposal is not None  # guaranteed by the parser
            adapter = _pivot_to_metad(
                decision.metad_proposal,
                source_trajectory=dcd,
                topology_path=adapter.topology_path,
                factory=biased_adapter_factory,
                plumed_dat_path=plumed_dat_path,
            )
            in_metad = True
            current_plumed_dat = plumed_dat_path
            n_steps = initial_steps
            continue

        extra_ns = min(decision.extra_ns or 0.5, max_extra_ns)
        n_steps = max(int(extra_ns * _STEPS_PER_NS), 1)

    return CampaignResult(work_dir, tuple(rounds), "max_rounds_reached")


def _pivot_to_metad(
    proposal: MetadProposal,
    *,
    source_trajectory: Path,
    topology_path: Path,
    factory: BiasedAdapterFactory,
    plumed_dat_path: Path,
) -> MDAdapter:
    """Resolve a CV proposal into a biased, started adapter.

    Renders plumed.dat from the proposal + the CV's fluctuation on
    `source_trajectory`, writes it to `plumed_dat_path` (the authoritative
    audit artifact), then builds and starts the biased adapter.
    """
    plumed_input = _build_plumed_input(proposal, source_trajectory, topology_path)
    plumed_dat_path.parent.mkdir(parents=True, exist_ok=True)
    plumed_dat_path.write_text(plumed_input)
    biased = factory(plumed_input)
    biased.prepare()
    biased.start()
    return biased


def _build_plumed_input(
    proposal: MetadProposal, trajectory_path: Path, topology_path: Path
) -> str:
    """Proposal → resolved CV → sized bias → rendered plumed.dat text."""
    topology = md.load_topology(str(topology_path))
    cv = design_cv(
        CVProposal(
            cv_type=proposal.cv_type,
            selections=tuple(proposal.selections),
            label=proposal.label,
        ),
        topology,
    )
    bias = design_bias(
        cv, trajectory_path, topology_path, temperature_k=_METAD_TEMPERATURE_K
    )
    return PlumedInput(cvs=(cv,), bias=bias).render()


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
    """Lean view of a prior round for the scientist's context — no raw report."""
    return {
        "round_index": r.index,
        "n_steps": r.n_steps,
        "trajectory_length_ns": r.report.get("trajectory_length_ns"),
        "ess": r.report.get("ess"),
        "plateau_reached": r.report.get("plateau_reached"),
        "decision": r.decision.decision,
        "reason": r.decision.reason,
        "biased": r.plumed_dat_path is not None,
    }


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
