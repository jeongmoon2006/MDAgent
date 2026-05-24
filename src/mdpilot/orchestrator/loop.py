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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.diagnostics.report import make_report
from mdpilot.memory import store
from mdpilot.orchestrator.scientist import Decision, decide

_TIMESTEP_FS = 2.0  # matches the OpenMM adapter's integrator timestep
_STEPS_PER_NS = int(1_000_000.0 / _TIMESTEP_FS)  # 500_000 steps/ns at 2 fs

StopReason = Literal["scientist_said_stop", "max_rounds_reached"]


@dataclass(frozen=True)
class RoundResult:
    index: int
    n_steps: int
    dcd_path: Path
    summary_path: Path
    report: dict[str, Any]
    decision: Decision


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
) -> CampaignResult:
    """Run the closed loop, resuming from `work_dir/state.db` if it exists.

    `adapter` defaults to `OpenMMAdapter(work_dir=work_dir, seed=seed)`.
    Pass a different `MDAdapter` to run through another engine.

    `max_rounds` and `max_extra_ns` are loop-control bounds and may differ
    between invocations; the physics-bound params (seed, initial_steps,
    report_interval_steps, equilibration_steps) are locked at first init
    and a mismatch on resume raises.
    """
    work_dir = Path(work_dir)
    rounds_dir = work_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": seed,
        "initial_steps": initial_steps,
        "report_interval_steps": report_interval_steps,
        "equilibration_steps": equilibration_steps,
    }
    store.init_campaign(work_dir, config)
    prior_rows = store.list_rounds(work_dir)
    rounds: list[RoundResult] = [_row_to_result(r) for r in prior_rows]

    if rounds and rounds[-1].decision.decision == "stop":
        return CampaignResult(work_dir, tuple(rounds), "scientist_said_stop")

    if adapter is None:
        adapter = OpenMMAdapter(work_dir=work_dir, seed=seed)
    adapter.prepare()
    adapter.start()
    top_pdb = adapter.topology_path

    if prior_rows:
        last = prior_rows[-1]
        if last.checkpoint_path is None or not last.checkpoint_path.exists():
            raise FileNotFoundError(
                f"round {last.round_index} in {work_dir} has no readable "
                f"checkpoint; cannot resume"
            )
        adapter.load_checkpoint(last.checkpoint_path)
        start_round = last.round_index + 1
        n_steps = max(int((last.extra_ns or 0.5) * _STEPS_PER_NS), 1)
    else:
        if equilibration_steps > 0:
            adapter.run_steps(equilibration_steps)
        start_round = 1
        n_steps = initial_steps

    for round_idx in range(start_round, max_rounds + 1):
        dcd = rounds_dir / f"round_{round_idx:03d}.dcd"
        adapter.run_steps(
            n_steps,
            trajectory_path=dcd,
            report_interval_steps=report_interval_steps,
        )
        report = make_report(dcd, top_pdb)
        prior_summaries = [_compact_prior(r) for r in rounds]
        decision = decide(report, prior_round_summaries=prior_summaries)

        ckpt = adapter.save_checkpoint(rounds_dir / f"round_{round_idx:03d}.chk")
        summary_path = rounds_dir / f"round_{round_idx:03d}.json"
        _persist_round_json(summary_path, round_idx, n_steps, dcd, report, decision)
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
        )
        rounds.append(RoundResult(round_idx, n_steps, dcd, summary_path, report, decision))

        if decision.decision == "stop":
            return CampaignResult(work_dir, tuple(rounds), "scientist_said_stop")
        extra_ns = min(decision.extra_ns or 0.5, max_extra_ns)
        n_steps = max(int(extra_ns * _STEPS_PER_NS), 1)

    return CampaignResult(work_dir, tuple(rounds), "max_rounds_reached")


def _row_to_result(row: store.RoundRow) -> RoundResult:
    return RoundResult(
        index=row.round_index,
        n_steps=row.n_steps,
        dcd_path=row.dcd_path,
        summary_path=row.dcd_path.with_suffix(".json"),
        report=row.report,
        decision=Decision(
            decision=cast(Literal["extend", "stop"], row.decision),
            reason=row.reason,
            extra_ns=row.extra_ns,
        ),
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
    }


def _persist_round_json(
    path: Path,
    round_idx: int,
    n_steps: int,
    dcd: Path,
    report: dict[str, Any],
    decision: Decision,
) -> None:
    payload = {
        "round_index": round_idx,
        "n_steps": n_steps,
        "dcd_path": str(dcd),
        "report": report,
        "decision": {
            "decision": decision.decision,
            "reason": decision.reason,
            "extra_ns": decision.extra_ns,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
