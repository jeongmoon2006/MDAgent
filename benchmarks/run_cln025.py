"""Run the Milestone-4 done-criterion campaign: CLN025 folding via metadynamics.

Reads `benchmarks/tasks/cln025_folding.yaml` through `mdpilot.task_file`,
which maps the tunable fields onto `run_campaign` arguments and *checks* the
ones that are declared but not yet tunable against the constants that really
govern them. Then drives the campaign through the OpenMM adapter.

Must be launched inside the conda environment, and via `micromamba run` rather
than by calling the environment's python directly — `metad_report` shells out to
`plumed sum_hills`, and invoking the interpreter by path leaves `plumed` off
PATH, so every biased round would raise `PlumedNotAvailable`:

    export MAMBA_ROOT_PREFIX=$HOME/.micromamba
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_cln025 --dry-run
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_cln025

`--dry-run` is a ~10 minute shakedown on a 0.1 ns biased budget. It proves the
pivot fires, PLUMED attaches, HILLS accumulates and the free-energy report parses
on *this* system. It does not attempt to cross the barrier — 0.1 ns cannot.

The done criterion is checked against CA-RMSD, which is *not* the biased CV: the
scientist proposes that from its own vocabulary, and `metad_report.recrossings`
counts transitions along whichever CV it chose. `verify_done_criterion` is the
separate scientific check that the CV crossing corresponds to real folding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np

from mdpilot.diagnostics.free_energy import count_recrossings
from mdpilot.orchestrator.loop import CampaignResult, run_campaign, steps_per_ns_for
from mdpilot.task_file import TaskFile, load_task_file

_TASK_FILE = Path("benchmarks/tasks/cln025_folding.yaml")


def load_task(path: Path = _TASK_FILE) -> TaskFile:
    return load_task_file(path)


def run(task: TaskFile, work_dir: Path, *, dry_run: bool) -> tuple[CampaignResult, int]:
    """Drive the campaign; return the result and the engine's steps-per-ns.

    The split is deliberate: the *task file* owns everything that defines what
    the campaign is — system, ensemble, expectation, state thresholds, wall,
    biased budget — and this runner owns only loop-control bounds, which may
    differ between a dry run and the real thing. `task.run_kwargs` merges the
    two and checks every key against `run_campaign`'s signature.

    The steps-per-ns conversion comes back with the result because the caller
    reports round lengths in nanoseconds, and it is the adapter — not this
    file — that knows the timestep.
    """
    adapter = task.build_adapter(work_dir, seed=42)
    steps_per_ns = steps_per_ns_for(adapter)

    if dry_run:
        # Short enough to fail fast, long enough that the diagnostics have
        # something to chew on. 0.05 ns vanilla, 0.1 ns of biased budget —
        # the budget override deliberately undercuts the file's 20 ns.
        overrides = dict(
            initial_steps=25_000,
            report_interval_steps=100,       # 0.2 ps/frame -> 250 frames
            max_rounds=4,
            max_biased_ns=0.1,
        )
    else:
        overrides = dict(
            initial_steps=1 * steps_per_ns,  # 1 ns vanilla, and the first biased round
            report_interval_steps=500,       # 1 ps/frame
            # The budget, not this, is meant to be the real bound. Raised from
            # 15 once `switch_cv` landed: a CV switch spends an extra round at
            # `initial_steps` and restarts the extend cadence, so a campaign
            # that revises its coordinate can reach the round cap with budget
            # unspent.
            max_rounds=20,
        )

    result = run_campaign(
        work_dir=work_dir,
        adapter=adapter,
        **task.run_kwargs(max_extra_ns=2.0, seed=42, **overrides),
    )
    return result, steps_per_ns


def verify_done_criterion(
    result: CampaignResult, task: TaskFile, steps_per_ns: int
) -> dict[str, Any]:
    """Check the campaign against the task's own state definitions, post-hoc.

    The loop judges the biased phase along the CV the scientist proposed. This
    asks the separate question the task actually poses: did the system visit
    both states the task names, and did it come back? Concatenating the biased
    rounds in order gives the whole biased trajectory, so a crossing that
    straddles a round boundary still counts.

    The thresholds come from `task.campaign["state_thresholds"]` — the same
    tuple the loop was given — rather than being re-read from the criterion, so
    the post-hoc check and the in-loop count cannot be taken against different
    bands.
    """
    criterion = task.done_criterion
    lo, hi = task.campaign["state_thresholds"]
    states = criterion["states"]

    biased = [r for r in result.rounds if r.plumed_dat_path is not None]
    pivoted = any(r.decision.decision == "switch_to_metad" for r in result.rounds)

    rmsd = _biased_rmsd_series(result, biased)
    # count_recrossings' state convention is low=-1 / high=+1, so the folded and
    # extended thresholds map onto it directly.
    recrossings = int(count_recrossings(rmsd, lo, hi)) if rmsd.size else 0

    return {
        "pivoted": pivoted,
        "states": {
            "low": {**states["low"], "reached": bool(rmsd.size and rmsd.min() < lo)},
            "high": {**states["high"], "reached": bool(rmsd.size and rmsd.max() > hi)},
        },
        "observable": task.observable_name,
        "n_biased_rounds": len(biased),
        "biased_ns": sum(r.n_steps for r in biased) / steps_per_ns,
        "observable_min": float(rmsd.min()) if rmsd.size else None,
        "observable_max": float(rmsd.max()) if rmsd.size else None,
        "recrossings": recrossings,
        "min_recrossings_required": int(criterion["min_recrossings"]),
        "passed": bool(
            pivoted and recrossings >= int(criterion["min_recrossings"])
        ),
    }


def _biased_rmsd_series(result: CampaignResult, biased: list) -> np.ndarray:
    """CA-RMSD to the campaign reference across all biased rounds, in order."""
    topology = result.work_dir / "topology.pdb"
    if not topology.exists() or not biased:
        return np.empty(0)
    reference = md.load(str(topology))
    ca = reference.topology.select("protein and name CA")
    reference = reference.atom_slice(ca)

    chunks: list[np.ndarray] = []
    for r in sorted(biased, key=lambda x: x.index):
        if not r.dcd_path.exists():
            continue
        traj = md.load(str(r.dcd_path), top=str(topology)).atom_slice(ca)
        chunks.append(md.rmsd(traj, reference, frame=0) * 10.0)  # nm -> A
    return np.concatenate(chunks) if chunks else np.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="short shakedown run")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    task = load_task()
    work_dir = args.work_dir or Path(
        "campaigns/cln025_dryrun" if args.dry_run else "campaigns/cln025_metad"
    )

    result, steps_per_ns = run(task, work_dir, dry_run=args.dry_run)

    print(f"\n=== campaign finished: {result.stop_reason} ===")
    for r in result.rounds:
        phase = "metad " if r.plumed_dat_path else "vanilla"
        ns = r.n_steps / steps_per_ns
        print(f"  round {r.index:2d} [{phase}] {ns:6.3f} ns -> {r.decision.decision}")
        print(f"      {r.decision.reason}")
        if r.decision.metad_proposal is not None:
            print(f"      CV: {r.decision.metad_proposal.to_dict()}")

    verdict = verify_done_criterion(result, task, steps_per_ns)
    print(f"\n=== done criterion ({task.observable_name}, post-hoc) ===")
    print(json.dumps(verdict, indent=2))
    (work_dir / "done_criterion.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
