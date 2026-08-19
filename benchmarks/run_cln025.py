"""Run the Milestone-4 done-criterion campaign: CLN025 folding via metadynamics.

Reads `benchmarks/tasks/cln025_folding.yaml` for the task expectation and the
done criterion, then drives `run_campaign` through the OpenMM adapter.

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
import yaml

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.diagnostics.free_energy import count_recrossings
from mdpilot.orchestrator.loop import CampaignResult, run_campaign

_TASK_FILE = Path("benchmarks/tasks/cln025_folding.yaml")
_STEPS_PER_NS = 500_000  # 2 fs timestep


def load_task(path: Path = _TASK_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def run(task: dict[str, Any], work_dir: Path, *, dry_run: bool) -> CampaignResult:
    criterion = task["done_criterion"]

    if dry_run:
        # Short enough to fail fast, long enough that the diagnostics have
        # something to chew on. 0.05 ns vanilla, 0.1 ns of biased budget.
        initial_steps, max_biased_ns, max_rounds = 25_000, 0.1, 4
        report_interval_steps = 100          # 0.2 ps/frame -> 250 frames
    else:
        initial_steps = 1 * _STEPS_PER_NS    # 1 ns vanilla, and the first biased round
        max_biased_ns = float(criterion["max_biased_ns"])
        max_rounds = 15                      # the budget, not this, is the real bound
        report_interval_steps = 500          # 1 ps/frame

    adapter = OpenMMAdapter(
        work_dir=work_dir, seed=42, spec=SystemSpec(pdb_id=task["system"]["starting_pdb"])
    )
    return run_campaign(
        work_dir=work_dir,
        adapter=adapter,
        initial_steps=initial_steps,
        report_interval_steps=report_interval_steps,
        max_rounds=max_rounds,
        max_extra_ns=2.0,
        max_biased_ns=max_biased_ns,
        min_recrossings=int(criterion["min_recrossings"]),
        task_expectation=task["task_expectation"],
        seed=42,
    )


def verify_done_criterion(
    result: CampaignResult, task: dict[str, Any]
) -> dict[str, Any]:
    """Check the campaign against the task's CA-RMSD criterion, post-hoc.

    The loop judges the biased phase along the CV the scientist proposed. This
    asks the separate question the task actually poses: did the system visit
    both the extended (CA-RMSD > 4.0 A) and native (< 1.5 A) states, and did it
    come back? Concatenating the biased rounds in order gives the whole biased
    trajectory, so a crossing that straddles a round boundary still counts.
    """
    criterion = task["done_criterion"]
    lo = float(criterion["folded_state_rmsd_angstrom"])
    hi = float(criterion["extended_state_rmsd_angstrom"])

    biased = [r for r in result.rounds if r.plumed_dat_path is not None]
    pivoted = any(r.decision.decision == "switch_to_metad" for r in result.rounds)

    rmsd = _biased_rmsd_series(result, biased)
    # count_recrossings' state convention is low=-1 / high=+1, so the folded and
    # extended thresholds map onto it directly.
    recrossings = int(count_recrossings(rmsd, lo, hi)) if rmsd.size else 0

    return {
        "pivoted": pivoted,
        "n_biased_rounds": len(biased),
        "biased_ns": sum(r.n_steps for r in biased) / _STEPS_PER_NS,
        "rmsd_min_angstrom": float(rmsd.min()) if rmsd.size else None,
        "rmsd_max_angstrom": float(rmsd.max()) if rmsd.size else None,
        "reached_folded": bool(rmsd.size and rmsd.min() < lo),
        "reached_extended": bool(rmsd.size and rmsd.max() > hi),
        "rmsd_recrossings": recrossings,
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

    result = run(task, work_dir, dry_run=args.dry_run)

    print(f"\n=== campaign finished: {result.stop_reason} ===")
    for r in result.rounds:
        phase = "metad " if r.plumed_dat_path else "vanilla"
        ns = r.n_steps / _STEPS_PER_NS
        print(f"  round {r.index:2d} [{phase}] {ns:6.3f} ns -> {r.decision.decision}")
        print(f"      {r.decision.reason}")
        if r.decision.metad_proposal is not None:
            print(f"      CV: {r.decision.metad_proposal.to_dict()}")

    verdict = verify_done_criterion(result, task)
    print("\n=== done criterion (CA-RMSD, post-hoc) ===")
    print(json.dumps(verdict, indent=2))
    (work_dir / "done_criterion.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
