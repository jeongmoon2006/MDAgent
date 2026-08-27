"""Compute-node entry point: perform one MD operation, then exit.

The campaign loop calls the Anthropic API every round. Chestnut's compute
nodes have no outbound network — DNS does not resolve there — so the loop
cannot run on them. `execution.slurm.SlurmAdapter` therefore keeps the loop
on the login node and submits *this* module as a batch job for each MD
operation the adapter Protocol defines.

Nothing crosses the boundary except files on the shared filesystem. There is
no in-memory `Simulation` to carry between rounds, so the OpenMM checkpoint —
which `OpenMMAdapter` already writes for resume — is what continues the run
from one job to the next. `start` seeds it from the cached post-equilibration
state; `run` loads it, advances, and writes it back.

Both subcommands call `prepare()` and `start()`. Both are idempotent and cheap
after the first time: `prepare()` finds the fixed PDB in `inputs/` and returns,
`start()` rebuilds the Simulation from `cache/system.xml` rather than
re-minimizing. The one thing `prepare()` would do that a compute node cannot —
fetch the structure from RCSB — has already been done on the login node before
the first job is submitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.task_file import load_task_file


def build_adapter(
    task_path: Path, work_dir: Path, plumed_path: Path | None
) -> OpenMMAdapter:
    """Reconstruct the campaign's adapter from the task file.

    The task file is the campaign contract (D7), so the compute node rebuilds
    the system from it rather than from arguments the submitter chose — the
    same reason `TaskFile.build_adapter` exists at all (F12: a caller that
    constructs an adapter by hand can drop the system spec and silently
    simulate a different molecule).

    A biased job differs only by the PLUMED input, which is a file on the
    shared filesystem rather than an argument: plumed.dat is the campaign's
    audit artifact and the job should run exactly the bias that was recorded.
    """
    task = load_task_file(task_path)
    if plumed_path is None:
        return task.build_adapter(work_dir)
    return OpenMMAdapter(
        work_dir=Path(work_dir),
        seed=task.seed,
        spec=task.spec,
        plumed_input=Path(plumed_path).read_text(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "run"))
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True,
                        help="checkpoint carrying engine state between jobs")
    parser.add_argument("--plumed", type=Path, default=None,
                        help="plumed.dat to attach; omit for the vanilla phase")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--report-interval", type=int, default=500)
    args = parser.parse_args(argv)

    adapter = build_adapter(args.task, args.work_dir, args.plumed)
    adapter.prepare()
    adapter.start()
    print(f"platform : {adapter.platform_name}", flush=True)

    if args.command == "run":
        # `start()` above rebuilt the Simulation from the cache, i.e. at the
        # post-equilibration state. The checkpoint is what makes this job a
        # continuation of the previous one rather than a restart.
        adapter.load_checkpoint(args.state)
        adapter.run_steps(
            args.steps,
            trajectory_path=args.trajectory,
            report_interval_steps=args.report_interval,
        )

    adapter.save_checkpoint(args.state)
    print(f"state    : {args.state}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
