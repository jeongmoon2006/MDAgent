"""Run any task file as a campaign, headless.

The counterpart to `app.py`. The Streamlit surface is for watching a campaign
decide; this is for leaving one running — on a workstation overnight, or on a
server over SSH — where a browser session is the wrong thing to depend on.

    python -m mdpilot.run benchmarks/tasks/cln025_contacts.yaml campaigns/run1

Resumable by construction: `run_campaign` reads completed rounds from
`<work_dir>/state.db` and continues from the last checkpoint, so re-running the
same command after an interruption picks up where it stopped. The loop-control
bounds below may differ between invocations; everything that defines what the
campaign *is* comes from the task file and is refused if it changes.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mdpilot.execution.slurm import SlurmAdapter, SlurmResources
from mdpilot.orchestrator.loop import run_campaign, steps_per_ns_for
from mdpilot.task_file import load_task_file

_INTERESTING = {
    "campaign_start", "preflight_ok", "round_start", "decision",
    "override", "pivot", "campaign_end",
}


def _log(name: str, payload: dict[str, Any]) -> None:
    """One line per event, flushed — so `tail -f` on a redirected log works."""
    if name not in _INTERESTING:
        return
    stamp = datetime.now().strftime("%H:%M:%S")
    body = "  ".join(
        f"{k}={str(v)[:160]}" for k, v in payload.items() if k != "report"
    )
    print(f"[{stamp}] {name:<15} {body}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_file", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--opening-ns", type=float, default=1.0,
                        help="length of round 1, and of the first round after a "
                             "pivot or CV switch. Locked at first run.")
    parser.add_argument("--max-extension-ns", type=float, default=2.0,
                        help="ceiling on a single extend round")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--biased-cap-ns", type=float, default=None,
                        help="cumulative metadynamics budget; replaces the task "
                             "file's own max_biased_ns. Defaults to the file's.")
    parser.add_argument("--frame-ps", type=float, default=1.0,
                        help="trajectory sampling interval. Locked at first run.")
    parser.add_argument("--slurm", metavar="PARTITION", default=None,
                        help="run the MD as Slurm jobs on this partition, "
                             "keeping the decision loop in this process. For "
                             "clusters whose compute nodes have no outbound "
                             "network, which is where the API call would fail.")
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument("--slurm-time", default="24:00:00",
                        help="walltime per MD job, not per campaign")
    parser.add_argument("--slurm-env", default="mdpilot_env",
                        help="conda environment to activate in the job")
    args = parser.parse_args(argv)

    task = load_task_file(args.task_file)
    extra: dict[str, Any] = {}
    if args.slurm is None:
        adapter = task.build_adapter(args.work_dir)
    else:
        adapter = SlurmAdapter(
            work_dir=args.work_dir,
            task=task,
            resources=SlurmResources(
                partition=args.slurm,
                cpus=args.slurm_cpus,
                time_limit=args.slurm_time,
                conda_env=args.slurm_env,
            ),
        )
        # The loop's default factory keys off the adapter's type and would
        # refuse the pivot; see SlurmAdapter.biased_factory.
        extra["biased_adapter_factory"] = adapter.biased_factory()
    steps_per_ns = steps_per_ns_for(adapter)

    overrides: dict[str, Any] = {
        "initial_steps": max(int(args.opening_ns * steps_per_ns), 1),
        "report_interval_steps": max(int(args.frame_ps * steps_per_ns / 1000), 1),
        "max_rounds": args.max_rounds,
        "max_extra_ns": args.max_extension_ns,
    }
    if args.biased_cap_ns is not None:
        overrides["max_biased_ns"] = args.biased_cap_ns

    print(f"task     : {task.name}  (sha {task.sha256[:12]})", flush=True)
    print(f"work dir : {args.work_dir}", flush=True)
    result = run_campaign(
        work_dir=args.work_dir, adapter=adapter, on_event=_log,
        **extra, **task.run_kwargs(**overrides)
    )
    print(f"\nfinished : {result.stop_reason}  ({len(result.rounds)} rounds)", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
