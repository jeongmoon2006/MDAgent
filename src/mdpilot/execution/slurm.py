"""`MDAdapter` that runs the MD in Slurm jobs instead of in this process.

Why this exists rather than `sbatch python -m mdpilot.run`: the loop calls the
Anthropic API once per round and Chestnut's compute nodes have no outbound
network (DNS does not resolve there), so the whole campaign cannot run inside
one job. The split follows the network boundary rather than a design
preference — the login node keeps the parts that need the internet and no
compute (the scientist's decision, and `prepare()`'s one-time structure
fetch), and every step of dynamics goes to a compute node.

This is a wrapper, not a second engine. It holds an `OpenMMAdapter` for the
things the Protocol asks for that are pure metadata — the spec, the timestep,
the thermostat temperature, where the topology lives — and it is the same
`OpenMMAdapter` class that runs on the compute node, rebuilt there from the
task file by `execution.worker`. Trajectories, checkpoints and PLUMED output
are all written straight into `work_dir` on the shared filesystem, so the
campaign directory a Slurm run produces is byte-identical in layout to one a
local run produces and every diagnostic reads it unchanged.

State between jobs is the OpenMM checkpoint at `<work_dir>/slurm/state.chk`.
`start()` always submits a job, even when the cache is warm: a started adapter
must leave that file holding *this* adapter's post-equilibration state, and
the pivot depends on it — a biased adapter cannot continue from the vanilla
phase's checkpoint (the added `PlumedForce` makes it a different System), so
its `start()` has to overwrite the file rather than inherit it.

The engine name recorded in the campaign config is `SlurmAdapter`, so a
campaign started on the cluster refuses to resume through a bare
`OpenMMAdapter` and vice versa. That guard reads conservatively here — the
checkpoints are ordinary OpenMM checkpoints and would in fact load — but the
right way to continue a cluster campaign is on the cluster, and a mismatch
that says so is better than one that has to be reasoned about.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.task_file import TaskFile

_STATE_NAME = "state.chk"


@dataclass(frozen=True)
class SlurmResources:
    """What to ask Slurm for, per MD job.

    `cpus` is also exported as `OPENMM_CPU_THREADS`: OpenMM's CPU platform
    otherwise sizes its thread pool from the machine's core count, which on a
    shared node is not the allocation.
    """

    partition: str
    cpus: int = 8
    time_limit: str = "24:00:00"
    conda_env: str = "mdpilot_env"
    conda_sh: Path | None = None

    def resolve_conda_sh(self) -> Path:
        """Path to `conda.sh`, which a batch shell must source before
        `conda activate` works — batch jobs do not read an interactive profile.

        Derived from the submitting process's own conda installation, because
        the login node the loop runs on is by construction the same
        installation the compute node will see over the shared filesystem.
        """
        if self.conda_sh is not None:
            return Path(self.conda_sh)
        exe = os.environ.get("CONDA_EXE")
        if not exe:
            raise RuntimeError(
                "SlurmResources: CONDA_EXE is unset, so conda.sh cannot be "
                "located. Activate the conda environment before running, or "
                "pass conda_sh= explicitly."
            )
        return Path(exe).parent.parent / "etc" / "profile.d" / "conda.sh"


class SlurmAdapter:
    """MDAdapter: OpenMM, executed through `sbatch`.

    `task` is the campaign contract the compute node rebuilds the system from;
    `work_dir` must be on a filesystem both the login and compute nodes see.
    `plumed_input`, when set, is written next to the campaign's other bias
    artifacts and attached by every subsequent job.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        task: TaskFile,
        resources: SlurmResources,
        plumed_input: str | None = None,
    ):
        self._work_dir = Path(work_dir)
        self._task = task
        self._resources = resources
        self._job_dir = self._work_dir / "slurm"
        self._state = self._job_dir / _STATE_NAME
        # Metadata only: this instance never starts a Simulation. It is the
        # same class and the same spec the worker builds, so timestep,
        # temperature and topology_path cannot drift from what actually ran.
        self._inner = OpenMMAdapter(
            work_dir=self._work_dir,
            seed=task.seed,
            spec=task.spec,
            plumed_input=plumed_input,
        )
        # Kept as its own file rather than pointing the job at the campaign's
        # plumed.dat: on a mid-metaD resume the loop hands the factory a
        # RESTART-enabled variant of that text without writing it back, so the
        # bias the job must run is not always the bias on disk.
        self._plumed_path: Path | None = None
        if plumed_input is not None:
            self._plumed_path = self._job_dir / "plumed_input.dat"
            self._job_dir.mkdir(parents=True, exist_ok=True)
            self._plumed_path.write_text(plumed_input)

    # --- Protocol metadata, delegated -------------------------------------

    @property
    def spec(self) -> SystemSpec:
        return self._inner.spec

    @property
    def timestep_fs(self) -> float:
        return self._inner.timestep_fs

    @property
    def temperature_k(self) -> float:
        return self._inner.temperature_k

    @property
    def trajectory_extension(self) -> str:
        return self._inner.trajectory_extension

    @property
    def topology_path(self) -> Path:
        return self._inner.topology_path

    # --- Protocol operations ----------------------------------------------

    def prepare(self) -> None:
        """Fetch and fix the structure — on this node, not on a compute node.

        This is the one lifecycle step that touches the network (PDBFixer
        pulls from RCSB when the spec names a PDB ID), and the one that must
        therefore stay here. It caches to `inputs/`, so the worker's own
        `prepare()` finds the file and does nothing.
        """
        self._inner.prepare()

    def start(self) -> None:
        self._submit("start", [])

    def run_steps(
        self,
        n_steps: int,
        *,
        trajectory_path: Path | None = None,
        report_interval_steps: int = 500,
    ) -> Path | None:
        argv = ["--steps", str(n_steps),
                "--report-interval", str(report_interval_steps)]
        if trajectory_path is not None:
            trajectory_path = Path(trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            argv += ["--trajectory", str(trajectory_path.resolve())]
        self._submit("run", argv)
        return trajectory_path

    def save_checkpoint(self, path: Path) -> Path:
        """Copy the job-carried state to where the loop wants it.

        The checkpoint the loop asks for and the one the jobs pass between
        themselves are the same bytes; only the path differs. Written by the
        last job to run, so there is nothing to serialize here.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._require_state(), path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SlurmAdapter: no checkpoint at {path}")
        self._job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, self._state)

    # --- Submission --------------------------------------------------------

    def _require_state(self) -> Path:
        if not self._state.exists():
            raise RuntimeError(
                f"SlurmAdapter: no engine state at {self._state}; "
                f"start() must run before checkpoints exist"
            )
        return self._state

    def _submit(self, command: str, argv: list[str]) -> None:
        """Write a batch script, submit it, and block until the job ends.

        `sbatch --wait` rather than a polling loop: the loop has nothing to do
        until the MD finishes, and Slurm already knows when that is. The login
        node process stays idle — it holds no simulation and burns no CPU.

        A job that fails takes the campaign down with it, carrying the tail of
        the job's own log. Continuing past a failed round would hand the
        scientist a truncated or absent trajectory and ask it to judge
        convergence on it.
        """
        self._job_dir.mkdir(parents=True, exist_ok=True)
        script = self._job_dir / f"{command}.sbatch"
        # Absolute: Slurm resolves a relative --output against the job's
        # working directory, which is the submission directory only by
        # default and is not something this adapter should depend on.
        log = self._job_dir.resolve() / f"{command}-%j.out"
        script.write_text(self._render(command, argv, log))
        result = subprocess.run(
            ["sbatch", "--wait", str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SlurmAdapter: {command} job failed (sbatch exit "
                f"{result.returncode})\n{result.stdout}{result.stderr}\n"
                f"{self._tail_log(command, result.stdout)}"
            )

    def _tail_log(self, command: str, sbatch_stdout: str) -> str:
        """Last lines of the failed job's own output, if it can be found.

        sbatch prints `Submitted batch job <id>`; the log name is built from
        the same id. Best-effort — a job that died before writing anything
        leaves nothing to show, and that is not itself worth an exception.
        """
        job_id = sbatch_stdout.strip().rsplit(" ", 1)[-1]
        log = self._job_dir / f"{command}-{job_id}.out"
        if not job_id.isdigit() or not log.exists():
            return f"(no job log at {self._job_dir}/{command}-*.out)"
        lines = log.read_text(errors="replace").splitlines()[-20:]
        return f"--- tail of {log} ---\n" + "\n".join(lines)

    def _render(self, command: str, argv: list[str], log: Path) -> str:
        r = self._resources
        worker = [
            "python", "-m", "mdpilot.execution.worker", command,
            "--task", str(self._task.path.resolve()),
            "--work-dir", str(self._work_dir.resolve()),
            "--state", str(self._state.resolve()),
        ]
        if self._plumed_path is not None:
            worker += ["--plumed", str(self._plumed_path.resolve())]
        worker += argv
        return "\n".join([
            "#!/bin/bash",
            f"#SBATCH --job-name=mdp-{command}",
            f"#SBATCH --partition={r.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={r.cpus}",
            f"#SBATCH --time={r.time_limit}",
            f"#SBATCH --output={log}",
            "set -euo pipefail",
            f"source {r.resolve_conda_sh()}",
            f"conda activate {r.conda_env}",
            f"export OPENMM_CPU_THREADS={r.cpus}",
            " ".join(worker),
            "",
        ])

    def biased_factory(self):
        """`biased_adapter_factory` for `run_campaign`.

        Needed explicitly: the loop's default factory recognizes
        `OpenMMAdapter` by type and refuses anything else, because a CV's atom
        indices are resolved against one engine's topology. Here the engine
        *is* OpenMM — the same class, the same spec, the same cached system —
        so the refusal would be about the submission mechanism rather than
        about the physics it exists to protect.
        """

        def factory(plumed_input: str) -> MDAdapter:
            return SlurmAdapter(
                work_dir=self._work_dir,
                task=self._task,
                resources=self._resources,
                plumed_input=plumed_input,
            )

        return factory
