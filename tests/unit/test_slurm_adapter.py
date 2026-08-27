"""The Slurm adapter is the seam between a login node and a compute node.

Nothing here submits a job. What is worth pinning is that the two sides agree:
the batch script asks the worker for exactly the operation the loop asked the
adapter for, the state that carries a simulation from one job to the next is
the checkpoint the loop already knows about, and the metadata the loop reasons
in (timestep, temperature, spec) is the task file's rather than a default that
happens to match.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from mdpilot.adapters.base import MDAdapter
from mdpilot.execution import worker
from mdpilot.execution.slurm import SlurmAdapter, SlurmResources
from mdpilot.task_file import load_task_file


def _task(tmp_path: Path, **doc):
    body = {
        "name": "t",
        "seed": 7,
        "system": {"starting_pdb": "1L2Y"},
        "integrator": {"temperature_K": 340.0, "timestep_fs": 2.0},
        **doc,
    }
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False))
    return load_task_file(p)


def _resources(**over) -> SlurmResources:
    return SlurmResources(
        partition="g_pamish",
        cpus=4,
        time_limit="02:00:00",
        conda_env="mdpilot_env",
        # Pinned so the test does not depend on the developer's own conda
        # install; the unpinned path is exercised separately.
        conda_sh=Path("/opt/conda/etc/profile.d/conda.sh"),
        **over,
    )


class _Sbatch:
    """Stands in for `sbatch --wait`, recording what it was asked to run."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.scripts: list[str] = []

    def __call__(self, argv, capture_output=False, text=False):
        assert argv[:2] == ["sbatch", "--wait"]
        self.scripts.append(Path(argv[2]).read_text())
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout="Submitted batch job 4242\n", stderr=""
        )


@pytest.fixture
def sbatch(monkeypatch):
    fake = _Sbatch()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def _adapter(tmp_path: Path, **over) -> SlurmAdapter:
    return SlurmAdapter(
        work_dir=tmp_path / "campaign",
        task=_task(tmp_path),
        resources=_resources(),
        **over,
    )


# ---------- the Protocol the loop talks to ----------

def test_satisfies_the_adapter_protocol(tmp_path: Path) -> None:
    assert isinstance(_adapter(tmp_path), MDAdapter)


def test_physics_metadata_comes_from_the_task_file(tmp_path: Path) -> None:
    """The loop sizes rounds against `timestep_fs` and hands `temperature_k`
    to PLUMED's well-tempered scaling. Both must be the file's, not the
    adapter default that happens to sit next to them."""
    adapter = _adapter(tmp_path)

    assert adapter.timestep_fs == 2.0
    assert adapter.temperature_k == 340.0
    assert adapter.spec.pdb_id == "1L2Y"
    assert adapter.trajectory_extension == ".dcd"
    assert adapter.topology_path == tmp_path / "campaign" / "topology.pdb"


# ---------- what the job is asked to do ----------

def test_start_submits_a_start_job(tmp_path: Path, sbatch) -> None:
    _adapter(tmp_path).start()

    (script,) = sbatch.scripts
    assert "mdpilot.execution.worker start" in script
    assert "--partition=g_pamish" in script
    assert "--cpus-per-task=4" in script
    assert "--time=02:00:00" in script
    # The CPU platform otherwise sizes its pool from the machine, not the
    # allocation, and would oversubscribe a shared node.
    assert "export OPENMM_CPU_THREADS=4" in script
    assert "source /opt/conda/etc/profile.d/conda.sh" in script
    assert "conda activate mdpilot_env" in script


def test_run_steps_passes_the_round_through_and_returns_its_trajectory(
    tmp_path: Path, sbatch
) -> None:
    adapter = _adapter(tmp_path)
    dcd = tmp_path / "campaign" / "rounds" / "round_001.dcd"

    returned = adapter.run_steps(12_500, trajectory_path=dcd, report_interval_steps=250)

    (script,) = sbatch.scripts
    assert "mdpilot.execution.worker run" in script
    assert "--steps 12500" in script
    assert "--report-interval 250" in script
    assert f"--trajectory {dcd.resolve()}" in script
    assert returned == dcd


def test_a_run_without_a_trajectory_asks_for_none(tmp_path: Path, sbatch) -> None:
    """The loop's optional equilibration round advances the state and records
    nothing; a `--trajectory` here would leave a stray file the diagnostics
    would later find in `rounds/`."""
    assert _adapter(tmp_path).run_steps(500) is None

    assert "--trajectory" not in sbatch.scripts[0]


def test_the_vanilla_phase_attaches_no_bias(tmp_path: Path, sbatch) -> None:
    _adapter(tmp_path).start()

    assert "--plumed" not in sbatch.scripts[0]


# ---------- state between jobs ----------

def test_checkpoints_round_trip_through_the_job_carried_state(
    tmp_path: Path, sbatch
) -> None:
    """`save_checkpoint` and `load_checkpoint` are file moves here: the job
    that ran the dynamics already wrote the bytes."""
    adapter = _adapter(tmp_path)
    state = tmp_path / "campaign" / "slurm" / "state.chk"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"round-1-state")

    saved = adapter.save_checkpoint(tmp_path / "campaign" / "rounds" / "r1.chk")
    assert saved.read_bytes() == b"round-1-state"

    state.write_bytes(b"round-2-state")
    adapter.load_checkpoint(saved)
    assert state.read_bytes() == b"round-1-state"


def test_saving_before_anything_ran_names_the_missing_state(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="start\\(\\) must run"):
        _adapter(tmp_path).save_checkpoint(tmp_path / "out.chk")


def test_loading_a_checkpoint_that_is_not_there_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _adapter(tmp_path).load_checkpoint(tmp_path / "nope.chk")


# ---------- failure ----------

def test_a_failed_job_stops_the_campaign_and_carries_its_log(
    tmp_path: Path, monkeypatch
) -> None:
    """Continuing past a failed round would hand the scientist a truncated
    trajectory and ask it to judge convergence on it."""
    fake = _Sbatch(returncode=1)
    monkeypatch.setattr(subprocess, "run", fake)
    adapter = _adapter(tmp_path)
    log = tmp_path / "campaign" / "slurm" / "run-4242.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Traceback\nRuntimeError: CUDA_ERROR_UNSUPPORTED_PTX_VERSION\n")

    with pytest.raises(RuntimeError, match="UNSUPPORTED_PTX"):
        adapter.run_steps(100)


def test_a_failure_with_no_log_still_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", _Sbatch(returncode=1))

    with pytest.raises(RuntimeError, match="no job log"):
        _adapter(tmp_path).run_steps(100)


# ---------- the biased phase ----------

def test_the_biased_factory_hands_the_job_the_bias_it_was_given(
    tmp_path: Path, sbatch
) -> None:
    """On a mid-metaD resume the loop passes a RESTART-enabled variant of
    plumed.dat without writing it back, so the job has to run the string it
    was handed rather than the file on disk."""
    biased = _adapter(tmp_path).biased_factory()("RESTART\nMETAD ...\n")
    biased.start()

    (script,) = sbatch.scripts
    plumed = tmp_path / "campaign" / "slurm" / "plumed_input.dat"
    assert f"--plumed {plumed.resolve()}" in script
    assert plumed.read_text() == "RESTART\nMETAD ...\n"


def test_the_biased_adapter_keeps_the_campaign_it_came_from(tmp_path: Path) -> None:
    """A CV's atom indices were resolved against this system; a biased adapter
    built from a different spec would bias whichever atoms sat at those
    positions."""
    vanilla = _adapter(tmp_path)
    biased = vanilla.biased_factory()("METAD ...")

    assert biased.spec == vanilla.spec
    assert biased.temperature_k == vanilla.temperature_k
    assert biased.topology_path == vanilla.topology_path


# ---------- the other side of the seam ----------

def test_the_worker_rebuilds_the_task_files_system(tmp_path: Path) -> None:
    """Not the adapter default: a compute node that reconstructs the wrong
    molecule is F12 again, with a queue in front of it."""
    task = _task(tmp_path)

    adapter = worker.build_adapter(task.path, tmp_path / "campaign", None)

    assert adapter.spec == task.spec
    assert adapter.timestep_fs == 2.0


def test_the_worker_attaches_the_bias_file_it_is_pointed_at(tmp_path: Path) -> None:
    task = _task(tmp_path)
    plumed = tmp_path / "plumed_input.dat"
    plumed.write_text("METAD ARG=d SIGMA=0.05\n")

    adapter = worker.build_adapter(task.path, tmp_path / "campaign", plumed)

    assert adapter._plumed_input == "METAD ARG=d SIGMA=0.05\n"
    assert adapter.spec == task.spec


def test_the_two_sides_seed_the_same_system(tmp_path: Path) -> None:
    """The adapter is built twice — here and on the compute node — and a seed
    resolved differently on the two sides is one campaign with two systems."""
    task = _task(tmp_path)

    assert task.seed == 7
    assert worker.build_adapter(task.path, tmp_path / "c", None)._seed == 7
    assert task.build_adapter(tmp_path / "c")._seed == 7


# ---------- conda discovery ----------

def test_conda_sh_is_derived_from_the_submitting_environment(monkeypatch) -> None:
    """A batch shell reads no interactive profile, so `conda activate` only
    works after sourcing conda.sh — and the login node's install is the one
    the compute node will see over the shared filesystem."""
    monkeypatch.setenv("CONDA_EXE", "/home/u/anaconda3/bin/conda")

    resolved = SlurmResources(partition="p").resolve_conda_sh()

    assert resolved == Path("/home/u/anaconda3/etc/profile.d/conda.sh")


def test_no_conda_in_the_environment_says_so(monkeypatch) -> None:
    monkeypatch.delenv("CONDA_EXE", raising=False)

    with pytest.raises(RuntimeError, match="CONDA_EXE"):
        SlurmResources(partition="p").resolve_conda_sh()
