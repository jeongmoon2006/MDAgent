"""The task file is the one artifact a user — or a setup agent — writes.

These pin the property that makes it worth generating: a task file cannot
silently disagree with the code. Fields that are tunable are mapped, fields
that are not are checked against the constant that governs them, and anything
unrecognised raises rather than being dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdpilot.adapters.system_spec import Ensemble
from mdpilot.task_file import load_task_file

_TASKS = Path("benchmarks/tasks")


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def _minimal(**system) -> dict:
    return {"name": "t", "system": {"starting_pdb": "1L2Y", **system}}


# ---------- the real files ----------

def test_both_shipped_task_files_load() -> None:
    """They were decorative until now; nothing ever parsed trpcage's, which is
    how its `diagnostics:` block stayed invalid YAML."""
    for path in sorted(_TASKS.glob("*.yaml")):
        assert load_task_file(path).name


def test_cln025_produces_the_kwargs_the_runner_hand_assembled() -> None:
    """Drop-in proof: the file owns exactly the campaign half, and the runner
    keeps the loop-control bounds (max_rounds, initial_steps, seed)."""
    task = load_task_file(_TASKS / "cln025_folding.yaml")

    assert task.spec.pdb_id == "5AWL"
    assert task.campaign == {
        "task_expectation": task.campaign["task_expectation"],
        "cv_upper_wall_nm": 0.8,
        "min_recrossings": 2,
        "max_biased_ns": 20.0,
        "state_thresholds": (1.5, 4.0),
    }
    # (folded, extended), the order run_campaign refuses to see inverted.
    assert task.campaign["state_thresholds"][0] < task.campaign["state_thresholds"][1]


def test_a_pure_convergence_task_yields_no_campaign_overrides() -> None:
    """No task_expectation and no done_criterion, so the loop's own defaults
    stand and `switch_to_metad` stays out of the action space."""
    task = load_task_file(_TASKS / "trpcage_convergence.yaml")

    assert task.campaign == {}
    assert task.done_criterion == {}


# ---------- mapped fields ----------

def test_the_integrator_block_reaches_the_ensemble(tmp_path: Path) -> None:
    doc = _minimal()
    doc["integrator"] = {"temperature_K": 240.0, "timestep_fs": 1.0}

    spec = load_task_file(_write(tmp_path, doc)).spec

    assert spec.ensemble == Ensemble(temperature_k=240.0, timestep_fs=1.0)
    assert "ensemble" in spec.to_dict()   # so the resume guard covers it


def test_an_ensemble_the_adapters_cannot_integrate_is_refused(tmp_path: Path) -> None:
    doc = _minimal()
    doc["integrator"] = {"timestep_fs": 4.0}

    with pytest.raises(ValueError, match="hydrogen mass"):
        load_task_file(_write(tmp_path, doc))


# ---------- declared-but-not-tunable ----------

def test_a_declared_value_that_disagrees_with_the_code_raises(tmp_path: Path) -> None:
    """The whole point. `padding_nm: 1.5` used to be documentation the adapter
    was free to contradict, and it ran at 1.0 with nothing said."""
    with pytest.raises(ValueError, match="not tunable yet"):
        load_task_file(_write(tmp_path, _minimal(padding_nm=1.5)))


def test_a_declared_value_that_agrees_is_accepted(tmp_path: Path) -> None:
    from mdpilot.adapters.openmm_adapter import _PADDING_NM

    assert load_task_file(_write(tmp_path, _minimal(padding_nm=_PADDING_NM)))


def test_the_verified_table_is_sourced_from_the_modules_that_own_the_values() -> None:
    """Retyped constants drift. These must be the live ones."""
    from mdpilot.adapters.openmm_adapter import _FORCEFIELD_FILES, _SALT_M
    from mdpilot.diagnostics.report import OBSERVABLE_NAME
    from mdpilot.task_file import _VERIFIED

    assert _VERIFIED[("system", "forcefield")] == list(_FORCEFIELD_FILES)
    assert _VERIFIED[("system", "ionic_strength_M")] == _SALT_M
    assert _VERIFIED[("observable", "name")] == OBSERVABLE_NAME
    assert _VERIFIED[("diagnostics", "target_ess")] == 50.0


# ---------- unknown keys: how a generated file's typos surface ----------

def test_an_unknown_key_raises_rather_than_being_dropped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown system key"):
        load_task_file(_write(tmp_path, _minimal(temperature_K=300.0)))  # wrong block

    doc = _minimal()
    doc["smapling"] = {"cv_upper_wall_nm": 0.8}   # plausible LLM typo
    with pytest.raises(ValueError, match="unknown <top level> key"):
        load_task_file(_write(tmp_path, doc))


def test_run_kwargs_rejects_an_override_run_campaign_would_not_take() -> None:
    task = load_task_file(_TASKS / "cln025_folding.yaml")

    assert task.run_kwargs(max_rounds=3)["max_rounds"] == 3
    with pytest.raises(ValueError, match="not run_campaign parameters"):
        task.run_kwargs(max_round=3)


def test_the_file_hashes_itself_for_provenance() -> None:
    import hashlib

    path = _TASKS / "cln025_folding.yaml"
    assert load_task_file(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
