"""Resume short-circuit: if the last persisted round said 'stop', the loop
returns immediately without rebuilding the OpenMM simulation.

This is the only part of resume that's testable without a real OpenMM
run; the round-1→round-2 case lives in tests/integration/test_resume_live.py.
"""

from __future__ import annotations

from pathlib import Path

from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.memory import store
from mdpilot.orchestrator.loop import run_campaign


def _config_kwargs() -> dict:
    return dict(
        initial_steps=5_000,
        seed=42,
        report_interval_steps=500,
        equilibration_steps=0,
    )


def _full_config(cfg: dict) -> dict:
    """The full config dict that run_campaign will compute and write."""
    return {
        "seed": cfg["seed"],
        "initial_steps": cfg["initial_steps"],
        "report_interval_steps": cfg["report_interval_steps"],
        "equilibration_steps": cfg["equilibration_steps"],
        "system_spec": SystemSpec.trpcage().to_dict(),
    }


def test_resume_with_prior_stop_returns_without_running_openmm(tmp_path: Path) -> None:
    cfg = _config_kwargs()
    store.init_campaign(tmp_path, _full_config(cfg))
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=5_000,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report={
            "trajectory_length_ns": 0.01,
            "ess": 120.0,
            "plateau_reached": True,
            "well_sampled": True,
        },
        decision="stop",
        reason="converged on prior run",
        extra_ns=None,
    )

    result = run_campaign(work_dir=tmp_path, max_rounds=5, **cfg)

    assert result.stop_reason == "scientist_said_stop"
    assert len(result.rounds) == 1
    assert result.rounds[0].decision.decision == "stop"
    assert result.rounds[0].decision.reason == "converged on prior run"
    # Nothing OpenMM-side ran: no PDB cache, no topology.
    assert not (tmp_path / "inputs").exists()
    assert not (tmp_path / "topology.pdb").exists()


def test_config_mismatch_on_resume_is_rejected(tmp_path: Path) -> None:
    import pytest

    cfg = _config_kwargs()
    store.init_campaign(tmp_path, _full_config(cfg))
    # Now invoke run_campaign with a different seed — must raise before
    # any OpenMM setup.
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=tmp_path, max_rounds=1, **{**cfg, "seed": 99})
    assert not (tmp_path / "inputs").exists()


def test_system_spec_mismatch_on_resume_is_rejected(tmp_path: Path) -> None:
    """Resuming a Trp-cage campaign with a chignolin spec must raise before
    any engine setup — the SystemSpec is part of the locked-in config."""
    import pytest

    from mdpilot.adapters.openmm_adapter import OpenMMAdapter

    cfg = _config_kwargs()
    store.init_campaign(tmp_path, _full_config(cfg))  # Trp-cage spec persisted

    chignolin_adapter = OpenMMAdapter(
        work_dir=tmp_path, seed=cfg["seed"], spec=SystemSpec(pdb_id="1UAO")
    )
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=tmp_path, max_rounds=1, adapter=chignolin_adapter, **cfg)
    assert not (tmp_path / "inputs").exists()
