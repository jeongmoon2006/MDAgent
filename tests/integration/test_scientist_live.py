"""Live Anthropic API integration test for scientist.decide().

Skipped automatically when ANTHROPIC_API_KEY is unset (so unit-only `pytest
tests/unit/` runs are unaffected). Costs a few cents per run on Haiku 4.5.
"""

from __future__ import annotations

import os

import pytest

from mdpilot.orchestrator.scientist import decide

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set in env",
)


def _under_converged_report() -> dict:
    return {
        "trajectory_path": "/tmp/under_converged.dcd",
        "topology_path": "/tmp/topology.pdb",
        "n_frames": 50,
        "frame_dt_ps": 1.0,
        "trajectory_length_ns": 0.05,
        "observable_name": "rmsd_ca_to_reference_angstrom",
        "mean": 1.42,
        "sem_blocked": 0.31,
        "sem_naive": 0.07,
        "plateau_reached": False,
        "statistical_inefficiency_block": 19.6,
        "statistical_inefficiency_autocorr": 18.4,
        "tau_int_frames": 9.2,
        "ess": 5.4,
        "well_sampled": False,
    }


def _converged_report() -> dict:
    return {
        "trajectory_path": "/tmp/converged.dcd",
        "topology_path": "/tmp/topology.pdb",
        "n_frames": 5000,
        "frame_dt_ps": 1.0,
        "trajectory_length_ns": 5.0,
        "observable_name": "rmsd_ca_to_reference_angstrom",
        "mean": 1.18,
        "sem_blocked": 0.012,
        "sem_naive": 0.009,
        "plateau_reached": True,
        "statistical_inefficiency_block": 1.78,
        "statistical_inefficiency_autocorr": 1.71,
        "tau_int_frames": 0.85,
        "ess": 2924.0,
        "well_sampled": True,
    }


def test_under_converged_report_yields_extend() -> None:
    result = decide(_under_converged_report())
    assert result.decision == "extend", result
    assert result.extra_ns is not None and result.extra_ns > 0
    assert result.reason  # non-empty


def test_converged_report_yields_stop() -> None:
    result = decide(_converged_report())
    assert result.decision == "stop", result
    assert result.extra_ns is None
    assert result.reason
