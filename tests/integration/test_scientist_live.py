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


def _trapped_biased_report() -> dict:
    """The real numbers from `campaigns/ui_campaign_chignolin` at round 5.

    The walker left the folded state and sat below 0.3 native contacts for two
    rounds while the deposited bias grew past 130 kJ/mol — an order of
    magnitude beyond any real folding free energy. A contact count maps every
    disordered conformation onto roughly the same value, so the bias fills one
    degenerate bin and cannot lead the chain back.
    """
    return {
        "phase": "metad",
        "cv_label": "native_contacts_fraction",
        "fes_drift_kj_per_mol": 36.7,
        "fes_depth_kj_per_mol": 131.0,
        "barrier_kj_per_mol": 118.0,
        "n_basins_fes": 1,
        "n_fes_estimates": 40,
        "recrossings": 1,
        "min_recrossings": 2,
        "barrier_crossed": True,
        "fes_converged": False,
        "recrossing_basis": "task_states",
        "recrossing_observable": "native_contacts_fraction",
        "recrossing_low": 0.3,
        "recrossing_high": 0.7,
        "cv_min": 0.39,
        "cv_max": 10.01,
        "cv_start": 9.85,
        "observable_min_this_round": 0.030,
        "observable_max_this_round": 0.126,
        "confined_to_state": "low",
        "rounds_confined": 2,
        "cv_switches_used": 0,
        "cv_switches_remaining": 1,
    }


def test_scientist_escapes_a_contact_space_trap_with_switch_cv() -> None:
    """The escape hatch, end to end through the real model.

    A trapped campaign cannot reach its own criterion by extending: the surface
    deepens without bound and the walker never returns. The right action is to
    revise the coordinate while an allowance remains.
    """
    decision = decide(
        _trapped_biased_report(),
        phase="metad",
        allow_cv_switch=True,
        task_expectation=(
            "Sample folding and unfolding of chignolin, connecting the native "
            "state (native contact fraction > 0.7) and the unfolded state "
            "(< 0.3), with at least 2 transitions. Budget 20 ns biased."
        ),
        prior_round_summaries=[
            {"round_index": 3, "phase": "metad", "decision": "extend",
             "fes_depth_kj_per_mol": 92.0, "recrossings": 1,
             "observable_min_this_round": 0.038, "observable_max_this_round": 0.436},
            {"round_index": 4, "phase": "metad", "decision": "extend",
             "fes_depth_kj_per_mol": 116.0, "recrossings": 1,
             "observable_min_this_round": 0.030, "observable_max_this_round": 0.126},
        ],
    )

    assert decision.decision == "switch_cv", decision.reason
    assert decision.metad_proposal is not None
    # Anything that still separates disordered structures a contact count
    # cannot — not another contact count on the same atoms.
    assert decision.metad_proposal.cv_type in {"rmsd", "gyration", "distance", "torsion"}


def test_a_healthy_biased_round_is_not_mistaken_for_a_trap() -> None:
    """The other half: an unconverged-but-moving surface must be extended, not
    thrown away. A switch spends compute that is never refunded."""
    healthy = _trapped_biased_report()
    healthy.update(
        fes_depth_kj_per_mol=18.0, fes_drift_kj_per_mol=6.0,
        observable_min_this_round=0.12, observable_max_this_round=0.82,
        confined_to_state=None, rounds_confined=0, n_basins_fes=2,
    )

    decision = decide(
        healthy, phase="metad", allow_cv_switch=True,
        task_expectation="Sample folding and unfolding of chignolin; 2 transitions.",
    )

    assert decision.decision == "extend", decision.reason
