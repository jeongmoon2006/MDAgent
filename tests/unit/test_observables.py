"""The campaign observable: what every round is judged on.

It was hardcoded to CA-RMSD, which made non-protein campaigns impossible at
round one. It is now a declared collective variable, computed through the same
engine that sizes the bias — these pin that generality and that shared engine.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpilot.observables import ObservableSpec, campaign_observable


def _peptide(n_res: int = 6, n_frames: int = 20) -> md.Trajectory:
    """A CA-only chain that stretches steadily along x."""
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, res)
    xyz = np.zeros((n_frames, n_res, 3), dtype=np.float32)
    for f in range(n_frames):
        spacing = 0.38 + 0.02 * f
        xyz[f, :, 0] = np.arange(n_res) * spacing
    return md.Trajectory(xyz=xyz, topology=top)


@pytest.fixture()
def system(tmp_path: Path) -> tuple[md.Trajectory, Path]:
    traj = _peptide()
    top_path = tmp_path / "top.pdb"
    traj[0].save_pdb(str(top_path))
    return traj, top_path


# ---------- the default is unchanged behaviour ----------

def test_the_default_observable_is_the_m1_one() -> None:
    spec = ObservableSpec.ca_rmsd_angstrom()

    assert spec.cv_type == "rmsd"
    assert spec.selections == ("protein and name CA",)
    assert spec.name == "rmsd_ca_to_reference_angstrom"
    assert spec.scale == 10.0            # mdtraj nm -> Angstrom
    assert ObservableSpec.from_dict(spec.to_dict()) == spec


def test_omitting_the_spec_gives_the_default(system) -> None:
    traj, top_path = system

    a, name_a = campaign_observable(traj, top_path)
    b, name_b = campaign_observable(traj, top_path, ObservableSpec.ca_rmsd_angstrom())

    assert name_a == name_b
    np.testing.assert_allclose(a, b)


def test_rmsd_is_measured_against_the_campaign_topology_not_frame_zero(
    system,
) -> None:
    """A per-round reference would make every round a different observable, so
    `ess` and `plateau_reached` would be compared across incomparable series."""
    traj, top_path = system

    first_half, _ = campaign_observable(traj[:10], top_path)
    second_half, _ = campaign_observable(traj[10:], top_path)

    # The chain only stretches, so the later window must sit further out. If
    # each window were scored against its own frame 0 both would start at ~0.
    assert second_half.min() > first_half.max()
    assert first_half[0] == pytest.approx(0.0, abs=1e-4)


# ---------- generality ----------

def test_a_distance_observable_computes(system) -> None:
    traj, top_path = system
    spec = ObservableSpec(
        cv_type="distance",
        selections=("resSeq 1 and name CA", "resSeq 6 and name CA"),
        name="termini_nm",
    )

    series, name = campaign_observable(traj, top_path, spec)

    assert name == "termini_nm"
    assert series.size == traj.n_frames
    assert np.all(np.diff(series) > 0)          # the chain stretches monotonically


def test_scale_converts_the_unit(system) -> None:
    traj, top_path = system
    base = ObservableSpec(
        cv_type="gyration", selections=("name CA",), name="rg_nm"
    )
    scaled = ObservableSpec(
        cv_type="gyration", selections=("name CA",), name="rg_A", scale=10.0
    )

    nm, _ = campaign_observable(traj, top_path, base)
    ang, _ = campaign_observable(traj, top_path, scaled)

    np.testing.assert_allclose(ang, nm * 10.0, rtol=1e-6)


def test_the_observable_and_the_bias_use_one_computation(system) -> None:
    """`bias_designer.cv_series` sizes SIGMA from a coordinate and
    `observables` judges the campaign on a coordinate. Two implementations
    would disagree on a live campaign with nothing to catch it."""
    from mdpilot.sampling.bias_designer import cv_series
    from mdpilot.sampling.cv_designer import CVProposal, design_cv

    traj, top_path = system
    selections = ("resSeq 1 and name CA", "resSeq 6 and name CA")

    via_observable, _ = campaign_observable(
        traj, top_path,
        ObservableSpec(cv_type="distance", selections=selections, name="d"),
    )
    cv = design_cv(
        CVProposal(cv_type="distance", selections=selections, label="d"),
        traj.topology,
    )
    np.testing.assert_allclose(via_observable, cv_series(cv, traj))


# ---------- refusals ----------

def test_an_unscoreable_selection_is_refused(system) -> None:
    """The observable is what every round is judged on, so an empty selection
    is a campaign that cannot be scored — it must fail loudly, at round one."""
    traj, top_path = system
    spec = ObservableSpec(
        cv_type="rmsd", selections=("resname LIG",), name="nothing"
    )

    with pytest.raises(ValueError, match="cannot be scored"):
        campaign_observable(traj, top_path, spec)


def test_a_malformed_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one selection"):
        ObservableSpec(cv_type="rmsd", selections=(), name="x")
    with pytest.raises(ValueError, match="name is required"):
        ObservableSpec(cv_type="rmsd", selections=("name CA",), name="")
    with pytest.raises(ValueError, match="scale must be positive"):
        ObservableSpec(cv_type="rmsd", selections=("name CA",), name="x", scale=0.0)


def test_wrong_arity_is_refused_by_the_cv_layer(system) -> None:
    """Arity rules live in `cv_designer` and are not restated here."""
    traj, top_path = system
    spec = ObservableSpec(
        cv_type="distance", selections=("name CA",), name="d"
    )

    with pytest.raises(ValueError, match="distance requires 2 selections"):
        campaign_observable(traj, top_path, spec)
