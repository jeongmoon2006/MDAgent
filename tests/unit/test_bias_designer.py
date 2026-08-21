"""bias_designer: resolved CV + trajectory → sized MetadynamicsBias.

Trajectories are built programmatically and round-tripped through disk (PDB +
DCD) so the real ``md.load`` path in ``design_bias`` is exercised — no network,
no OpenMM. The property under test is the sizing arithmetic (SIGMA = spread/3,
HEIGHT = 0.5 kT, PACE default, floor, circular spread for torsions).
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpilot.adapters.plumed_writer import DistanceCV, GyrationCV, RmsdCV, TorsionCV
from mdpilot.sampling.bias_designer import (
    _DEFAULT_BIAS_FACTOR,
    _HEIGHT_KT_FRACTION,
    _KB_KJ_PER_MOL_K,
    design_bias,
    design_upper_wall,
    sigma_floor,
    size_sigma,
)


def _write_traj(tmp_path: Path, top: md.Topology, xyz: np.ndarray) -> tuple[Path, Path]:
    """Save a trajectory to PDB (topology) + DCD (frames); return their paths."""
    traj = md.Trajectory(xyz=xyz.astype(np.float32), topology=top)
    pdb = tmp_path / "top.pdb"
    dcd = tmp_path / "traj.dcd"
    traj[0].save_pdb(str(pdb))
    traj.save_dcd(str(dcd))
    return dcd, pdb


def _two_atom_top() -> md.Topology:
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain, resSeq=1)
    top.add_atom("A", md.element.carbon, res)
    top.add_atom("B", md.element.carbon, res)
    return top


def _height() -> float:
    return _HEIGHT_KT_FRACTION * _KB_KJ_PER_MOL_K * 300.0


def test_distance_sigma_is_stddev_over_three(tmp_path: Path) -> None:
    top = _two_atom_top()
    # Atom A at origin; atom B along x at these distances (nm).
    ds = np.array([1.0, 1.2, 0.8, 1.0])
    xyz = np.zeros((len(ds), 2, 3))
    xyz[:, 1, 0] = ds
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(DistanceCV(label="d", atoms=(0, 1)), dcd, pdb)

    expected_sigma = float(np.std(ds)) / 3.0
    assert bias.cv_labels == ("d",)
    assert bias.sigma[0] == pytest.approx(expected_sigma, rel=1e-4)
    assert bias.height == pytest.approx(_height(), rel=1e-6)
    assert bias.pace == 500


def test_pinned_cv_sigma_is_floored(tmp_path: Path) -> None:
    """A CV that never moves (the exact switch-to-metad case) must not yield
    SIGMA=0 — PLUMED rejects that. It clamps to the floor."""
    top = _two_atom_top()
    xyz = np.zeros((5, 2, 3))
    xyz[:, 1, 0] = 1.0  # constant distance across all frames
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(DistanceCV(label="d", atoms=(0, 1)), dcd, pdb)

    assert bias.sigma[0] == pytest.approx(sigma_floor(DistanceCV(label="d", atoms=(0, 1))))
    assert bias.sigma_floored is True


def test_gyration_sizes_from_the_atom_subset(tmp_path: Path) -> None:
    """SIGMA for a gyration CV is std(Rg over the selected atoms)/3, computed on
    the sliced group — not the whole system."""
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("GLY", chain, resSeq=1)
    for name in ("A", "B", "C"):
        top.add_atom(name, md.element.carbon, res)
    # Three atoms breathing in/out along x over 4 frames.
    scales = np.array([1.0, 1.5, 0.5, 1.2])
    xyz = np.zeros((len(scales), 3, 3))
    for f, s in enumerate(scales):
        xyz[f, 0, 0] = -s
        xyz[f, 2, 0] = +s  # atom B stays at origin
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    cv = GyrationCV(label="rg", atoms=(0, 1, 2))
    bias = design_bias(cv, dcd, pdb)

    traj = md.load(str(dcd), top=str(pdb))
    expected_sigma = float(np.std(md.compute_rg(traj.atom_slice([0, 1, 2])))) / 3.0
    assert bias.sigma[0] == pytest.approx(expected_sigma, rel=1e-4)


def test_constant_torsion_uses_circular_spread_and_floors(tmp_path: Path) -> None:
    """A torsion held fixed has zero circular spread → floored SIGMA, and the
    torsion path (compute_dihedrals) is exercised rather than a linear stddev."""
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain, resSeq=1)
    for name in ("N", "CA", "C", "O"):
        top.add_atom(name, md.element.carbon, res)
    # A fixed non-planar geometry, identical across frames → constant dihedral.
    frame = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.1]]
    )
    xyz = np.stack([frame, frame, frame])
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(TorsionCV(label="phi", atoms=(0, 1, 2, 3)), dcd, pdb)

    assert bias.sigma[0] == pytest.approx(
        sigma_floor(TorsionCV(label="phi", atoms=(0, 1, 2, 3)))
    )
    assert bias.sigma_floored is True
    assert bias.height == pytest.approx(_height(), rel=1e-6)


def test_temperature_and_pace_overrides_flow_through(tmp_path: Path) -> None:
    top = _two_atom_top()
    xyz = np.zeros((3, 2, 3))
    xyz[:, 1, 0] = [1.0, 1.1, 0.9]
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(
        DistanceCV(label="d", atoms=(0, 1)), dcd, pdb, temperature_k=350.0, pace=200
    )

    assert bias.height == pytest.approx(_HEIGHT_KT_FRACTION * _KB_KJ_PER_MOL_K * 350.0)
    assert bias.pace == 200
    # PLUMED needs TEMP to evaluate the well-tempered factor; it has to track
    # the thermostat, not a hardcoded 300 K.
    assert bias.temperature_k == pytest.approx(350.0)


def test_designed_bias_is_well_tempered(tmp_path: Path) -> None:
    top = _two_atom_top()
    xyz = np.zeros((3, 2, 3))
    xyz[:, 1, 0] = [1.0, 1.1, 0.9]
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(DistanceCV(label="d", atoms=(0, 1)), dcd, pdb)

    assert bias.bias_factor == pytest.approx(_DEFAULT_BIAS_FACTOR)
    assert bias.bias_factor > 1.0
    assert "BIASFACTOR=" in bias.render()
    assert "TEMP=300" in bias.render()


def test_bias_factor_override_flows_through(tmp_path: Path) -> None:
    top = _two_atom_top()
    xyz = np.zeros((3, 2, 3))
    xyz[:, 1, 0] = [1.0, 1.1, 0.9]
    dcd, pdb = _write_traj(tmp_path, top, xyz)

    bias = design_bias(DistanceCV(label="d", atoms=(0, 1)), dcd, pdb, bias_factor=15.0)

    assert bias.bias_factor == pytest.approx(15.0)


# ---------- F4: the SIGMA floor ----------

def test_pinned_cv_gets_the_floor_not_thermal_jitter(tmp_path: Path) -> None:
    """The F4 regression, with the real measured number.

    Backbone Rg over 10 ps of Trp-cage gave spread/3 = 0.0018 nm. Hills that
    narrow never overlap, so the accumulated bias stays ~0, well-tempering has
    nothing to damp, and WT-MetaD degenerates into plain metaD that fills
    nothing. The floor has to bind here.
    """
    cv = GyrationCV(label="rg", atoms=(0, 1, 2))
    sigma, floored = size_sigma(cv, spread=0.0018 * 3.0)

    assert floored is True
    assert sigma == pytest.approx(sigma_floor(cv))
    assert sigma > 0.01, "a hill this narrow cannot overlap its neighbours"


def test_well_sampled_cv_keeps_its_measured_width() -> None:
    """The floor is a floor, not an override — a CV that genuinely explored a
    wide basin must not be widened to a rule of thumb."""
    cv = GyrationCV(label="rg", atoms=(0, 1, 2))
    sigma, floored = size_sigma(cv, spread=0.30)

    assert floored is False
    assert sigma == pytest.approx(0.10)


def test_floor_binds_below_the_boundary_and_releases_above() -> None:
    """Behaviour either side of the floor. Exact-equality at the boundary is
    left unspecified — `floor * 3 / 3` lands an ulp off and either answer is
    correct there."""
    cv = DistanceCV(label="d", atoms=(0, 1))
    floor = sigma_floor(cv)

    above, floored_above = size_sigma(cv, spread=floor * 3.0 * 1.05)
    below, floored_below = size_sigma(cv, spread=floor * 3.0 * 0.95)

    assert floored_above is False
    assert above == pytest.approx(floor * 1.05)
    assert floored_below is True
    assert below == pytest.approx(floor)


def test_torsion_floor_is_in_radians_not_nanometres() -> None:
    """Torsions span 2π; a 0.02 rad hill would be absurdly narrow. Sharing one
    floor across CV types would silently mis-size dihedral biases."""
    angular = sigma_floor(TorsionCV(label="phi", atoms=(0, 1, 2, 3)))
    linear = sigma_floor(DistanceCV(label="d", atoms=(0, 1)))

    assert angular > linear
    assert 0.1 <= angular <= 0.35


def test_unknown_cv_type_has_no_silent_default() -> None:
    class FakeCV:
        label = "x"

    with pytest.raises(TypeError, match="no SIGMA floor"):
        sigma_floor(FakeCV())  # type: ignore[arg-type]


# ---------- upper walls ----------


def test_upper_wall_bounds_a_length_dimensioned_cv() -> None:
    """RMSD is unbounded on the unfolded side, so well-tempered metaD drives the
    walker outward forever instead of returning. The first CLN025 campaign
    deposited 129 kJ/mol — an order of magnitude past the real folding free
    energy — and got one crossing with no return trip."""
    cv = RmsdCV(label="rmsd_ca", atoms=(1, 2, 3), reference_path=Path("/tmp/r.pdb"))
    wall = design_upper_wall(cv, 0.8)
    assert wall is not None
    assert wall.cv_label == "rmsd_ca"
    assert wall.at == 0.8
    assert "UPPER_WALLS" in wall.render() and "AT=0.8" in wall.render()


def test_no_wall_when_none_requested() -> None:
    cv = RmsdCV(label="r", atoms=(1, 2, 3), reference_path=Path("/tmp/r.pdb"))
    assert design_upper_wall(cv, None) is None


def test_no_wall_on_a_torsion() -> None:
    """A wall position in nm is meaningless on a torsion, which is already
    bounded on [-pi, pi]."""
    assert design_upper_wall(TorsionCV(label="phi", atoms=(1, 2, 3, 4)), 0.8) is None


def test_contact_count_has_its_own_sigma_floor() -> None:
    """A contact count is dimensionless, so the nanometre floors do not apply.
    Sharing the length floor would size hills for this CV a factor of 25 too
    narrow — the same class of error the torsion floor exists to prevent."""
    from mdpilot.adapters.plumed_writer import ContactsCV, DistanceCV

    contacts = ContactsCV(label="q", pairs=((0, 5), (1, 6)), r0_nm=0.75)
    assert sigma_floor(contacts) == 0.5
    assert sigma_floor(contacts) != sigma_floor(DistanceCV(label="d", atoms=(0, 1)))
