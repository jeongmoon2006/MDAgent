"""The upper wall, measured from the simulation rather than guessed.

Two failures motivate this. F6: an unbounded coordinate with no wall lets
well-tempered metadynamics drive the walker outward forever. F11: a wall placed
beyond what the box can hold lets the solute reach its own periodic image
before the wall ever pushes back — the first CLN025 campaign ran with a
hand-chosen 0.8 nm wall in a box that could only hold 0.55.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpilot.adapters.plumed_writer import ContactsCV, PlumedInput, RmsdCV, TorsionCV
from mdpilot.sampling.bias_designer import box_limited_wall, design_upper_wall


def _stretching_traj(tmp_path: Path, n_frames: int = 60, box_nm: float = 4.0):
    """A chain that stretches steadily, in a fixed box — so CV and extent are
    tightly coupled and the box limit is exactly computable."""
    top = md.Topology()
    chain = top.add_chain()
    for i in range(6):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, res)
    xyz = np.zeros((n_frames, 6, 3), dtype=np.float32)
    for f in range(n_frames):
        xyz[f, :, 0] = np.arange(6) * (0.30 + 0.008 * f)
    traj = md.Trajectory(xyz=xyz, topology=top)
    traj.unitcell_lengths = np.tile([box_nm] * 3, (n_frames, 1))
    traj.unitcell_angles = np.tile([90.0] * 3, (n_frames, 1))
    pdb, dcd = tmp_path / "top.pdb", tmp_path / "traj.dcd"
    traj[0].save_pdb(str(pdb))
    traj.save_dcd(str(dcd))
    return dcd, pdb


def _rmsd_cv(pdb: Path, tmp_path: Path) -> RmsdCV:
    ref = md.load(str(pdb))
    from mdpilot.sampling.cv_designer import CVProposal, design_cv

    return design_cv(
        CVProposal("rmsd", ("name CA",), "rmsd_ca"), ref.topology,
        reference=ref, output_dir=tmp_path,
    )


# ---------- deriving the limit ----------

def test_the_limit_is_where_the_solute_would_reach_its_own_image(
    tmp_path: Path,
) -> None:
    dcd, pdb = _stretching_traj(tmp_path, box_nm=4.0)
    cv = _rmsd_cv(pdb, tmp_path)

    limit = box_limited_wall(cv, dcd, pdb, clearance_nm=1.0)

    assert limit is not None and limit > 0
    # Sanity: a bigger box must tolerate a larger CV value.
    big_dir = tmp_path / "big"
    big_dir.mkdir()
    bigger, _ = _stretching_traj(big_dir, box_nm=6.0)
    assert box_limited_wall(cv, bigger, pdb, clearance_nm=1.0) > limit


def test_a_bounded_cv_has_no_wall_at_all(tmp_path: Path) -> None:
    """A wall at "0.8" means nothing on a torsion or a contact count."""
    dcd, pdb = _stretching_traj(tmp_path)

    assert box_limited_wall(TorsionCV(label="t", atoms=(0, 1, 2, 3)), dcd, pdb) is None
    assert design_upper_wall(
        ContactsCV(label="q", pairs=((0, 1),), r0_nm=0.75), 0.8
    ) is None


def test_an_uninformative_trajectory_declines_rather_than_guessing(
    tmp_path: Path,
) -> None:
    """A folded run barely varies, so the extrapolation has nothing to stand on.
    A bad ceiling is worse than none — verified on a real 0.05 ns vanilla round,
    which correctly yields nothing."""
    top = md.Topology()
    chain = top.add_chain()
    for i in range(4):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, res)
    xyz = np.tile(np.array([[0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]]),
                  (40, 1, 1))
    traj = md.Trajectory(xyz=xyz.astype(np.float32), topology=top)
    traj.unitcell_lengths = np.tile([4.0] * 3, (40, 1))
    traj.unitcell_angles = np.tile([90.0] * 3, (40, 1))
    pdb, dcd = tmp_path / "t.pdb", tmp_path / "t.dcd"
    traj[0].save_pdb(str(pdb)); traj.save_dcd(str(dcd))
    cv = _rmsd_cv(pdb, tmp_path)

    assert box_limited_wall(cv, dcd, pdb) is None


# ---------- choosing the wall ----------

def test_an_explicit_wall_wins_but_is_still_measured_against_the_box(
    tmp_path: Path,
) -> None:
    """The task knows what counts as unfolded; this is not the place to
    overrule it. But a wall the box cannot honour must not pass silently."""
    dcd, pdb = _stretching_traj(tmp_path, box_nm=4.0)
    cv = _rmsd_cv(pdb, tmp_path)
    limit = box_limited_wall(cv, dcd, pdb)

    wall = design_upper_wall(cv, limit * 3, trajectory_path=dcd, topology_path=pdb)

    assert wall.at == pytest.approx(limit * 3)      # honoured
    assert wall.derived_from_box is False
    assert wall.exceeds_box_limit is True           # and flagged


def test_no_configured_wall_falls_back_to_the_measured_limit(tmp_path: Path) -> None:
    dcd, pdb = _stretching_traj(tmp_path)
    cv = _rmsd_cv(pdb, tmp_path)

    wall = design_upper_wall(cv, None, trajectory_path=dcd, topology_path=pdb)

    assert wall is not None
    assert wall.derived_from_box is True
    assert wall.exceeds_box_limit is False
    assert wall.at == pytest.approx(box_limited_wall(cv, dcd, pdb))


def test_without_a_trajectory_the_old_behaviour_is_unchanged(tmp_path: Path) -> None:
    """`design_upper_wall(cv, 0.8)` must still work for callers that have no
    trajectory to measure from."""
    dcd, pdb = _stretching_traj(tmp_path)
    cv = _rmsd_cv(pdb, tmp_path)

    wall = design_upper_wall(cv, 0.8)

    assert wall.at == pytest.approx(0.8)
    assert wall.box_limit_nm is None and wall.exceeds_box_limit is False


# ---------- what reaches the audit artifact ----------

def test_plumed_dat_records_why_the_wall_is_where_it_is(tmp_path: Path) -> None:
    dcd, pdb = _stretching_traj(tmp_path)
    cv = _rmsd_cv(pdb, tmp_path)
    from mdpilot.adapters.plumed_writer import MetadynamicsBias

    bias = MetadynamicsBias(cv_labels=("rmsd_ca",), sigma=(0.05,), height=1.2, pace=500)
    derived = design_upper_wall(cv, None, trajectory_path=dcd, topology_path=pdb)
    too_far = design_upper_wall(
        cv, derived.at * 4, trajectory_path=dcd, topology_path=pdb
    )

    text = PlumedInput(cvs=(cv,), bias=bias, walls=(derived,),
                       output_dir=tmp_path.resolve()).render()
    assert "box limit measured from the source trajectory" in text

    text = PlumedInput(cvs=(cv,), bias=bias, walls=(too_far,),
                       output_dir=tmp_path.resolve()).render()
    assert "WARNING" in text and "periodic image" in text
