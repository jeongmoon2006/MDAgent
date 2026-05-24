"""GROMACS adapter end-to-end: setup → run → checkpoint → resume.

Live test against a real `gmx` binary and a real Trp-cage system. Skipped
when `gmx` is not on PATH. Single test by design — Trp-cage setup
(pdb2gmx → solvate → genion → minimize) is the dominant wall-time cost
and we don't want to pay it three times.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mdtraj as md
import pytest

from mdpilot.adapters.gromacs_adapter import GROMACSAdapter

pytestmark = pytest.mark.skipif(
    shutil.which("gmx") is None,
    reason="gmx binary not on PATH",
)


def test_setup_run_checkpoint_resume(tmp_path: Path) -> None:
    adapter = GROMACSAdapter(work_dir=tmp_path, seed=42)

    # Setup
    adapter.prepare()
    adapter.start()
    assert adapter.trajectory_extension == ".xtc"
    assert adapter.topology_path.exists()
    top = md.load(str(adapter.topology_path))
    # Trp-cage (20 residues) + waters + ions
    n_protein = len(top.topology.select("protein"))
    assert n_protein > 100  # Trp-cage has ~300 atoms heavy + hydrogens
    assert top.n_atoms > n_protein  # solvent present

    rounds_dir = tmp_path / "rounds"
    rounds_dir.mkdir()

    # First MD round — cold start, generates velocities
    xtc1 = rounds_dir / "round_001.xtc"
    adapter.run_steps(500, trajectory_path=xtc1, report_interval_steps=100)
    assert xtc1.exists()
    traj1 = md.load(str(xtc1), top=str(adapter.topology_path))
    assert traj1.n_frames >= 5  # 500 / 100 = 5 frames, maybe ±1 for initial frame
    assert traj1.n_atoms == top.n_atoms

    # Save checkpoint, advance further
    cpt = tmp_path / "checkpoint.cpt"
    adapter.save_checkpoint(cpt)
    assert cpt.exists() and cpt.stat().st_size > 0

    xtc2 = rounds_dir / "round_002.xtc"
    adapter.run_steps(500, trajectory_path=xtc2, report_interval_steps=100)
    traj2 = md.load(str(xtc2), top=str(adapter.topology_path))
    assert traj2.n_frames == traj1.n_frames

    # Resume from the saved checkpoint and re-run the same 500 steps. We do
    # *not* require bit-equality with xtc2 — GROMACS mdrun is not bit-
    # reproducible under default settings (OpenMP force-summation order +
    # XTC ~pm quantization). What we require is that resume produces a
    # physically-equivalent trajectory: the protein CA atoms in the
    # resumed trajectory should track the unresumed continuation closely.
    adapter.load_checkpoint(cpt)
    xtc2_resumed = rounds_dir / "round_002_resumed.xtc"
    adapter.run_steps(500, trajectory_path=xtc2_resumed, report_interval_steps=100)
    traj_resumed = md.load(str(xtc2_resumed), top=str(adapter.topology_path))
    assert traj_resumed.n_frames == traj2.n_frames
    assert traj_resumed.n_atoms == traj2.n_atoms

    ca = traj2.topology.select("protein and name CA")
    assert ca.size > 0
    ca_rmsd_nm = md.rmsd(traj_resumed.atom_slice(ca), traj2.atom_slice(ca))
    # 100 steps (0.2 ps) of Langevin from the same checkpoint with OpenMP
    # non-determinism gives CA-RMSD well under 1 Å. Tight enough to catch
    # a real resume bug (e.g., loading the wrong cpt → totally different
    # trajectory → RMSD several Å), loose enough to tolerate thread noise.
    assert (ca_rmsd_nm < 0.1).all(), (
        f"resumed CA-RMSD vs continuous exceeded 1 Å: {ca_rmsd_nm}"
    )
