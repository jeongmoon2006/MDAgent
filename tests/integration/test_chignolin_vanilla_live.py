"""Chignolin via the GROMACS adapter — validates SystemSpec end-to-end.

If this test passes, MDPilot's adapter generalization (D6 step 2) actually
works on a non-Trp-cage system, which is the prerequisite for using
chignolin as the M4 metadynamics forcing function.

Uses GROMACS rather than OpenMM purely for speed (OpenMM on CPU is slow
even for tiny systems; GROMACS does the full Trp-cage cycle in <30s and
chignolin is smaller).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import mdtraj as md
import pytest

from mdpilot.adapters.gromacs_adapter import GROMACSAdapter
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.orchestrator.loop import run_campaign

pytestmark = pytest.mark.skipif(
    shutil.which("gmx") is None or not os.getenv("ANTHROPIC_API_KEY"),
    reason="gmx not on PATH or ANTHROPIC_API_KEY unset",
)


def test_chignolin_vanilla_single_round(tmp_path: Path) -> None:
    """One short vanilla MD round on chignolin (PDB 1UAO). Verifies that the
    SystemSpec abstraction actually lets us swap the system without touching
    adapter code."""
    adapter = GROMACSAdapter(
        work_dir=tmp_path,
        seed=42,
        spec=SystemSpec(pdb_id="1UAO"),
    )
    result = run_campaign(
        work_dir=tmp_path,
        adapter=adapter,
        initial_steps=500,            # 1 ps — far from converged
        max_rounds=1,
        seed=42,
        report_interval_steps=50,
    )

    assert len(result.rounds) == 1
    round_0 = result.rounds[0]

    # Trajectory was actually produced
    assert round_0.dcd_path.exists()
    assert round_0.dcd_path.suffix == ".xtc"

    # Diagnostic report ran on a non-Trp-cage system
    report = round_0.report
    assert report["observable_name"] == "rmsd_ca_to_reference_angstrom"
    assert report["n_frames"] > 0

    # Topology has chignolin's protein atoms (10 residues, ~150 heavy + H)
    top = md.load(str(adapter.topology_path))
    n_residues = sum(1 for r in top.topology.residues if r.is_protein)
    assert n_residues == 10, f"expected 10 chignolin residues, got {n_residues}"

    # 1 ps from minimized state is nowhere near converged → scientist extends
    assert round_0.decision.decision == "extend", round_0.decision
