"""Rendered CVs and bias inputs against the real PLUMED parser.

The unit tests check that `cv_designer` resolves selections to the right atom
indices and that `plumed_writer` formats them as expected. What they cannot
check is whether PLUMED *accepts* the result: the continuation syntax, the
switching-function block, and the 1-based serial convention are all things the
binary validates and a string comparison does not.

`plumed driver` is used rather than a biased MD run, so these stay fast — no
integrator, no forces, just parse-and-evaluate on a structure.

Trajectories are fed as DCD with an explicit `--natoms`. The `--mf_pdb` path
looks like it works and does not: it parses, runs, and reports every pairwise
distance as zero, so a contact map comes back saturated at one contact per
pair. That failure is invisible to an "is it close to the number of pairs"
assertion, which is precisely why the cross-check against a Python evaluation
below is the test that carries the weight.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpilot.diagnostics.free_energy import plumed_available
from mdpilot.sampling.bias_designer import _cv_series
from mdpilot.sampling.cv_designer import CVProposal, design_cv

_FIXED_PDB = Path("benchmarks/data/trpcage/1L2Y_fixed.pdb")

pytestmark = [
    pytest.mark.skipif(not plumed_available(), reason="no PLUMED runtime on PATH"),
    pytest.mark.skipif(not _FIXED_PDB.exists(), reason=f"{_FIXED_PDB} missing"),
]


def _drive(cv, traj: md.Trajectory, cwd: Path) -> np.ndarray:
    """Evaluate `cv` over `traj` with `plumed driver`; return the CV series."""
    cwd.mkdir(parents=True, exist_ok=True)
    dcd = cwd / "traj.dcd"
    traj.save_dcd(str(dcd))
    plumed_dat = cwd / "plumed.dat"
    plumed_dat.write_text(cv.render() + "\nPRINT ARG=q FILE=QOUT STRIDE=1\n")

    result = subprocess.run(
        [
            "plumed", "driver",
            "--mf_dcd", str(dcd),
            "--natoms", str(traj.n_atoms),
            "--plumed", str(plumed_dat),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    out = cwd / "QOUT"
    assert out.exists(), result.stdout + result.stderr
    return np.array([
        float(line.split()[1])
        for line in out.read_text().splitlines()
        if not line.startswith("#")
    ])


def _native_and_extended() -> tuple[md.Trajectory, md.Trajectory]:
    native = md.load(str(_FIXED_PDB))
    extended = native[0]
    # Pull the chain apart along x so every native contact breaks.
    extended.xyz[0, :, 0] += 3.0 * np.arange(extended.n_atoms, dtype=np.float32)
    return native, extended


def _contacts_cv(native: md.Trajectory):
    return design_cv(
        CVProposal(cv_type="contacts", selections=("name CA",), label="q"),
        native.topology,
        reference=native,
    )


def test_contacts_cv_separates_the_native_structure_from_an_extended_one(
    tmp_path: Path,
) -> None:
    """The CV has to actually move between the states it will be asked to bias.

    Pairs are selected as those within the cutoff in the reference, so each
    switching function returns at least 0.5 there and the native count cannot
    fall below half the pair count by construction. What is *not* guaranteed —
    and is what this asserts — is that pulling the chain apart drives it to
    nearly zero, so the coordinate spans its range rather than sitting pinned.
    """
    native, extended = _native_and_extended()
    cv = _contacts_cv(native)

    q_native = _drive(cv, native, tmp_path / "n")
    q_extended = _drive(cv, extended, tmp_path / "e")

    assert q_native[0] > 0.6 * len(cv.pairs)
    assert q_extended[0] < 0.05 * len(cv.pairs)


def test_python_and_plumed_agree_on_the_contact_count(tmp_path: Path) -> None:
    """`bias_designer` re-implements the switching function in Python to size
    SIGMA in the CV's own units. If that drifts from what PLUMED computes, the
    hill width is sized against a coordinate the run does not actually bias —
    silently, since both numbers look reasonable on their own."""
    native, extended = _native_and_extended()
    cv = _contacts_cv(native)
    both = md.join([native[0], extended])

    from_plumed = _drive(cv, both, tmp_path / "both")
    from_python = _cv_series(cv, both)

    assert from_plumed.size == both.n_frames
    # atol as well as rtol: the extended frame sits at Q ~ 1e-13, where a
    # relative tolerance compares two different roundings of zero.
    np.testing.assert_allclose(from_python, from_plumed, rtol=1e-5, atol=1e-9)


def test_a_full_switched_bias_input_is_accepted_by_plumed(tmp_path: Path) -> None:
    """The composition the `switch_cv` path produces, end to end.

    A CONTACTMAP block uses PLUMED's line-continuation syntax, and the bias and
    PRINT directives follow it in the same file. Each piece parses alone; this
    pins that they parse *together*, which is the arrangement `switch_cv`
    actually writes and the one a unit test on rendered text cannot check.
    """
    from mdpilot.orchestrator.loop import _build_plumed_input
    from mdpilot.orchestrator.scientist import MetadProposal

    native, extended = _native_and_extended()
    source = md.join([native[0], extended, native[0]])
    traj_path = tmp_path / "source.dcd"
    source.save_dcd(str(traj_path))

    text = _build_plumed_input(
        MetadProposal(cv_type="contacts", selections=("name CA",), label="q_native"),
        traj_path,
        _FIXED_PDB,
        tmp_path,
    )
    plumed_dat = tmp_path / "plumed.dat"
    plumed_dat.write_text(text)

    result = subprocess.run(
        [
            "plumed", "driver",
            "--mf_dcd", str(traj_path),
            "--natoms", str(source.n_atoms),
            "--plumed", str(plumed_dat),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "Action CONTACTMAP" in combined, combined
    assert "Action METAD" in combined, combined
    assert (tmp_path / "HILLS").exists()
    # A bounded CV takes no upper wall, and there is nothing to bound.
    assert "UPPER_WALLS" not in text
