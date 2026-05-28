"""cv_designer: MDTraj selection strings → typed PLUMED CV objects.

Topology built programmatically: a 5-residue ALA mini-peptide (20 atoms,
N/CA/C/O per residue, resSeq 1..5). No PDB load, no network — the property
under test is selection resolution + arity validation, not parsing real
structures.
"""

from __future__ import annotations

import mdtraj as md
import pytest

from mdpilot.adapters.plumed_writer import DistanceCV, GyrationCV, TorsionCV
from mdpilot.sampling.cv_designer import CVProposal, design_cv


def _mini_peptide() -> md.Topology:
    top = md.Topology()
    chain = top.add_chain()
    for resid in range(1, 6):
        res = top.add_residue("ALA", chain, resSeq=resid)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
    return top


# ---------- happy paths ----------

def test_distance_resolves_to_pair_of_atom_indices() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="distance",
        selections=("resSeq 1 and name CA", "resSeq 5 and name CA"),
        label="d_term",
    )
    cv = design_cv(prop, top)
    assert isinstance(cv, DistanceCV)
    assert cv.label == "d_term"
    assert cv.atoms == (1, 17)
    assert cv.render() == "d_term: DISTANCE ATOMS=2,18"


def test_torsion_resolves_to_four_atom_indices() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="torsion",
        selections=(
            "resSeq 2 and name N",
            "resSeq 2 and name CA",
            "resSeq 2 and name C",
            "resSeq 3 and name N",
        ),
        label="psi2",
    )
    cv = design_cv(prop, top)
    assert isinstance(cv, TorsionCV)
    assert cv.atoms == (4, 5, 6, 8)


def test_gyration_resolves_to_atom_group() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="gyration",
        selections=("backbone and resSeq 1 to 3",),
        label="rg_back",
    )
    cv = design_cv(prop, top)
    assert isinstance(cv, GyrationCV)
    assert cv.atoms == tuple(range(12))  # 3 residues × 4 backbone atoms
    assert "GYRATION TYPE=RADIUS" in cv.render()


# ---------- validation failures ----------

def test_unknown_cv_type_lists_available_types() -> None:
    top = _mini_peptide()
    prop = CVProposal(cv_type="rmsd", selections=(), label="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown cv_type"):
        design_cv(prop, top)


def test_empty_selection_names_the_string() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="distance",
        selections=("resSeq 99 and name CA", "resSeq 1 and name CA"),
        label="d",
    )
    with pytest.raises(ValueError, match="resolved to 0 atoms"):
        design_cv(prop, top)


def test_distance_wrong_arity_raises() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="distance",
        selections=("resSeq 1 and name CA",),
        label="d",
    )
    with pytest.raises(ValueError, match="distance requires 2 selections"):
        design_cv(prop, top)


def test_distance_multi_atom_selection_raises() -> None:
    """A scientist who passes `name CA` (all CAs) for one endpoint must fail
    fast — distance needs a single atom per endpoint, not a group."""
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="distance",
        selections=("name CA", "resSeq 5 and name CA"),
        label="d",
    )
    with pytest.raises(ValueError, match="exactly 1 atom"):
        design_cv(prop, top)


def test_torsion_wrong_arity_raises() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="torsion",
        selections=("resSeq 1 and name N", "resSeq 1 and name CA"),
        label="phi",
    )
    with pytest.raises(ValueError, match="torsion requires 4 selections"):
        design_cv(prop, top)


def test_gyration_wrong_arity_raises() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="gyration",
        selections=("name CA", "name N"),
        label="rg",
    )
    with pytest.raises(ValueError, match="gyration requires 1 selection"):
        design_cv(prop, top)


def test_gyration_single_atom_selection_raises() -> None:
    top = _mini_peptide()
    prop = CVProposal(
        cv_type="gyration",
        selections=("resSeq 1 and name CA",),
        label="rg",
    )
    with pytest.raises(ValueError, match=">=2 atoms"):
        design_cv(prop, top)
