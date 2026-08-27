"""Resolve a structured CV proposal to a concrete PLUMED CV object.

This module is the boundary the architecture demands. The scientist (LLM)
reasons about *which* coordinate is slow for this system — a chemistry
judgment — and emits a structured proposal. This module validates that
proposal against the actual topology, looks up the atom indices
deterministically, and returns a typed CV object that ``plumed_writer`` can
render. Atom indices never pass through the LLM; CVs are never pre-curated
per system. The system-agnostic vocabulary is what ``plumed_writer`` can
render today (``distance``, ``torsion``, ``gyration``, ``rmsd``,
``contacts``); new types are added when a real campaign needs one — not
before.

Selection language: MDTraj selection strings (``"backbone and resid 1 to 10"``,
``"name CA and resid 1"``). MDTraj is already a project dependency and its
grammar is well-documented and consistent across the codebase.

A proposal of an unknown CV type, an empty selection, or a wrong arity for
the chosen type raises ``ValueError`` with a message naming the failure — so
the caller (the loop, ultimately the scientist via tool-use feedback) can act
on it deterministically rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mdtraj as md

import numpy as np

from mdpilot.adapters.plumed_writer import (
    CV,
    ContactsCV,
    DistanceCV,
    GyrationCV,
    RmsdCV,
    TorsionCV,
)

CVType = Literal["distance", "torsion", "gyration", "rmsd", "contacts"]
_CV_TYPES: tuple[str, ...] = (
    "distance", "torsion", "gyration", "rmsd", "contacts",
)

# A CA-CA pair closer than this in the reference structure counts as a native
# contact. 0.75 nm is the conventional CA-level cutoff; heavy-atom definitions
# use ~0.45 nm, which does not transfer to a CA-only selection.
_CONTACT_CUTOFF_NM = 0.75
# Contacts between residues near each other in sequence are formed in any
# conformation, folded or not, so they carry no information about nativeness
# and would only add a constant offset to the count.
_MIN_SEQUENCE_SEPARATION = 3


@dataclass(frozen=True)
class CVProposal:
    """A structured CV proposal prior to topology resolution.

    ``selections`` are MDTraj selection strings, one per logical position the
    CV needs. Arity is type-dependent: ``distance`` takes 2 single-atom
    selections, ``torsion`` takes 4 single-atom selections, ``gyration`` and
    ``rmsd`` each take 1 multi-atom selection (>=2 and >=3 atoms
    respectively).
    """

    cv_type: CVType
    selections: tuple[str, ...]
    label: str


def design_cv(
    proposal: CVProposal,
    topology: md.Topology,
    *,
    reference: md.Trajectory | None = None,
    output_dir: Path | None = None,
) -> CV:
    """Resolve a CVProposal against a topology; return a typed PLUMED CV.

    ``reference`` is required for ``cv_type="rmsd"`` and ``cv_type="contacts"``
    — the two CVs that need coordinates rather than just connectivity, since
    both are defined relative to a native structure. ``output_dir`` is required
    only for ``rmsd``, which is the one that has to write a reference PDB for
    PLUMED to read; ``contacts`` bakes the resolved pairs into plumed.dat
    directly. Everything else resolves from topology alone.
    """
    if proposal.cv_type not in _CV_TYPES:
        raise ValueError(
            f"cv_designer: unknown cv_type {proposal.cv_type!r}; "
            f"available: {sorted(_CV_TYPES)}"
        )

    indices = tuple(_resolve(sel, topology) for sel in proposal.selections)

    if proposal.cv_type == "distance":
        return _build_distance(proposal.label, indices)
    if proposal.cv_type == "torsion":
        return _build_torsion(proposal.label, indices)
    if proposal.cv_type == "gyration":
        return _build_gyration(proposal.label, indices)
    if proposal.cv_type == "rmsd":
        return _build_rmsd(proposal.label, indices, reference, output_dir)
    if proposal.cv_type == "contacts":
        return _build_contacts(proposal.label, indices, reference)
    raise AssertionError(f"unreachable cv_type {proposal.cv_type!r}")


def _resolve(selection: str, topology: md.Topology) -> tuple[int, ...]:
    atoms = topology.select(selection)
    if atoms.size == 0:
        raise ValueError(
            f"cv_designer: selection {selection!r} resolved to 0 atoms"
        )
    return tuple(int(a) for a in atoms)


def _build_distance(
    label: str, sels: tuple[tuple[int, ...], ...]
) -> DistanceCV:
    if len(sels) != 2:
        raise ValueError(
            f"cv_designer: distance requires 2 selections, got {len(sels)}"
        )
    for i, atoms in enumerate(sels):
        if len(atoms) != 1:
            raise ValueError(
                f"cv_designer: distance selection {i} must resolve to "
                f"exactly 1 atom (got {len(atoms)})"
            )
    return DistanceCV(label=label, atoms=(sels[0][0], sels[1][0]))


def _build_torsion(
    label: str, sels: tuple[tuple[int, ...], ...]
) -> TorsionCV:
    if len(sels) != 4:
        raise ValueError(
            f"cv_designer: torsion requires 4 selections, got {len(sels)}"
        )
    for i, atoms in enumerate(sels):
        if len(atoms) != 1:
            raise ValueError(
                f"cv_designer: torsion selection {i} must resolve to "
                f"exactly 1 atom (got {len(atoms)})"
            )
    return TorsionCV(
        label=label,
        atoms=(sels[0][0], sels[1][0], sels[2][0], sels[3][0]),
    )


def _build_gyration(
    label: str, sels: tuple[tuple[int, ...], ...]
) -> GyrationCV:
    if len(sels) != 1:
        raise ValueError(
            f"cv_designer: gyration requires 1 selection, got {len(sels)}"
        )
    atoms = sels[0]
    if len(atoms) < 2:
        raise ValueError(
            f"cv_designer: gyration selection must resolve to >=2 atoms "
            f"(got {len(atoms)})"
        )
    return GyrationCV(label=label, atoms=atoms)


def _build_rmsd(
    label: str,
    sels: tuple[tuple[int, ...], ...],
    reference: "md.Trajectory | None",
    output_dir: Path | None,
) -> RmsdCV:
    if len(sels) != 1:
        raise ValueError(
            f"cv_designer: rmsd requires 1 selection, got {len(sels)}"
        )
    atoms = sels[0]
    if len(atoms) < 3:
        raise ValueError(
            f"cv_designer: rmsd selection must resolve to >=3 atoms for optimal "
            f"superposition (got {len(atoms)})"
        )
    if reference is None or output_dir is None:
        raise ValueError(
            "cv_designer: cv_type='rmsd' needs `reference` (the structure to "
            "measure against) and `output_dir` (where to write the reference "
            "PDB); pass both to design_cv"
        )
    path = (Path(output_dir) / f"{label}_reference.pdb").resolve()
    _write_reference_pdb(path, reference, atoms)
    return RmsdCV(label=label, atoms=atoms, reference_path=path)


def _build_contacts(
    label: str,
    sels: tuple[tuple[int, ...], ...],
    reference: "md.Trajectory | None",
) -> ContactsCV:
    if len(sels) != 1:
        raise ValueError(
            f"cv_designer: contacts requires 1 selection, got {len(sels)}"
        )
    atoms = sels[0]
    if len(atoms) < 2:
        raise ValueError(
            f"cv_designer: contacts selection must resolve to >=2 atoms "
            f"(got {len(atoms)})"
        )
    if reference is None:
        raise ValueError(
            "cv_designer: cv_type='contacts' needs `reference` — the structure "
            "whose contacts count as native; pass it to design_cv"
        )
    pairs = _native_pairs(atoms, reference)
    if not pairs:
        raise ValueError(
            f"cv_designer: no native contacts among the {len(atoms)} selected "
            f"atoms (cutoff {_CONTACT_CUTOFF_NM} nm, sequence separation "
            f">= {_MIN_SEQUENCE_SEPARATION} residues); the reference may be "
            f"extended, or the selection too small"
        )
    return ContactsCV(label=label, pairs=pairs, r0_nm=_CONTACT_CUTOFF_NM)


def _native_pairs(
    atoms: tuple[int, ...], reference: "md.Trajectory"
) -> tuple[tuple[int, int], ...]:
    """Pairs among `atoms` that are in contact in the reference structure.

    mdtraj coordinates are in nanometres, which is also PLUMED's length unit,
    so the cutoff needs no conversion.

    Distances come from ``md.compute_distances``, which applies the minimum
    image convention when the reference carries a unit cell. That matters
    because this is one of *three* places the same pair distance gets measured:
    here (which pairs are native), in ``bias_designer.cv_series`` (sizing
    SIGMA), and in PLUMED's own ``CONTACTMAP`` at run time. The other two apply
    PBC, so measuring raw displacements here would define the native set under
    a different convention than the one that evaluates it. The reference is
    written by the adapter with ``enforcePeriodicBox=True``, which wraps per
    *molecule* — a single chain stays whole, but two chains are wrapped
    independently, so an interchain native contact is exactly the case a raw
    displacement would miss.
    """
    topology = reference.topology
    candidates = [
        (i, j)
        for a, i in enumerate(atoms)
        for j in atoms[a + 1 :]
        if abs(topology.atom(i).residue.index - topology.atom(j).residue.index)
        >= _MIN_SEQUENCE_SEPARATION
    ]
    if not candidates:
        return ()
    distances = md.compute_distances(reference[0], np.array(candidates))[0]
    return tuple(
        pair
        # strict: one distance per candidate pair, by construction.
        for pair, d in zip(candidates, distances, strict=True)
        if d <= _CONTACT_CUTOFF_NM
    )


def _write_reference_pdb(
    path: Path, reference: "md.Trajectory", atoms: tuple[int, ...]
) -> None:
    """Write the reference structure PLUMED's RMSD action reads.

    Hand-rolled rather than `reference.atom_slice(atoms).save_pdb(path)`,
    because PLUMED identifies reference atoms by **PDB serial number** and
    mdtraj renumbers serials from 1 when it writes a sliced trajectory. That
    would map the reference onto atoms 1..N of the simulated system — the first
    N atoms of the protein, not the selected ones — and align against the wrong
    thing without any error. Serials here are the original 0-based indices + 1,
    matching the convention the rest of `plumed_writer` uses.

    Occupancy and B-factor are both 1.00: PLUMED reads occupancy as the
    alignment weight and B-factor as the displacement weight, so this measures
    RMSD over exactly the selected atoms, aligned on the same set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    top = reference.topology
    xyz = reference.xyz[0] * 10.0  # nm -> Angstrom, as PDB requires
    lines = []
    for atom_index in atoms:
        atom = top.atom(atom_index)
        x, y, z = xyz[atom_index]
        element = (atom.element.symbol if atom.element is not None else "").upper()
        lines.append(
            f"ATOM  {atom_index + 1:>5d} {atom.name:<4.4s}{'':1s}"
            f"{atom.residue.name:>3.3s} {'A':1s}{atom.residue.resSeq:>4d}{'':1s}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{1.00:>6.2f}"
            f"{'':10s}{element:>2.2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
