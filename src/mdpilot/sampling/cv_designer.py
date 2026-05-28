"""Resolve a structured CV proposal to a concrete PLUMED CV object.

This module is the boundary the architecture demands. The scientist (LLM)
reasons about *which* coordinate is slow for this system — a chemistry
judgment — and emits a structured proposal. This module validates that
proposal against the actual topology, looks up the atom indices
deterministically, and returns a typed CV object that ``plumed_writer`` can
render. Atom indices never pass through the LLM; CVs are never pre-curated
per system. The system-agnostic vocabulary is what ``plumed_writer`` can
render today (``distance``, ``torsion``, ``gyration``); new types are added
when a real campaign needs one — not before.

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
from typing import Literal

import mdtraj as md

from mdpilot.adapters.plumed_writer import CV, DistanceCV, GyrationCV, TorsionCV

CVType = Literal["distance", "torsion", "gyration"]
_CV_TYPES: tuple[str, ...] = ("distance", "torsion", "gyration")


@dataclass(frozen=True)
class CVProposal:
    """A structured CV proposal prior to topology resolution.

    ``selections`` are MDTraj selection strings, one per logical position the
    CV needs. Arity is type-dependent: ``distance`` takes 2 single-atom
    selections, ``torsion`` takes 4 single-atom selections, ``gyration``
    takes 1 multi-atom selection.
    """

    cv_type: CVType
    selections: tuple[str, ...]
    label: str


def design_cv(proposal: CVProposal, topology: md.Topology) -> CV:
    """Resolve a CVProposal against a topology; return a typed PLUMED CV."""
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
