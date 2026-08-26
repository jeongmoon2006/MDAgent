"""What the campaign measures, and how to compute it from a trajectory.

The campaign observable is the one 1-D series every phase is judged on: the
vanilla convergence bundle summarizes it, and the biased phase counts
recrossings against the task's own state thresholds *on it* rather than on
whichever CV the scientist chose to bias (F7, F9). Until now it was hardcoded
as CA-RMSD to the campaign topology, so `campaign_observable` began with
``topology.select("protein and name CA")`` and raised on an empty selection —
which made every non-protein campaign impossible at round one.

The generalization is deliberately not a new abstraction. An observable is a
collective variable, and `sampling/` already knows how to resolve and compute
five of them; `bias_designer.cv_series` is the engine. So an `ObservableSpec`
is a `CVProposal` plus a display name and a unit scale, and this module is the
thin layer that keeps the two uses — "size the bias from this coordinate" and
"judge the campaign on this coordinate" — computing the same thing.

`rmsd` is the one type computed directly here rather than through
`cv_designer`. PLUMED's RMSD action needs a reference *file*, which
`design_cv` writes; the campaign observable measures against the campaign
topology already in memory, so routing it through a written PDB would add a
file and an opportunity for the two references to differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mdtraj as md
import numpy as np

from mdpilot.sampling.bias_designer import cv_series
from mdpilot.sampling.cv_designer import CVProposal, CVType, design_cv

# The M1-era observable, kept as the default so a campaign that says nothing
# behaves exactly as it did. Angstrom because that is the unit the task files,
# the done criteria and every recorded campaign state their thresholds in;
# mdtraj works in nanometres, hence the scale.
_CA_RMSD_NAME = "rmsd_ca_to_reference_angstrom"
_NM_TO_ANGSTROM = 10.0


@dataclass(frozen=True)
class ObservableSpec:
    """The campaign observable, as a CV plus a name and a unit scale.

    `selections` are MDTraj selection strings with the same arity rules
    `cv_designer` enforces, because for every type but `rmsd` this *is*
    `cv_designer`. `scale` multiplies the raw value: mdtraj returns nanometres
    and radians, and a campaign may state its thresholds in something else.
    """

    cv_type: CVType
    selections: tuple[str, ...]
    name: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.selections:
            raise ValueError("ObservableSpec: at least one selection required")
        if not self.name:
            raise ValueError("ObservableSpec: name is required — it labels the "
                             "series in every report and done criterion")
        if self.scale <= 0:
            raise ValueError(
                f"ObservableSpec: scale must be positive (got {self.scale})"
            )

    @classmethod
    def ca_rmsd_angstrom(cls) -> "ObservableSpec":
        """CA-RMSD to the campaign reference, in Angstrom — the M1 default."""
        return cls(
            cv_type="rmsd",
            selections=("protein and name CA",),
            name=_CA_RMSD_NAME,
            scale=_NM_TO_ANGSTROM,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_type": self.cv_type,
            "selections": list(self.selections),
            "name": self.name,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservableSpec":
        return cls(
            cv_type=data["cv_type"],
            selections=tuple(data["selections"]),
            name=data["name"],
            scale=float(data.get("scale", 1.0)),
        )


def campaign_observable(
    traj: md.Trajectory,
    top_path: "Any",
    spec: ObservableSpec | None = None,
) -> tuple[np.ndarray, str]:
    """The campaign observable over `traj`, plus its name.

    The reference is the campaign topology — written once by the adapter's
    `start()` and constant for the life of the campaign — not this round's
    first frame. A per-round reference makes every round a different
    observable: round 3 would measure displacement from wherever round 3
    happened to start, while the scientist is shown `ess` and `plateau_reached`
    across rounds as if they described one time series.
    """
    spec = spec or ObservableSpec.ca_rmsd_angstrom()
    reference = md.load(str(top_path))
    return _series(spec, traj, reference), spec.name


def _series(
    spec: ObservableSpec, traj: md.Trajectory, reference: md.Trajectory
) -> np.ndarray:
    if spec.cv_type == "rmsd":
        atoms = _resolve_one(spec, traj.topology, minimum=3)
        # Same atom indices on both sides: `traj` was loaded with this topology.
        return (
            md.rmsd(traj.atom_slice(atoms), reference.atom_slice(atoms), frame=0)
            * spec.scale
        )
    cv = design_cv(
        CVProposal(
            cv_type=spec.cv_type, selections=tuple(spec.selections), label=spec.name
        ),
        traj.topology,
        reference=reference,
    )
    return cv_series(cv, traj) * spec.scale


def _resolve_one(
    spec: ObservableSpec, topology: md.Topology, *, minimum: int
) -> np.ndarray:
    if len(spec.selections) != 1:
        raise ValueError(
            f"observable: {spec.cv_type} takes 1 selection, got "
            f"{len(spec.selections)}"
        )
    atoms = topology.select(spec.selections[0])
    if atoms.size < minimum:
        raise ValueError(
            f"observable: selection {spec.selections[0]!r} resolved to "
            f"{atoms.size} atom(s); {spec.cv_type} needs at least {minimum}. "
            f"The campaign observable is what every round is judged on, so an "
            f"empty or undersized selection is a campaign that cannot be scored."
        )
    return atoms
