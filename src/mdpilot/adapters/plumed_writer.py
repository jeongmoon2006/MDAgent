"""Generate PLUMED input files from typed CV + bias specifications.

PLUMED is the de facto standard for biased MD in academic research:
metadynamics, umbrella sampling, well-tempered metaD, free-energy
calculations all share the same input-file format. MDPilot's scientist
proposes CVs and bias parameters; this writer turns those proposals
into the plumed.dat text the engines read.

Runtime PLUMED is *not* a build-time dependency of this module — it
generates text. The OpenMM adapter handles the optional runtime import
(``openmmplumed``) separately. This split lets us develop and unit-test
the writer locally without requiring a working PLUMED install; the
"bias actually acts" verification happens on AWS where we control the
environment (D6 step 5).

Atom-indexing convention: this module accepts **0-based** atom indices
to match the rest of the codebase (OpenMM, mdtraj). PLUMED's text
format is 1-based; the writer converts on output. Callers should never
hand-encode the +1 themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class DistanceCV:
    """A distance collective variable between two atoms (0-based indices)."""

    label: str
    atoms: tuple[int, int]

    def render(self) -> str:
        a1, a2 = self.atoms[0] + 1, self.atoms[1] + 1
        return f"{self.label}: DISTANCE ATOMS={a1},{a2}"


@dataclass(frozen=True)
class TorsionCV:
    """A torsion (dihedral) collective variable across four atoms (0-based)."""

    label: str
    atoms: tuple[int, int, int, int]

    def render(self) -> str:
        idx = ",".join(str(a + 1) for a in self.atoms)
        return f"{self.label}: TORSION ATOMS={idx}"


@dataclass(frozen=True)
class GyrationCV:
    """Radius of gyration of an atom group (0-based atom indices).

    PLUMED's ``GYRATION TYPE=RADIUS`` action. Useful as a global compaction /
    folding order parameter — small Rg is compact/folded, large Rg is extended.
    """

    label: str
    atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.atoms) < 2:
            raise ValueError(
                f"GyrationCV: need at least 2 atoms (got {len(self.atoms)})"
            )

    def render(self) -> str:
        idx = ",".join(str(a + 1) for a in self.atoms)
        return f"{self.label}: GYRATION TYPE=RADIUS ATOMS={idx}"


CV = Union[DistanceCV, TorsionCV, GyrationCV]


@dataclass(frozen=True)
class MetadynamicsBias:
    """Metadynamics on one or more CVs.

    SIGMA is the Gaussian width (one per CV). HEIGHT and PACE control
    deposition. HILLS is the file PLUMED writes deposited Gaussians to —
    `plumed sum_hills` later integrates this into a free-energy surface.
    """

    cv_labels: tuple[str, ...]
    sigma: tuple[float, ...]
    height: float            # kJ/mol
    pace: int                # steps between deposits
    hills_file: str = "HILLS"
    bias_label: str = "metad"

    def __post_init__(self) -> None:
        if len(self.cv_labels) != len(self.sigma):
            raise ValueError(
                f"metaD: cv_labels and sigma must have same length "
                f"({len(self.cv_labels)} vs {len(self.sigma)})"
            )
        if len(self.cv_labels) == 0:
            raise ValueError("metaD: at least one CV required")
        if self.height <= 0:
            raise ValueError(f"metaD: height must be positive (got {self.height})")
        if self.pace <= 0:
            raise ValueError(f"metaD: pace must be positive (got {self.pace})")

    def render(self) -> str:
        args = ",".join(self.cv_labels)
        sigmas = ",".join(f"{s:g}" for s in self.sigma)
        return (
            f"{self.bias_label}: METAD ARG={args} "
            f"SIGMA={sigmas} HEIGHT={self.height:g} PACE={self.pace} "
            f"FILE={self.hills_file}"
        )

    @property
    def bias_value(self) -> str:
        return f"{self.bias_label}.bias"


@dataclass(frozen=True)
class HarmonicRestraint:
    """Harmonic restraint on a single CV — used for umbrella sampling
    windows and as the simplest PLUMED bias (handy for smoke tests)."""

    cv_label: str
    at: float
    kappa: float             # kJ/mol per CV-unit²
    restraint_label: str = "restraint"

    def __post_init__(self) -> None:
        if self.kappa < 0:
            raise ValueError(f"restraint: kappa must be non-negative (got {self.kappa})")

    def render(self) -> str:
        return (
            f"{self.restraint_label}: RESTRAINT "
            f"ARG={self.cv_label} AT={self.at:g} KAPPA={self.kappa:g}"
        )

    @property
    def bias_value(self) -> str:
        return f"{self.restraint_label}.bias"


Bias = Union[MetadynamicsBias, HarmonicRestraint]


@dataclass(frozen=True)
class PlumedInput:
    """A complete plumed.dat: CVs + bias + periodic COLVAR output."""

    cvs: tuple[CV, ...]
    bias: Bias
    print_stride: int = 500
    colvar_file: str = "COLVAR"

    def __post_init__(self) -> None:
        if len(self.cvs) == 0:
            raise ValueError("PlumedInput: at least one CV required")
        cv_labels = {cv.label for cv in self.cvs}
        if len(cv_labels) != len(self.cvs):
            raise ValueError(f"PlumedInput: CV labels must be unique")
        used = self._bias_cv_labels()
        missing = used - cv_labels
        if missing:
            raise ValueError(
                f"PlumedInput: bias references undefined CV label(s): "
                f"{sorted(missing)}; defined: {sorted(cv_labels)}"
            )

    def _bias_cv_labels(self) -> set[str]:
        if isinstance(self.bias, MetadynamicsBias):
            return set(self.bias.cv_labels)
        return {self.bias.cv_label}

    def render(self) -> str:
        lines: list[str] = [
            "# PLUMED input generated by MDPilot",
            "",
            "# Collective variables",
        ]
        for cv in self.cvs:
            lines.append(cv.render())
        lines += ["", "# Bias", self.bias.render(), "", "# Periodic output"]
        print_args = ",".join([cv.label for cv in self.cvs] + [self.bias.bias_value])
        lines.append(
            f"PRINT ARG={print_args} STRIDE={self.print_stride} FILE={self.colvar_file}"
        )
        return "\n".join(lines) + "\n"
