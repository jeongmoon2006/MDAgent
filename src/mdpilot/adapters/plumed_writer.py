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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Union


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


@dataclass(frozen=True)
class RmsdCV:
    """RMSD to a reference structure after optimal superposition (0-based atoms).

    PLUMED's ``RMSD ... TYPE=OPTIMAL`` action — Kearsley superposition, so this
    is RMSD *after* removing rigid-body motion, the folding order parameter
    people actually mean by "RMSD to native".

    ``reference_path`` points at a PDB holding only ``atoms``. PLUMED maps that
    file onto the running system by **PDB serial number**, not by file order, so
    the reference cannot be written with `mdtraj.save_pdb` on a sliced
    trajectory (which renumbers serials from 1 and would silently align against
    the wrong atoms). `cv_designer.design_cv` writes it with the original
    1-based serials preserved.

    The path must be absolute, for the same reason ``PlumedInput.output_dir``
    must be: PLUMED resolves relative paths against the *process* working
    directory, not the location of plumed.dat (F5).
    """

    label: str
    atoms: tuple[int, ...]
    reference_path: Path

    def __post_init__(self) -> None:
        if len(self.atoms) < 3:
            raise ValueError(
                f"RmsdCV: need at least 3 atoms for optimal superposition "
                f"(got {len(self.atoms)})"
            )
        if not Path(self.reference_path).is_absolute():
            raise ValueError(
                f"RmsdCV: reference_path must be absolute (got "
                f"{self.reference_path!r}); PLUMED resolves relative paths "
                f"against the process working directory"
            )

    def render(self) -> str:
        return f"{self.label}: RMSD REFERENCE={self.reference_path} TYPE=OPTIMAL"


@dataclass(frozen=True)
class ContactsCV:
    """Number of native contacts formed, via PLUMED's ``CONTACTMAP ... SUM``.

    Each pair contributes a rational switching function that goes to 1 when the
    two atoms are closer than ``r0_nm`` and to 0 when they are far apart. The
    raw ``SUM`` is a count, and the CV exposed under ``label`` is that count
    divided by the number of pairs — a **fraction on [0, 1]**.

    Normalised deliberately, and always. A campaign judged its states on a
    normalised contact fraction (0.3 / 0.7) while biasing the raw count, and
    both carried the name ``native_contacts_fraction``: the free-energy axis
    ran to 10 where every threshold in the campaign lived below 1. A count and
    a fraction of the same contacts are the same coordinate scaled, so there is
    nothing to choose between them except which one can be compared with the
    rest of the campaign.

    Unlike RMSD-to-native this coordinate is **bounded on both sides**, which is
    what makes it usable with well-tempered metadynamics on a folding problem:
    there is no open unfolded tail for the bias to drive the walker into
    indefinitely.

    ``pairs`` are 0-based atom index pairs, resolved from the reference
    structure by ``cv_designer``; indices are converted to PLUMED's 1-based
    convention at render time like every other CV here.
    """

    label: str
    pairs: tuple[tuple[int, int], ...]
    r0_nm: float

    # Rational switching exponents. `bias_designer` evaluates this same
    # function in Python to size SIGMA, using the identity
    # (1 - x**NN) / (1 - x**MM) == 1 / (1 + x**NN), which holds exactly when
    # MM == 2 * NN and avoids the 0/0 the raw form hits at x == 1. Preserve
    # that relation if these ever change, or the two evaluations diverge.
    NN: ClassVar[int] = 6
    MM: ClassVar[int] = 12

    def __post_init__(self) -> None:
        if len(self.pairs) < 1:
            raise ValueError("ContactsCV: need at least 1 contact pair")
        if self.r0_nm <= 0.0:
            raise ValueError(
                f"ContactsCV: r0_nm must be positive (got {self.r0_nm})"
            )

    @property
    def raw_label(self) -> str:
        """The unnormalised count, an intermediate PLUMED value."""
        return f"{self.label}_count"

    def render(self) -> str:
        # PLUMED's line-continuation form: one ATOMS<n> per pair, then a single
        # global SWITCH that applies to all of them, then SUM to collapse the
        # map into one number. NN/MM are the conventional rational exponents.
        lines = [f"{self.raw_label}: CONTACTMAP ..."]
        for n, (i, j) in enumerate(self.pairs, start=1):
            lines.append(f"  ATOMS{n}={i + 1},{j + 1}")
        lines.append(
            f"  SWITCH={{RATIONAL R_0={self.r0_nm} NN={self.NN} MM={self.MM}}}"
        )
        lines.append("  SUM")
        # Bare "..." to close the continuation. PLUMED accepts a second word
        # there only if it repeats the *label* ("... q:"), not the action name —
        # `... CONTACTMAP` is an assertion failure, not a comment.
        lines.append("...")
        # Divide by the pair count so the biased coordinate, the COLVAR trace
        # and the free-energy axis are all the fraction the campaign's state
        # thresholds are stated in.
        lines.append(
            f"{self.label}: COMBINE ARG={self.raw_label} "
            f"COEFFICIENTS={1.0 / len(self.pairs):.8g} PERIODIC=NO"
        )
        return "\n".join(lines)


# The closed form `bias_designer` evaluates in Python is equal to the rational
# switch PLUMED evaluates here *only* when MM == 2*NN. Stated as code rather
# than left to the comment above: changing either exponent is silent in both
# modules, and the two evaluations would then disagree on a live campaign with
# no error anywhere — SIGMA sized against a coordinate the run does not bias.
if ContactsCV.MM != 2 * ContactsCV.NN:
    raise AssertionError(
        f"ContactsCV: bias_designer's closed-form switching function requires "
        f"MM == 2*NN (got NN={ContactsCV.NN}, MM={ContactsCV.MM})"
    )


CV = Union[DistanceCV, TorsionCV, GyrationCV, RmsdCV, ContactsCV]


@dataclass(frozen=True)
class MetadynamicsBias:
    """Well-tempered metadynamics on one or more CVs.

    SIGMA is the Gaussian width (one per CV). HEIGHT is the *initial* hill
    height and PACE the deposition stride. HILLS is the file PLUMED writes
    deposited Gaussians to — `plumed sum_hills` later integrates this into a
    free-energy surface.

    Well-tempered only: `bias_factor` (γ, PLUMED's BIASFACTOR) makes the
    deposition rate decay as exp(-V(s,t)/k_B ΔT) with ΔT = (γ-1)T, so the
    bias converges to -(1 - 1/γ)F(s) instead of overfilling the basin the
    way plain metadynamics does. γ must exceed 1; γ → ∞ recovers plain
    metaD, which is why it is not constructible here. PLUMED needs TEMP to
    evaluate the tempering factor, so `temperature_k` must match the
    thermostat the engine is running.
    """

    cv_labels: tuple[str, ...]
    sigma: tuple[float, ...]
    height: float            # kJ/mol, initial hill height W0
    pace: int                # steps between deposits
    bias_factor: float = 10.0        # γ, dimensionless; must be > 1
    temperature_k: float = 300.0     # must match the thermostat
    sigma_floored: bool = False      # SIGMA came from the floor, not the data
    sigma_ceiled: bool = False       # SIGMA came from the ceiling, not the data
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
        if self.bias_factor <= 1.0:
            raise ValueError(
                f"metaD: bias_factor (γ) must be > 1 for well-tempered "
                f"metadynamics (got {self.bias_factor})"
            )
        if self.temperature_k <= 0:
            raise ValueError(
                f"metaD: temperature_k must be positive (got {self.temperature_k})"
            )

    def render(self) -> str:
        args = ",".join(self.cv_labels)
        sigmas = ",".join(f"{s:g}" for s in self.sigma)
        return (
            f"{self.bias_label}: METAD ARG={args} "
            f"SIGMA={sigmas} HEIGHT={self.height:g} PACE={self.pace} "
            f"BIASFACTOR={self.bias_factor:g} TEMP={self.temperature_k:g} "
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


@dataclass(frozen=True)
class UpperWall:
    """One-sided harmonic wall bounding a CV from above (PLUMED ``UPPER_WALLS``).

    Needed for CVs that are unbounded on one side. RMSD-to-native is the clear
    case: the folded side is pinned near zero, but the unfolded side runs to
    arbitrarily large values, so well-tempered metadynamics keeps driving the
    walker outward into an ever-larger unfolded space instead of coming back.
    The first CLN025 campaign deposited 129 kJ/mol of bias — an order of
    magnitude past the real folding free energy — and got one crossing with no
    return trip. A wall past the "unfolded" threshold bounds the space the bias
    has to fill and concentrates sampling in the transition region.

    Flat below ``at``; above it the energy rises as ``kappa * (s - at)^exp``.
    """

    cv_label: str
    at: float                 # CV units (nm for length-dimensioned CVs)
    kappa: float              # kJ/mol per CV-unit^exp
    exp: int = 2
    wall_label: str = "uwall"
    # The CV value at which the solute would reach the edge of its box,
    # measured from the source trajectory by `bias_designer.box_limited_wall`.
    # None when it could not be derived.
    box_limit_nm: float | None = None
    # True when `at` *is* that limit because the campaign gave no position.
    derived_from_box: bool = False

    @property
    def exceeds_box_limit(self) -> bool:
        """A wall the box cannot honour. The bias will drive the solute into
        its own periodic image before the wall ever pushes back (F11)."""
        return self.box_limit_nm is not None and self.at > self.box_limit_nm

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"UpperWall: kappa must be positive (got {self.kappa})")
        if self.exp < 1:
            raise ValueError(f"UpperWall: exp must be >= 1 (got {self.exp})")

    def render(self) -> str:
        return (
            f"{self.wall_label}: UPPER_WALLS ARG={self.cv_label} "
            f"AT={self.at:g} KAPPA={self.kappa:g} EXP={self.exp}"
        )


Bias = Union[MetadynamicsBias, HarmonicRestraint]


_RESTART_DIRECTIVE = "RESTART"


def enable_restart(plumed_input: str) -> str:
    """Return `plumed_input` with PLUMED's RESTART directive enabled.

    Without it, PLUMED backs the existing HILLS up to ``bck.0.HILLS`` and
    starts a fresh file with zero accumulated bias — verified against PLUMED
    2.9: ``metad.bias`` reads exactly 0.0 on the first frame. A biased phase
    resumed that way continues from a *biased* configuration under *no* bias.
    The walker sits in a basin the campaign already filled, refills it, and
    the surface eventually integrated is the sum of two disjoint fillings.
    With RESTART, METAD reads HILLS back at initialization ("Restarting from
    HILLS: N Gaussians read") and appends to it, and PRINT appends to COLVAR
    instead of backing that up too — which is what keeps the recrossing count
    covering the whole biased phase rather than only the last leg.

    Text-level rather than a `PlumedInput` field because the persisted
    ``plumed.dat`` *is* the interface on resume: the loop rebuilds the biased
    adapter from that file's contents, not from the proposal that produced it.

    Idempotent. The adapter rewrites plumed.dat on every ``start()``, so a
    second resume reads text this function has already touched.
    """
    for line in plumed_input.splitlines():
        stripped = line.strip()
        if stripped == _RESTART_DIRECTIVE or stripped.startswith(
            _RESTART_DIRECTIVE + " "
        ):
            return plumed_input
    return (
        f"{_RESTART_DIRECTIVE}   "
        f"# resumed biased phase: read HILLS back and append\n{plumed_input}"
    )


@dataclass(frozen=True)
class PlumedInput:
    """A complete plumed.dat: CVs + bias + periodic COLVAR output.

    ``output_dir`` must be absolute and is prefixed onto every file PLUMED
    writes. PLUMED resolves relative ``FILE=`` paths against the *process*
    working directory, not against the location of plumed.dat — so a bare
    ``FILE=HILLS`` drops the campaign's deposited bias wherever python
    happened to be started, colliding across concurrent campaigns and leaving
    resume unable to find the previous HILLS. Requiring an absolute directory
    here makes that unrepresentable.
    """

    cvs: tuple[CV, ...]
    bias: Bias
    output_dir: Path
    walls: tuple[UpperWall, ...] = ()
    print_stride: int = 500
    colvar_file: str = "COLVAR"

    def __post_init__(self) -> None:
        if len(self.cvs) == 0:
            raise ValueError("PlumedInput: at least one CV required")
        if not Path(self.output_dir).is_absolute():
            raise ValueError(
                f"PlumedInput: output_dir must be absolute (got "
                f"{self.output_dir!r}); PLUMED resolves relative FILE= paths "
                f"against the process working directory"
            )
        cv_labels = {cv.label for cv in self.cvs}
        if len(cv_labels) != len(self.cvs):
            raise ValueError("PlumedInput: CV labels must be unique")
        used = self._bias_cv_labels() | {w.cv_label for w in self.walls}
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

    def _bias_with_resolved_paths(self) -> Bias:
        """Rebind the bias's output file under `output_dir`. The bias renders
        itself, so the directory has to be pushed into it before rendering."""
        if isinstance(self.bias, MetadynamicsBias):
            return replace(
                self.bias,
                hills_file=str(Path(self.output_dir) / self.bias.hills_file),
            )
        return self.bias

    def render(self) -> str:
        lines: list[str] = [
            "# PLUMED input generated by MDPilot",
            "",
            "# Collective variables",
        ]
        for cv in self.cvs:
            lines.append(cv.render())
        lines += ["", "# Bias"]
        if isinstance(self.bias, MetadynamicsBias) and self.bias.sigma_floored:
            # The audit artifact has to say when SIGMA did not come from the
            # data, or a floored width reads later as a measured one.
            lines += [
                "# NOTE: SIGMA was raised to this CV type's floor — the source",
                "#       trajectory's spread measured narrower than the",
                "#       narrowest hill worth depositing. Treat the vanilla",
                "#       phase as having under-sampled this coordinate.",
            ]
        if isinstance(self.bias, MetadynamicsBias) and self.bias.sigma_ceiled:
            # Same argument from the other side: a width taken from an
            # already-biased trajectory is inflated, and a hill wider than the
            # features it should resolve flattens the surface by construction.
            lines += [
                "# NOTE: SIGMA was lowered to this CV type's ceiling — the",
                "#       source trajectory's spread measured wider than the",
                "#       widest hill worth depositing. Expect this when a CV is",
                "#       sized on a trajectory that was already biased.",
            ]
        lines.append(self._bias_with_resolved_paths().render())
        if self.walls:
            lines += ["", "# Bounds"]
            for wall in self.walls:
                if wall.derived_from_box:
                    lines += [
                        "# NOTE: no wall position was given, so this one is the",
                        "#       box limit measured from the source trajectory —",
                        "#       the CV value at which the solute would reach its",
                        "#       own periodic image.",
                    ]
                elif wall.exceeds_box_limit:
                    lines += [
                        f"# WARNING: this wall sits at {wall.at:g}, beyond the",
                        f"#          {wall.box_limit_nm:.2f} the box can hold. The bias will",
                        "#          drive the solute into its own periodic image",
                        "#          before the wall pushes back (F11). Enlarge the",
                        "#          box or lower the wall.",
                    ]
                lines.append(wall.render())
        lines += ["", "# Periodic output"]
        print_args = ",".join([cv.label for cv in self.cvs] + [self.bias.bias_value])
        colvar = Path(self.output_dir) / self.colvar_file
        lines.append(
            f"PRINT ARG={print_args} STRIDE={self.print_stride} FILE={colvar}"
        )
        # PLUMED buffers its output and only flushes when the context is
        # finalized. Without this, a campaign killed mid-round loses every
        # deposited hill, and a resume reads a HILLS file that looks empty.
        lines += ["", "# Durability", f"FLUSH STRIDE={self.print_stride}"]
        return "\n".join(lines) + "\n"
