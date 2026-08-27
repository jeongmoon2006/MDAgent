"""Deterministic metadynamics bias parameters from a resolved CV + prior trajectory.

The scientist (LLM) proposes *which* coordinate is slow; ``cv_designer`` resolves
that proposal to concrete atom indices. This module reads how that CV actually
fluctuated on the just-run (vanilla) trajectory and sizes the Gaussian width,
then fills height and deposition pace from temperature and a rule-of-thumb
stride. Physics-unit numbers (SIGMA, HEIGHT, PACE) never pass through the LLM —
the same boundary ``cv_designer`` draws for atom indices.

Sizing:

- ``SIGMA`` ≈ (spread of the CV over the trajectory) / 3, subject to a physical
  per-CV-type floor. Metadynamics hills should be narrower than the basin they
  fill; the /3 rule of thumb is standard *once the basin has actually been
  sampled*. The floor exists because that precondition is systematically
  violated here: ``switch_to_metad`` fires precisely when the CV looks pinned,
  which is exactly when its measured spread is least trustworthy. Measured on a
  10 ps Trp-cage run, backbone Rg gave spread/3 ≈ 0.0018 nm — thermal jitter,
  not basin width. Hills that narrow never overlap, so the accumulated bias at
  each new deposit stays ≈ 0, well-tempering has nothing to damp, and WT-MetaD
  silently degenerates into plain metadynamics that fills nothing. Recorded as
  F4 in ``docs/activity-log.md``.

  The floors are the narrowest hill worth depositing for each coordinate type,
  not safety epsilons: below them you are resolving structure finer than the
  free-energy features anyone is trying to recover. A genuinely well-sampled CV
  measures wider than its floor and keeps its measured value.
- ``HEIGHT`` ≈ 0.5·k_B·T — the *initial* hill height W0. Conservative: deep
  enough to climb out of a basin over many deposits, shallow enough not to heat
  the system. Tied to the thermostat temperature so it tracks the run conditions
  (~1.25 kJ/mol at 300 K). Under well-tempering it decays from here.
- ``PACE`` = 500 steps. Standard deposition stride; system-independent.
- ``BIASFACTOR`` (γ) = 10. Well-tempered metadynamics: the deposition rate decays
  as exp(-V/k_B ΔT) with ΔT = (γ-1)T, so the bias converges to -(1 - 1/γ)F(s)
  rather than overfilling the basin indefinitely the way plain metaD does. γ=10
  puts ΔT = 2700 K, i.e. the sampled distribution flattens barriers up to
  ~γ·k_B·T ≈ 25 kJ/mol — the right order for the conformational barriers that
  trigger a ``switch_to_metad`` in the first place. Larger γ explores more
  aggressively but converges more slowly.

Torsion CVs use circular statistics for the spread (an ordinary stddev is wrong
across the ±π wrap); distance and gyration CVs use an ordinary stddev.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from mdpilot.adapters.plumed_writer import (
    CV,
    ContactsCV,
    DistanceCV,
    GyrationCV,
    MetadynamicsBias,
    RmsdCV,
    UpperWall,
    TorsionCV,
)

_KB_KJ_PER_MOL_K = 0.0083144621  # Boltzmann constant, kJ/mol/K
_DEFAULT_TEMPERATURE_K = 300.0
_DEFAULT_PACE = 500
_SIGMA_FRACTION = 1.0 / 3.0
_HEIGHT_KT_FRACTION = 0.5
_DEFAULT_BIAS_FACTOR = 10.0

# Narrowest hill worth depositing, per CV type. Distances and radii of gyration
# are nm; a 0.02 nm hill puts ~5 deposits across a 0.1 nm basin, which is enough
# overlap to fill. Torsions are radians; 0.15 rad (~8.6°) sits at the low end of
# the 0.1-0.35 rad range conventionally used for dihedral metadynamics.
# RMSD-to-native is a *global* coordinate: it spans ~1 nm between folded and
# unfolded for a small protein, where a distance or Rg between two chosen points
# spans a fraction of that. 0.05 nm (0.5 A) is the low end of the 0.05-0.1 nm
# range conventionally used for RMSD metadynamics, the same way 0.15 rad sits at
# the low end of the dihedral range.
_SIGMA_FLOORS: tuple[tuple[type, float], ...] = (
    (DistanceCV, 0.02),
    (GyrationCV, 0.02),
    (TorsionCV, 0.15),
    (RmsdCV, 0.05),
    # A contact fraction is dimensionless, not a length. A folded basin spans
    # roughly two to three formed contacts, so 0.5 *contacts* gives the same ~5
    # deposits across a basin that the length floors above are sized for — and
    # because the CV is a fraction, `sigma_floor` divides this by the pair
    # count to express it on the coordinate actually being biased.
    (ContactsCV, 0.5),
)

# The open tail of F4. The floors above catch a spread measured on a trajectory
# that never left its basin. A *replacement* CV proposed mid-metaD has the
# opposite pathology: it is sized on a biased trajectory, where the walker has
# been actively driven across the coordinate, so the spread is inflated and
# SIGMA comes out too wide. A hill wider than the features it is meant to
# resolve fills the surface uniformly and resolves nothing — the coordinate
# looks sampled and the free-energy profile is flat by construction.
#
# Sized as "the widest hill that still leaves ~5 deposits across the range the
# coordinate can span", the same rule the floors use from the other side.
_SIGMA_CEILINGS: tuple[tuple[type, float], ...] = (
    (DistanceCV, 0.2),
    (GyrationCV, 0.2),
    (RmsdCV, 0.2),
    (TorsionCV, 1.0),
    (ContactsCV, 2.0),
)


def design_bias(
    cv: CV,
    trajectory_path: Path,
    topology_path: Path,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
    pace: int = _DEFAULT_PACE,
    bias_factor: float = _DEFAULT_BIAS_FACTOR,
) -> MetadynamicsBias:
    """Size a single-CV metadynamics bias from the CV's fluctuation on a run.

    ``cv`` is a resolved PLUMED CV (atom indices already looked up by
    ``cv_designer``). ``trajectory_path`` is the vanilla trajectory the switch
    decision was made on; its spread along ``cv`` sets SIGMA, subject to the
    per-CV-type floor.
    """
    traj = md.load(str(trajectory_path), top=str(topology_path))
    series = cv_series(cv, traj)
    spread = _spread(cv, series)
    sigma, floored, ceiled = size_sigma(cv, spread)
    height = _HEIGHT_KT_FRACTION * _KB_KJ_PER_MOL_K * temperature_k
    return MetadynamicsBias(
        cv_labels=(cv.label,),
        sigma=(sigma,),
        height=height,
        pace=pace,
        sigma_floored=floored,
        sigma_ceiled=ceiled,
        bias_factor=bias_factor,
        temperature_k=temperature_k,
    )


def sigma_floor(cv: CV) -> float:
    """Narrowest hill width worth depositing for this CV type.

    Contacts bounds are stated in *contacts* and converted to the fraction the
    CV actually is, so they keep meaning the same thing whatever the pair count.
    """
    for cv_type, floor in _SIGMA_FLOORS:
        if isinstance(cv, cv_type):
            return floor / len(cv.pairs) if isinstance(cv, ContactsCV) else floor
    raise TypeError(f"bias_designer: no SIGMA floor for {type(cv).__name__}")


def sigma_ceiling(cv: CV) -> float:
    """Widest hill still worth depositing for this CV type."""
    for cv_type, ceiling in _SIGMA_CEILINGS:
        if isinstance(cv, cv_type):
            return ceiling / len(cv.pairs) if isinstance(cv, ContactsCV) else ceiling
    raise TypeError(f"bias_designer: no SIGMA ceiling for {type(cv).__name__}")


def size_sigma(cv: CV, spread: float) -> tuple[float, bool, bool]:
    """Hill width from an observed CV spread, and which bound clamped it.

    Returns ``(sigma, floored, ceiled)``. ``floored=True`` means the source
    trajectory measured narrower than the CV type's physical floor — a signal
    that the source phase under-sampled this coordinate, not that the
    coordinate is genuinely that stiff. ``ceiled=True`` means it measured
    *wider* than the widest hill worth depositing, which is what a spread taken
    from an already-biased trajectory looks like. Callers surface both rather
    than swallowing them: a silently substituted SIGMA is how F4 went unnoticed
    in the first place, and the same argument applies from either side.
    """
    measured = spread * _SIGMA_FRACTION
    floor = sigma_floor(cv)
    if measured < floor:
        return floor, True, False
    ceiling = sigma_ceiling(cv)
    if measured > ceiling:
        return ceiling, False, True
    return measured, False, False


def cv_series(cv: CV, traj: md.Trajectory) -> np.ndarray:
    """Per-frame CV value using the mdtraj primitive matching the CV type.

    Public because `mdpilot.observables` computes the campaign observable with
    it. The bias is sized from a coordinate and the campaign is judged on a
    coordinate; those must be the same computation or the two disagree with
    nothing to catch it.
    """
    if isinstance(cv, DistanceCV):
        return md.compute_distances(traj, np.array([cv.atoms]))[:, 0]
    if isinstance(cv, TorsionCV):
        return md.compute_dihedrals(traj, np.array([cv.atoms]))[:, 0]
    if isinstance(cv, GyrationCV):
        return md.compute_rg(traj.atom_slice(list(cv.atoms)))
    if isinstance(cv, RmsdCV):
        # The reference PDB holds exactly `cv.atoms`, in the same ascending
        # order `atom_slice` produces, so the two line up index-for-index.
        # md.rmsd superposes before measuring, matching PLUMED TYPE=OPTIMAL.
        reference = md.load(str(cv.reference_path))
        return md.rmsd(traj.atom_slice(list(cv.atoms)), reference, frame=0)
    if isinstance(cv, ContactsCV):
        # The same rational switching function PLUMED applies, evaluated here so
        # SIGMA is sized in the CV's own units (contacts) rather than in nm.
        # With MM == 2*NN the closed form 1/(1+x**NN) is exact and has no
        # singularity at x == 1, where (1-x**NN)/(1-x**MM) is 0/0.
        distances = md.compute_distances(traj, np.array(cv.pairs))
        switched = 1.0 / (1.0 + (distances / cv.r0_nm) ** cv.NN)
        # Divided by the pair count, matching what `ContactsCV.render` biases:
        # PLUMED's COMBINE turns the SUM into a fraction, so SIGMA has to be
        # sized in fractions too or the hills are `len(pairs)` times too wide.
        return switched.sum(axis=1) / len(cv.pairs)
    raise TypeError(f"bias_designer: unsupported CV type {type(cv).__name__}")


def _spread(cv: CV, series: np.ndarray) -> float:
    if isinstance(cv, TorsionCV):
        return _circular_std(series)
    return float(np.std(series))


def _circular_std(angles: np.ndarray) -> float:
    """Circular standard deviation of angles (radians).

    A linear stddev is wrong near the ±π wrap. Uses the mean resultant length
    R: std = sqrt(-2 ln R). R→0 (maximally dispersed) is clamped to π.
    """
    r = float(np.abs(np.mean(np.exp(1j * angles))))
    if r <= 0.0:
        return float(np.pi)
    return float(np.sqrt(-2.0 * np.log(r)))


# Stiff enough that the walker turns around within ~0.1 nm of the wall
# (1000 * 0.1^2 = 10 kJ/mol, ~4 kT), soft enough not to shock the dynamics.
_WALL_KAPPA_KJ_PER_MOL_NM2 = 1000.0

# CVs measured in nanometres. A wall position in nm is meaningful only for
# these; a torsion is already bounded on [-pi, pi], and a contact count is
# bounded on [0, n_pairs] — neither needs a wall.
_LENGTH_DIMENSIONED = (DistanceCV, GyrationCV, RmsdCV)


# Solvent that must remain between the solute and its periodic image. The
# nonbonded cutoff: closer than this and the solute interacts with its own
# image directly, which is F11.
_MIN_IMAGE_CLEARANCE_NM = 1.0
# Below this correlation the CV does not predict the solute's extent well
# enough to extrapolate a box limit from, and guessing is worse than declining.
_MIN_SPAN_CORRELATION = 0.15


def box_limited_wall(
    cv: CV,
    trajectory_path: Path,
    topology_path: Path,
    *,
    clearance_nm: float = _MIN_IMAGE_CLEARANCE_NM,
) -> float | None:
    """The CV value at which the solute would reach the edge of its own box.

    Measured, not assumed. For every frame of the source trajectory this takes
    the CV and the solute's widest extent, fits extent against CV, and solves
    for the extent at which only `clearance_nm` of solvent would remain between
    the solute and its periodic image.

    This exists because the relationship between the padding a campaign asks
    for and the clearance it actually gets is not obvious — measured on CLN025,
    padding of 1.0 / 1.5 / 2.0 nm produced 0.88 / 1.10 / 1.38 nm of clearance
    around the *folded* structure, and the unfolded ensemble a biased run goes
    looking for is far wider than that. A wall chosen by hand from the task's
    own state thresholds knows none of this: the 0.8 nm wall on the first
    CLN025 campaign sat well above what its box could hold.

    Returns None when the fit cannot support the extrapolation — a folded
    trajectory barely varies, and a bad ceiling is worse than none.
    """
    if not isinstance(cv, _LENGTH_DIMENSIONED):
        return None
    traj = md.load(str(trajectory_path), top=str(topology_path))
    if traj.unitcell_lengths is None or traj.n_frames < 8:
        return None
    solute = traj.topology.select("protein and element != H")
    if solute.size < 2:
        solute = traj.topology.select("protein")
    if solute.size < 2:
        return None

    values = cv_series(cv, traj)
    span = np.array([np.ptp(f, axis=0).max() for f in traj.atom_slice(solute).xyz])
    if np.std(values) <= 0 or np.std(span) <= 0:
        return None
    if abs(float(np.corrcoef(values, span)[0, 1])) < _MIN_SPAN_CORRELATION:
        return None

    slope, intercept = np.polyfit(values, span, 1)
    if slope <= 0:
        return None
    tolerable_span = float(traj.unitcell_lengths[:, 0].min()) - clearance_nm
    limit = (tolerable_span - float(intercept)) / float(slope)
    return limit if limit > 0 else None


def design_upper_wall(
    cv: CV,
    at_nm: float | None,
    *,
    trajectory_path: Path | None = None,
    topology_path: Path | None = None,
) -> UpperWall | None:
    """Bound an unbounded CV from above, or None if no wall applies.

    Returns None when the CV is not length-dimensioned — a wall at "0.8" means
    nothing on a torsion, which is already bounded on [-pi, pi], nor on a
    contact count bounded on [0, n_pairs].

    An explicit `at_nm` from the campaign wins: the task knows what counts as
    unfolded, and this is not the place to overrule it. What is added here is
    the *box* limit, measured from the trajectory — used as the position when
    the campaign gave none, and recorded either way so a wall placed beyond
    what the box can hold does not pass silently.
    """
    if not isinstance(cv, _LENGTH_DIMENSIONED):
        return None
    limit = (
        box_limited_wall(cv, trajectory_path, topology_path)
        if trajectory_path is not None and topology_path is not None
        else None
    )
    position = at_nm if at_nm is not None else limit
    if position is None:
        return None
    return UpperWall(
        cv_label=cv.label,
        at=position,
        kappa=_WALL_KAPPA_KJ_PER_MOL_NM2,
        box_limit_nm=limit,
        derived_from_box=at_nm is None,
    )
