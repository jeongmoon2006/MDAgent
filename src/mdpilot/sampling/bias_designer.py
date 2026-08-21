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
    # A contact count is dimensionless, not a length. A folded basin spans
    # roughly two to three formed contacts, so 0.5 gives the same ~5 deposits
    # across a basin that the length floors above are sized for.
    (ContactsCV, 0.5),
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
    series = _cv_series(cv, traj)
    spread = _spread(cv, series)
    sigma, floored = size_sigma(cv, spread)
    height = _HEIGHT_KT_FRACTION * _KB_KJ_PER_MOL_K * temperature_k
    return MetadynamicsBias(
        cv_labels=(cv.label,),
        sigma=(sigma,),
        height=height,
        pace=pace,
        sigma_floored=floored,
        bias_factor=bias_factor,
        temperature_k=temperature_k,
    )


def sigma_floor(cv: CV) -> float:
    """Narrowest hill width worth depositing for this CV type."""
    for cv_type, floor in _SIGMA_FLOORS:
        if isinstance(cv, cv_type):
            return floor
    raise TypeError(f"bias_designer: no SIGMA floor for {type(cv).__name__}")


def size_sigma(cv: CV, spread: float) -> tuple[float, bool]:
    """Hill width from an observed CV spread, and whether the floor bound.

    Returns ``(sigma, floored)``. ``floored=True`` means the source trajectory
    measured narrower than the CV type's physical floor — a signal that the
    vanilla phase under-sampled this coordinate, not that the coordinate is
    genuinely that stiff. Callers surface it rather than swallowing it: a
    silently substituted SIGMA is how F4 went unnoticed in the first place.
    """
    measured = spread * _SIGMA_FRACTION
    floor = sigma_floor(cv)
    if measured < floor:
        return floor, True
    return measured, False


def _cv_series(cv: CV, traj: md.Trajectory) -> np.ndarray:
    """Per-frame CV value using the mdtraj primitive matching the CV type."""
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
        return switched.sum(axis=1)
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


def design_upper_wall(cv: CV, at_nm: float | None) -> UpperWall | None:
    """Bound an unbounded CV from above, or None if no wall applies.

    Returns None when no wall was requested, and also when the CV is not
    length-dimensioned — a wall at "0.8" means nothing on a torsion, which is
    already bounded on [-pi, pi]. The position comes from the campaign (the
    task knows what counts as unfolded); only the stiffness is a constant here,
    the same boundary `design_bias` draws for SIGMA and HEIGHT.
    """
    if at_nm is None or not isinstance(cv, _LENGTH_DIMENSIONED):
        return None
    return UpperWall(
        cv_label=cv.label, at=at_nm, kappa=_WALL_KAPPA_KJ_PER_MOL_NM2
    )
