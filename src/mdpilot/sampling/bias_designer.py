"""Deterministic metadynamics bias parameters from a resolved CV + prior trajectory.

The scientist (LLM) proposes *which* coordinate is slow; ``cv_designer`` resolves
that proposal to concrete atom indices. This module reads how that CV actually
fluctuated on the just-run (vanilla) trajectory and sizes the Gaussian width,
then fills height and deposition pace from temperature and a rule-of-thumb
stride. Physics-unit numbers (SIGMA, HEIGHT, PACE) never pass through the LLM —
the same boundary ``cv_designer`` draws for atom indices.

Sizing:

- ``SIGMA`` ≈ (spread of the CV over the trajectory) / 3. Metadynamics hills
  should be narrower than the basin they fill; the /3 rule of thumb is standard.
  Floored to a small epsilon so a tightly-pinned CV (the exact case that
  triggers ``switch_to_metad``) cannot yield ``SIGMA=0``, which PLUMED rejects.
- ``HEIGHT`` ≈ 0.5·k_B·T. Conservative: deep enough to climb out of a basin over
  many deposits, shallow enough not to heat the system. Tied to the thermostat
  temperature so it tracks the run conditions (~1.25 kJ/mol at 300 K).
- ``PACE`` = 500 steps. Standard deposition stride; system-independent.

Torsion CVs use circular statistics for the spread (an ordinary stddev is wrong
across the ±π wrap); distance and gyration CVs use an ordinary stddev.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from mdpilot.adapters.plumed_writer import (
    CV,
    DistanceCV,
    GyrationCV,
    MetadynamicsBias,
    TorsionCV,
)

_KB_KJ_PER_MOL_K = 0.0083144621  # Boltzmann constant, kJ/mol/K
_DEFAULT_TEMPERATURE_K = 300.0
_DEFAULT_PACE = 500
_SIGMA_FRACTION = 1.0 / 3.0
_SIGMA_FLOOR = 1e-3  # CV units (nm or rad); keeps SIGMA>0 for a pinned CV
_HEIGHT_KT_FRACTION = 0.5


def design_bias(
    cv: CV,
    trajectory_path: Path,
    topology_path: Path,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
    pace: int = _DEFAULT_PACE,
) -> MetadynamicsBias:
    """Size a single-CV metadynamics bias from the CV's fluctuation on a run.

    ``cv`` is a resolved PLUMED CV (atom indices already looked up by
    ``cv_designer``). ``trajectory_path`` is the vanilla trajectory the switch
    decision was made on; its spread along ``cv`` sets SIGMA.
    """
    traj = md.load(str(trajectory_path), top=str(topology_path))
    series = _cv_series(cv, traj)
    spread = _spread(cv, series)
    sigma = max(spread * _SIGMA_FRACTION, _SIGMA_FLOOR)
    height = _HEIGHT_KT_FRACTION * _KB_KJ_PER_MOL_K * temperature_k
    return MetadynamicsBias(
        cv_labels=(cv.label,),
        sigma=(sigma,),
        height=height,
        pace=pace,
    )


def _cv_series(cv: CV, traj: md.Trajectory) -> np.ndarray:
    """Per-frame CV value using the mdtraj primitive matching the CV type."""
    if isinstance(cv, DistanceCV):
        return md.compute_distances(traj, np.array([cv.atoms]))[:, 0]
    if isinstance(cv, TorsionCV):
        return md.compute_dihedrals(traj, np.array([cv.atoms]))[:, 0]
    if isinstance(cv, GyrationCV):
        return md.compute_rg(traj.atom_slice(list(cv.atoms)))
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
