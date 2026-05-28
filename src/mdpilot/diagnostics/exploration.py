"""Is the observable pinned in one basin, or exploring multiple states?

This is the signal the convergence diagnostics (block-averaging,
autocorrelation) cannot supply. A metastable basin reads as a *converged
plateau*: low variance, a flat mean, an apparently settled observable. The
convergence rubric would call that "looks done" — which is exactly backwards
when the question requires a transition the trajectory never made. The job
here is to separate "pinned in a single basin" from "visited multiple basins",
so the scientist can tell *not-yet-converged* (extend) from *vanilla MD is
inadequate* (switch to enhanced sampling) once a task-level expectation says a
transition was required.

Method: Sarle's bimodality coefficient (BC) on the observable's marginal
distribution. BC ≈ 1/3 for a normal, 5/9 for a uniform, and → 1 for a
well-separated two-state mixture; 5/9 is the conventional unimodal/bimodal
cutoff. BC is chosen over a k-means split because k-means manufactures a
"separation" even on unimodal data (splitting a standard normal at its mean
yields cluster means ~2.6σ apart), so a separation-of-means criterion would
false-positive on a single basin. When BC clears the cutoff, a 1D 2-means
gives the minor-basin occupancy as a populated-second-state safety gate.

Reference: Sarle, SAS/STAT User's Guide (1990); Pfister et al., "Good things
peak in pairs", Front. Psychol. 4:700 (2013), for the finite-sample form.

Limitations (documented, not bugs): a single barrier crossing (unfolded →
folded, then stable) produces a bimodal marginal and reads as `exploring` —
which is correct, vanilla MD *was* able to cross, so it is not inadequate,
merely possibly under-sampled (the ESS in the convergence report catches
that). A pure monotonic drift with no basin structure can also inflate BC; in
practice MD observables dwell, they do not ramp linearly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BC_CUTOFF = 5.0 / 9.0          # Sarle's unimodal/bimodal threshold ≈ 0.555
_MIN_OCCUPANCY = 0.05           # minor basin must hold ≥5% of frames to count


@dataclass(frozen=True)
class ExplorationResult:
    bimodality_coefficient: float
    n_basins: int                   # 1 (pinned) or 2 (visited a second state)
    minor_basin_occupancy: float    # fraction of frames in the minority basin
    exploring: bool                 # n_basins >= 2
    n_samples: int


def exploration(x: np.ndarray, *, bc_cutoff: float = _BC_CUTOFF) -> ExplorationResult:
    """Single-basin (pinned) vs multi-basin (exploring) verdict for a 1D series.

    `bimodality_coefficient` is the deciding statistic; `bc_cutoff` (default
    5/9) is the unimodal/bimodal threshold. The verdict is `exploring` when BC
    clears the cutoff *and* a 1D 2-means split puts at least `_MIN_OCCUPANCY`
    of frames in the minority basin (so a transient blip that inflates BC
    without populating a second state stays "pinned").

    BC is unreliable below ~30 samples (sample kurtosis is noisy); callers
    should gate on frame count before trusting the verdict.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    if n < 8:
        raise ValueError(f"need at least 8 samples, got {n}")

    centered = arr - arr.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:  # constant series — one (degenerate) basin
        return ExplorationResult(
            bimodality_coefficient=0.0,
            n_basins=1,
            minor_basin_occupancy=0.0,
            exploring=False,
            n_samples=n,
        )

    bc = _bimodality_coefficient(centered, m2, n)
    minor_occ = _minor_basin_occupancy(arr) if bc > bc_cutoff else 0.0
    bimodal = bc > bc_cutoff and minor_occ >= _MIN_OCCUPANCY

    return ExplorationResult(
        bimodality_coefficient=bc,
        n_basins=2 if bimodal else 1,
        minor_basin_occupancy=minor_occ,
        exploring=bimodal,
        n_samples=n,
    )


def _bimodality_coefficient(centered: np.ndarray, m2: float, n: int) -> float:
    """Sarle's BC with finite-sample-corrected skewness G1 and kurtosis G2."""
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    g1 = m3 / m2**1.5            # biased sample skewness
    g2 = m4 / m2**2 - 3.0        # biased sample excess kurtosis
    G1 = g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)
    G2 = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    return float((G1**2 + 1.0) / (G2 + 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))))


def _minor_basin_occupancy(arr: np.ndarray, *, max_iter: int = 50) -> float:
    """Fraction of frames in the smaller of two 1D k-means clusters."""
    lo, hi = float(arr.min()), float(arr.max())
    centroids = np.array([lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)])
    labels = np.zeros(arr.size, dtype=int)
    for _ in range(max_iter):
        labels = (np.abs(arr - centroids[1]) < np.abs(arr - centroids[0])).astype(int)
        new = np.array([
            arr[labels == 0].mean() if np.any(labels == 0) else centroids[0],
            arr[labels == 1].mean() if np.any(labels == 1) else centroids[1],
        ])
        if np.allclose(new, centroids):
            break
        centroids = new
    frac1 = float(np.mean(labels == 1))
    return min(frac1, 1.0 - frac1)
