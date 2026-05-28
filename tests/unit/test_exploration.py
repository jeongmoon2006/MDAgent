"""Exploration diagnostic separates a pinned single basin from a multi-basin series.

The decisive science test for M4: a metastable single basin (which the
convergence diagnostics mistake for a converged plateau) must read as
`exploring=False`, while a trajectory that visits a second state must read as
`exploring=True`. Fixtures are synthetic 1D series, matching the rest of the
diagnostics suite — no slow real-trajectory load needed for the property under
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdpilot.diagnostics.exploration import exploration


def _ar1(n: int, phi: float, *, seed: int) -> np.ndarray:
    """Stationary AR(1): autocorrelated but unimodal — one basin."""
    rng = np.random.default_rng(seed)
    sigma_eps = np.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_eps * rng.normal()
    return x


def _two_state_telegraph(
    n: int, *, sep: float, p_switch: float, seed: int
) -> np.ndarray:
    """Two-basin series: dwells near 0 or `sep`, switching with prob p_switch,
    with within-basin Gaussian noise. Long dwells => strongly bimodal marginal."""
    rng = np.random.default_rng(seed)
    state = 0
    out = np.empty(n)
    for t in range(n):
        if rng.random() < p_switch:
            state = 1 - state
        out[t] = state * sep + 0.3 * rng.normal()
    return out


def test_single_basin_iid_normal_is_pinned() -> None:
    """k-means would split a normal at its mean; BC must not be fooled."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000)
    r = exploration(x)
    assert r.bimodality_coefficient < 5.0 / 9.0, r.bimodality_coefficient
    assert r.n_basins == 1
    assert r.exploring is False


def test_single_basin_slow_ar1_is_pinned() -> None:
    """Strong autocorrelation (slow wander within one basin, the Trp-cage 5 ns
    regime) is still a single basin — low ESS must not be read as multi-state."""
    x = _ar1(50_000, phi=0.95, seed=123)
    r = exploration(x)
    assert r.bimodality_coefficient < 5.0 / 9.0, r.bimodality_coefficient
    assert r.n_basins == 1
    assert r.exploring is False


def test_two_basin_telegraph_is_exploring() -> None:
    x = _two_state_telegraph(20_000, sep=6.0, p_switch=0.002, seed=7)
    r = exploration(x)
    assert r.bimodality_coefficient > 5.0 / 9.0, r.bimodality_coefficient
    assert r.n_basins == 2
    assert r.minor_basin_occupancy > 0.1
    assert r.exploring is True


def test_lopsided_minor_basin_below_occupancy_gate_is_pinned() -> None:
    """A brief excursion to a second value (≈1% of frames) is not a populated
    second basin: BC may rise but the occupancy gate keeps the verdict pinned."""
    rng = np.random.default_rng(3)
    x = 0.3 * rng.standard_normal(20_000)
    x[:150] += 8.0  # ~0.75% of frames parked far away
    r = exploration(x)
    assert r.minor_basin_occupancy < 0.05
    assert r.n_basins == 1
    assert r.exploring is False


def test_constant_series_is_single_basin() -> None:
    r = exploration(np.full(100, 2.5))
    assert r.bimodality_coefficient == 0.0
    assert r.n_basins == 1
    assert r.exploring is False


def test_too_few_samples_raises() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        exploration(np.arange(5.0))
