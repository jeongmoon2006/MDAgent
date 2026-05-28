"""Report bundle composes diagnostics into compact JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mdpilot.diagnostics.report import _summarize, to_json


def _ar1(n: int, phi: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma_eps = np.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_eps * rng.normal()
    return x


def test_summarize_under_eight_frames_degrades_gracefully(tmp_path: Path) -> None:
    obs = np.array([1.0, 1.1, 1.05])
    report = _summarize(
        observable=obs,
        observable_name="rmsd",
        dcd_path=tmp_path / "x.dcd",
        top_path=tmp_path / "top.pdb",
        frame_dt_ps=1.0,
        length_ns=0.003,
    )
    assert report["n_frames"] == 3
    assert report["plateau_reached"] is False
    assert report["well_sampled"] is False
    assert report["exploring"] is None  # key present even when statistics skipped
    assert "too few frames" in report["note"]


def test_summarize_correlated_series_flags_under_converged(tmp_path: Path) -> None:
    obs = _ar1(60, phi=0.95, seed=7)  # short + highly correlated → ESS small
    report = _summarize(
        observable=obs,
        observable_name="rmsd",
        dcd_path=tmp_path / "x.dcd",
        top_path=tmp_path / "top.pdb",
        frame_dt_ps=1.0,
        length_ns=0.060,
    )
    assert report["ess"] < 50
    assert report["well_sampled"] is False


def test_summarize_long_iid_series_flags_well_sampled(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(5_000)
    report = _summarize(
        observable=obs,
        observable_name="rmsd",
        dcd_path=tmp_path / "x.dcd",
        top_path=tmp_path / "top.pdb",
        frame_dt_ps=1.0,
        length_ns=5.0,
    )
    assert report["well_sampled"] is True
    assert report["plateau_reached"] is True
    assert report["ess"] > 1000
    assert report["exploring"] is False  # unimodal single basin
    assert report["n_basins"] == 1


def test_to_json_is_compact_and_deterministic(tmp_path: Path) -> None:
    obs = _ar1(2_000, phi=0.7, seed=1)
    report = _summarize(
        observable=obs,
        observable_name="rmsd",
        dcd_path=tmp_path / "x.dcd",
        top_path=tmp_path / "top.pdb",
        frame_dt_ps=1.0,
        length_ns=2.0,
    )
    s = to_json(report)
    assert len(s) < 2048, f"report JSON is {len(s)} bytes; should be < 2 KB"
    assert json.loads(s)["n_frames"] == 2_000
    # determinism: keys sorted
    assert s == to_json(report)
