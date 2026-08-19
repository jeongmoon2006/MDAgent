"""`plumed sum_hills` against real PLUMED, on HILLS files with known shape.

The unit tests build free-energy surfaces analytically and never invoke PLUMED.
What they cannot check is the part that talks to the binary: the strided output
naming (`fes.dat0.dat`, not `fes_0.dat`), the header dialect sum_hills emits,
and whether a HILLS file this codebase produced can actually be read back.

HILLS is synthesized here rather than simulated. Hills deposited in two
clusters integrate to a two-basin surface, which gives barrier and recrossing
detection something real to find without waiting on a barrier crossing in MD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdpilot.diagnostics.free_energy import (
    load_fes,
    metad_report,
    plumed_available,
    sum_hills,
)

pytestmark = pytest.mark.skipif(
    not plumed_available(), reason="no PLUMED runtime on PATH"
)

_SIGMA = 0.15
_GAMMA = 10.0


def _write_hills(path: Path, centres: np.ndarray, label: str = "cv") -> Path:
    """A minimal well-tempered HILLS file in PLUMED's own format."""
    lines = [
        f"#! FIELDS time {label} sigma_{label} height biasf",
        "#! SET multivariate false",
        "#! SET kerneltype stretched-gaussian",
    ]
    for i, c in enumerate(centres):
        lines.append(
            f"  {float(i + 1):>10.3f} {c:>20.12f} {_SIGMA:>14.6f} "
            f"{1.2:>20.12f} {_GAMMA:>20.1f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _two_basin_centres(seed: int = 3) -> np.ndarray:
    """Hills clustered at -1 and +1, more at -1 so that basin ends up deeper."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(-1.0, 0.12, 90), rng.normal(+1.0, 0.12, 45)]
    )


def _write_colvar(path: Path, series: np.ndarray, label: str = "cv") -> Path:
    lines = [f"#! FIELDS time {label} metad.bias"]
    for i, x in enumerate(series):
        lines.append(f"{float(i):.6f} {x:.6f} {0.5 * i:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_sum_hills_produces_a_readable_surface(tmp_path: Path) -> None:
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())

    surfaces = sum_hills(hills, tmp_path / "out")

    assert len(surfaces) == 1
    fes = load_fes(surfaces[0])
    assert fes.cv_label == "cv"
    assert fes.cv.size > 10
    assert fes.free_energy.min() == pytest.approx(0.0, abs=1e-6)


def test_strided_sum_hills_returns_surfaces_in_time_order(tmp_path: Path) -> None:
    """PLUMED appends the index and another `.dat` to --outfile, so the files
    are `fes.dat0.dat`… — a glob written for `fes_0.dat` silently finds none."""
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())

    surfaces = sum_hills(hills, tmp_path / "out", stride=30)

    assert len(surfaces) >= 3
    assert [p.name for p in surfaces] == sorted(
        (p.name for p in surfaces),
        key=lambda n: int(n.rsplit(".dat", 2)[1] or 0),
    )
    # Later estimates integrate more hills, so the surface gets deeper.
    depths = [load_fes(p).free_energy.max() for p in surfaces]
    assert depths[-1] > depths[0]


def test_clustered_hills_integrate_to_two_basins(tmp_path: Path) -> None:
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())

    fes = load_fes(sum_hills(hills, tmp_path / "out")[0])
    minima = fes.minima()

    assert len(minima) == 2
    # Twice as many hills landed near -1, so that basin must be the deeper one.
    assert float(fes.cv[minima[0]]) < 0.0
    barrier = fes.barrier_kj_per_mol()
    assert barrier is not None and barrier > 0.0


def test_report_flags_a_crossing_run_as_converged(tmp_path: Path) -> None:
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())
    # A walker that shuttles between the two basins several times.
    series = np.concatenate([np.full(40, s) for s in (-1, 1, -1, 1, -1)])
    colvar = _write_colvar(tmp_path / "COLVAR", series)

    report = metad_report(hills, colvar, tmp_path / "out", stride=30)

    assert report["n_basins_fes"] == 2
    assert report["barrier_kj_per_mol"] > 0.0
    assert report["recrossings"] >= 3
    assert report["barrier_crossed"] is True


def test_report_refuses_to_call_a_trapped_run_converged(tmp_path: Path) -> None:
    """The guard that matters: a walker pinned in one basin still produces a
    surface whose successive estimates agree, because nothing new is sampled.
    Low drift alone must not read as convergence."""
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())
    colvar = _write_colvar(tmp_path / "COLVAR", np.full(200, -1.0))

    report = metad_report(hills, colvar, tmp_path / "out", stride=30)

    assert report["recrossings"] == 0
    assert report["barrier_crossed"] is False
    assert report["fes_converged"] is False


def test_report_withholds_a_verdict_without_colvar(tmp_path: Path) -> None:
    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())

    report = metad_report(hills, None, tmp_path / "out", stride=30)

    assert report["recrossings"] is None
    assert report["fes_converged"] is None
    assert report["fes_drift_kj_per_mol"] is not None


def test_report_is_json_serializable(tmp_path: Path) -> None:
    """Same contract as the vanilla diagnostic bundle — scalars and paths, so
    it can go into SQLite and the agent's context unchanged."""
    import json

    hills = _write_hills(tmp_path / "HILLS", _two_basin_centres())
    colvar = _write_colvar(tmp_path / "COLVAR", np.full(50, -1.0))

    report = metad_report(hills, colvar, tmp_path / "out", stride=30)

    assert json.loads(json.dumps(report)) == report
