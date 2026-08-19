"""Read a metadynamics HILLS file back into a free-energy surface.

Until now the deposited bias was write-only: HILLS accumulated on disk and
nothing ever read it, so rounds after a `switch_to_metad` were still judged by
the *unbiased* convergence rubric — block-averaged RMSD, ESS, bimodality. Those
statistics describe an equilibrium ensemble. A biased trajectory is not one:
the bias actively drives the observable, so a long τ_int means the bias is
still filling, and a bimodal marginal means the bias worked rather than that
the system is exploring freely. Reading them as convergence evidence is a
category error.

What replaces them is the free-energy surface itself, plus the standard
well-tempered convergence check: integrate HILLS at increasing time strides and
ask whether the surface has stopped changing.

Summation is delegated to `plumed sum_hills` rather than reimplemented. The
arithmetic looks trivial — sum Gaussians on a grid — but PLUMED writes
well-tempered heights pre-scaled by γ/(γ-1) and deposits *stretched* Gaussians
by default (`#! SET kerneltype stretched-gaussian`), so a hand-rolled sum would
silently disagree with the file it claims to integrate.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_KB_KJ_PER_MOL_K = 0.0083144621
_DEFAULT_TEMPERATURE_K = 300.0
_SUM_HILLS_TIMEOUT_S = 600.0
# A dip shallower than this is thermal noise on the surface, not a basin.
_MIN_PROMINENCE_KT = 1.0


class PlumedNotAvailable(RuntimeError):
    """`plumed` is not on PATH, so HILLS cannot be integrated."""


def _uphill_peak(f: np.ndarray, start: int, step: int) -> float:
    """Walk uphill from `start` in direction `step`; return the crest reached.

    Stops at the first turn downward, so this is the local maximum enclosing
    `start` on that side rather than the highest point anywhere beyond it.
    """
    i = start
    while 0 <= i + step < f.size and f[i + step] >= f[i]:
        i += step
    return float(f[i])


@dataclass(frozen=True)
class FreeEnergySurface:
    """A 1-D free-energy profile along one collective variable, in kJ/mol."""

    cv_label: str
    cv: np.ndarray
    free_energy: np.ndarray
    periodic: bool

    def minima(self, temperature_k: float = _DEFAULT_TEMPERATURE_K) -> list[int]:
        """Indices of local minima deep enough to be basins, deepest first.

        Prominence is measured against the *enclosing* local maximum on each
        side — the crest you would actually climb to leave this dip — and the
        shallower of the two is taken. Using the global maximum on each side
        instead would give every ripple at the bottom of a deep well the
        prominence of the whole well, manufacturing basins out of sub-kT
        sampling noise.
        """
        f = self.free_energy
        if f.size < 3:
            return []
        cutoff = _MIN_PROMINENCE_KT * _KB_KJ_PER_MOL_K * temperature_k
        found: list[int] = []
        for i in range(1, f.size - 1):
            if f[i] <= f[i - 1] and f[i] < f[i + 1]:
                left = _uphill_peak(f, i, -1)
                right = _uphill_peak(f, i, +1)
                if min(left, right) - float(f[i]) >= cutoff:
                    found.append(i)
        # The deepest point on the surface is a basin by definition. Prominence
        # filtering exists to suppress *extra* basins, and it would otherwise
        # discard the global minimum whenever that minimum happens to sit in a
        # shallow ripple — or lie on a boundary the interior scan never visits.
        global_min = int(np.argmin(f))
        if global_min not in found:
            found.append(global_min)
        found.sort(key=lambda i: float(f[i]))
        return found

    def barrier_kj_per_mol(
        self, temperature_k: float = _DEFAULT_TEMPERATURE_K
    ) -> float | None:
        """Height of the highest point between the two deepest basins,
        measured from the deeper one. None when only one basin is resolved."""
        m = self.minima(temperature_k)
        if len(m) < 2:
            return None
        deepest = m[0]  # minima() returns deepest-first
        lo, hi = sorted(m[:2])
        crest = float(self.free_energy[lo : hi + 1].max())
        return crest - float(self.free_energy[deepest])

    def depth_kj_per_mol(self) -> float:
        return float(self.free_energy.max() - self.free_energy.min())


def plumed_available() -> bool:
    return shutil.which("plumed") is not None


def sum_hills(
    hills_path: Path,
    out_dir: Path,
    *,
    stride: int | None = None,
    basename: str = "fes",
) -> list[Path]:
    """Integrate HILLS into one or more free-energy surfaces.

    With `stride`, PLUMED emits a cumulative surface every `stride` hills. It
    builds those names by appending the index *and* another `.dat` to whatever
    `--outfile` was given, so `--outfile fes.dat` yields `fes.dat0.dat`,
    `fes.dat1.dat`, … — not the `fes_0.dat` the naming suggests. Returns the
    surfaces in time order, so the last element is the most complete.

    `--mintozero` is always passed: absolute free energies from sum_hills carry
    an arbitrary offset, and successive estimates cannot be compared without a
    common reference.
    """
    if not plumed_available():
        raise PlumedNotAvailable(
            "`plumed` is not on PATH, so HILLS cannot be integrated into a "
            "free-energy surface. Install a PLUMED runtime (conda-forge "
            "`plumed`); see the README."
        )
    hills_path = Path(hills_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{basename}.dat"

    cmd = [
        "plumed", "sum_hills",
        "--hills", str(hills_path),
        "--outfile", str(outfile),
        "--mintozero",
    ]
    if stride is not None:
        cmd += ["--stride", str(stride)]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        timeout=_SUM_HILLS_TIMEOUT_S,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            f"`plumed sum_hills` failed (exit {result.returncode}) on "
            f"{hills_path}\n--- stderr tail ---\n{tail}"
        )

    if stride is None:
        return [outfile]
    pattern = re.compile(rf"^{re.escape(outfile.name)}(\d+)\.dat$")
    indexed: list[tuple[int, Path]] = []
    for candidate in out_dir.iterdir():
        match = pattern.match(candidate.name)
        if match:
            indexed.append((int(match.group(1)), candidate))
    if not indexed:
        return [outfile]
    indexed.sort()
    surfaces = [p for _, p in indexed]
    # When the hill count is a multiple of `stride`, sum_hills writes the final
    # complete surface *and* the last strided one, and they are byte-identical.
    # A drift measured across that pair is exactly 0.0 by construction — which
    # is what silently made the drift half of the convergence test vacuous.
    if len(surfaces) >= 2 and surfaces[-1].read_bytes() == surfaces[-2].read_bytes():
        surfaces.pop()
    return surfaces


def load_fes(path: Path) -> FreeEnergySurface:
    """Parse a sum_hills output file into a FreeEnergySurface."""
    path = Path(path)
    label = "cv"
    periodic = False
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#! FIELDS"):
            fields = line.split()[2:]
            if fields:
                label = fields[0]
        elif line.startswith("#! SET periodic"):
            periodic = line.split()[-1].strip().lower() == "true"
        elif line.startswith("#") or not line.strip():
            continue
        else:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError(f"no free-energy rows parsed from {path}")
    arr = np.array(rows)
    return FreeEnergySurface(
        cv_label=label, cv=arr[:, 0], free_energy=arr[:, 1], periodic=periodic
    )


def fes_drift_kj_per_mol(
    early: FreeEnergySurface, late: FreeEnergySurface
) -> float | None:
    """Largest change between two successive surface estimates.

    The standard well-tempered convergence test: once the surface stops moving
    as more hills land, it is converged. Compared only over the region both
    estimates resolve — sum_hills grids the range the walker has visited, so a
    later estimate is usually wider and the extra region has no counterpart.
    """
    lo = max(early.cv.min(), late.cv.min())
    hi = min(early.cv.max(), late.cv.max())
    if not (hi > lo):
        return None
    grid = np.linspace(lo, hi, 200)
    a = np.interp(grid, early.cv, early.free_energy)
    b = np.interp(grid, late.cv, late.free_energy)
    # Re-reference: --mintozero anchors each estimate at its own minimum, which
    # may sit in a different basin once the surface changes shape.
    a = a - a.min()
    b = b - b.min()
    return float(np.abs(b - a).max())


def load_colvar(path: Path) -> dict[str, np.ndarray]:
    """Parse a PLUMED COLVAR file into named columns."""
    path = Path(path)
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#! FIELDS"):
            names = line.split()[2:]
        elif line.startswith("#") or not line.strip():
            continue
        else:
            parts = line.split()
            if names and len(parts) == len(names):
                rows.append([float(p) for p in parts])
    if not names or not rows:
        raise ValueError(f"no COLVAR rows parsed from {path}")
    arr = np.array(rows)
    return {name: arr[:, i] for i, name in enumerate(names)}


def basin_thresholds(
    fes: FreeEnergySurface, temperature_k: float = _DEFAULT_TEMPERATURE_K
) -> tuple[float, float] | None:
    """CV values marking "has arrived in" each of the two deepest basins.

    Placed halfway between the barrier crest and each basin minimum, not at the
    minima themselves. A walker oscillating inside its basin rarely sits exactly
    on the minimum, so anchoring the test there would miss transitions the
    walker genuinely completed; anchoring it at the crest would count every
    thermal excursion onto the barrier. Halfway is the usual committor-style
    compromise. Returns None when fewer than two basins are resolved.
    """
    m = fes.minima(temperature_k)
    if len(m) < 2:
        return None
    lo_i, hi_i = sorted(m[:2])
    crest_i = lo_i + int(np.argmax(fes.free_energy[lo_i : hi_i + 1]))
    lo_cv, hi_cv = float(fes.cv[lo_i]), float(fes.cv[hi_i])
    crest_cv = float(fes.cv[crest_i])
    return (lo_cv + crest_cv) / 2.0, (crest_cv + hi_cv) / 2.0


def count_recrossings(cv_series: np.ndarray, low: float, high: float) -> int:
    """Transitions between two basins, with the region between them as
    hysteresis.

    A bare midpoint threshold counts every thermal jiggle across it as a
    crossing. Requiring the walker to actually reach the far basin before the
    next transition counts is what makes this a measure of barrier crossing
    rather than of noise.
    """
    if not (high > low):
        return 0
    state = 0
    crossings = 0
    for x in cv_series:
        if x <= low:
            new = -1
        elif x >= high:
            new = +1
        else:
            continue
        if state != 0 and new != state:
            crossings += 1
        state = new
    return crossings


def metad_report(
    hills_path: Path,
    colvar_path: Path | None,
    out_dir: Path,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
    stride: int = 10,
    min_recrossings: int = 1,
) -> dict[str, Any]:
    """Compact diagnostic bundle for a biased round.

    Same contract as `diagnostics.report.make_report`: JSON-serializable
    scalars plus filesystem paths, no arrays and no trajectory bytes.
    """
    surfaces = sum_hills(hills_path, out_dir, stride=stride)
    final = load_fes(surfaces[-1])
    minima = final.minima(temperature_k)

    # Drift against the half-way surface, not the previous one. With a stride
    # of 10 over thousands of hills, consecutive estimates are ~0.25% of the run
    # apart, so their difference is near zero however unconverged the surface
    # is. The standard well-tempered test compares estimates that are
    # meaningfully separated in time.
    drift = None
    if len(surfaces) >= 2:
        baseline = load_fes(surfaces[len(surfaces) // 2])
        drift = fes_drift_kj_per_mol(baseline, final)

    report: dict[str, Any] = {
        "hills_path": str(hills_path),
        "fes_path": str(surfaces[-1]),
        "cv_label": final.cv_label,
        "cv_min": float(final.cv.min()),
        "cv_max": float(final.cv.max()),
        "n_fes_estimates": len(surfaces),
        "n_basins_fes": len(minima),
        "barrier_kj_per_mol": final.barrier_kj_per_mol(temperature_k),
        "fes_depth_kj_per_mol": final.depth_kj_per_mol(),
        "fes_drift_kj_per_mol": drift,
        "recrossings": None,
        "barrier_crossed": None,
        "fes_converged": None,
    }

    if colvar_path is not None and Path(colvar_path).exists():
        colvar = load_colvar(Path(colvar_path))
        series = colvar.get(final.cv_label)
        thresholds = basin_thresholds(final, temperature_k)
        if series is not None and thresholds is not None:
            low, high = thresholds
            n = count_recrossings(series, low, high)
            report["recrossings"] = n
            # "Did the walker cross the barrier at all", which is n >= 1 —
            # deliberately NOT gated on min_recrossings. Tying it to the
            # round-trip threshold made the field contradict its own name
            # (recrossings=1 reported as barrier_crossed=false), and the
            # scientist flagged exactly that as inconsistent mid-campaign.
            # `fes_converged` is where min_recrossings belongs.
            report["barrier_crossed"] = n >= 1
        elif series is not None:
            report["recrossings"] = 0
            report["barrier_crossed"] = False
        report["colvar_path"] = str(colvar_path)

    report["min_recrossings"] = min_recrossings
    report["fes_converged"] = _fes_converged(report, temperature_k, min_recrossings)
    return report


def _fes_converged(
    report: dict[str, Any], temperature_k: float, min_recrossings: int = 1
) -> bool | None:
    """Small drift is necessary for convergence but nowhere near sufficient.

    A walker that never left its starting basin produces a surface that stops
    changing almost immediately — the last two estimates agree because nothing
    new is being sampled, not because the surface is right. Reporting that as
    converged is the same error as a metastable basin reading as a converged
    plateau, one level up.

    So convergence also requires the walker to have gone back and forth over
    the barrier: a well-tempered surface is trustworthy once the CV is
    diffusing across the region, not merely once the first basin is full.

    `min_recrossings` is how many transitions count as "back and forth".
    `count_recrossings` increments once per transition, so a one-way A -> B
    trip scores 1 and a full A -> B -> A round trip scores 2. The default of
    1 accepts a one-way crossing; a task whose criterion is a full round trip
    (the reverse barrier is unsampled otherwise) should pass 2.
    """
    drift = report.get("fes_drift_kj_per_mol")
    recrossings = report.get("recrossings")
    if drift is None or recrossings is None:
        return None
    return bool(
        drift < _KB_KJ_PER_MOL_K * temperature_k and recrossings >= min_recrossings
    )
