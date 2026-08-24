"""Free-energy surface analysis: parsing, basins, barriers, drift, recrossings.

Surfaces are built analytically and written in sum_hills' own text format, so
the parser and every downstream statistic are exercised without a PLUMED
runtime. The `plumed sum_hills` call itself needs the real binary and lives in
`tests/integration/test_free_energy_live.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdpilot.diagnostics.free_energy import (
    FreeEnergySurface,
    _fes_converged,
    basin_thresholds,
    count_recrossings,
    fes_drift_kj_per_mol,
    load_colvar,
    load_fes,
)

_KT_300 = 0.0083144621 * 300.0


def _double_well(barrier: float = 20.0, n: int = 201) -> FreeEnergySurface:
    """Two basins of unequal depth separated by `barrier` kJ/mol."""
    x = np.linspace(-2.0, 2.0, n)
    f = barrier * (x**2 - 1) ** 2 / 1.0
    f = f + 3.0 * x           # tilt: right basin shallower than left
    f = f - f.min()
    return FreeEnergySurface("cv", x, f, periodic=False)


def _single_well(n: int = 201) -> FreeEnergySurface:
    x = np.linspace(-2.0, 2.0, n)
    f = 30.0 * x**2
    return FreeEnergySurface("cv", x, f - f.min(), periodic=False)


def _write_fes(path: Path, s: FreeEnergySurface) -> Path:
    lines = [f"#! FIELDS {s.cv_label} file.free der_{s.cv_label}",
             f"#! SET min_{s.cv_label} {s.cv.min()}",
             f"#! SET max_{s.cv_label} {s.cv.max()}",
             f"#! SET periodic_{s.cv_label} {'true' if s.periodic else 'false'}"]
    for x, f in zip(s.cv, s.free_energy):
        lines.append(f"  {x:.9f}   {f:.9f}   0.0")
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------- parsing ----------

def test_load_fes_round_trips_grid_and_label(tmp_path: Path) -> None:
    original = _double_well()
    loaded = load_fes(_write_fes(tmp_path / "fes.dat", original))

    assert loaded.cv_label == "cv"
    assert loaded.periodic is False
    assert loaded.cv == pytest.approx(original.cv, rel=1e-6)
    assert loaded.free_energy == pytest.approx(original.free_energy, rel=1e-6)


def test_load_fes_reads_periodicity(tmp_path: Path) -> None:
    s = FreeEnergySurface("phi", np.linspace(-np.pi, np.pi, 20),
                          np.zeros(20), periodic=True)
    assert load_fes(_write_fes(tmp_path / "fes.dat", s)).periodic is True


def test_load_fes_rejects_a_file_with_no_data(tmp_path: Path) -> None:
    p = tmp_path / "empty.dat"
    p.write_text("#! FIELDS cv file.free\n")
    with pytest.raises(ValueError, match="no free-energy rows"):
        load_fes(p)


# ---------- basins and barriers ----------

def test_double_well_resolves_two_basins() -> None:
    assert len(_double_well().minima()) == 2


def test_single_well_resolves_one_basin() -> None:
    assert len(_single_well().minima()) == 1


def test_minima_are_returned_deepest_first() -> None:
    s = _double_well()
    m = s.minima()
    assert s.free_energy[m[0]] < s.free_energy[m[1]]


def test_barrier_is_measured_from_the_deeper_basin() -> None:
    """The barrier is the crest *between* the basins, not the highest point on
    the grid — the surface rises without bound at the domain edges, and reading
    that as the barrier would overstate it by an order of magnitude."""
    s = _double_well(barrier=20.0)
    m = sorted(s.minima())
    crest = float(s.free_energy[m[0] : m[1] + 1].max())
    deepest = float(s.free_energy.min())

    assert s.barrier_kj_per_mol() == pytest.approx(crest - deepest, rel=1e-6)
    assert s.barrier_kj_per_mol() < float(s.free_energy.max()) - deepest


def test_single_basin_has_no_barrier() -> None:
    """One basin means the surface has not resolved a transition — reporting a
    number here would invent a barrier out of noise."""
    assert _single_well().barrier_kj_per_mol() is None


def test_shallow_ripples_are_not_counted_as_basins() -> None:
    """Sub-kT wiggles on a surface are sampling noise. Counting them would
    manufacture barriers and basins that do not exist."""
    x = np.linspace(-2.0, 2.0, 401)
    f = 30.0 * x**2 + 0.3 * np.sin(40 * x)   # ripple amplitude << kT
    s = FreeEnergySurface("cv", x, f - f.min(), periodic=False)

    assert len(s.minima()) == 1


# ---------- drift ----------

def test_identical_surfaces_have_zero_drift() -> None:
    s = _double_well()
    assert fes_drift_kj_per_mol(s, s) == pytest.approx(0.0, abs=1e-9)


def test_drift_detects_a_changing_surface() -> None:
    assert fes_drift_kj_per_mol(_double_well(20.0), _double_well(28.0)) > 1.0


def test_drift_compares_only_the_overlapping_region() -> None:
    """sum_hills grids whatever the walker has visited, so a later estimate is
    usually wider. Comparing outside the overlap would read the widening as
    drift."""
    wide = _double_well()
    narrow_mask = np.abs(wide.cv) <= 1.0
    narrow = FreeEnergySurface(
        "cv", wide.cv[narrow_mask], wide.free_energy[narrow_mask], periodic=False
    )
    drift = fes_drift_kj_per_mol(narrow, wide)

    assert drift is not None
    assert drift == pytest.approx(0.0, abs=0.5)


def test_drift_is_none_when_surfaces_do_not_overlap() -> None:
    a = FreeEnergySurface("cv", np.linspace(0, 1, 10), np.zeros(10), False)
    b = FreeEnergySurface("cv", np.linspace(5, 6, 10), np.zeros(10), False)
    assert fes_drift_kj_per_mol(a, b) is None


# ---------- recrossings ----------

def test_recrossings_counts_completed_transitions() -> None:
    series = np.array([-1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    assert count_recrossings(series, low=-0.5, high=0.5) == 3


def test_jitter_at_the_midpoint_is_not_a_crossing() -> None:
    """A bare threshold would count every thermal wobble. Requiring the walker
    to reach the far basin is what makes this a barrier-crossing measure."""
    series = np.array([-1.0, -0.1, 0.1, -0.1, 0.1, -0.1, -1.0])
    assert count_recrossings(series, low=-0.5, high=0.5) == 0


def test_staying_in_one_basin_is_zero_crossings() -> None:
    assert count_recrossings(np.full(50, -1.0), low=-0.5, high=0.5) == 0


# ---------- the convergence verdict ----------

def test_small_drift_alone_does_not_mean_converged() -> None:
    """The real failure this guard exists for: a walker that never left its
    basin produces a surface that stops changing immediately. Drift is tiny
    because nothing new is being sampled."""
    verdict = _fes_converged(
        {"fes_drift_kj_per_mol": 0.1, "recrossings": 0}, 300.0
    )
    assert verdict is False


def test_converged_requires_both_low_drift_and_recrossing() -> None:
    assert _fes_converged({"fes_drift_kj_per_mol": 0.1, "recrossings": 3}, 300.0) is True
    assert _fes_converged({"fes_drift_kj_per_mol": 9.0, "recrossings": 3}, 300.0) is False


def test_drift_threshold_is_kt() -> None:
    below = _fes_converged(
        {"fes_drift_kj_per_mol": _KT_300 * 0.9, "recrossings": 2}, 300.0)
    above = _fes_converged(
        {"fes_drift_kj_per_mol": _KT_300 * 1.1, "recrossings": 2}, 300.0)
    assert below is True and above is False


def test_verdict_is_none_when_evidence_is_missing() -> None:
    """No COLVAR means no recrossing count, and a bare drift number cannot
    carry the verdict on its own."""
    assert _fes_converged({"fes_drift_kj_per_mol": 0.1, "recrossings": None}, 300.0) is None
    assert _fes_converged({"fes_drift_kj_per_mol": None, "recrossings": 3}, 300.0) is None


# ---------- COLVAR ----------

def test_load_colvar_returns_named_columns(tmp_path: Path) -> None:
    p = tmp_path / "COLVAR"
    p.write_text(
        "#! FIELDS time rg metad.bias\n"
        "0.000000 0.700000 0.000000\n"
        "1.000000 0.710000 1.200000\n"
    )
    cols = load_colvar(p)

    assert set(cols) == {"time", "rg", "metad.bias"}
    assert cols["rg"] == pytest.approx([0.70, 0.71])
    assert cols["metad.bias"][-1] == pytest.approx(1.2)


def test_global_minimum_always_counts_as_a_basin() -> None:
    """Prominence filtering suppresses extra basins; it must not delete the
    deepest point. A ripple at the bottom of a well would otherwise leave a
    surface with no basins at all."""
    x = np.linspace(-2.0, 2.0, 401)
    f = 30.0 * x**2 + 0.3 * np.sin(40 * x)
    s = FreeEnergySurface("cv", x, f - f.min(), periodic=False)

    m = s.minima()
    assert len(m) == 1
    assert m[0] == int(np.argmin(s.free_energy))


def test_monotonic_surface_reports_its_boundary_minimum() -> None:
    """A partially sampled surface can be monotonic — its minimum sits on the
    edge, where the interior local-minimum scan never looks."""
    x = np.linspace(0.0, 1.0, 50)
    s = FreeEnergySurface("cv", x, 20.0 * x, periodic=False)

    assert s.minima() == [0]
    assert s.barrier_kj_per_mol() is None


def test_thresholds_sit_between_each_basin_and_the_crest() -> None:
    """Not at the basin minima: a walker oscillating inside its basin seldom
    sits exactly on the minimum, so anchoring there would miss transitions it
    genuinely completed."""
    s = _double_well()
    result = basin_thresholds(s)

    assert result is not None
    low, high = result
    m = sorted(s.minima())
    left_min, right_min = float(s.cv[m[0]]), float(s.cv[m[1]])

    assert left_min < low < 0.0
    assert 0.0 < high < right_min


def test_thresholds_are_none_without_two_basins() -> None:
    assert basin_thresholds(_single_well()) is None


def test_oscillating_walker_still_registers_its_transitions() -> None:
    """The regression behind the threshold placement: a walker shuttling
    between basins but never landing exactly on a minimum scored zero
    crossings when the minima themselves were used as thresholds."""
    s = _double_well()
    low, high = basin_thresholds(s)
    m = sorted(s.minima())
    left, right = float(s.cv[m[0]]), float(s.cv[m[1]])

    # Oscillate around each basin without hitting the minimum exactly.
    series = np.concatenate([
        np.full(20, left + 0.05), np.full(20, right - 0.05),
        np.full(20, left + 0.05), np.full(20, right - 0.05),
    ])
    assert count_recrossings(series, low, high) == 3


def test_min_recrossings_can_require_a_full_round_trip() -> None:
    """`count_recrossings` increments per transition, so a one-way A -> B trip
    scores 1 and a full A -> B -> A round trip scores 2. A task whose criterion
    is a round trip (otherwise the reverse barrier is never sampled) must be
    able to demand 2 rather than accept the one-way default."""
    one_way = {"fes_drift_kj_per_mol": 0.5, "recrossings": 1}
    round_trip = {"fes_drift_kj_per_mol": 0.5, "recrossings": 2}

    assert _fes_converged(one_way, 300.0) is True
    assert _fes_converged(one_way, 300.0, min_recrossings=2) is False
    assert _fes_converged(round_trip, 300.0, min_recrossings=2) is True


def test_barrier_crossed_is_not_gated_on_min_recrossings(tmp_path: Path) -> None:
    """`barrier_crossed` answers "did the walker cross at all" (n >= 1).

    Gating it on min_recrossings made it report False on a run that had
    demonstrably crossed once — the field contradicting its own name. The
    scientist caught it live and called it inconsistent, then reasoned poorly
    from it. min_recrossings belongs to `fes_converged` alone.
    """
    from mdpilot.diagnostics.free_energy import count_recrossings

    # One-way A -> B trip: crossed the barrier, but not a round trip.
    series = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    n = count_recrossings(series, 0.2, 0.8)
    assert n == 1
    assert (n >= 1) is True                                   # barrier_crossed
    assert _fes_converged({"fes_drift_kj_per_mol": 0.5, "recrossings": n},
                          300.0, min_recrossings=2) is False  # not yet converged


def test_drift_baseline_is_never_the_final_surface_itself() -> None:
    """`n // 2` is the *last* index when there are exactly two estimates, so
    the surface was compared against itself and drift came out 0.0 by
    construction. The baseline has to stay strictly below the last index."""
    from mdpilot.diagnostics.free_energy import _baseline_index

    for n in range(2, 12):
        assert _baseline_index(n) < n - 1
    assert _baseline_index(2) == 0        # the only choice left
    assert _baseline_index(3) == 1
    assert _baseline_index(10) == 5       # unchanged where the clamp is slack


def test_two_estimates_still_measure_a_real_drift(tmp_path: Path, monkeypatch) -> None:
    """The regression this guards: a first biased round short enough to yield
    two estimates reported drift=0.0 on a surface still moving by tens of
    kJ/mol, and `_fes_converged` needs only low drift plus one recrossing — so
    `fes_converged=true` came back and `_refuse_premature_stop` let the
    scientist's `stop` through with the budget unspent.
    """
    from mdpilot.diagnostics import free_energy as fe

    early = _write_fes(tmp_path / "early.dat", _double_well(barrier=20.0))
    late = _write_fes(tmp_path / "late.dat", _double_well(barrier=40.0))
    monkeypatch.setattr(fe, "sum_hills", lambda *a, **k: [early, late])

    # One folded -> extended transition on the task's own coordinate, so the
    # recrossing half of the convergence gate is satisfied and only drift is
    # left to decide the verdict.
    report = fe.metad_report(
        tmp_path / "HILLS",
        None,
        tmp_path / "out",
        observable=np.repeat([1.0, 6.0], 50),
        observable_name="rmsd_ca_to_reference_angstrom",
        state_thresholds=(2.0, 5.0),
    )

    assert report["n_fes_estimates"] == 2
    assert report["recrossings"] == 1
    assert report["fes_drift_kj_per_mol"] == pytest.approx(
        fes_drift_kj_per_mol(load_fes(early), load_fes(late))
    )
    assert report["fes_drift_kj_per_mol"] > _KT_300
    assert report["fes_converged"] is False


def _fake_plumed(monkeypatch, out: Path, bodies: list[str]) -> None:
    """Stand in for `plumed sum_hills --stride`: write one surface per block.

    The files are written *by the call*, as the real binary does, rather than
    pre-placed in `out_dir` — `sum_hills` clears stale surfaces before invoking
    plumed, so anything staged beforehand is (correctly) gone by the time the
    subprocess would run.
    """
    from mdpilot.diagnostics import free_energy as fe

    def run(*_args, **_kwargs):
        for i, body in enumerate(bodies):
            (out / f"fes.dat{i}.dat").write_text(body)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(fe, "plumed_available", lambda: True)
    monkeypatch.setattr(fe.subprocess, "run", run)


def test_sum_hills_drops_a_trailing_duplicate_surface(tmp_path: Path, monkeypatch) -> None:
    """When the hill count is a multiple of --stride, sum_hills emits the final
    complete surface *and* an identical last strided one. Comparing that pair
    yields a drift of exactly 0.0 by construction, which silently made the
    drift half of the convergence gate vacuous for every biased round."""
    from mdpilot.diagnostics import free_energy as fe

    out = tmp_path / "out"
    out.mkdir()
    first = "#! FIELDS cv file.free\n0.0 0.0\n1.0 5.0\n"
    rest = "#! FIELDS cv file.free\n0.0 0.0\n1.0 7.0\n"
    # fes.dat1 and fes.dat2 are byte-identical -> the trailing duplicate.
    _fake_plumed(monkeypatch, out, [first, rest, rest])

    surfaces = fe.sum_hills(tmp_path / "HILLS", out, stride=10)
    assert [p.name for p in surfaces] == ["fes.dat0.dat", "fes.dat1.dat"]


def test_sum_hills_clears_surfaces_from_an_earlier_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    """A round re-run after a crash integrates the *restored* (shorter) HILLS,
    so it writes fewer indexed surfaces than the aborted attempt did. Without
    cleanup the leftover higher indices survive, `indexed.sort()` puts one of
    them last, and `surfaces[-1]` — the profile every statistic is taken on —
    is a surface from the previous attempt.
    """
    from mdpilot.diagnostics import free_energy as fe

    out = tmp_path / "out"
    out.mkdir()
    for i in range(5):                      # leftovers from the aborted attempt
        (out / f"fes.dat{i}.dat").write_text("#! FIELDS cv file.free\n0.0 99.0\n")
    (out / "fes.dat").write_text("#! FIELDS cv file.free\n0.0 99.0\n")

    body = "#! FIELDS cv file.free\n0.0 0.0\n1.0 5.0\n"
    _fake_plumed(monkeypatch, out, [body, body.replace("5.0", "6.0")])

    surfaces = fe.sum_hills(tmp_path / "HILLS", out, stride=10)

    assert [p.name for p in surfaces] == ["fes.dat0.dat", "fes.dat1.dat"]
    assert load_fes(surfaces[-1]).free_energy.max() == pytest.approx(6.0)


# ---------- restriction to the sampled region ----------

def _padded_well(n: int = 201) -> FreeEnergySurface:
    """A well over [-1, 1] with a flat, very high shelf on either side.

    This is the shape `sum_hills` produces: it grids a few SIGMA past the
    outermost hill, and out there the surface is extrapolation that no hill
    ever touched.
    """
    x = np.linspace(-2.0, 2.0, n)
    f = np.where(np.abs(x) <= 1.0, 10.0 * x**2, 500.0)
    return FreeEnergySurface("cv", x, f - f.min(), periodic=False)


def test_restricted_to_excludes_unsampled_padding() -> None:
    s = _padded_well()
    assert s.depth_kj_per_mol() == pytest.approx(500.0)
    assert s.restricted_to(-1.0, 1.0).depth_kj_per_mol() == pytest.approx(10.0)


def test_restricted_to_keeps_the_grid_when_too_little_survives() -> None:
    """Fewer than three points cannot carry a minimum, a barrier or a drift,
    so an over-narrow restriction returns the surface untouched rather than a
    stub that every downstream statistic would have to special-case."""
    s = _padded_well()
    assert s.restricted_to(0.0, 0.0) is s


def test_restricted_to_preserves_label_and_periodicity() -> None:
    s = FreeEnergySurface("phi", np.linspace(-3, 3, 51), np.zeros(51), periodic=True)
    r = s.restricted_to(-1.0, 1.0)
    assert (r.cv_label, r.periodic) == ("phi", True)
    assert r.cv.min() >= -1.0 and r.cv.max() <= 1.0
