"""PLUMED writer: typed CV/bias → plumed.dat text, with validation.

Atom-indexing convention: callers pass 0-based indices everywhere;
the writer adds +1 on output to match PLUMED's 1-based text format.
"""

from __future__ import annotations

import pytest

from mdpilot.adapters.plumed_writer import (
    DistanceCV,
    GyrationCV,
    HarmonicRestraint,
    MetadynamicsBias,
    PlumedInput,
    TorsionCV,
)


# ---------- CVs ----------

def test_distance_cv_emits_1based_atoms() -> None:
    cv = DistanceCV(label="d1", atoms=(4, 9))
    assert cv.render() == "d1: DISTANCE ATOMS=5,10"


def test_torsion_cv_emits_1based_atoms() -> None:
    cv = TorsionCV(label="phi", atoms=(0, 1, 2, 3))
    assert cv.render() == "phi: TORSION ATOMS=1,2,3,4"


def test_gyration_cv_emits_1based_atoms() -> None:
    cv = GyrationCV(label="rg", atoms=(0, 1, 4, 5))
    assert cv.render() == "rg: GYRATION TYPE=RADIUS ATOMS=1,2,5,6"


def test_gyration_cv_rejects_under_two_atoms() -> None:
    with pytest.raises(ValueError, match="at least 2 atoms"):
        GyrationCV(label="rg", atoms=(3,))


# ---------- Metadynamics ----------

def test_metad_renders_args_sigma_height_pace_file() -> None:
    bias = MetadynamicsBias(
        cv_labels=("d1",),
        sigma=(0.1,),
        height=1.2,
        pace=500,
        bias_factor=10.0,
        temperature_k=300.0,
    )
    rendered = bias.render()
    assert rendered == (
        "metad: METAD ARG=d1 SIGMA=0.1 HEIGHT=1.2 PACE=500 "
        "BIASFACTOR=10 TEMP=300 FILE=HILLS"
    )


def test_metad_is_always_well_tempered() -> None:
    """Plain metaD (no BIASFACTOR) must not be constructible — it does not
    converge to the FES. The default carries a finite gamma."""
    bias = MetadynamicsBias(cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=500)
    assert "BIASFACTOR=" in bias.render()
    assert bias.bias_factor > 1.0


def test_metad_rejects_bias_factor_at_or_below_one() -> None:
    for gamma in (1.0, 0.5, -3.0):
        with pytest.raises(ValueError, match=r"bias_factor"):
            MetadynamicsBias(
                cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=500,
                bias_factor=gamma,
            )


def test_metad_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature_k must be positive"):
        MetadynamicsBias(
            cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=500,
            temperature_k=0.0,
        )


def test_metad_supports_multiple_cvs() -> None:
    bias = MetadynamicsBias(
        cv_labels=("d1", "phi"),
        sigma=(0.1, 0.2),
        height=0.5,
        pace=250,
        hills_file="HILLS.dat",
    )
    assert "ARG=d1,phi" in bias.render()
    assert "SIGMA=0.1,0.2" in bias.render()
    assert "FILE=HILLS.dat" in bias.render()


def test_metad_rejects_sigma_cv_label_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        MetadynamicsBias(cv_labels=("d1", "phi"), sigma=(0.1,), height=1.0, pace=500)


def test_metad_rejects_zero_cvs() -> None:
    with pytest.raises(ValueError, match="at least one CV"):
        MetadynamicsBias(cv_labels=(), sigma=(), height=1.0, pace=500)


def test_metad_rejects_nonpositive_height() -> None:
    with pytest.raises(ValueError, match="height must be positive"):
        MetadynamicsBias(cv_labels=("d1",), sigma=(0.1,), height=0.0, pace=500)


def test_metad_rejects_nonpositive_pace() -> None:
    with pytest.raises(ValueError, match="pace must be positive"):
        MetadynamicsBias(cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=0)


def test_metad_bias_value_is_dot_bias() -> None:
    bias = MetadynamicsBias(cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=500)
    assert bias.bias_value == "metad.bias"


# ---------- Harmonic restraint ----------

def test_harmonic_restraint_renders() -> None:
    r = HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0)
    assert r.render() == "restraint: RESTRAINT ARG=d1 AT=0.5 KAPPA=1000"


def test_harmonic_restraint_rejects_negative_kappa() -> None:
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        HarmonicRestraint(cv_label="d1", at=0.5, kappa=-1.0)


# ---------- PlumedInput composite ----------

def test_plumed_input_renders_metad_with_two_cvs() -> None:
    pi = PlumedInput(
        cvs=(
            DistanceCV(label="d1", atoms=(4, 9)),
            TorsionCV(label="phi", atoms=(0, 1, 2, 3)),
        ),
        bias=MetadynamicsBias(
            cv_labels=("d1", "phi"),
            sigma=(0.1, 0.2),
            height=1.0,
            pace=500,
        ),
        print_stride=100,
    )
    text = pi.render()
    assert "d1: DISTANCE ATOMS=5,10" in text
    assert "phi: TORSION ATOMS=1,2,3,4" in text
    assert "metad: METAD ARG=d1,phi" in text
    assert "PRINT ARG=d1,phi,metad.bias STRIDE=100 FILE=COLVAR" in text


def test_plumed_input_renders_harmonic_restraint() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
    )
    text = pi.render()
    assert "d1: DISTANCE ATOMS=1,10" in text
    assert "restraint: RESTRAINT ARG=d1 AT=0.5 KAPPA=1000" in text
    assert "PRINT ARG=d1,restraint.bias" in text


def test_plumed_input_rejects_undefined_cv_reference() -> None:
    with pytest.raises(ValueError, match="undefined CV label"):
        PlumedInput(
            cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
            bias=MetadynamicsBias(
                cv_labels=("ghost",),
                sigma=(0.1,),
                height=1.0,
                pace=500,
            ),
        )


def test_plumed_input_rejects_duplicate_cv_labels() -> None:
    with pytest.raises(ValueError, match="CV labels must be unique"):
        PlumedInput(
            cvs=(
                DistanceCV(label="d1", atoms=(0, 9)),
                DistanceCV(label="d1", atoms=(1, 10)),
            ),
            bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
        )


def test_plumed_input_rejects_zero_cvs() -> None:
    with pytest.raises(ValueError, match="at least one CV"):
        PlumedInput(
            cvs=(),
            bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
        )


def test_plumed_input_render_includes_header_comments() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
    )
    text = pi.render()
    assert "# PLUMED input generated by MDPilot" in text
    assert "# Collective variables" in text
    assert "# Bias" in text
    assert "# Periodic output" in text
