"""PLUMED writer: typed CV/bias → plumed.dat text, with validation.

Atom-indexing convention: callers pass 0-based indices everywhere;
the writer adds +1 on output to match PLUMED's 1-based text format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot.adapters.plumed_writer import (
    DistanceCV,
    GyrationCV,
    HarmonicRestraint,
    MetadynamicsBias,
    PlumedInput,
    TorsionCV,
    enable_restart,
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
        output_dir=Path("/campaigns/demo"),
        print_stride=100,
    )
    text = pi.render()
    assert "d1: DISTANCE ATOMS=5,10" in text
    assert "phi: TORSION ATOMS=1,2,3,4" in text
    assert "metad: METAD ARG=d1,phi" in text
    assert "PRINT ARG=d1,phi,metad.bias STRIDE=100 FILE=/campaigns/demo/COLVAR" in text


def test_plumed_input_renders_harmonic_restraint() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
        output_dir=Path("/campaigns/demo"),
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
            output_dir=Path("/campaigns/demo"),
        )


def test_plumed_input_rejects_duplicate_cv_labels() -> None:
    with pytest.raises(ValueError, match="CV labels must be unique"):
        PlumedInput(
            cvs=(
                DistanceCV(label="d1", atoms=(0, 9)),
                DistanceCV(label="d1", atoms=(1, 10)),
            ),
            bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
            output_dir=Path("/campaigns/demo"),
        )


def test_plumed_input_rejects_zero_cvs() -> None:
    with pytest.raises(ValueError, match="at least one CV"):
        PlumedInput(
            cvs=(),
            bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
            output_dir=Path("/campaigns/demo"),
        )


def test_plumed_input_render_includes_header_comments() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
        output_dir=Path("/campaigns/demo"),
    )
    text = pi.render()
    assert "# PLUMED input generated by MDPilot" in text
    assert "# Collective variables" in text
    assert "# Bias" in text
    assert "# Periodic output" in text


def test_plumed_input_rejects_relative_output_dir() -> None:
    """PLUMED resolves relative FILE= against the process working directory,
    so a relative output_dir silently drops HILLS/COLVAR outside the campaign.
    This actually happened: the first live run wrote them to the repo root."""
    with pytest.raises(ValueError, match="output_dir must be absolute"):
        PlumedInput(
            cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
            bias=HarmonicRestraint(cv_label="d1", at=0.5, kappa=1000.0),
            output_dir=Path("campaigns/demo"),
        )


def test_plumed_input_places_hills_under_output_dir() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=MetadynamicsBias(
            cv_labels=("d1",), sigma=(0.1,), height=1.0, pace=500
        ),
        output_dir=Path("/campaigns/demo/rounds"),
    )
    text = pi.render()
    assert "FILE=/campaigns/demo/rounds/HILLS" in text
    assert "FILE=/campaigns/demo/rounds/COLVAR" in text


def test_floored_sigma_is_declared_in_the_plumed_input() -> None:
    """A floored SIGMA must be visible in the audit artifact. Reading plumed.dat
    later, a substituted width is indistinguishable from a measured one unless
    the file says so."""
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=MetadynamicsBias(
            cv_labels=("d1",), sigma=(0.02,), height=1.0, pace=500,
            sigma_floored=True,
        ),
        output_dir=Path("/campaigns/demo"),
    )
    text = pi.render()
    assert "NOTE" in text and "floor" in text
    assert "under-sampled" in text


def test_measured_sigma_carries_no_note() -> None:
    pi = PlumedInput(
        cvs=(DistanceCV(label="d1", atoms=(0, 9)),),
        bias=MetadynamicsBias(
            cv_labels=("d1",), sigma=(0.06,), height=1.0, pace=500
        ),
        output_dir=Path("/campaigns/demo"),
    )
    assert "NOTE" not in pi.render()


# ---------- RESTART (resumed biased phase) ----------


def test_enable_restart_prepends_the_directive() -> None:
    """Verified against PLUMED 2.9: without RESTART, METAD backs HILLS up to
    bck.0.HILLS and metad.bias reads 0.0 on the first frame; with it, METAD
    reports 'Restarting from HILLS: N Gaussians read' and appends."""
    rendered = enable_restart("# PLUMED input generated by MDPilot\nd: DISTANCE ATOMS=1,2\n")
    assert rendered.splitlines()[0].startswith("RESTART")
    # The original input is preserved verbatim beneath it.
    assert rendered.endswith("# PLUMED input generated by MDPilot\nd: DISTANCE ATOMS=1,2\n")


def test_enable_restart_is_idempotent() -> None:
    """The adapter rewrites plumed.dat on every start(), so a second resume
    reads text this function already touched. Enabling twice must not stack."""
    once = enable_restart("d: DISTANCE ATOMS=1,2\n")
    assert enable_restart(once) == once
    assert once.count("RESTART") == 1


def test_enable_restart_ignores_a_commented_directive() -> None:
    """A '#'-commented RESTART is inert to PLUMED, so it must not be mistaken
    for an already-enabled restart."""
    rendered = enable_restart("# RESTART deliberately off here\nd: DISTANCE ATOMS=1,2\n")
    assert rendered.splitlines()[0].startswith("RESTART")


# ---------- contacts ----------

def test_contacts_renders_a_summed_contactmap() -> None:
    from mdpilot.adapters.plumed_writer import ContactsCV

    cv = ContactsCV(label="q", pairs=((4, 46), (10, 52)), r0_nm=0.75)
    rendered = cv.render()

    # The raw SUM is an intermediate; `q` is the normalised fraction.
    assert rendered.startswith("q_count: CONTACTMAP ...")
    # 1-based, like every other CV here.
    assert "ATOMS1=5,47" in rendered
    assert "ATOMS2=11,53" in rendered
    assert "SWITCH={RATIONAL R_0=0.75 NN=6 MM=12}" in rendered
    assert "  SUM" in rendered
    # PLUMED asserts on `... CONTACTMAP`: a second word there must repeat the
    # *label*, not the action name. Bare "..." is what closes the block.
    assert "..." in rendered.splitlines()
    # …and the label exposes the count divided by the pair count, so the biased
    # coordinate, COLVAR and the free-energy axis are all the same fraction the
    # campaign states its thresholds in.
    assert rendered.splitlines()[-1] == (
        "q: COMBINE ARG=q_count COEFFICIENTS=0.5 PERIODIC=NO"
    )


def test_contacts_rejects_an_empty_map() -> None:
    from mdpilot.adapters.plumed_writer import ContactsCV
    import pytest as _pytest

    with _pytest.raises(ValueError, match="at least 1 contact pair"):
        ContactsCV(label="q", pairs=(), r0_nm=0.75)


def test_contacts_rejects_a_nonpositive_r0() -> None:
    """R_0 is the denominator of the switching function; zero or negative makes
    it undefined rather than merely unusual."""
    from mdpilot.adapters.plumed_writer import ContactsCV
    import pytest as _pytest

    with _pytest.raises(ValueError, match="r0_nm must be positive"):
        ContactsCV(label="q", pairs=((0, 5),), r0_nm=0.0)
