"""The force-field vocabulary is a promise: every entry can actually be built.

A combination that loads but cannot be solvated, or that names an XML the
installed OpenMM does not ship, fails at setup time — after a structure has
been fetched and a campaign started. These check the promise up front.
"""

from __future__ import annotations

import openmm.app as app
import pytest

from mdpilot import forcefields

# What `Modeller.addSolvent` can actually build, per its own docstring. A water
# model outside this set loads as XML and then fails at solvation — which is
# why `amber14/opc` is not in the vocabulary despite parsing fine.
_SOLVATABLE = {"tip3p", "spce", "tip4pew", "tip5p", "swm4ndp"}


@pytest.mark.parametrize("key", forcefields.available())
def test_every_entry_loads_and_is_solvatable(key: str) -> None:
    entry = forcefields.resolve(key)

    app.ForceField(*entry.openmm_files)          # raises if an XML is missing
    assert entry.openmm_water_model in _SOLVATABLE, entry.openmm_water_model
    assert entry.summary


def test_the_default_is_what_every_recorded_campaign_ran_on() -> None:
    assert forcefields.DEFAULT_KEY == "amber14/tip3p"
    assert forcefields.resolve(forcefields.DEFAULT_KEY).openmm_files == (
        "amber14-all.xml",
        "amber14/tip3p.xml",
    )


def test_an_unknown_combination_lists_what_exists() -> None:
    with pytest.raises(ValueError, match="unknown combination"):
        forcefields.resolve("amber14/opc")       # loads, but cannot be solvated
    with pytest.raises(ValueError, match="amber14/tip3p"):
        forcefields.resolve("charmm27/tip3p")


# ---------- engine coverage is uneven, and refuses rather than substitutes ----

def test_only_one_combination_is_buildable_by_both_engines() -> None:
    """Which makes it the only one a genuine cross-engine study can use. The
    two adapters previously defaulted to *different* force fields, with nothing
    recording that they differed."""
    shared = [k for k in forcefields.available() if forcefields.resolve(k).gromacs]

    assert shared == ["amber99sbildn/tip3p"]
    assert forcefields.for_gromacs("amber99sbildn/tip3p") == ("amber99sb-ildn", "tip3p")


def test_gromacs_refuses_rather_than_substituting_the_nearest_parameter_set() -> None:
    """`amber99sb-ildn` is a different force field, not a translation of
    ff14SB. Substituting it would turn a cross-engine comparison into a
    comparison of two force fields with nothing recording it."""
    with pytest.raises(NotImplementedError, match="no GROMACS equivalent"):
        forcefields.for_gromacs("amber14/tip3p")

    message = pytest.raises(
        NotImplementedError, forcefields.for_gromacs, "charmm36/tip3p"
    ).value.args[0]
    assert "amber99sbildn/tip3p" in message      # names the way out
