"""The force fields a campaign may ask for, and how each engine names them.

A closed vocabulary, not free text. Force-field choice is high-stakes and
silent when wrong — a mismatched protein/water pair produces a system that
builds, runs, and reports plausible numbers while sampling the wrong ensemble.
So the campaign names a *combination* from this table and the engines resolve
it, exactly as `cv_designer` resolves a CV type from a five-item enum rather
than accepting whatever string arrives.

Every entry here was verified against the installed OpenMM on two axes: the XML
set loads, **and** its water model is one `Modeller.addSolvent` can actually
build. Both matter. `amber14/opc` passes the first and fails the second —
`addSolvent` supports only tip3p, spce, tip4pew, tip5p and swm4ndp, so OPC
would need a tip4pew-geometry workaround — and it is left out rather than
offered untested. `tests/unit/test_forcefields.py` re-checks the whole table.

**Engine coverage is genuinely uneven, and that is not hidden.** The stock
GROMACS install ships amber94/96/99/99sb/99sb-ildn/03, charmm27, gromos and
oplsaa — no ff14SB, no charmm36. `amber14-all.xml` therefore has no GROMACS
counterpart, and `amber99sb-ildn` is a *different force field*, not a
translation of it. An unmapped combination raises rather than substituting the
nearest thing, because substituting is how a cross-engine comparison silently
becomes a comparison of two force fields.

`amber99sbildn/tip3p` is the only entry both engines can build, which makes it
the only combination a genuine cross-engine study can use. The two adapters
have until now defaulted to *different* force fields (amber14-all vs
amber99sb-ildn) with nothing recording that they differed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForceField:
    """One validated protein + water combination, per engine."""

    key: str
    openmm_files: tuple[str, ...]
    openmm_water_model: str
    # None where the engine has no equivalent. Not "the closest thing" —
    # see the module docstring.
    gromacs: tuple[str, str] | None
    summary: str


_FORCE_FIELDS: tuple[ForceField, ...] = (
    ForceField(
        key="amber14/tip3p",
        openmm_files=("amber14-all.xml", "amber14/tip3p.xml"),
        openmm_water_model="tip3p",
        gromacs=None,
        summary="ff14SB protein + TIP3P water. The default and the best-tested "
                "choice for folded and small-peptide protein work.",
    ),
    ForceField(
        key="amber19/tip3p",
        openmm_files=("amber19-all.xml", "amber14/tip3p.xml"),
        openmm_water_model="tip3p",
        gromacs=None,
        summary="ff19SB protein + TIP3P. Newer amino-acid parameters than "
                "ff14SB; note ff19SB was parameterised with OPC water, so "
                "pairing it with TIP3P is a compromise.",
    ),
    ForceField(
        key="amber14/tip4pew",
        openmm_files=("amber14-all.xml", "amber14/tip4pew.xml"),
        openmm_water_model="tip4pew",
        gromacs=None,
        summary="ff14SB + TIP4P-Ew, a 4-site water with better bulk structure "
                "and dynamics than TIP3P. Costs more per step.",
    ),
    ForceField(
        key="amber14/spce",
        openmm_files=("amber14-all.xml", "amber14/spce.xml"),
        openmm_water_model="spce",
        gromacs=None,
        summary="ff14SB + SPC/E. A 3-site water with better density and "
                "diffusion than TIP3P at similar cost.",
    ),
    ForceField(
        key="charmm36/tip3p",
        openmm_files=("charmm36.xml", "charmm36/water.xml"),
        openmm_water_model="tip3p",
        gromacs=None,
        summary="CHARMM36 + CHARMM-modified TIP3P. A different force-field "
                "family from AMBER — useful to check that a result is not an "
                "artefact of one parameter set.",
    ),
    ForceField(
        key="amber99sbildn/tip3p",
        openmm_files=("amber99sbildn.xml", "tip3p.xml"),
        openmm_water_model="tip3p",
        gromacs=("amber99sb-ildn", "tip3p"),
        summary="ff99SB-ILDN + TIP3P. Older, but the only entry both OpenMM "
                "and GROMACS can build — required for any real cross-engine "
                "comparison.",
    ),
)

_BY_KEY = {ff.key: ff for ff in _FORCE_FIELDS}

DEFAULT_KEY = "amber14/tip3p"


def available() -> tuple[str, ...]:
    return tuple(_BY_KEY)


def resolve(key: str) -> ForceField:
    """The named combination, or a ValueError listing what exists."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ValueError(
            f"forcefields: unknown combination {key!r}; available: "
            f"{sorted(_BY_KEY)}. Combinations are validated pairs, not free "
            f"text — add one here (and to the vocabulary the setup agent sees) "
            f"before a campaign can ask for it."
        ) from None


def for_gromacs(key: str) -> tuple[str, str]:
    """`(forcefield, water)` names for `gmx pdb2gmx`, or a refusal.

    Refuses rather than substituting: the stock GROMACS install has no ff14SB
    and no charmm36, and `amber99sb-ildn` is a different force field rather
    than a translation of `amber14-all.xml`. Silently swapping one for the
    other turns a cross-engine comparison into a comparison of two force
    fields, with nothing anywhere recording that it happened.
    """
    ff = resolve(key)
    if ff.gromacs is None:
        raise NotImplementedError(
            f"forcefields: {key!r} has no GROMACS equivalent in a stock "
            f"install, and the nearest available parameter set is a different "
            f"force field rather than a translation. Use "
            f"{'amber99sbildn/tip3p'!r}, which both engines can build, or run "
            f"this campaign through OpenMM."
        )
    return ff.gromacs
