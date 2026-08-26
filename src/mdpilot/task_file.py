"""Load a task file into the arguments `run_campaign` takes.

A task file is the one thing a user (or, later, a setup agent) writes to
describe a campaign. It already existed as `benchmarks/tasks/*.yaml` and was
almost entirely decorative: `run_cln025.py` read `starting_pdb`,
`task_expectation`, `sampling` and `done_criterion`, and every other declared
field — force field, water model, padding, ionic strength, timestep,
constraints, observable, target ESS — was documentation the adapters were free
to contradict. Nothing checked that `padding_nm: 1.0` in the file was the
`_PADDING_NM = 1.0` in the adapter.

This module makes the file load-bearing, in three modes per field:

- **Mapped** — turned into a `SystemSpec`/`Ensemble` or a `run_campaign`
  keyword. These are the parameters that are actually tunable today.
- **Verified** — not yet tunable, but declared in the file, so the value is
  checked against the constant that really governs it and a mismatch raises.
  This is the point of the module: a task file may not quietly disagree with
  the code. `padding_nm: 1.5` gets an error naming the fixed value rather than
  silently running at 1.0.
- **Informational** — prose and provenance (`description`, `starting_state`,
  `observable.selection`). Carried, not interpreted.

Anything else raises. Unknown keys are how a generated file's typos surface,
and a setup agent will produce those.

The verified values are the *OpenMM* adapter's, since that is `run_campaign`'s
default engine. `forcefield` in particular is engine-specific — GROMACS runs
amber99sb-ildn, which is a different force field, not a translation of
`amber14-all.xml` — so a GROMACS campaign declaring it needs an engine-keyed
table here. Deferred until a task file actually targets GROMACS.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mdpilot.adapters import openmm_adapter as _omm
from mdpilot.adapters.system_spec import Ensemble, SystemSpec
from mdpilot.diagnostics.autocorrelation import autocorrelation
from mdpilot.observables import ObservableSpec

# Declared in task files, governed by a constant elsewhere. Sourced from the
# module that owns each value rather than retyped, so the check cannot drift
# into asserting a number nothing uses any more.
_VERIFIED: dict[tuple[str, str], Any] = {
    ("system", "ionic_strength_M"): _omm._SALT_M,
    # Nonbonded treatment. Invisible until now and reproducibility-critical:
    # a free-energy surface is not interpretable without knowing the cutoff and
    # how long-range electrostatics were handled.
    ("system", "nonbonded_cutoff_nm"): _omm._NONBONDED_CUTOFF_NM,
    ("system", "electrostatics"): "PME",
    # Coupling. The *algorithms* stay engine-owned — OpenMM's LangevinMiddle
    # and MonteCarloBarostat are idiomatic equivalents of GROMACS's `sd` and
    # C-rescale, not interchangeable names — so they are declared here and
    # checked rather than chosen. The thermodynamic state they target
    # (temperature, pressure) is a real parameter and lives on `Ensemble`.
    ("integrator", "type"): "LangevinMiddle",
    ("integrator", "friction_per_ps"): _omm._FRICTION_PER_PS,
    ("integrator", "barostat"): "MonteCarloBarostat",
    ("integrator", "barostat_interval_steps"): _omm._BAROSTAT_INTERVAL,
    ("integrator", "constraints"): "HBonds",
    ("diagnostics", "target_ess"): inspect.signature(
        autocorrelation
    ).parameters["target_ess"].default,
}

# Carried for humans, not interpreted.
_INFORMATIONAL = {
    ("system", "starting_state"),
    # Only one reference is possible today — the campaign topology — so this is
    # documentation until a second one exists.
    ("observable", "reference"),
    ("diagnostics", "methods"),
    ("done_criterion", "pivot_required"),
}

_TOP_LEVEL = {
    "name", "description", "system", "integrator", "observable",
    "diagnostics", "sampling", "expectation", "done_criterion",
}
_EXPECTATION_KEYS = {"objective", "characteristic_timescale_ns", "timescale_source"}
_DONE_CRITERION_KEYS = {"states", "min_recrossings", "max_biased_ns", "pivot_required"}
_STATE_KEYS = {"name", "threshold"}
_OBSERVABLE_KEYS = {"cv_type", "selections", "name", "scale", "normalize", "reference"}


@dataclass(frozen=True)
class TaskFile:
    """A parsed, checked task file."""

    name: str
    spec: SystemSpec
    campaign: dict[str, Any]
    done_criterion: dict[str, Any]
    sha256: str
    path: Path
    observable_name: str

    def build_adapter(self, work_dir: Path, *, seed: int = 42) -> Any:
        """The engine adapter this task describes.

        `run_kwargs` deliberately does not carry the system spec — the spec
        belongs to the adapter, which the caller constructs. That split is a
        footgun: `run_campaign(adapter=None)` silently falls back to
        `SystemSpec.trpcage()`, so a caller who forgets gets a Trp-cage
        campaign no matter what the task file said, with nothing to indicate
        the file was ignored. The Streamlit app did exactly that, and ran
        1L2Y for forty minutes under a task file describing chignolin.

        Using this instead of constructing an adapter by hand keeps the two
        halves of a task file — what to simulate, and how to judge it —
        from being separable by accident.
        """
        from mdpilot.adapters.openmm_adapter import OpenMMAdapter

        return OpenMMAdapter(work_dir=Path(work_dir), seed=seed, spec=self.spec)

    def run_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Keyword arguments for `run_campaign`, with caller overrides applied.

        Overrides are for loop-control bounds the file does not own —
        `max_rounds`, `initial_steps`, a dry-run's shortened budget. Keys are
        checked against `run_campaign`'s actual signature so a typo raises here
        instead of arriving as an opaque TypeError several frames down.
        """
        from mdpilot.orchestrator.loop import run_campaign

        merged = {**self.campaign, **overrides}
        valid = set(inspect.signature(run_campaign).parameters)
        unknown = sorted(set(merged) - valid)
        if unknown:
            raise ValueError(
                f"task_file: {unknown} are not run_campaign parameters; "
                f"valid: {sorted(valid - {'work_dir'})}"
            )
        return merged


def load_task_file(path: Path) -> TaskFile:
    """Parse and check a task file. Raises on anything it cannot honour."""
    path = Path(path)
    raw = path.read_bytes()
    doc = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"task_file: {path} is not a YAML mapping")

    _reject_unknown("<top level>", set(doc), _TOP_LEVEL, path)
    _reject_unknown(
        "observable", set(doc.get("observable") or {}), _OBSERVABLE_KEYS, path
    )
    _check_expectation(doc, path)
    _verify_declared_constants(doc, path)

    return TaskFile(
        name=doc.get("name", path.stem),
        spec=_build_spec(doc, path),
        campaign=_build_campaign(doc),
        done_criterion=dict(doc.get("done_criterion") or {}),
        sha256=hashlib.sha256(raw).hexdigest(),
        path=path,
        observable_name=_build_observable(doc).name,
    )


def _build_spec(doc: dict[str, Any], path: Path) -> SystemSpec:
    system = doc.get("system") or {}
    integrator = doc.get("integrator") or {}
    _reject_unknown(
        "system",
        set(system),
        {"starting_pdb", "structure_path", "padding_nm", "forcefield"}
        | {k for s, k in _VERIFIED if s == "system"}
        | {k for s, k in _INFORMATIONAL if s == "system"},
        path,
    )
    _reject_unknown(
        "integrator",
        set(integrator),
        {"temperature_K", "pressure_bar", "timestep_fs"}
        | {k for s, k in _VERIFIED if s == "integrator"},
        path,
    )

    ensemble_kwargs: dict[str, Any] = {}
    if "temperature_K" in integrator:
        ensemble_kwargs["temperature_k"] = float(integrator["temperature_K"])
    if "pressure_bar" in integrator:
        ensemble_kwargs["pressure_bar"] = float(integrator["pressure_bar"])
    if "timestep_fs" in integrator:
        ensemble_kwargs["timestep_fs"] = float(integrator["timestep_fs"])
    ensemble = Ensemble(**ensemble_kwargs)

    # SystemSpec enforces exactly-one-of and a positive padding; let its
    # messages stand rather than restating them here.
    spec_kwargs: dict[str, Any] = {}
    if "padding_nm" in system:
        spec_kwargs["padding_nm"] = float(system["padding_nm"])
    if "forcefield" in system:
        # A key into `mdpilot.forcefields`, which pairs the protein force field
        # with a water model that engine can actually solvate. There is no
        # separate `water_model` field: two fields could disagree, and the
        # combination is what was validated, not either half.
        spec_kwargs["forcefield"] = system["forcefield"]
    return SystemSpec(
        pdb_id=system.get("starting_pdb"),
        structure_path=(
            Path(system["structure_path"]) if system.get("structure_path") else None
        ),
        ensemble=ensemble,
        **spec_kwargs,
    )


def render_task_expectation(doc: dict[str, Any]) -> str:
    """Build the `task_expectation` string from the task file's typed fields.

    `task_expectation` is the only input gating `switch_to_metad`, and it was
    the one load-bearing input that escaped structuring: free prose that
    restated the state thresholds, the round-trip requirement and the compute
    budget in words, so three of its four decision-driving numbers existed
    twice with nothing linking the copies. Rendering makes drift
    unrepresentable — the prose is a *view* of the fields, not a second copy.

    Deliberately state-name-agnostic. `low` and `high` are positions on the
    campaign observable, not folding roles: a ligand-binding campaign names
    them "bound"/"unbound" on a distance and this function is unchanged.

    The one genuinely free fact is the characteristic timescale, which is why
    it carries a source. The budget-vs-timescale comparison the pivot rule asks
    the scientist to make is computed here rather than left as arithmetic on
    prose.
    """
    expectation = doc.get("expectation") or {}
    criterion = doc.get("done_criterion") or {}
    observable = (doc.get("observable") or {}).get("name", "the campaign observable")
    ensemble = doc.get("integrator") or {}

    parts: list[str] = []

    system = doc.get("system") or {}
    identity = (
        f"PDB {system['starting_pdb']}"
        if system.get("starting_pdb")
        else f"structure {Path(system['structure_path']).name}"
        if system.get("structure_path")
        else "the configured system"
    )
    conditions = ""
    if "temperature_K" in ensemble:
        conditions = f", simulated at {float(ensemble['temperature_K']):g} K"
    parts.append(f"Target: {doc.get('name', 'campaign')} ({identity}){conditions}.")

    if expectation.get("objective"):
        parts.append(f"Objective: {str(expectation['objective']).strip()}")

    states = criterion.get("states") or {}
    if states:
        low, high = states["low"], states["high"]
        n = int(criterion.get("min_recrossings", 1))
        trip = (
            "A full round trip out and back is required; a one-way crossing "
            "leaves the reverse barrier unsampled and the surface under-filled "
            "on one side."
            if n >= 2
            else "A single one-way crossing satisfies this."
        )
        parts.append(
            f"Required transition: the campaign must connect the "
            f"\"{high['name']}\" state ({observable} > {high['threshold']:g}) and "
            f"the \"{low['name']}\" state ({observable} < {low['threshold']:g}), "
            f"recording at least {n} transition(s) between them. {trip}"
        )

    timescale = expectation.get("characteristic_timescale_ns")
    if timescale is not None:
        source = expectation.get("timescale_source")
        cite = f" ({source})" if source else ""
        parts.append(
            f"Characteristic timescale: {float(timescale):g} ns{cite}."
        )

    budget = criterion.get("max_biased_ns")
    if budget is not None:
        line = f"Compute budget: {float(budget):g} ns for the biased phase."
        if timescale:
            ratio = float(timescale) / float(budget)
            if ratio >= 1.5:
                line += (
                    f" That is {ratio:.0f}x shorter than the characteristic "
                    f"timescale, so unbiased MD within this budget is expected "
                    f"to stay trapped in whichever state it starts from."
                )
            else:
                # The budget reaches the timescale. Saying "0x shorter ...
                # expected to stay trapped" here was not merely awkward, it
                # asserted the opposite of what the numbers say, and the pivot
                # rule reads this sentence.
                line += (
                    " That is comparable to or longer than the characteristic "
                    "timescale, so unbiased MD within this budget may reach "
                    "the transition without enhanced sampling."
                )
        parts.append(line)

    return "\n\n".join(parts) + "\n"


def _build_observable(doc: dict[str, Any]) -> ObservableSpec:
    """The declared campaign observable, or the CA-RMSD default.

    An observable is a collective variable, so the block is the same shape as a
    CV proposal. A file that says nothing gets the M1 observable, which is what
    every campaign recorded so far actually ran on.
    """
    block = dict(doc.get("observable") or {})
    block.pop("reference", None)   # informational; only one reference exists
    if not block:
        return ObservableSpec.ca_rmsd_angstrom()
    missing = {"cv_type", "selections", "name"} - set(block)
    if missing:
        raise ValueError(
            f"task_file: observable block is missing {sorted(missing)}. An "
            f"observable is declared as a collective variable: cv_type, "
            f"selections and name are all required."
        )
    return ObservableSpec(
        cv_type=block["cv_type"],
        selections=tuple(block["selections"]),
        name=block["name"],
        scale=float(block.get("scale", 1.0)),
        normalize=bool(block.get("normalize", False)),
    )


def _build_campaign(doc: dict[str, Any]) -> dict[str, Any]:
    """The `run_campaign` keywords the file owns. Absent keys are omitted
    rather than passed as None, so the loop's own defaults stay in charge."""
    campaign: dict[str, Any] = {}

    observable = _build_observable(doc)
    if observable != ObservableSpec.ca_rmsd_angstrom():
        campaign["observable"] = observable

    # Prose, threaded through only so `preflight` can check it against the
    # structure that actually gets fetched. Not part of the campaign's
    # identity — editing the description must not stop a resume.
    if doc.get("description"):
        campaign["description"] = str(doc["description"])

    if doc.get("expectation"):
        campaign["task_expectation"] = render_task_expectation(doc)

    sampling = doc.get("sampling") or {}
    for key in ("cv_upper_wall_nm", "bias_pace", "bias_factor"):
        if key in sampling:
            campaign[key] = sampling[key]

    criterion = doc.get("done_criterion") or {}
    if "min_recrossings" in criterion:
        campaign["min_recrossings"] = int(criterion["min_recrossings"])
    if "max_biased_ns" in criterion:
        campaign["max_biased_ns"] = float(criterion["max_biased_ns"])
    # `state_thresholds` is (low, high) on the campaign observable — the band
    # biased-round recrossings are counted between. run_campaign refuses an
    # inverted pair, so the ordering here is load-bearing, not cosmetic.
    states = criterion.get("states")
    if states:
        campaign["state_thresholds"] = (
            float(states["low"]["threshold"]),
            float(states["high"]["threshold"]),
        )
    return campaign


def _check_expectation(doc: dict[str, Any], path: Path) -> None:
    """Structural checks on the blocks `task_expectation` is rendered from."""
    expectation = doc.get("expectation") or {}
    criterion = doc.get("done_criterion") or {}
    _reject_unknown("expectation", set(expectation), _EXPECTATION_KEYS, path)
    _reject_unknown("done_criterion", set(criterion), _DONE_CRITERION_KEYS, path)

    states = criterion.get("states")
    if states is not None:
        _reject_unknown("done_criterion.states", set(states), {"low", "high"}, path)
        for side in ("low", "high"):
            if side not in states:
                raise ValueError(
                    f"task_file: {path} done_criterion.states is missing "
                    f"{side!r}; both bands are needed to count a transition"
                )
            _reject_unknown(
                f"done_criterion.states.{side}", set(states[side]), _STATE_KEYS, path
            )
        if not float(states["high"]["threshold"]) > float(states["low"]["threshold"]):
            raise ValueError(
                f"task_file: {path} done_criterion.states must have "
                f"high.threshold > low.threshold on the campaign observable; got "
                f"low={states['low']['threshold']!r}, "
                f"high={states['high']['threshold']!r}. `count_recrossings` "
                f"silently returns 0 for an inverted band, so a swapped pair "
                f"reads as a run that never crossed."
            )

    _check_timescale_units(expectation, path)

    # `run_campaign` refuses a pivot-capable campaign with no task states,
    # because the biased phase would fall back to counting recrossings between
    # whichever two basins are currently deepest (F9). Catch it here, where the
    # message can name the task-file block instead of the loop argument.
    if expectation and not criterion.get("states"):
        raise ValueError(
            f"task_file: {path} has an `expectation:` block, so this campaign "
            f"can pivot to metadynamics, but done_criterion.states is missing. "
            f"A biased phase needs the task's own state definitions to count "
            f"recrossings against."
        )


# Timescales the source *states*, as (number, unit) pairs converted to
# nanoseconds. Matching a quantity rather than a bare unit is the point: an
# earlier version looked for any unit anywhere in the text, so a source that
# mentioned a slower process for context — "folds on the ~1 us timescale, far
# below the ms timescale of larger proteins" — was refused although its number
# was right. Scientific prose does that constantly.
#
# Tolerance is one order of magnitude either way: 0.6 us written as 600 ns is
# correct and must pass, while 1 us written as 1 ns is off by 1000x and must not.
_ORDER_OF_MAGNITUDE = 10.0
_UNIT_NS: dict[str, float] = {
    "ns": 1.0, "nanosecond": 1.0, "nanoseconds": 1.0,
    "us": 1e3, "\u00b5s": 1e3, "\u03bcs": 1e3,
    "microsecond": 1e3, "microseconds": 1e3,
    "ms": 1e6, "millisecond": 1e6, "milliseconds": 1e6,
}
_QUANTITY = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(nanoseconds?|microseconds?|milliseconds?|ns|us|\u00b5s|\u03bcs|ms)\b",
    re.IGNORECASE,
)


def _quoted_timescales_ns(source: str) -> list[float]:
    """Every magnitude the source actually states, in nanoseconds."""
    return [
        float(number) * _UNIT_NS[unit.lower()]
        for number, unit in _QUANTITY.findall(source)
        if unit.lower() in _UNIT_NS
    ]


def _check_timescale_units(expectation: dict[str, Any], path: Path) -> None:
    """Refuse a timescale that contradicts the unit its own source states.

    `characteristic_timescale_ns` is the one decision-driving number with no
    second copy anywhere, and it is what the pivot rule compares elapsed
    simulation time against. A unit slip there does not look wrong — it looks
    like a small number — and it inverts the decision: a transition stated as
    1 ns instead of 1000 ns tells the scientist unbiased MD reaches it easily,
    so the campaign never pivots and spends its whole budget on sampling that
    cannot work.

    Compared against the magnitudes the source *states*, passing if any one of
    them is within an order of magnitude. A source is free to mention other
    timescales for context; an earlier version keyed on bare units and refused
    good sources for doing exactly that.
    """
    value = expectation.get("characteristic_timescale_ns")
    source = str(expectation.get("timescale_source") or "")
    if value is None or not source:
        return
    quoted = _quoted_timescales_ns(source)
    if not quoted:
        return
    value = float(value)
    if any(
        q / _ORDER_OF_MAGNITUDE <= value <= q * _ORDER_OF_MAGNITUDE for q in quoted
    ):
        return
    closest = min(quoted, key=lambda q: abs(q - value))
    raise ValueError(
        f"task_file: {path} states characteristic_timescale_ns={value:g}, but "
        f"its own timescale_source quotes {sorted(set(quoted))} ns, the "
        f"nearest of which ({closest:g} ns) is more than an order of magnitude "
        f"away. One of the two is wrong, and this is the number the pivot "
        f"decision is taken against — state the timescale in nanoseconds."
    )


def _verify_declared_constants(doc: dict[str, Any], path: Path) -> None:
    """A declared value that is not yet tunable must match what actually runs."""
    for (section, key), fixed in _VERIFIED.items():
        block = doc.get(section) or {}
        if key not in block:
            continue
        declared = block[key]
        if declared == fixed:
            continue
        raise ValueError(
            f"task_file: {path} declares {section}.{key} = {declared!r}, but "
            f"that is not tunable yet and the value actually used is "
            f"{fixed!r}. Either set it to {fixed!r}, drop it from the file, or "
            f"make it a real parameter before declaring it — a task file that "
            f"disagrees with the code is worse than one that stays silent."
        )


def _reject_unknown(
    where: str, got: set[str], allowed: set[str], path: Path
) -> None:
    unknown = sorted(got - allowed)
    if unknown:
        raise ValueError(
            f"task_file: {path} has unknown {where} key(s) {unknown}; "
            f"allowed: {sorted(allowed)}"
        )
