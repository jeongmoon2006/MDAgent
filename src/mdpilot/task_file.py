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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mdpilot.adapters import openmm_adapter as _omm
from mdpilot.adapters.system_spec import Ensemble, SystemSpec
from mdpilot.diagnostics.autocorrelation import autocorrelation
from mdpilot.diagnostics.report import OBSERVABLE_NAME

# Declared in task files, governed by a constant elsewhere. Sourced from the
# module that owns each value rather than retyped, so the check cannot drift
# into asserting a number nothing uses any more.
_VERIFIED: dict[tuple[str, str], Any] = {
    ("system", "forcefield"): list(_omm._FORCEFIELD_FILES),
    ("system", "water_model"): "tip3p",
    ("system", "padding_nm"): _omm._PADDING_NM,
    ("system", "ionic_strength_M"): _omm._SALT_M,
    ("integrator", "type"): "LangevinMiddle",
    ("integrator", "friction_per_ps"): _omm._FRICTION_PER_PS,
    ("integrator", "constraints"): "HBonds",
    ("observable", "name"): OBSERVABLE_NAME,
    ("diagnostics", "target_ess"): inspect.signature(
        autocorrelation
    ).parameters["target_ess"].default,
}

# Carried for humans, not interpreted.
_INFORMATIONAL = {
    ("system", "starting_state"),
    ("observable", "selection"),
    ("observable", "reference"),
    ("diagnostics", "methods"),
    ("done_criterion", "pivot_required"),
}

_TOP_LEVEL = {
    "name", "description", "system", "integrator", "observable",
    "diagnostics", "sampling", "task_expectation", "done_criterion",
}


@dataclass(frozen=True)
class TaskFile:
    """A parsed, checked task file."""

    name: str
    spec: SystemSpec
    campaign: dict[str, Any]
    done_criterion: dict[str, Any]
    sha256: str
    path: Path

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
    _verify_declared_constants(doc, path)

    return TaskFile(
        name=doc.get("name", path.stem),
        spec=_build_spec(doc, path),
        campaign=_build_campaign(doc),
        done_criterion=dict(doc.get("done_criterion") or {}),
        sha256=hashlib.sha256(raw).hexdigest(),
        path=path,
    )


def _build_spec(doc: dict[str, Any], path: Path) -> SystemSpec:
    system = doc.get("system") or {}
    integrator = doc.get("integrator") or {}
    _reject_unknown(
        "system",
        set(system),
        {"starting_pdb", "structure_path"}
        | {k for s, k in _VERIFIED if s == "system"}
        | {k for s, k in _INFORMATIONAL if s == "system"},
        path,
    )
    _reject_unknown(
        "integrator",
        set(integrator),
        {"temperature_K", "timestep_fs"}
        | {k for s, k in _VERIFIED if s == "integrator"},
        path,
    )

    ensemble_kwargs: dict[str, Any] = {}
    if "temperature_K" in integrator:
        ensemble_kwargs["temperature_k"] = float(integrator["temperature_K"])
    if "timestep_fs" in integrator:
        ensemble_kwargs["timestep_fs"] = float(integrator["timestep_fs"])
    ensemble = Ensemble(**ensemble_kwargs)

    # SystemSpec enforces exactly-one-of; let its message stand.
    return SystemSpec(
        pdb_id=system.get("starting_pdb"),
        structure_path=(
            Path(system["structure_path"]) if system.get("structure_path") else None
        ),
        ensemble=ensemble,
    )


def _build_campaign(doc: dict[str, Any]) -> dict[str, Any]:
    """The `run_campaign` keywords the file owns. Absent keys are omitted
    rather than passed as None, so the loop's own defaults stay in charge."""
    campaign: dict[str, Any] = {}

    if doc.get("task_expectation"):
        campaign["task_expectation"] = doc["task_expectation"]

    sampling = doc.get("sampling") or {}
    for key in ("cv_upper_wall_nm", "bias_pace", "bias_factor"):
        if key in sampling:
            campaign[key] = sampling[key]

    criterion = doc.get("done_criterion") or {}
    if "min_recrossings" in criterion:
        campaign["min_recrossings"] = int(criterion["min_recrossings"])
    if "max_biased_ns" in criterion:
        campaign["max_biased_ns"] = float(criterion["max_biased_ns"])
    # `state_thresholds` is (folded, extended) — the band biased-round
    # recrossings are counted between. run_campaign refuses an inverted pair,
    # so the ordering here is load-bearing, not cosmetic.
    if {"folded_state_rmsd_angstrom", "extended_state_rmsd_angstrom"} <= set(
        criterion
    ):
        campaign["state_thresholds"] = (
            float(criterion["folded_state_rmsd_angstrom"]),
            float(criterion["extended_state_rmsd_angstrom"]),
        )
    return campaign


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
