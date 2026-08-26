"""The task file is the one artifact a user — or a setup agent — writes.

These pin the property that makes it worth generating: a task file cannot
silently disagree with the code. Fields that are tunable are mapped, fields
that are not are checked against the constant that governs them, and anything
unrecognised raises rather than being dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mdpilot.adapters.system_spec import Ensemble
from mdpilot.task_file import load_task_file

_TASKS = Path("benchmarks/tasks")


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def _minimal(**system) -> dict:
    return {"name": "t", "system": {"starting_pdb": "1L2Y", **system}}


# ---------- the real files ----------

def test_both_shipped_task_files_load() -> None:
    """They were decorative until now; nothing ever parsed trpcage's, which is
    how its `diagnostics:` block stayed invalid YAML."""
    for path in sorted(_TASKS.glob("*.yaml")):
        assert load_task_file(path).name


def test_cln025_produces_the_kwargs_the_runner_hand_assembled() -> None:
    """Drop-in proof: the file owns exactly the campaign half, and the runner
    keeps the loop-control bounds (max_rounds, initial_steps, seed)."""
    task = load_task_file(_TASKS / "cln025_folding.yaml")

    assert task.spec.pdb_id == "5AWL"
    assert task.campaign == {
        "task_expectation": task.campaign["task_expectation"],
        "cv_upper_wall_nm": 0.8,
        "min_recrossings": 2,
        "max_biased_ns": 20.0,
        "state_thresholds": (1.5, 4.0),
        # Prose, carried only for the pre-flight structure check.
        "description": task.campaign["description"],
    }
    # (low, high) on the observable — the order run_campaign refuses inverted.
    assert task.campaign["state_thresholds"][0] < task.campaign["state_thresholds"][1]


def test_a_pure_convergence_task_yields_no_campaign_overrides() -> None:
    """No task_expectation and no done_criterion, so the loop's own defaults
    stand and `switch_to_metad` stays out of the action space."""
    task = load_task_file(_TASKS / "trpcage_convergence.yaml")

    # `description` is prose for the pre-flight check, not a campaign
    # parameter; nothing here changes what the loop does.
    assert set(task.campaign) <= {"description"}
    assert task.done_criterion == {}


# ---------- mapped fields ----------

def test_the_integrator_block_reaches_the_ensemble(tmp_path: Path) -> None:
    doc = _minimal()
    doc["integrator"] = {"temperature_K": 240.0, "timestep_fs": 1.0}

    spec = load_task_file(_write(tmp_path, doc)).spec

    assert spec.ensemble == Ensemble(temperature_k=240.0, timestep_fs=1.0)
    assert "ensemble" in spec.to_dict()   # so the resume guard covers it


def test_an_ensemble_the_adapters_cannot_integrate_is_refused(tmp_path: Path) -> None:
    doc = _minimal()
    doc["integrator"] = {"timestep_fs": 4.0}

    with pytest.raises(ValueError, match="hydrogen mass"):
        load_task_file(_write(tmp_path, doc))


# ---------- declared-but-not-tunable ----------

def test_a_declared_value_that_disagrees_with_the_code_raises(tmp_path: Path) -> None:
    """The whole point. `padding_nm: 1.5` used to be documentation the adapter
    was free to contradict, and it ran at 1.0 with nothing said."""
    with pytest.raises(ValueError, match="not tunable yet"):
        load_task_file(_write(tmp_path, _minimal(ionic_strength_M=0.3)))


def test_a_declared_value_that_agrees_is_accepted(tmp_path: Path) -> None:
    from mdpilot.adapters.openmm_adapter import _SALT_M

    assert load_task_file(_write(tmp_path, _minimal(ionic_strength_M=_SALT_M)))


def test_the_verified_table_is_sourced_from_the_modules_that_own_the_values() -> None:
    """Retyped constants drift. These must be the live ones."""
    from mdpilot.adapters.openmm_adapter import _SALT_M
    from mdpilot.task_file import _VERIFIED

    assert _VERIFIED[("system", "ionic_strength_M")] == _SALT_M
    assert _VERIFIED[("diagnostics", "target_ess")] == 50.0
    # Fields leave this table as they become declarable; a stale entry would
    # refuse every campaign that sets its own value. The observable went first,
    # then padding, then the force field.
    assert not any(section == "observable" for section, _ in _VERIFIED)
    for gone in ("padding_nm", "forcefield", "water_model"):
        assert ("system", gone) not in _VERIFIED


# ---------- unknown keys: how a generated file's typos surface ----------

def test_an_unknown_key_raises_rather_than_being_dropped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown system key"):
        load_task_file(_write(tmp_path, _minimal(temperature_K=300.0)))  # wrong block

    doc = _minimal()
    doc["smapling"] = {"cv_upper_wall_nm": 0.8}   # plausible LLM typo
    with pytest.raises(ValueError, match="unknown <top level> key"):
        load_task_file(_write(tmp_path, doc))


def test_run_kwargs_rejects_an_override_run_campaign_would_not_take() -> None:
    task = load_task_file(_TASKS / "cln025_folding.yaml")

    assert task.run_kwargs(max_rounds=3)["max_rounds"] == 3
    with pytest.raises(ValueError, match="not run_campaign parameters"):
        task.run_kwargs(max_round=3)


def test_the_file_hashes_itself_for_provenance() -> None:
    import hashlib

    path = _TASKS / "cln025_folding.yaml"
    assert load_task_file(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- task_expectation is rendered, not authored ----------
#
# It is the only input gating `switch_to_metad`, and it was the one
# load-bearing input that escaped structuring. These pin that the prose and the
# typed fields agree *by construction* rather than by an author keeping two
# copies in step.

def _campaign_doc(**criterion_overrides) -> dict:
    doc = _minimal()
    doc["observable"] = {
        "cv_type": "rmsd",
        "selections": ["protein and name CA"],
        "name": "rmsd_ca_to_reference_angstrom",
        "scale": 10.0,
    }
    doc["integrator"] = {"temperature_K": 300.0}
    doc["expectation"] = {
        "objective": "Sample the transition in both directions.",
        "characteristic_timescale_ns": 800.0,
        "timescale_source": "Someone et al. 2011",
    }
    doc["done_criterion"] = {
        "states": {
            "low": {"name": "native beta-hairpin", "threshold": 1.5},
            "high": {"name": "extended", "threshold": 4.0},
        },
        "min_recrossings": 2,
        "max_biased_ns": 20.0,
        **criterion_overrides,
    }
    return doc


def test_every_decision_driving_number_appears_in_the_rendered_prose() -> None:
    task = load_task_file(_TASKS / "cln025_folding.yaml")
    prose = task.campaign["task_expectation"]

    lo, hi = task.campaign["state_thresholds"]
    for value in (lo, hi, task.campaign["min_recrossings"],
                  task.campaign["max_biased_ns"]):
        assert f"{value:g}" in prose, (value, prose)


def test_changing_a_typed_field_changes_the_prose(tmp_path: Path) -> None:
    """The drift test. Under the old free-text expectation the thresholds were
    restated in words, so editing done_criterion left the prose stale and
    nothing noticed."""
    from mdpilot.task_file import render_task_expectation

    base = render_task_expectation(_campaign_doc())
    for override in (
        {"min_recrossings": 1},
        {"max_biased_ns": 5.0},
        {"states": {
            "low": {"name": "native beta-hairpin", "threshold": 2.0},
            "high": {"name": "extended", "threshold": 4.0},
        }},
    ):
        assert render_task_expectation(_campaign_doc(**override)) != base, override


def test_the_budget_versus_timescale_comparison_is_computed(tmp_path: Path) -> None:
    """`phase_vanilla` asks the scientist to compare cumulative simulation time
    against the characteristic timescale. Doing the arithmetic here means it is
    not doing it on prose."""
    from mdpilot.task_file import render_task_expectation

    prose = render_task_expectation(_campaign_doc())

    assert "40x shorter" in prose          # 800 ns / 20 ns
    assert "Someone et al. 2011" in prose  # the one free fact carries its source


def test_an_authored_task_expectation_is_not_accepted(tmp_path: Path) -> None:
    """Allowing both would reintroduce exactly the drift rendering removes."""
    doc = _campaign_doc()
    doc["task_expectation"] = "hand-written prose"

    with pytest.raises(ValueError, match="unknown <top level> key"):
        load_task_file(_write(tmp_path, doc))


# ---------- the schema is not folding-shaped ----------

def _ligand_doc() -> dict:
    """A protein-ligand binding campaign. Same mechanics, different states."""
    doc = _minimal(starting_pdb="1STP")
    doc["observable"] = {
        "cv_type": "distance",
        "selections": ["resname LIG and name C1", "protein and resid 42 and name CA"],
        "name": "ligand_com_distance_nm",
    }
    doc["expectation"] = {
        "objective": "Sample binding and unbinding of the ligand.",
        "characteristic_timescale_ns": 50_000.0,
        "timescale_source": "residence time ~50 us, SPR",
    }
    doc["done_criterion"] = {
        "states": {
            "low": {"name": "bound", "threshold": 0.4},
            "high": {"name": "unbound", "threshold": 1.5},
        },
        "min_recrossings": 2,
        "max_biased_ns": 100.0,
    }
    return doc


def test_the_renderer_is_not_folding_shaped() -> None:
    """The goal is a general MD agent. `low`/`high` name bands on whatever the
    campaign observable is; a binding campaign uses the same mechanics on a
    distance and `count_recrossings` is unchanged."""
    from mdpilot.task_file import render_task_expectation

    prose = render_task_expectation(_ligand_doc())

    assert '"unbound" state (ligand_com_distance_nm > 1.5)' in prose
    assert '"bound" state (ligand_com_distance_nm < 0.4)' in prose
    assert "500x shorter" in prose          # 50 us residence vs 100 ns budget
    assert "fold" not in prose.lower()      # no folding vocabulary anywhere


def test_a_non_protein_observable_now_loads(tmp_path: Path) -> None:
    """This test used to assert the opposite.

    `campaign_observable` was hardcoded to `protein and name CA`, so a binding
    campaign could not be scored at all and the loader refused any other
    `observable.name` rather than let one be silently judged on protein RMSD.
    The observable is now a declared collective variable, so the refusal is
    gone and the declaration carries through to `run_campaign`.
    """
    from mdpilot.observables import ObservableSpec

    task = load_task_file(_write(tmp_path, _ligand_doc()))

    assert task.observable_name == "ligand_com_distance_nm"
    assert task.campaign["observable"] == ObservableSpec(
        cv_type="distance",
        selections=("resname LIG and name C1", "protein and resid 42 and name CA"),
        name="ligand_com_distance_nm",
    )


# ---------- structural refusals a generated file will hit ----------

def test_an_inverted_state_band_raises(tmp_path: Path) -> None:
    doc = _campaign_doc(states={
        "low": {"name": "extended", "threshold": 4.0},
        "high": {"name": "native", "threshold": 1.5},
    })
    with pytest.raises(ValueError, match="high.threshold > low.threshold"):
        load_task_file(_write(tmp_path, doc))


def test_an_expectation_without_states_raises_naming_the_block(
    tmp_path: Path,
) -> None:
    """run_campaign refuses this too, but from the loop's vantage the message
    names an argument the task-file author never sees."""
    doc = _campaign_doc()
    del doc["done_criterion"]["states"]

    with pytest.raises(ValueError, match="done_criterion.states is missing"):
        load_task_file(_write(tmp_path, doc))


def test_an_unknown_expectation_key_raises(tmp_path: Path) -> None:
    doc = _campaign_doc()
    doc["expectation"]["timescale_us"] = 0.8   # plausible unit slip

    with pytest.raises(ValueError, match="unknown expectation key"):
        load_task_file(_write(tmp_path, doc))


# ---------- the timescale is the field with no second copy ----------

def test_a_timescale_contradicting_its_own_source_is_refused(tmp_path: Path) -> None:
    """Caught on the setup agent's first live run: it cited a ~1 microsecond
    folding time and wrote `characteristic_timescale_ns: 1.0`. A unit slip here
    does not look wrong, it looks like a small number, and it inverts the pivot
    decision — the scientist would conclude unbiased MD reaches the transition
    easily and never switch to enhanced sampling."""
    doc = _campaign_doc()
    doc["expectation"]["characteristic_timescale_ns"] = 1.0
    doc["expectation"]["timescale_source"] = "folds on the ~1 microsecond timescale"

    with pytest.raises(ValueError, match="more than an order of magnitude"):
        load_task_file(_write(tmp_path, doc))


def test_a_timescale_agreeing_with_its_source_passes(tmp_path: Path) -> None:
    doc = _campaign_doc()
    doc["expectation"]["characteristic_timescale_ns"] = 1000.0
    doc["expectation"]["timescale_source"] = "folds on the ~1 microsecond timescale"

    assert load_task_file(_write(tmp_path, doc))


def test_a_budget_that_reaches_the_timescale_is_not_described_as_short() -> None:
    """`{ratio:.0f}x shorter` rendered "0x shorter ... expected to stay trapped"
    when the budget exceeded the timescale — asserting the opposite of the
    numbers, in the sentence the pivot rule reads."""
    from mdpilot.task_file import render_task_expectation

    doc = _campaign_doc(max_biased_ns=5000.0)      # budget >> 800 ns timescale
    prose = render_task_expectation(doc)

    assert "0x shorter" not in prose
    assert "may reach the transition without enhanced sampling" in prose
    assert "expected to stay trapped" not in prose


def test_padding_is_declarable_and_reaches_the_spec(tmp_path: Path) -> None:
    """It moved out of the verified-but-fixed table when it became tunable; a
    stale entry there would refuse every campaign that sets its own box."""
    from mdpilot.task_file import _VERIFIED

    assert ("system", "padding_nm") not in _VERIFIED
    assert load_task_file(_write(tmp_path, _minimal(padding_nm=2.0))).spec.padding_nm == 2.0
    assert load_task_file(_write(tmp_path, _minimal())).spec.padding_nm == 1.5


def test_the_shipped_task_files_use_the_post_f11_box() -> None:
    for name in ("cln025_folding", "trpcage_convergence"):
        assert load_task_file(_TASKS / f"{name}.yaml").spec.padding_nm == 1.5


def test_build_adapter_carries_the_whole_system_spec(tmp_path: Path) -> None:
    """`run_kwargs` deliberately excludes the spec — it belongs to the adapter.
    That split is a footgun, because `run_campaign(adapter=None)` falls back to
    Trp-cage rather than complaining, so a caller who forgets gets a different
    molecule with no signal at all."""
    from mdpilot.adapters.system_spec import Ensemble

    doc = _minimal(starting_pdb="5AWL", padding_nm=2.0, forcefield="charmm36/tip3p")
    doc["integrator"] = {"temperature_K": 240.0, "timestep_fs": 1.0}
    task = load_task_file(_write(tmp_path, doc))

    adapter = task.build_adapter(tmp_path, seed=7)

    assert adapter.spec.pdb_id == "5AWL"
    assert adapter.spec.padding_nm == 2.0
    assert adapter.spec.forcefield == "charmm36/tip3p"
    assert adapter.spec.ensemble == Ensemble(temperature_k=240.0, timestep_fs=1.0)
    assert adapter.temperature_k == 240.0 and adapter.timestep_fs == 1.0
    # None of this is in run_kwargs, which is exactly why the helper exists.
    assert "spec" not in task.run_kwargs()


def test_a_source_may_mention_other_timescales_for_context(tmp_path: Path) -> None:
    """Regression. The check keyed on bare units, so a source that mentioned a
    slower process for comparison — which scientific prose does constantly —
    was refused although its own number was right. Hit live: a draft citing
    "~1 µs" alongside a millisecond comparison was rejected at 1000 ns."""
    doc = _campaign_doc()
    doc["expectation"]["characteristic_timescale_ns"] = 1000.0
    doc["expectation"]["timescale_source"] = (
        "Chignolin folds on the ~1 microsecond timescale, far below the "
        "millisecond timescale of larger proteins."
    )

    assert load_task_file(_write(tmp_path, doc))


def test_a_source_with_no_stated_magnitude_is_not_judged(tmp_path: Path) -> None:
    """Nothing to compare against is not evidence of a mistake."""
    doc = _campaign_doc()
    doc["expectation"]["timescale_source"] = "Honda et al. (2004) JACS 126:15318"

    assert load_task_file(_write(tmp_path, doc))


def test_the_refusal_names_what_the_source_actually_quoted(tmp_path: Path) -> None:
    doc = _campaign_doc()
    doc["expectation"]["characteristic_timescale_ns"] = 1.0
    doc["expectation"]["timescale_source"] = "folds on the ~1 microsecond timescale"

    with pytest.raises(ValueError) as excinfo:
        load_task_file(_write(tmp_path, doc))

    assert "quotes [1000.0] ns" in str(excinfo.value)
