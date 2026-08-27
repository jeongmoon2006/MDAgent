"""The Streamlit app's non-widget logic.

The rendering is exercised headlessly by `streamlit.testing.v1.AppTest`; what
is worth unit-testing is the part that has consequences — the log formatter
(which runs inside the campaign observer and must never raise), the campaign
reader, and the worker thread that actually starts a run.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="UI extra not installed")
pytest.importorskip("matplotlib", reason="UI extra not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import app  # noqa: E402


# ---------- the log formatter runs inside the observer ----------

_EVENTS = [
    ("campaign_start", {"work_dir": "campaigns/x", "engine": "OpenMMAdapter",
                        "forcefield": "amber14/tip3p", "temperature_k": 300.0,
                        "timestep_fs": 2.0, "padding_nm": 1.5,
                        "observable": "rmsd_ca_to_reference_angstrom",
                        "start_round": 1, "max_rounds": 4, "resuming_from_round": 0}),
    ("round_start", {"round_index": 2, "n_steps": 25_000, "ns": 0.05, "phase": "metad"}),
    ("simulated", {"round_index": 2, "seconds": 12.3, "trajectory": "r/round_002.dcd"}),
    ("report", {"round_index": 2, "report": {"phase": "metad", "recrossings": 1,
                                             "fes_drift_kj_per_mol": 7.8}}),
    ("decision", {"round_index": 2, "decision": "extend", "reason": "still filling",
                  "extra_ns": 2.0, "metad_proposal": None}),
    ("override", {"round_index": 2, "note": "stop refused: fes_converged=False"}),
    ("pivot", {"round_index": 1, "kind": "switch_to_metad", "plumed_dat": "x/plumed.dat",
               "cv": {"cv_type": "contacts", "selections": ["name CA"], "label": "q"}}),
    ("campaign_end", {"stop_reason": "scientist_said_stop", "n_rounds": 2,
                      "biased_rounds": 1}),
]


@pytest.mark.parametrize("name,payload", _EVENTS)
def test_every_event_formats_to_lines(name: str, payload: dict) -> None:
    lines = app.format_event(name, payload)

    assert lines and all(isinstance(line, str) for line in lines)
    assert all(line.startswith("[") for line in lines)


def test_the_formatter_survives_events_it_does_not_know() -> None:
    """It runs inside `run_campaign`'s observer. A KeyError here would be
    swallowed by `_emit`, but the line would be lost silently — better to
    render something than to render nothing."""
    assert app.format_event("something_new", {"a": 1})
    assert app.format_event("round_start", {"round_index": 1, "n_steps": 1,
                                            "ns": 0.1, "phase": "vanilla"})
    assert app.format_event("decision", {"decision": "stop", "round_index": 1})


def test_the_decision_line_carries_the_reasoning() -> None:
    lines = "\n".join(app.format_event("decision", {
        "round_index": 3, "decision": "switch_to_metad",
        "reason": "pinned in one basin and the budget cannot reach the transition",
        "extra_ns": None,
        "metad_proposal": {"cv_type": "contacts", "selections": ["name CA"], "label": "q"},
    }))

    assert "SWITCH_TO_METAD" in lines
    assert "cannot reach the transition" in lines
    assert "contacts" in lines


# ---------- reading a campaign off disk ----------

def test_latest_fes_path_prefers_the_most_integrated_surface(tmp_path: Path) -> None:
    """`sum_hills` writes `fes.dat0.dat`, `fes.dat1.dat`, … so the highest index
    is the most hills integrated — and plain string sort puts `fes.dat9` after
    `fes.dat10`."""
    for name in ("fes.dat", "fes.dat2.dat", "fes.dat10.dat", "fes.dat9.dat"):
        (tmp_path / name).write_text("#\n")

    assert app.latest_fes_path(tmp_path).name == "fes.dat10.dat"


def test_latest_fes_path_falls_back_and_gives_up_cleanly(tmp_path: Path) -> None:
    assert app.latest_fes_path(tmp_path / "missing") is None
    assert app.latest_fes_path(tmp_path) is None
    (tmp_path / "fes.dat").write_text("#\n")
    assert app.latest_fes_path(tmp_path).name == "fes.dat"


def test_completed_rounds_is_quiet_about_a_directory_that_is_not_a_campaign(
    tmp_path: Path,
) -> None:
    """The viewer lists whatever is in `campaigns/`; a half-made directory must
    not take the page down."""
    assert app.completed_rounds(tmp_path) == []


# ---------- the worker thread ----------

def _drain_to_text(run: app.CampaignRun) -> str:
    return "\n".join(run.drain())


def test_the_worker_runs_the_campaign_with_the_task_files_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        kwargs["on_event"]("campaign_end", {"stop_reason": "max_rounds_reached",
                                            "n_rounds": 0, "biased_rounds": 0})
        class R: stop_reason = "max_rounds_reached"
        return R()

    monkeypatch.setattr(app, "run_campaign", fake_run_campaign)
    task_path = Path("benchmarks/tasks/cln025_folding.yaml")
    run = app.CampaignRun(work_dir=tmp_path)

    app.start_campaign(run, task_path, {"max_rounds": 3, "initial_steps": 100})
    run.thread.join(timeout=30)

    assert run.finished and run.error is None
    assert run.stop_reason == "max_rounds_reached"
    # The system the task file describes must reach the engine. Calling
    # `run_campaign` without an adapter silently falls back to
    # `SystemSpec.trpcage()`, so a chignolin task file ran 1L2Y for forty
    # minutes with nothing anywhere indicating the file had been ignored.
    assert captured["adapter"].spec.pdb_id == "5AWL"
    # Loop-control overrides win; the file still owns what the campaign is.
    assert captured["max_rounds"] == 3
    assert captured["state_thresholds"] == (1.5, 4.0)
    assert captured["min_recrossings"] == 2
    assert "campaign end" in _drain_to_text(run)


def test_a_failing_campaign_surfaces_in_the_log_rather_than_vanishing(
    tmp_path: Path, monkeypatch
) -> None:
    """A background thread that dies quietly leaves the UI showing 'running'
    forever."""
    def boom(**kwargs):
        raise RuntimeError("PLUMED is not on PATH")

    monkeypatch.setattr(app, "run_campaign", boom)
    run = app.CampaignRun(work_dir=tmp_path)

    app.start_campaign(run, Path("benchmarks/tasks/cln025_folding.yaml"), {})
    run.thread.join(timeout=30)

    assert run.finished
    assert "PLUMED is not on PATH" in (run.error or "")
    assert "PLUMED is not on PATH" in _drain_to_text(run)


def test_an_invalid_task_file_never_reaches_run_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(app, "run_campaign", lambda **k: pytest.fail("should not run"))
    bad = tmp_path / "task.yaml"
    bad.write_text("name: x\nsystem:\n  starting_pdb: 1L2Y\n  nonsense: 1\n")
    run = app.CampaignRun(work_dir=tmp_path)

    app.start_campaign(run, bad, {})
    run.thread.join(timeout=30)

    assert run.finished and "unknown system key" in (run.error or "")


def test_the_log_is_safe_to_write_from_the_campaign_thread(tmp_path: Path) -> None:
    """Streamlit forbids `st.*` off the main thread, so the worker only touches
    this queue and the script drains it on rerun."""
    run = app.CampaignRun(work_dir=tmp_path)
    writers = [
        threading.Thread(target=lambda i=i: [run.note(f"{i}:{j}") for j in range(50)])
        for i in range(4)
    ]
    for w in writers:
        w.start()
    for w in writers:
        w.join(timeout=30)

    assert len(run.drain()) == 200


# ---------- the viewer must not render anything unasked ----------

def test_the_viewer_starts_idle(tmp_path: Path) -> None:
    """A fresh page opened onto an animating trajectory from whatever campaign
    happened to sort last on disk — a run the user had not asked to see and had
    no context for. Agent-driven mode follows *this session's* campaign; with
    none, the viewer shows nothing.
    """
    from streamlit.testing.v1 import AppTest

    repo = Path(__file__).resolve().parents[2]
    at = AppTest.from_file(str(repo / "app.py"), default_timeout=180)
    at.run()

    assert not at.exception, at.exception
    # The round picker and the tabs that render structure and free energy are
    # not reached at all. Asserted this way rather than on the exact widget
    # list because `campaigns/` is gitignored: on a fresh checkout there are no
    # campaigns to pick from and the viewer stops one step earlier still. The
    # invariant is the same either way — nothing renders unasked.
    assert "Round" not in [s.label for s in at.selectbox]
    assert len(at.tabs) == 0


def test_choosing_a_campaign_still_renders_it() -> None:
    """The other half: manual override must actually work."""
    from streamlit.testing.v1 import AppTest

    from mdpilot.memory import store

    repo = Path(__file__).resolve().parents[2]
    # A campaign that actually has completed rounds. Picking whichever sorts
    # last is brittle: a campaign mid-setup has a state.db and no rounds, so
    # the viewer stops before the round picker and the assertions below fail
    # for a reason that has nothing to do with the app.
    with_rounds = sorted(
        d.name for d in (repo / "campaigns").glob("*")
        if (d / "state.db").exists() and store.list_rounds(d)
    )
    if not with_rounds:
        pytest.skip("no campaign with completed rounds to inspect")

    at = AppTest.from_file(str(repo / "app.py"), default_timeout=180)
    at.run()
    at.selectbox[0].select(with_rounds[-1]).run()

    assert not at.exception, at.exception
    assert [s.label for s in at.selectbox] == ["Campaign", "Round"]
    assert len(at.tabs) == 3


# ---------- the run-bounds panel must state what it will actually cost ----------

def test_the_envelope_states_the_real_ceiling() -> None:
    """`max rounds` alone says nothing about time. A round is either an opening
    round or an extension, and the scientist picks extension lengths within a
    ceiling that the panel did not expose at all."""
    note = app._envelope_note(1.0, 2.0, 20, 20.0)

    assert "up to 20 rounds" in note
    assert "~39 ns total" in note          # 1 + 19*2
    assert "20 ns of that may be biased" in note


def test_the_envelope_says_the_unbiased_phase_is_uncapped() -> None:
    """`run_campaign` gates its budget check on `in_metad`, so a campaign that
    keeps extending without pivoting is bounded by the round count alone. That
    is invisible from the controls."""
    assert "unbiased phase has no ns budget" in app._envelope_note(0.05, 2.0, 4, 0.1)


def test_the_budget_note_says_the_cap_replaces_the_files_value() -> None:
    """Overrides always win, in both directions. Someone could raise the cap to
    5, believe the file's 20 ns was in force, and get a quarter of a campaign."""
    task_yaml = Path("benchmarks/tasks/cln025_contacts.yaml").read_text()

    assert "20 ns" in app._budget_note(task_yaml, 0.1)
    assert "0.1 ns" in app._budget_note(task_yaml, 0.1)
    assert "matching the task file" in app._budget_note(task_yaml, 20.0)
    # An unparseable editor buffer must not take the panel down.
    assert app._budget_note("not: [valid", 0.1)
    assert app._budget_note("", 0.1)
