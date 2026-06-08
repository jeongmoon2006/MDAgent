"""SQLite memory store: schema, idempotent init, round append, ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot.memory import store


def _config() -> dict:
    return {"initial_steps": 25_000, "seed": 42, "max_rounds": 10}


def _report(n_frames: int, ess: float) -> dict:
    return {
        "trajectory_path": "rounds/round_001.dcd",
        "n_frames": n_frames,
        "ess": ess,
        "plateau_reached": ess > 50,
        "well_sampled": ess > 50,
    }


def test_init_creates_db_and_campaign_row(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    assert store.db_path(tmp_path).exists()
    assert store.get_campaign_config(tmp_path) == _config()


def test_init_is_idempotent_on_matching_config(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    store.init_campaign(tmp_path, _config())
    assert store.get_campaign_config(tmp_path) == _config()


def test_init_rejects_config_change(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    with pytest.raises(ValueError, match="different config"):
        store.init_campaign(tmp_path, {**_config(), "seed": 99})


def test_get_campaign_config_returns_none_when_missing(tmp_path: Path) -> None:
    assert store.get_campaign_config(tmp_path) is None


def test_append_and_list_rounds_in_order(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    for i in (1, 2, 3):
        store.append_round(
            tmp_path,
            round_index=i,
            n_steps=25_000 * i,
            dcd_path=tmp_path / f"rounds/round_{i:03d}.dcd",
            checkpoint_path=tmp_path / f"rounds/round_{i:03d}.chk",
            report=_report(n_frames=50 * i, ess=10.0 * i),
            decision="extend" if i < 3 else "stop",
            reason=f"round {i} reasoning",
            extra_ns=0.5 if i < 3 else None,
        )

    rounds = store.list_rounds(tmp_path)
    assert [r.round_index for r in rounds] == [1, 2, 3]
    assert rounds[0].decision == "extend"
    assert rounds[2].decision == "stop"
    assert rounds[2].extra_ns is None
    assert rounds[1].report["ess"] == 20.0
    assert rounds[0].checkpoint_path == tmp_path / "rounds/round_001.chk"


def test_get_last_round(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    assert store.get_last_round(tmp_path) is None
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=1000,
        dcd_path=tmp_path / "r1.dcd",
        checkpoint_path=None,
        report=_report(n_frames=10, ess=2.0),
        decision="extend",
        reason="early",
        extra_ns=0.5,
    )
    last = store.get_last_round(tmp_path)
    assert last is not None and last.round_index == 1
    assert last.checkpoint_path is None


def test_duplicate_round_index_raises(tmp_path: Path) -> None:
    import sqlite3

    store.init_campaign(tmp_path, _config())
    kwargs = dict(
        round_index=1,
        n_steps=1000,
        dcd_path=tmp_path / "r1.dcd",
        checkpoint_path=None,
        report=_report(n_frames=10, ess=2.0),
        decision="extend",
        reason="x",
        extra_ns=0.5,
    )
    store.append_round(tmp_path, **kwargs)
    with pytest.raises(sqlite3.IntegrityError):
        store.append_round(tmp_path, **kwargs)


def test_invalid_decision_rejected_by_check_constraint(tmp_path: Path) -> None:
    import sqlite3

    store.init_campaign(tmp_path, _config())
    with pytest.raises(sqlite3.IntegrityError):
        store.append_round(
            tmp_path,
            round_index=1,
            n_steps=1000,
            dcd_path=tmp_path / "r1.dcd",
            checkpoint_path=None,
            report=_report(n_frames=10, ess=2.0),
            decision="bogus",
            reason="x",
            extra_ns=None,
        )


def test_list_rounds_empty_when_no_db(tmp_path: Path) -> None:
    assert store.list_rounds(tmp_path) == []


def test_ledger_empty_when_no_db(tmp_path: Path) -> None:
    assert store.list_ledger_notes(tmp_path) == []


def test_append_and_list_ledger_notes_in_order(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    store.append_ledger_note(tmp_path, round_index=1, text="trajectory still relaxing")
    store.append_ledger_note(tmp_path, round_index=2, text="ess plateauing low")
    store.append_ledger_note(tmp_path, round_index=2, text="possible slow torsion")

    notes = store.list_ledger_notes(tmp_path)
    assert [(n.round_index, n.text) for n in notes] == [
        (1, "trajectory still relaxing"),
        (2, "ess plateauing low"),
        (2, "possible slow torsion"),
    ]


def test_ledger_independent_of_rounds_table(tmp_path: Path) -> None:
    """Ledger notes can reference round indices that don't yet exist in the
    rounds table (e.g. a note recorded before the round row is inserted, or
    a hypothetical-future note). No FK constraint is enforced."""
    store.init_campaign(tmp_path, _config())
    store.append_ledger_note(tmp_path, round_index=99, text="future-hypothesis stub")
    notes = store.list_ledger_notes(tmp_path)
    assert len(notes) == 1 and notes[0].round_index == 99


# ---------- M4: switch_to_metad decision + metad_proposal_json column ----------

def test_switch_to_metad_round_trips_with_proposal(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    proposal = {
        "cv_type": "gyration",
        "selections": ["backbone and resSeq 1 to 10"],
        "label": "rg_back",
    }
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=25_000,
        dcd_path=tmp_path / "r1.dcd",
        checkpoint_path=tmp_path / "r1.chk",
        report=_report(n_frames=50, ess=3.0),
        decision="switch_to_metad",
        reason="exploring=False; task wants µs fold; budget short",
        extra_ns=None,
        metad_proposal=proposal,
    )
    rows = store.list_rounds(tmp_path)
    assert len(rows) == 1
    assert rows[0].decision == "switch_to_metad"
    assert rows[0].extra_ns is None
    assert rows[0].metad_proposal == proposal


def test_extend_decision_persists_null_metad_proposal(tmp_path: Path) -> None:
    """The metad_proposal column must hold null for non-switch decisions."""
    store.init_campaign(tmp_path, _config())
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=25_000,
        dcd_path=tmp_path / "r1.dcd",
        checkpoint_path=None,
        report=_report(n_frames=50, ess=3.0),
        decision="extend",
        reason="ess=3",
        extra_ns=0.5,
    )
    rows = store.list_rounds(tmp_path)
    assert rows[0].metad_proposal is None


def test_pre_m4_db_is_migrated_to_metad_aware_schema(tmp_path: Path) -> None:
    """A SQLite DB created with the pre-M4 schema (no metad_proposal_json
    column, decision CHECK without 'switch_to_metad') is migrated in place by
    init_campaign so old campaigns can resume and accept the new decision."""
    import sqlite3

    import json as _json

    pre_m4_schema = """
    CREATE TABLE IF NOT EXISTS campaign (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS rounds (
        round_index INTEGER PRIMARY KEY,
        n_steps INTEGER NOT NULL,
        dcd_path TEXT NOT NULL,
        checkpoint_path TEXT,
        report_json TEXT NOT NULL,
        decision TEXT NOT NULL CHECK (decision IN ('extend', 'stop')),
        reason TEXT NOT NULL,
        extra_ns REAL,
        created_at TEXT NOT NULL
    );
    """
    db = store.db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(pre_m4_schema)
        conn.execute(
            """
            INSERT INTO rounds (round_index, n_steps, dcd_path, checkpoint_path,
                report_json, decision, reason, extra_ns, created_at)
            VALUES (1, 1000, 'r1.dcd', NULL, '{}', 'extend', 'x', 0.5, '2026-01-01')
            """
        )
        conn.execute(
            "INSERT INTO campaign (id, config_json, created_at) VALUES (1, ?, ?)",
            (_json.dumps(_config(), sort_keys=True), "2026-01-01"),
        )

    store.init_campaign(tmp_path, _config())  # triggers migration

    rows = store.list_rounds(tmp_path)
    assert len(rows) == 1 and rows[0].decision == "extend"
    assert rows[0].metad_proposal is None

    # The widened CHECK constraint should now accept switch_to_metad.
    store.append_round(
        tmp_path,
        round_index=2,
        n_steps=1000,
        dcd_path=tmp_path / "r2.dcd",
        checkpoint_path=None,
        report=_report(n_frames=50, ess=3.0),
        decision="switch_to_metad",
        reason="pinned + task wants transition",
        extra_ns=None,
        metad_proposal={"cv_type": "gyration", "selections": ["all"], "label": "rg"},
    )
    rows = store.list_rounds(tmp_path)
    assert rows[-1].decision == "switch_to_metad"
