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


# ---------- M4 step 4: plumed_dat_path column (biased-round marker) ----------

def test_biased_round_persists_plumed_dat_path(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=25_000,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=_report(n_frames=50, ess=80.0),
        decision="extend",
        reason="biased run, still filling",
        extra_ns=0.5,
        plumed_dat_path=tmp_path / "plumed.dat",
    )
    rows = store.list_rounds(tmp_path)
    assert rows[0].plumed_dat_path == tmp_path / "plumed.dat"


def test_vanilla_round_has_null_plumed_dat_path(tmp_path: Path) -> None:
    store.init_campaign(tmp_path, _config())
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=25_000,
        dcd_path=tmp_path / "r1.dcd",
        checkpoint_path=None,
        report=_report(n_frames=50, ess=3.0),
        decision="extend",
        reason="vanilla",
        extra_ns=0.5,
    )
    assert store.list_rounds(tmp_path)[0].plumed_dat_path is None


def test_pre_plumed_db_gains_column_and_keeps_rows(tmp_path: Path) -> None:
    """An M4-step-3 DB (has metad_proposal_json, no plumed_dat_path) is migrated
    in place: the column is added, existing rows read back with a NULL path, and
    a new biased round persists its path."""
    import sqlite3

    import json as _json

    pre_plumed_schema = """
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
        decision TEXT NOT NULL CHECK (decision IN ('extend', 'stop', 'switch_to_metad')),
        reason TEXT NOT NULL,
        extra_ns REAL,
        metad_proposal_json TEXT,
        created_at TEXT NOT NULL
    );
    """
    db = store.db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(pre_plumed_schema)
        conn.execute(
            """
            INSERT INTO rounds (round_index, n_steps, dcd_path, checkpoint_path,
                report_json, decision, reason, extra_ns, metad_proposal_json, created_at)
            VALUES (1, 1000, 'r1.dcd', NULL, '{}', 'extend', 'x', 0.5, NULL, '2026-01-01')
            """
        )
        conn.execute(
            "INSERT INTO campaign (id, config_json, created_at) VALUES (1, ?, ?)",
            (_json.dumps(_config(), sort_keys=True), "2026-01-01"),
        )

    store.init_campaign(tmp_path, _config())  # triggers migration

    rows = store.list_rounds(tmp_path)
    assert len(rows) == 1 and rows[0].plumed_dat_path is None

    store.append_round(
        tmp_path,
        round_index=2,
        n_steps=1000,
        dcd_path=tmp_path / "r2.dcd",
        checkpoint_path=None,
        report=_report(n_frames=50, ess=90.0),
        decision="extend",
        reason="biased",
        extra_ns=0.5,
        plumed_dat_path=tmp_path / "plumed.dat",
    )
    assert store.list_rounds(tmp_path)[-1].plumed_dat_path == tmp_path / "plumed.dat"


def test_pre_switch_cv_database_is_migrated_in_place(tmp_path: Path) -> None:
    """A campaign started before CV revision existed must accept a switch_cv
    round after upgrading, without losing the rounds already recorded. SQLite
    cannot ALTER a CHECK constraint, so this is a rename-create-copy-drop and
    the copy is the part that can silently drop data."""
    import sqlite3

    from mdpilot.memory.store import _connect, _migrate_rounds_for_cv_switch

    work_dir = tmp_path / "campaign"
    work_dir.mkdir(parents=True)
    with _connect(work_dir) as conn:
        conn.executescript(
            """
            CREATE TABLE rounds (
                round_index INTEGER PRIMARY KEY,
                n_steps INTEGER NOT NULL,
                dcd_path TEXT NOT NULL,
                checkpoint_path TEXT,
                report_json TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (
                    decision IN ('extend', 'stop', 'switch_to_metad')
                ),
                reason TEXT NOT NULL,
                extra_ns REAL,
                metad_proposal_json TEXT,
                plumed_dat_path TEXT,
                created_at TEXT NOT NULL
            );
            INSERT INTO rounds VALUES
                (1, 100, 'a.dcd', 'a.chk', '{}', 'extend', 'r', 0.5, NULL, NULL, 'now');
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rounds VALUES (2, 100, 'b.dcd', NULL, '{}', "
                "'switch_cv', 'r', NULL, NULL, NULL, 'now')"
            )
        conn.rollback()

        _migrate_rounds_for_cv_switch(conn)

        conn.execute(
            "INSERT INTO rounds VALUES (2, 100, 'b.dcd', NULL, '{}', "
            "'switch_cv', 'r', NULL, NULL, NULL, 'now')"
        )
        assert conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 2
        # The pre-existing round survived the copy.
        assert conn.execute(
            "SELECT dcd_path FROM rounds WHERE round_index = 1"
        ).fetchone()[0] == "a.dcd"


def test_cv_switch_migration_is_idempotent(tmp_path: Path) -> None:
    """It keys on the stored DDL rather than a new column, so a second run must
    detect that the constraint is already wide and do nothing."""
    from mdpilot.memory.store import _connect, _migrate_rounds_for_cv_switch, init_campaign

    work_dir = tmp_path / "campaign"
    init_campaign(work_dir, {"seed": 1})
    with _connect(work_dir) as conn:
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'rounds'"
        ).fetchone()[0]
        _migrate_rounds_for_cv_switch(conn)
        _migrate_rounds_for_cv_switch(conn)
        after = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'rounds'"
        ).fetchone()[0]
    assert before == after
    assert "switch_cv" in after


# ---------- campaign config compatibility across schema changes ----------
#
# The resume guard used to compare config JSON byte-for-byte, so *adding* a
# config key stranded every campaign already on disk — their stored config
# could not contain a key that did not exist when they started. Four real
# campaigns were lost that way. These pin the fix and, just as importantly,
# pin that the guard still refuses a genuine change.

def _legacy_config() -> dict:
    """A config as written before state_thresholds / min_recrossings existed."""
    return {
        "seed": 42,
        "initial_steps": 5_000,
        "report_interval_steps": 500,
        "equilibration_steps": 0,
        "system_spec": {"pdb_id": "1L2Y", "structure_path": None},
        "engine": "OpenMMAdapter",
        "task_expectation": None,
    }


def test_a_campaign_predating_a_key_resumes_when_the_value_matches_history(
    tmp_path: Path,
) -> None:
    """`min_recrossings` did not exist; the behaviour then was 1. Reopening
    with 1 describes the same campaign, so it must not be refused."""
    store.init_campaign(tmp_path, _legacy_config())

    store.init_campaign(
        tmp_path, {**_legacy_config(), "min_recrossings": 1, "state_thresholds": None}
    )


def test_a_campaign_predating_a_key_is_refused_when_the_value_changed(
    tmp_path: Path,
) -> None:
    """The other half. That campaign counted one crossing as done; reopening at
    2 changes when it is allowed to end, retroactively, for rounds already
    judged under the old rule."""
    store.init_campaign(tmp_path, _legacy_config())

    with pytest.raises(ValueError, match="different config"):
        store.init_campaign(tmp_path, {**_legacy_config(), "min_recrossings": 2})


def test_a_genuine_change_to_a_recorded_key_is_still_refused(tmp_path: Path) -> None:
    """The guard's actual job, unaffected by the compatibility layer."""
    store.init_campaign(tmp_path, _legacy_config())

    with pytest.raises(ValueError, match="different config"):
        store.init_campaign(tmp_path, {**_legacy_config(), "seed": 7})


def test_the_refusal_names_the_differing_field(tmp_path: Path) -> None:
    """It used to print two whole JSON blobs and leave the reader to diff."""
    store.init_campaign(tmp_path, _legacy_config())

    with pytest.raises(ValueError) as excinfo:
        store.init_campaign(
            tmp_path, {**_legacy_config(), "seed": 7, "min_recrossings": 2}
        )
    message = str(excinfo.value)
    assert "seed: stored=42 requested=7" in message
    assert "min_recrossings: stored=1" in message and "requested=2" in message
    assert "initial_steps" not in message   # unchanged fields stay out of it


def test_a_key_the_code_no_longer_writes_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    """Removing a config key is as much a semantic change as adding one; the
    stored value has no counterpart to agree with."""
    store.init_campaign(tmp_path, {**_legacy_config(), "retired_key": 3})

    with pytest.raises(ValueError, match="retired_key"):
        store.init_campaign(tmp_path, _legacy_config())


def test_not_recorded_is_distinguishable_from_none() -> None:
    """`None` is a real, meaningful value for several of these keys, so the
    message must not render an absent key as if it had been stored as null."""
    diff = store.config_differences({}, {"engine": "OpenMMAdapter"})

    assert set(diff) == {"engine"}
    stored, _ = diff["engine"]
    assert stored is not None
    assert "<not recorded>" in store._describe(stored)


def test_the_stored_config_is_never_rewritten(tmp_path: Path) -> None:
    """Read-time normalization only — a campaign's record should say what was
    actually recorded, not what a later version of the code would have written."""
    store.init_campaign(tmp_path, _legacy_config())
    store.init_campaign(tmp_path, {**_legacy_config(), "min_recrossings": 1})

    assert store.get_campaign_config(tmp_path) == _legacy_config()


def test_an_inferred_value_is_marked_as_inferred(tmp_path: Path) -> None:
    """Reporting a filled legacy default bare would claim a value the campaign
    never stored. The reader needs to know which side of that line it is on."""
    store.init_campaign(tmp_path, _legacy_config())

    with pytest.raises(ValueError) as excinfo:
        store.init_campaign(
            tmp_path, {**_legacy_config(), "seed": 7, "min_recrossings": 2}
        )
    message = str(excinfo.value)
    assert "min_recrossings: stored=1 (not recorded;" in message
    assert "seed: stored=42 requested=7" in message   # genuinely recorded, unmarked
