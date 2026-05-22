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
