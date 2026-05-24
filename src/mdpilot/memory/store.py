"""SQLite-backed campaign state for resume-from-disk.

One SQLite file per campaign at ``<work_dir>/state.db``. Three tables:

- ``campaign`` — singleton row (id = 1) holding the immutable run config.
  Re-opening with a different config raises so a half-finished campaign
  can't be silently restarted under new parameters.
- ``rounds`` — one row per completed round (PK = round_index). Existence
  of a row is the source of truth for "this round is done". The matching
  OpenMM checkpoint must be written *before* the row is inserted so we
  never see a row without a usable resume point.
- ``ledger`` — the hypothesis ledger. Append-only structured notes the
  scientist writes during decide() to track persistent observations
  across rounds (e.g. "vanilla MD will not fold chignolin in 100 ns",
  "phi torsion is the slow coordinate"). Multiple notes per round
  allowed; the next decide call is shown all prior notes as context.

Trajectories and checkpoints stay on the filesystem; this DB only stores
their paths plus the compact diagnostic report, decision, and ledger.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_DB_FILENAME = "state.db"

_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class RoundRow:
    round_index: int
    n_steps: int
    dcd_path: Path
    checkpoint_path: Path | None
    report: dict[str, Any]
    decision: str
    reason: str
    extra_ns: float | None


@dataclass(frozen=True)
class LedgerNote:
    round_index: int
    text: str


def db_path(work_dir: Path) -> Path:
    return Path(work_dir) / _DB_FILENAME


@contextmanager
def _connect(work_dir: Path) -> Iterator[sqlite3.Connection]:
    path = db_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


def init_campaign(work_dir: Path, config: dict[str, Any]) -> None:
    """Create the DB + campaign row if absent; verify config matches if present.

    Idempotent. Raises ``ValueError`` if a campaign already exists with a
    different config — resuming under different parameters would silently
    invalidate prior rounds.
    """
    work_dir = Path(work_dir)
    config_json = json.dumps(config, sort_keys=True)
    with _connect(work_dir) as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute("SELECT config_json FROM campaign WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO campaign (id, config_json, created_at) VALUES (1, ?, ?)",
                (config_json, _now_iso()),
            )
            return
        existing = row[0]
        if existing != config_json:
            raise ValueError(
                f"campaign at {work_dir} was created with a different config; "
                f"refusing to resume.\n  existing: {existing}\n  requested: {config_json}"
            )


def get_campaign_config(work_dir: Path) -> dict[str, Any] | None:
    """Return the stored campaign config, or None if no campaign exists yet."""
    if not db_path(work_dir).exists():
        return None
    with _connect(work_dir) as conn:
        row = conn.execute("SELECT config_json FROM campaign WHERE id = 1").fetchone()
    return json.loads(row[0]) if row else None


def append_round(
    work_dir: Path,
    *,
    round_index: int,
    n_steps: int,
    dcd_path: Path,
    checkpoint_path: Path | None,
    report: dict[str, Any],
    decision: str,
    reason: str,
    extra_ns: float | None,
) -> None:
    """Insert one completed round. Round_index must be unique within a campaign."""
    with _connect(work_dir) as conn:
        conn.execute(
            """
            INSERT INTO rounds (
                round_index, n_steps, dcd_path, checkpoint_path,
                report_json, decision, reason, extra_ns, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_index,
                n_steps,
                str(dcd_path),
                str(checkpoint_path) if checkpoint_path else None,
                json.dumps(report, sort_keys=True, default=_json_default),
                decision,
                reason,
                extra_ns,
                _now_iso(),
            ),
        )


def list_rounds(work_dir: Path) -> list[RoundRow]:
    """Return all completed rounds for the campaign at work_dir, ordered."""
    if not db_path(work_dir).exists():
        return []
    with _connect(work_dir) as conn:
        cursor = conn.execute(
            """
            SELECT round_index, n_steps, dcd_path, checkpoint_path,
                   report_json, decision, reason, extra_ns
            FROM rounds ORDER BY round_index ASC
            """
        )
        return [
            RoundRow(
                round_index=r[0],
                n_steps=r[1],
                dcd_path=Path(r[2]),
                checkpoint_path=Path(r[3]) if r[3] else None,
                report=json.loads(r[4]),
                decision=r[5],
                reason=r[6],
                extra_ns=r[7],
            )
            for r in cursor.fetchall()
        ]


def get_last_round(work_dir: Path) -> RoundRow | None:
    """Return the highest-indexed completed round, or None if none exist."""
    rounds = list_rounds(work_dir)
    return rounds[-1] if rounds else None


def append_ledger_note(
    work_dir: Path,
    *,
    round_index: int,
    text: str,
) -> None:
    """Add one hypothesis-ledger note. Multiple notes per round are allowed."""
    with _connect(work_dir) as conn:
        conn.execute(
            "INSERT INTO ledger (round_index, text, created_at) VALUES (?, ?, ?)",
            (round_index, text, _now_iso()),
        )


def list_ledger_notes(work_dir: Path) -> list[LedgerNote]:
    """Return all ledger notes ordered by insertion order (id ASC)."""
    if not db_path(work_dir).exists():
        return []
    with _connect(work_dir) as conn:
        cursor = conn.execute(
            "SELECT round_index, text FROM ledger ORDER BY id ASC"
        )
        return [LedgerNote(round_index=r[0], text=r[1]) for r in cursor.fetchall()]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
