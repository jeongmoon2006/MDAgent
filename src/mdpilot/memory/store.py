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

# --- campaign config compatibility -----------------------------------------
#
# The campaign config is the resume guard: `init_campaign` refuses to reopen a
# campaign whose parameters have changed, because prior rounds were produced
# under the old ones. Comparing the serialized JSON byte-for-byte made that
# guard brittle in one specific way — *adding* a config key stranded every
# campaign already on disk, since their stored config could not contain a key
# that did not exist when they started. Four real campaigns were stranded that
# way before this existed, by the commits that added `state_thresholds` and
# `min_recrossings`.
#
# The fix is to compare *meaning* rather than bytes: fill in, for each key that
# postdates the campaign table, the value that was in force before that key
# existed. A campaign that predates a key then resumes when the requested value
# matches what it actually ran under, and is still refused when it does not.
#
# This is read-time normalization only. The stored config is never rewritten —
# a campaign's record should say what was actually recorded, not what a later
# version of the code would have written.

# Written since the campaign table existed. Their absence from a stored config
# is not something that can be interpreted, so it is left to fail loudly.
_ORIGINAL_CONFIG_KEYS = frozenset({
    "seed",
    "initial_steps",
    "report_interval_steps",
    "equilibration_steps",
    "system_spec",
    "engine",
})

# Added later. Each maps to the behaviour in force *before* the key existed —
# not to the parameter's current default, which is a different thing and would
# be wrong the moment a default changes.
#
# `system_spec` is a nested dict and gets its own table below.
_LEGACY_CONFIG_DEFAULTS: dict[str, Any] = {
    # Before this, a campaign could not pivot; no expectation was recorded.
    "task_expectation": None,
    # Before this, the biased CV was unbounded above (F6).
    "cv_upper_wall_nm": None,
    # Before this, recrossings were counted between the two deepest basins of
    # the current surface rather than against task states (F9).
    "state_thresholds": None,
    # Before this, one crossing was enough to satisfy `fes_converged`.
    "min_recrossings": 1,
    # Before these, the bias took `bias_designer`'s own defaults; None still
    # means exactly that, so absent and unset agree.
    "bias_pace": None,
    "bias_factor": None,
    # Before this, the campaign observable was hardcoded to CA-RMSD in
    # Angstrom. `run_campaign` omits the key when the observable *is* that
    # default, so absent and unset agree here too.
    "observable": None,
}

# Nested under `system_spec`, same rule one level down. These are *historical*
# values, so they are literals rather than references to the current defaults:
# what a campaign was actually built with cannot change later, and pointing at
# a default that has since moved is precisely the bug this prevents.
#
# `padding_nm` is why this table exists. `SystemSpec.to_dict` omits a default
# `ensemble` because its default equals what pre-ensemble campaigns ran under,
# so absent and default agree. Padding's default was *raised* from 1.0 to 1.5
# (F11), so absent and default disagree — a pre-F11 campaign was built at 1.0,
# and resuming it at the new default must be refused, not waved through.
_LEGACY_SYSTEM_SPEC_DEFAULTS: dict[str, Any] = {
    "padding_nm": 1.0,
}

_MISSING = object()

_LEGACY_NOTE = " (not recorded; the behaviour before this key existed)"


def _with_legacy_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Config as it would read if every later-added key had been recorded."""
    filled = {**_LEGACY_CONFIG_DEFAULTS, **config}
    spec = filled.get("system_spec")
    if isinstance(spec, dict):
        filled["system_spec"] = {**_LEGACY_SYSTEM_SPEC_DEFAULTS, **spec}
    return filled


def config_differences(
    stored: dict[str, Any], requested: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Field-level disagreements between two configs, legacy defaults applied.

    Empty means the two describe the same campaign. Returned per field rather
    than as a boolean so the refusal can name what actually changed instead of
    printing two JSON blobs and leaving the reader to diff them.
    """
    a, b = _with_legacy_defaults(stored), _with_legacy_defaults(requested)
    return {
        key: (a.get(key, _MISSING), b.get(key, _MISSING))
        for key in sorted(set(a) | set(b))
        if a.get(key, _MISSING) != b.get(key, _MISSING)
    }


def _describe(value: Any) -> str:
    return "<not recorded>" if value is _MISSING else repr(value)


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
    decision TEXT NOT NULL CHECK (decision IN ('extend', 'stop', 'switch_to_metad', 'switch_cv')),
    reason TEXT NOT NULL,
    extra_ns REAL,
    metad_proposal_json TEXT,
    plumed_dat_path TEXT,
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
    metad_proposal: dict[str, Any] | None = None
    plumed_dat_path: Path | None = None


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
    invalidate prior rounds. The comparison is by meaning, not by bytes: keys
    that postdate the stored campaign are filled with the behaviour that was in
    force before they existed, so adding a config key does not strand campaigns
    already on disk. See `_LEGACY_CONFIG_DEFAULTS`.
    """
    work_dir = Path(work_dir)
    config_json = json.dumps(config, sort_keys=True)
    with _connect(work_dir) as conn:
        conn.executescript(_SCHEMA)
        _migrate_rounds_for_metad(conn)
        _migrate_rounds_for_plumed(conn)
        _migrate_rounds_for_cv_switch(conn)
        row = conn.execute("SELECT config_json FROM campaign WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO campaign (id, config_json, created_at) VALUES (1, ?, ?)",
                (config_json, _now_iso()),
            )
            return
        recorded = json.loads(row[0])
        differences = config_differences(recorded, config)
        if differences:
            # A key filled from `_LEGACY_CONFIG_DEFAULTS` is reported with what
            # the campaign effectively ran under, marked as inferred — printing
            # it bare would claim a value the campaign never actually stored.
            detail = "\n".join(
                f"  {key}: stored={_describe(was)}"
                + ("" if key in recorded else _LEGACY_NOTE)
                + f" requested={_describe(now)}"
                for key, (was, now) in differences.items()
            )
            raise ValueError(
                f"campaign at {work_dir} was created with a different config; "
                f"refusing to resume. Differing field(s):\n{detail}"
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
    metad_proposal: dict[str, Any] | None = None,
    plumed_dat_path: Path | None = None,
) -> None:
    """Insert one completed round. Round_index must be unique within a campaign.

    ``metad_proposal`` is the structured CV proposal when ``decision`` is
    ``switch_to_metad`` (and None otherwise); persisted as a JSON blob so the
    loop can reconstruct it on resume.

    ``plumed_dat_path`` marks a round run under a metadynamics bias: NULL for a
    vanilla round, the path to the plumed.dat that drove it otherwise. It is
    both the phase marker and the audit artifact — reading it back recovers the
    exact bias config for that round.
    """
    with _connect(work_dir) as conn:
        conn.execute(
            """
            INSERT INTO rounds (
                round_index, n_steps, dcd_path, checkpoint_path,
                report_json, decision, reason, extra_ns,
                metad_proposal_json, plumed_dat_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(metad_proposal, sort_keys=True) if metad_proposal else None,
                str(plumed_dat_path) if plumed_dat_path else None,
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
                   report_json, decision, reason, extra_ns, metad_proposal_json,
                   plumed_dat_path
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
                metad_proposal=json.loads(r[8]) if r[8] else None,
                plumed_dat_path=Path(r[9]) if r[9] else None,
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


def _migrate_rounds_for_metad(conn: sqlite3.Connection) -> None:
    """Bring pre-M4 ``rounds`` tables up to the metaD-aware schema.

    Two things change: the ``decision`` CHECK constraint widens to include
    ``switch_to_metad``, and a nullable ``metad_proposal_json`` column appears.
    SQLite cannot ``ALTER`` a CHECK constraint in place, so we rename-create-
    copy-drop. Idempotent: a no-op when the column already exists.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(rounds)").fetchall()}
    if not cols or "metad_proposal_json" in cols:
        return
    conn.executescript(
        """
        BEGIN;
        ALTER TABLE rounds RENAME TO rounds_pre_m4;
        CREATE TABLE rounds (
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
        INSERT INTO rounds (
            round_index, n_steps, dcd_path, checkpoint_path,
            report_json, decision, reason, extra_ns,
            metad_proposal_json, created_at
        )
        SELECT round_index, n_steps, dcd_path, checkpoint_path,
               report_json, decision, reason, extra_ns,
               NULL, created_at
        FROM rounds_pre_m4;
        DROP TABLE rounds_pre_m4;
        COMMIT;
        """
    )


def _migrate_rounds_for_plumed(conn: sqlite3.Connection) -> None:
    """Add the nullable ``plumed_dat_path`` column to pre-pivot ``rounds`` tables.

    Unlike the metaD migration this changes no CHECK constraint, so a plain
    ``ALTER TABLE ... ADD COLUMN`` suffices — no rename-create-copy-drop.
    Idempotent: a no-op once the column exists. Runs after
    ``_migrate_rounds_for_metad`` so it also covers the table that migration
    freshly recreated (which lacks the column).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(rounds)").fetchall()}
    if not cols or "plumed_dat_path" in cols:
        return
    conn.execute("ALTER TABLE rounds ADD COLUMN plumed_dat_path TEXT")


def _migrate_rounds_for_cv_switch(conn: sqlite3.Connection) -> None:
    """Widen the ``decision`` CHECK constraint to admit ``switch_cv``.

    A CHECK change again, so rename-create-copy-drop as in the metaD
    migration. Unlike the other two this adds no column, so idempotency is
    detected from the stored DDL rather than from ``PRAGMA table_info``. Runs
    last, so it also covers a table either earlier migration just recreated.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rounds'"
    ).fetchone()
    if row is None or "switch_cv" in (row[0] or ""):
        return
    conn.executescript(
        """
        BEGIN;
        ALTER TABLE rounds RENAME TO rounds_pre_cv_switch;
        CREATE TABLE rounds (
            round_index INTEGER PRIMARY KEY,
            n_steps INTEGER NOT NULL,
            dcd_path TEXT NOT NULL,
            checkpoint_path TEXT,
            report_json TEXT NOT NULL,
            decision TEXT NOT NULL CHECK (
                decision IN ('extend', 'stop', 'switch_to_metad', 'switch_cv')
            ),
            reason TEXT NOT NULL,
            extra_ns REAL,
            metad_proposal_json TEXT,
            plumed_dat_path TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO rounds (
            round_index, n_steps, dcd_path, checkpoint_path,
            report_json, decision, reason, extra_ns,
            metad_proposal_json, plumed_dat_path, created_at
        )
        SELECT round_index, n_steps, dcd_path, checkpoint_path,
               report_json, decision, reason, extra_ns,
               metad_proposal_json, plumed_dat_path, created_at
        FROM rounds_pre_cv_switch;
        DROP TABLE rounds_pre_cv_switch;
        COMMIT;
        """
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
