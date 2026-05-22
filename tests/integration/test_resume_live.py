"""End-to-end resume: stop after round 1, restart, round 1 must not re-run.

Uses real OpenMM + real Claude calls. Skipped without ANTHROPIC_API_KEY.
Tiny step counts keep the two-round wall time near the single-round
budget of test_loop_live.py.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mdpilot.memory import store
from mdpilot.orchestrator.loop import run_campaign

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set in env",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_second_invocation_resumes_without_redoing_round_1(tmp_path: Path) -> None:
    first = run_campaign(
        work_dir=tmp_path,
        initial_steps=5_000,   # 10 ps at 2 fs
        max_rounds=1,
        seed=42,
    )
    assert first.stop_reason == "max_rounds_reached"
    assert len(first.rounds) == 1

    round1_dcd = first.rounds[0].dcd_path
    hash_before = _sha(round1_dcd)
    mtime_before = round1_dcd.stat().st_mtime_ns

    rows = store.list_rounds(tmp_path)
    assert len(rows) == 1
    assert rows[0].checkpoint_path is not None and rows[0].checkpoint_path.exists()

    second = run_campaign(
        work_dir=tmp_path,
        initial_steps=5_000,
        max_rounds=2,
        seed=42,
    )

    # Round 1 file untouched — proves resume, not re-run.
    assert _sha(round1_dcd) == hash_before, "round 1 DCD was rewritten on resume"
    assert round1_dcd.stat().st_mtime_ns == mtime_before

    # Round 2 actually happened and was persisted.
    rows = store.list_rounds(tmp_path)
    assert [r.round_index for r in rows] == [1, 2]
    round2_dcd = second.rounds[1].dcd_path
    assert round2_dcd.exists() and round2_dcd != round1_dcd
    assert [r.index for r in second.rounds] == [1, 2]
