"""M3 done-criterion: the same campaign loop runs through GROMACS without
any changes to scientist or loop code — only the adapter differs.

OpenMM-through-loop is already covered by `test_loop_live.py` and
`test_milestone1_live.py` from M1. This test exercises the
not-yet-tested half: GROMACS adapter wired through `run_campaign`,
producing a well-formed diagnostic report and a real scientist decision.

Requires both `gmx` on PATH and `ANTHROPIC_API_KEY` set.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from mdpilot.adapters.gromacs_adapter import GROMACSAdapter
from mdpilot.orchestrator.loop import run_campaign

pytestmark = pytest.mark.skipif(
    shutil.which("gmx") is None or not os.getenv("ANTHROPIC_API_KEY"),
    reason="gmx not on PATH or ANTHROPIC_API_KEY unset",
)


def test_single_round_through_gromacs_adapter(tmp_path: Path) -> None:
    """One short MD round → diagnostics → scientist decision, end-to-end.

    Tiny step count (500 steps = 1 ps) keeps wall time bounded; the
    trajectory is therefore far from converged and the scientist must
    say "extend".
    """
    adapter = GROMACSAdapter(work_dir=tmp_path, seed=42)
    result = run_campaign(
        work_dir=tmp_path,
        adapter=adapter,
        initial_steps=500,
        max_rounds=1,
        seed=42,
        report_interval_steps=50,   # 10 frames per 500-step round
    )

    assert len(result.rounds) == 1
    round_0 = result.rounds[0]

    # Engine-specific trajectory extension flowed correctly through the loop
    assert round_0.dcd_path.exists()
    assert round_0.dcd_path.suffix == ".xtc"

    # Diagnostic report is well-formed regardless of which engine produced
    # the trajectory — the whole point of the MDAdapter abstraction.
    report = round_0.report
    assert report["observable_name"] == "rmsd_ca_to_first_angstrom"
    assert report["n_frames"] > 0
    assert "ess" in report
    assert "plateau_reached" in report

    # 1 ps from a freshly minimized + Langevin-warmed state is far from
    # converged on any reasonable rubric → scientist should extend.
    assert round_0.decision.decision == "extend", round_0.decision
    assert round_0.decision.extra_ns is not None and round_0.decision.extra_ns > 0
