"""In-place metaD pivot orchestration (M4 step 4), with fake adapters.

The pivot's *physics* (bias sizing, PLUMED rendering, biased MD) is covered by
bias_designer / plumed_writer / cv_designer unit tests and, end-to-end, by the
PLUMED-enabled live test. Here we pin the *orchestration*: on switch_to_metad
the loop builds a biased adapter, writes plumed.dat, marks subsequent rounds
with plumed_dat_path, and resumes correctly into the metaD phase — none of which
needs OpenMM or a PLUMED runtime. Collaborators that touch disk/LLM
(`make_report`, `decide`, `_build_plumed_input`) are stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.memory import store
from mdpilot.orchestrator import loop as loop_mod
from mdpilot.orchestrator.loop import run_campaign
from mdpilot.orchestrator.scientist import Decision, MetadProposal


class _FakeAdapter:
    """Minimal MDAdapter that writes marker files and records calls."""

    def __init__(self, work_dir: Path, *, spec: SystemSpec, plumed_input: str | None = None):
        self._work_dir = Path(work_dir)
        self._spec = spec
        self.plumed_input = plumed_input
        self.run_calls: list[int] = []
        self.loaded: list[Path] = []
        self.started = False
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def trajectory_extension(self) -> str:
        return ".dcd"

    @property
    def topology_path(self) -> Path:
        return self._topology_path

    def prepare(self) -> None:
        pass

    def start(self) -> None:
        self.started = True
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        self._topology_path.write_text("PDB")

    def run_steps(self, n_steps, *, trajectory_path=None, report_interval_steps=500):
        self.run_calls.append(n_steps)
        if trajectory_path is not None:
            Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
            Path(trajectory_path).write_text("DCD")
        return trajectory_path

    def save_checkpoint(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("CHK")
        return Path(path)

    def load_checkpoint(self, path: Path) -> None:
        self.loaded.append(Path(path))


_REPORT = {
    "trajectory_length_ns": 0.01,
    "ess": 5.0,
    "plateau_reached": True,
    "exploring": False,
    "n_basins": 1,
}


def _stub_collaborators(monkeypatch, decisions: list[Decision]) -> list[str]:
    """Patch decide/make_report/_build_plumed_input; return list plumed calls append to."""
    queue = list(decisions)
    monkeypatch.setattr(loop_mod, "make_report", lambda *a, **k: dict(_REPORT))
    monkeypatch.setattr(loop_mod, "decide", lambda *a, **k: queue.pop(0))
    built: list[str] = []

    def fake_build(proposal, traj, top, output_dir):  # noqa: ANN001
        text = "PLUMED-TEXT\n"
        built.append(text)
        return text

    monkeypatch.setattr(loop_mod, "_build_plumed_input", fake_build)
    return built


def _proposal() -> MetadProposal:
    return MetadProposal(
        cv_type="gyration", selections=("backbone",), label="rg_back"
    )


def _switch() -> Decision:
    return Decision(
        decision="switch_to_metad",
        reason="pinned single basin; task needs a transition the budget can't reach",
        extra_ns=None,
        metad_proposal=_proposal(),
    )


def _stop() -> Decision:
    return Decision(decision="stop", reason="biased run has sampled the transition", extra_ns=None)


def _full_config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "initial_steps": 100,
        "report_interval_steps": 50,
        "equilibration_steps": 0,
        "system_spec": SystemSpec.trpcage().to_dict(),
    }


def _run_kwargs() -> dict:
    return dict(initial_steps=100, seed=42, report_interval_steps=50, equilibration_steps=0)


def test_in_call_pivot_runs_biased_phase(tmp_path: Path, monkeypatch) -> None:
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built_biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built_biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"
    assert [r.decision.decision for r in result.rounds] == ["switch_to_metad", "stop"]
    # The switch round is vanilla; the metaD round is marked with the bias path.
    assert result.rounds[0].plumed_dat_path is None
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"

    # plumed.dat was written with exactly the rendered text, once.
    assert (tmp_path / "plumed.dat").read_text() == "PLUMED-TEXT\n"
    assert len(built_biased) == 1
    assert built_biased[0].plumed_input == "PLUMED-TEXT\n"
    assert built_biased[0].started
    # The biased adapter ran the metaD round (initial_steps), not the base one.
    assert built_biased[0].run_calls == [100]
    assert base.run_calls == [100]  # only the vanilla switch round

    # Persistence agrees: round 2 carries the plumed_dat_path.
    rows = store.list_rounds(tmp_path)
    assert rows[0].plumed_dat_path is None
    assert rows[1].plumed_dat_path == tmp_path / "plumed.dat"


def test_resume_after_switch_pivots_into_metad_phase(tmp_path: Path, monkeypatch) -> None:
    """A campaign whose last persisted round said switch_to_metad now resumes by
    building the biased adapter from the stored proposal + trajectory and running
    the metaD phase — it is no longer terminal."""
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="switch_to_metad",
        reason="pinned + task wants a transition",
        extra_ns=None,
        metad_proposal=_proposal().to_dict(),
    )

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built_biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built_biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"
    # Round 1 (the stored switch) + round 2 (first metaD round).
    assert [r.index for r in result.rounds] == [1, 2]
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"
    assert len(built_biased) == 1
    # The metaD phase starts fresh from cache — no vanilla checkpoint reload.
    assert base.loaded == []
    assert built_biased[0].loaded == []
    assert built_biased[0].run_calls == [100]


def test_second_switch_in_metad_phase_terminates(tmp_path: Path, monkeypatch) -> None:
    """Once biased, a further switch_to_metad ends the campaign for human review
    rather than rebuilding the bias in a loop."""
    _stub_collaborators(monkeypatch, [_switch(), _switch()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "switch_to_metad_requested"
    assert [r.decision.decision for r in result.rounds] == [
        "switch_to_metad",
        "switch_to_metad",
    ]
    # The second (metaD-phase) switch round is marked biased.
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"
