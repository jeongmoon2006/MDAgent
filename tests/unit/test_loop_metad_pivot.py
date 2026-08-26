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


# A real (if tiny) topology, not a marker file. `run_campaign`'s pre-flight
# loads this and computes the campaign observable on it before any dynamics,
# so a fake that writes "PDB" would exercise a path no real adapter takes.
# Four alanines in a line: enough CA atoms for an rmsd observable to resolve.
_MINIMAL_PDB = "".join(
    f"ATOM  {i * 2 + 1:>5d}  N   ALA A{i + 1:>4d}    "
    f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           N\n"
    f"ATOM  {i * 2 + 2:>5d}  CA  ALA A{i + 1:>4d}    "
    f"{i * 3.8 + 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    for i in range(4)
) + "END\n"


class _FakeAdapter:
    """Minimal MDAdapter that writes marker files and records calls."""

    def __init__(
        self,
        work_dir: Path,
        *,
        spec: SystemSpec,
        plumed_input: str | None = None,
        timestep_fs: float = 2.0,
        temperature_k: float = 300.0,
    ):
        self._work_dir = Path(work_dir)
        self._spec = spec
        self.plumed_input = plumed_input
        self._timestep_fs = timestep_fs
        self._temperature_k = temperature_k
        self.run_calls: list[int] = []
        self.loaded: list[Path] = []
        self.started = False
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def timestep_fs(self) -> float:
        return self._timestep_fs

    @property
    def temperature_k(self) -> float:
        return self._temperature_k

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
        self._topology_path.write_text(_MINIMAL_PDB)

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

# What `diagnostics.free_energy.metad_report` returns for a biased round. Note
# what is *not* here: no ess, no plateau_reached, no exploring. That omission is
# the contract under test, not an abbreviation of the fixture.
_METAD_REPORT = {
    "fes_drift_kj_per_mol": 0.9,
    "recrossings": 3,
    "recrossing_low": -0.4,
    "recrossing_high": 0.6,
    "barrier_crossed": True,
    "fes_converged": True,
    "n_basins_fes": 2,
    "barrier_kj_per_mol": 21.4,
}


def _stub_collaborators(monkeypatch, decisions: list[Decision]) -> dict:
    """Patch decide / both report builders / _build_plumed_input.

    Returns a record of what the loop asked for: `plumed_texts` (one entry per
    rendered bias), `phases` (the phase passed to each decide call),
    `allow_cv_switch` (whether CV revision was on the table that round) and
    `cv_labels` (the label of each CV the loop asked to have rendered), so
    tests can assert the loop routed each round to the right contract.
    """
    queue = list(decisions)
    record: dict = {
        "plumed_texts": [], "phases": [], "allow_cv_switch": [], "cv_labels": [],
    }

    monkeypatch.setattr(loop_mod, "make_report", lambda *a, **k: dict(_REPORT))
    monkeypatch.setattr(loop_mod, "metad_report", lambda *a, **k: dict(_METAD_REPORT))

    def fake_decide(report, **kwargs):  # noqa: ANN001
        record["phases"].append(kwargs.get("phase"))
        record["allow_cv_switch"].append(kwargs.get("allow_cv_switch"))
        return queue.pop(0)

    monkeypatch.setattr(loop_mod, "decide", fake_decide)

    def fake_build(proposal, traj, top, output_dir, **kwargs):  # noqa: ANN001
        text = "PLUMED-TEXT\n"
        record["plumed_texts"].append(text)
        record["cv_labels"].append(proposal.label)
        return text

    monkeypatch.setattr(loop_mod, "_build_plumed_input", fake_build)
    return record


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


def _extend() -> Decision:
    return Decision(decision="extend", reason="surface still moving", extra_ns=0.5)


def _stop() -> Decision:
    return Decision(decision="stop", reason="biased run has sampled the transition", extra_ns=None)


def _full_config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "initial_steps": 100,
        "report_interval_steps": 50,
        "equilibration_steps": 0,
        "system_spec": SystemSpec.trpcage().to_dict(),
        "engine": "_FakeAdapter",
        "task_expectation": None,
        "cv_upper_wall_nm": None,
        "state_thresholds": None,
        "min_recrossings": 1,
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


def test_resume_after_second_switch_stays_terminal(tmp_path: Path, monkeypatch) -> None:
    """A switch_to_metad recorded on an already-biased round is terminal on
    resume too, not just in-process.

    The stored row carries decision='switch_to_metad' *and* a plumed_dat_path.
    Without the phase check, resume would take the pivot branch and rebuild the
    bias from that biased round's trajectory — sizing SIGMA off a spread the
    bias itself produced, which is exactly the second pivot the live loop
    declined to perform.
    """
    store.init_campaign(tmp_path, _full_config(tmp_path))
    plumed_dat = tmp_path / "plumed.dat"
    plumed_dat.write_text("PLUMED-TEXT\n")
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="switch_to_metad",
        reason="second switch, already biased",
        extra_ns=None,
        metad_proposal=_proposal().to_dict(),
        plumed_dat_path=plumed_dat,
    )

    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    result = run_campaign(
        work_dir=tmp_path, adapter=base, max_rounds=5, **_run_kwargs()
    )

    assert result.stop_reason == "switch_to_metad_requested"
    assert [r.index for r in result.rounds] == [1]
    # Nothing ran: no engine work, no rebuilt bias.
    assert base.run_calls == []
    assert base.started is False


def test_default_biased_factory_refuses_a_non_openmm_engine(
    tmp_path: Path, monkeypatch
) -> None:
    """Without an injected factory, a pivot from a non-OpenMM adapter raises.

    The CV's atom indices were resolved against the vanilla engine's topology.
    Silently building an OpenMM biased adapter would bias whichever atoms sat
    at those indices in a differently-solvated, differently-ordered system.
    """
    _stub_collaborators(monkeypatch, [_switch()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(NotImplementedError, match="_FakeAdapter"):
        run_campaign(work_dir=tmp_path, adapter=base, max_rounds=5, **_run_kwargs())


def test_resumed_extend_round_is_clamped_to_max_extra_ns(
    tmp_path: Path, monkeypatch
) -> None:
    """SQLite stores the model's raw extra_ns, so the clamp has to be re-applied
    on read. Applying it only in the live loop let a resumed campaign run a
    longer round than the uninterrupted one would have.
    """
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="extend",
        reason="far from converged",
        extra_ns=20.0,  # well past any sane max_extra_ns
    )
    (tmp_path / "rounds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rounds/round_001.chk").write_text("CHK")

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        max_rounds=2,
        max_extra_ns=2.0,
        **_run_kwargs(),
    )

    # 2.0 ns at 2 fs = 1_000_000 steps, not the 10_000_000 the row asked for.
    assert base.run_calls == [1_000_000]


def test_biased_round_gets_the_free_energy_report_not_the_equilibrium_one(
    tmp_path: Path, monkeypatch
) -> None:
    """The pivot swaps the diagnostic contract, not just the adapter.

    A biased trajectory is not an equilibrium ensemble, so the vanilla
    convergence fields must be *absent* from a biased round's report rather
    than present-and-ignorable. This pins the omission, the phase label, and
    the action space handed to the scientist in each phase.
    """
    record = _stub_collaborators(monkeypatch, [_switch(), _stop()])
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

    vanilla, biased = result.rounds[0].report, result.rounds[1].report
    assert vanilla["phase"] == "vanilla"
    assert vanilla["ess"] == 5.0

    assert biased["phase"] == "metad"
    assert biased["fes_converged"] is True
    assert biased["recrossings"] == 3
    for equilibrium_field in ("ess", "plateau_reached", "exploring", "n_basins"):
        assert equilibrium_field not in biased, equilibrium_field

    # The scientist was offered the matching action space each round.
    assert record["phases"] == ["vanilla", "metad"]


def test_prior_round_summaries_do_not_leak_equilibrium_fields_from_biased_rounds(
    tmp_path: Path, monkeypatch
) -> None:
    """The compact prior-round view is phase-keyed too — otherwise the fields
    the biased report omits would reappear via the campaign history."""
    _stub_collaborators(monkeypatch, [_switch(), _extend(), _stop()])
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

    summaries = [loop_mod._compact_prior(r) for r in result.rounds]
    vanilla_summary, biased_summary = summaries[0], summaries[1]

    assert vanilla_summary["phase"] == "vanilla"
    assert "ess" in vanilla_summary

    assert biased_summary["phase"] == "metad"
    assert biased_summary["fes_converged"] is True
    assert "ess" not in biased_summary
    assert "plateau_reached" not in biased_summary


def _seed_metad_phase_round(tmp_path: Path, *, round_index: int = 1) -> Path:
    """A campaign already inside the metaD phase: one completed biased round,
    its checkpoint, its plumed.dat, and the bias snapshot paired with it."""
    plumed_dat = tmp_path / "plumed.dat"
    plumed_dat.write_text("PLUMED-TEXT\n")
    rounds = tmp_path / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    (rounds / f"round_{round_index:03d}.chk").write_text("CHK")
    (rounds / f"round_{round_index:03d}.hills").write_text("SNAPSHOT-HILLS\n")
    (rounds / f"round_{round_index:03d}.colvar").write_text("SNAPSHOT-COLVAR\n")

    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=round_index,
        n_steps=100,
        dcd_path=rounds / f"round_{round_index:03d}.dcd",
        checkpoint_path=rounds / f"round_{round_index:03d}.chk",
        report=dict(_METAD_REPORT),
        decision="extend",
        reason="surface still moving",
        extra_ns=0.5,
        plumed_dat_path=plumed_dat,
    )
    return plumed_dat


def test_mid_metad_resume_enables_restart_and_restores_the_bias(
    tmp_path: Path, monkeypatch
) -> None:
    """Resuming inside the biased phase must continue the deposited bias.

    Without RESTART, PLUMED backs HILLS up to bck.0.HILLS and refills from
    zero while the coordinates carry on from a biased configuration — so the
    surface the campaign integrates is the sum of two disjoint fillings. The
    HILLS/COLVAR snapshot is the other half: it puts the bias back to the
    point the restored checkpoint corresponds to.
    """
    _seed_metad_phase_round(tmp_path)
    # A live HILLS left over from a round that crashed before it was recorded.
    (tmp_path / "HILLS").write_text("SNAPSHOT-HILLS\nSTALE-EXTRA-HILL\n")

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    handed: list[str] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        handed.append(plumed_input)
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=2,
        **_run_kwargs(),
    )

    assert len(handed) == 1
    assert handed[0].splitlines()[0].startswith("RESTART")
    assert "PLUMED-TEXT" in handed[0]
    # The stale hill from the unrecorded round is gone; the bias matches the
    # checkpoint it is paired with.
    assert (tmp_path / "HILLS").read_text() == "SNAPSHOT-HILLS\n"
    assert (tmp_path / "COLVAR").read_text() == "SNAPSHOT-COLVAR\n"


def test_fresh_pivot_does_not_enable_restart(tmp_path: Path, monkeypatch) -> None:
    """A pivot has no prior bias for this campaign. RESTART there would read
    back whatever HILLS happened to be lying in the campaign directory."""
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    handed: list[str] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        handed.append(plumed_input)
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert handed == ["PLUMED-TEXT\n"]


def test_biased_round_snapshots_the_bias_beside_its_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """Every biased round leaves a bias snapshot next to its checkpoint, so a
    later resume has a consistent pair to restore."""
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    def factory(plumed_input: str) -> _FakeAdapter:
        # Stand in for PLUMED depositing hills during the biased round.
        (tmp_path / "HILLS").write_text("HILL-1\nHILL-2\n")
        (tmp_path / "COLVAR").write_text("ROW-1\n")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    rounds = tmp_path / "rounds"
    # Round 1 was vanilla — no bias existed yet, so nothing was snapshotted.
    assert not (rounds / "round_001.hills").exists()
    # Round 2 was biased.
    assert (rounds / "round_002.hills").read_text() == "HILL-1\nHILL-2\n"
    assert (rounds / "round_002.colvar").read_text() == "ROW-1\n"


def test_biased_budget_clamps_the_last_round_and_ends_the_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    """`max_biased_ns` is a hard cap on cumulative biased simulation time.

    A budget stated only in `task_expectation` is advisory — the model reads
    it and can still ask for more — which is exactly the wrong property for an
    unattended multi-hour run. The round that would overshoot is shortened to
    land on the budget rather than skipped, so its hills still count.
    """
    _stub_collaborators(monkeypatch, [_switch(), _extend(), _extend()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=10,
        max_biased_ns=0.0003,          # 150 steps at 2 fs
        **_run_kwargs(),               # initial_steps=100
    )

    assert result.stop_reason == "biased_budget_exhausted"
    # 100 steps, then the second biased round clamped from 250_000 to 50.
    assert biased[0].run_calls == [100, 50]
    assert sum(biased[0].run_calls) == 150
    # The vanilla switch round is not charged to the biased budget.
    assert base.run_calls == [100]


def test_biased_budget_survives_a_resume(tmp_path: Path, monkeypatch) -> None:
    """The meter is recomputed from the persisted biased rounds, so restarting
    a campaign cannot silently buy it another full budget."""
    _seed_metad_phase_round(tmp_path)   # one completed biased round, 100 steps

    _stub_collaborators(monkeypatch, [_extend(), _extend()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=10,
        max_biased_ns=0.0003,          # 150 steps; 100 already spent before the kill
        **_run_kwargs(),
    )

    assert result.stop_reason == "biased_budget_exhausted"
    assert biased[0].run_calls == [50]   # only the 50 steps still owed


def test_prior_summaries_carry_the_boundaries_a_recrossing_count_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two basins are re-derived from the current surface each round, so
    the boundaries move. A count carried into the campaign history without them
    is not comparable across rounds — the same defect that phase-keying fixed
    for `ess`, arriving through the history channel instead."""
    _stub_collaborators(
        monkeypatch,
        [
            Decision("switch_to_metad", "pivot", None, metad_proposal=_proposal()),
            Decision("stop", "converged", None),
        ],
    )
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    biased = [s for s in map(loop_mod._compact_prior, result.rounds)
              if s["phase"] == "metad"]
    assert biased
    for summary in biased:
        assert summary["recrossings"] == 3
        assert summary["recrossing_low"] == -0.4
        assert summary["recrossing_high"] == 0.6


# ---------- switch_cv: CV revision inside the biased phase ----------

def _replacement() -> MetadProposal:
    return MetadProposal(
        cv_type="contacts", selections=("name CA",), label="q_native"
    )


def _switch_cv() -> Decision:
    return Decision(
        decision="switch_cv",
        reason="recrossings counted between boundaries both above the task's "
               "extended threshold; the walker never returned to cv_start",
        extra_ns=None,
        metad_proposal=_replacement(),
    )


def _hills(work_dir: Path) -> Path:
    return work_dir / "HILLS"


def test_switch_cv_rebuilds_the_bias_on_the_replacement_cv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core of CV revision: a second biased adapter is built against the
    new proposal, and the campaign continues in the same call rather than
    terminating for a human to restart."""
    _stub_collaborators(monkeypatch, [_switch(), _switch_cv(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    # Two biased adapters: the pivot's, and the replacement's.
    assert len(built) == 2
    assert result.stop_reason == "scientist_said_stop"
    assert [r.decision.decision for r in result.rounds] == [
        "switch_to_metad", "switch_cv", "stop",
    ]
    # The round after the switch is still biased.
    assert result.rounds[2].plumed_dat_path is not None


def test_switch_cv_clears_the_outgoing_hills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hills on the old coordinate must not carry into the new bias — PLUMED
    would read them back as if they described the new CV. The round snapshot
    is what preserves them."""
    _stub_collaborators(monkeypatch, [_switch(), _switch_cv(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    work_dir = tmp_path / "campaign"

    def factory(plumed_input: str) -> _FakeAdapter:
        # Simulate PLUMED depositing as soon as a biased adapter starts.
        _hills(work_dir).parent.mkdir(parents=True, exist_ok=True)
        _hills(work_dir).write_text("old-cv hills")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=work_dir,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    # The replacement factory rewrote HILLS after the clear, so what matters is
    # that the *snapshot* of the pre-switch round survives as the record.
    snapshot = work_dir / "rounds" / "round_002.hills"
    assert snapshot.exists()
    assert snapshot.read_text() == "old-cv hills"


def test_switch_cv_is_withdrawn_once_the_allowance_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the cap the action is dropped from the tool schema rather than
    refused after the fact, so the model never emits a decision the loop will
    not honour — the same reason a second switch_to_metad is unrepresentable."""
    record = _stub_collaborators(
        monkeypatch, [_switch(), _switch_cv(), _extend(), _stop()]
    )
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=6,
        max_cv_switches=1,
        **_run_kwargs(),
    )

    # Round 1 vanilla: not offered. Round 2 biased with one switch left: offered.
    # Rounds 3+ biased with the allowance spent: withdrawn.
    assert record["allow_cv_switch"] == [False, True, False, False]


def test_switch_cv_allowance_is_recomputed_from_disk_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart must not hand the scientist a second allowance, for the same
    reason it must not hand it a second biased budget."""
    work_dir = tmp_path / "campaign"
    factory = lambda p: _FakeAdapter(  # noqa: E731
        tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
    )

    _stub_collaborators(monkeypatch, [_switch(), _switch_cv()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=2,          # stop right after the switch round
        max_cv_switches=1,
        **_run_kwargs(),
    )

    record = _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=4,
        max_cv_switches=1,
        **_run_kwargs(),
    )

    assert record["allow_cv_switch"] == [False]


def test_resumed_switch_cv_round_builds_the_replacement_not_the_rejected_cv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Branch-ordering guard. A switch_cv round is itself biased, so the
    generic `plumed_dat_path is not None` resume branch matches it too — and
    would restart the campaign on the coordinate the scientist just rejected,
    with RESTART reading its hills back."""
    work_dir = tmp_path / "campaign"
    factory = lambda p: _FakeAdapter(  # noqa: E731
        tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
    )

    _stub_collaborators(monkeypatch, [_switch(), _switch_cv()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=2,
        **_run_kwargs(),
    )

    record = _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=4,
        **_run_kwargs(),
    )

    # The bias rebuilt on resume is the replacement CV, not the original.
    assert record["cv_labels"] == [_replacement().label]


# ---------- strictness: a biased phase needs the task's states ----------

def test_a_campaign_that_can_pivot_must_supply_state_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without them the biased phase would fall back to counting between the
    two deepest basins of the current surface, which F9 showed is not
    comparable round to round. `task_expectation` is the sole input gating
    `switch_to_metad`, so it is the predicate for "can reach a biased phase"."""
    _stub_collaborators(monkeypatch, [_stop()])

    with pytest.raises(ValueError, match="state_thresholds is None"):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
            task_expectation="cross the barrier",
            **_run_kwargs(),
        )


def test_the_guard_fires_before_any_simulation_is_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raising at the pivot instead would throw away the whole vanilla phase —
    hours of GPU time on a real campaign."""
    _stub_collaborators(monkeypatch, [_stop()])
    adapter = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(ValueError):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=adapter,
            task_expectation="cross the barrier",
            **_run_kwargs(),
        )

    assert adapter.run_calls == []
    assert not (tmp_path / "campaign" / "rounds").exists()


def test_a_pure_convergence_campaign_needs_no_state_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict, not indiscriminate. With no task_expectation the scientist
    cannot propose a pivot, so no biased phase can occur and there is nothing
    for the thresholds to anchor.

    What makes that premise true is
    `test_a_campaign_with_no_expectation_cannot_express_a_pivot` in
    test_scientist.py: `switch_to_metad` leaves the tool enum entirely. Before
    that it was only asserted here and discouraged in the prompt, so the
    action stayed emittable and a pivot could still reach a biased phase with
    no band to count recrossings against.
    """
    _stub_collaborators(monkeypatch, [_stop()])

    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"


def test_inverted_state_thresholds_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count_recrossings` returns 0 for an inverted band rather than raising,
    so a swapped pair would read as a campaign that never crossed."""
    _stub_collaborators(monkeypatch, [_stop()])

    with pytest.raises(ValueError, match="high > low"):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
            task_expectation="cross the barrier",
            state_thresholds=(4.0, 1.5),
            **_run_kwargs(),
        )


def test_state_thresholds_are_locked_against_a_changed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming with a different band would splice two definitions of "a
    transition" into one campaign's history."""
    work_dir = tmp_path / "campaign"
    kwargs = dict(
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        task_expectation="cross the barrier",
        **_run_kwargs(),
    )
    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(work_dir=work_dir, state_thresholds=(1.5, 4.0), **kwargs)

    _stub_collaborators(monkeypatch, [_stop()])
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=work_dir, state_thresholds=(2.0, 5.0), **kwargs)


def test_min_recrossings_is_locked_against_a_changed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same definition. `state_thresholds` says where the
    states are; `min_recrossings` says how many transitions between them count
    as done — it is the threshold `fes_converged` compares against, and
    `_refuse_premature_stop` reads that verdict to decide whether the scientist
    may stop. Changing it mid-campaign re-judges rounds already decided under
    the old value.
    """
    work_dir = tmp_path / "campaign"
    kwargs = dict(
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        task_expectation="cross the barrier",
        state_thresholds=(1.5, 4.0),
        **_run_kwargs(),
    )
    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(work_dir=work_dir, min_recrossings=2, **kwargs)

    _stub_collaborators(monkeypatch, [_stop()])
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=work_dir, min_recrossings=1, **kwargs)


# ---------- the loop reads its physics constants off the adapter ----------

def test_round_length_follows_the_adapter_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    """`extra_ns` is nanoseconds, and the step count has to be derived from the
    engine's own dt. Against a hardcoded 2 fs, an engine at 4 fs would have run
    every round at twice the requested length with `extra_ns` silently no
    longer meaning nanoseconds anywhere in the campaign record.
    """
    _stub_collaborators(monkeypatch, [_extend(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), timestep_fs=4.0)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        max_rounds=2,
        max_extra_ns=2.0,
        **_run_kwargs(),
    )

    # `_extend()` asks for 0.5 ns. At 4 fs that is 125_000 steps, not the
    # 250_000 a 2 fs assumption would have produced.
    assert base.run_calls == [100, 125_000]


def test_biased_phase_uses_the_adapter_thermostat_temperature(
    tmp_path: Path, monkeypatch
) -> None:
    """PLUMED's `METAD ... TEMP=` must match the thermostat or the
    well-tempered scaling factor is computed against the wrong temperature and
    the bias converges to something other than -(1 - 1/gamma)F(s) — with no
    error anywhere. The same temperature sets the kT the free-energy
    convergence threshold is taken against.
    """
    record = _stub_collaborators(monkeypatch, [_switch(), _stop()])
    seen: dict = {}

    def spy_build(proposal, traj, top, output_dir, **kwargs):  # noqa: ANN001
        seen["build"] = kwargs["temperature_k"]
        record["cv_labels"].append(proposal.label)
        return "PLUMED-TEXT\n"

    def spy_metad_report(*_args, **kwargs):
        seen["report"] = kwargs["temperature_k"]
        return dict(_METAD_REPORT)

    monkeypatch.setattr(loop_mod, "_build_plumed_input", spy_build)
    monkeypatch.setattr(loop_mod, "metad_report", spy_metad_report)

    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), temperature_k=277.0)
    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda text: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=text, temperature_k=277.0
        ),
        max_rounds=3,
        **_run_kwargs(),
    )

    assert seen == {"build": 277.0, "report": 277.0}


# ---------- bias shape overrides ----------
#
# PACE and BIASFACTOR were already keyword arguments on `design_bias`; the loop
# simply never passed them, so no campaign could change the shape of its own
# bias. These pin the threading and the resume lock that has to come with it.

def _two_atom_traj(tmp_path: Path) -> tuple[Path, Path]:
    import mdtraj as md
    import numpy as np

    top = md.Topology()
    res = top.add_residue("ALA", top.add_chain(), resSeq=1)
    top.add_atom("A", md.element.carbon, res)
    top.add_atom("B", md.element.carbon, res)
    xyz = np.zeros((20, 2, 3))
    xyz[:, 1, 0] = np.linspace(0.9, 1.4, 20)   # a real, non-degenerate spread
    traj = md.Trajectory(xyz=xyz.astype(np.float32), topology=top)
    pdb, dcd = tmp_path / "top.pdb", tmp_path / "traj.dcd"
    traj[0].save_pdb(str(pdb))
    traj.save_dcd(str(dcd))
    return dcd, pdb


def _render(tmp_path: Path, **overrides) -> str:
    from mdpilot.orchestrator.loop import _build_plumed_input

    dcd, pdb = _two_atom_traj(tmp_path)
    return _build_plumed_input(
        MetadProposal(cv_type="distance", selections=("name A", "name B"), label="d"),
        dcd,
        pdb,
        tmp_path.resolve(),
        temperature_k=300.0,
        **overrides,
    )


def test_bias_overrides_reach_the_rendered_plumed_dat(tmp_path: Path) -> None:
    rendered = _render(tmp_path, bias_pace=200, bias_factor=15.0)

    assert "PACE=200" in rendered
    assert "BIASFACTOR=15" in rendered


def test_unset_bias_overrides_leave_the_designer_defaults(tmp_path: Path) -> None:
    """None means "let bias_designer decide" — the loop must not restate its
    defaults, or the two drift apart silently."""
    from mdpilot.sampling.bias_designer import _DEFAULT_BIAS_FACTOR, _DEFAULT_PACE

    rendered = _render(tmp_path)

    assert f"PACE={_DEFAULT_PACE}" in rendered
    assert f"BIASFACTOR={_DEFAULT_BIAS_FACTOR:g}" in rendered


def test_bias_shape_locks_into_the_campaign_config_only_when_set(
    tmp_path: Path, monkeypatch
) -> None:
    """Biased-phase physics, so it has to lock — but adding the keys
    unconditionally would break resume for every campaign predating them."""
    from mdpilot.memory import store

    for kwargs, expected in (
        ({}, False),
        ({"bias_factor": 15.0}, True),
    ):
        work = tmp_path / f"c{int(expected)}"
        _stub_collaborators(monkeypatch, [_stop()])
        adapter = _FakeAdapter(work, spec=SystemSpec.trpcage())
        run_campaign(
            work_dir=work, adapter=adapter, max_rounds=1, **_run_kwargs(), **kwargs
        )
        config = store.get_campaign_config(work)
        assert ("bias_factor" in config) is expected, kwargs


def test_every_config_key_is_covered_by_the_compatibility_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Forget-proofing for the resume guard.

    Adding a key to the campaign config without a compatibility entry silently
    strands every campaign already on disk — the bug `_LEGACY_CONFIG_DEFAULTS`
    exists to prevent, and one that had already cost four real campaigns before
    it existed. If this fails, add the new key to
    `store._LEGACY_CONFIG_DEFAULTS` with the behaviour that was in force
    *before* the key existed, which is not necessarily its current default.
    """
    from mdpilot.memory import store
    from mdpilot.memory.store import _LEGACY_CONFIG_DEFAULTS, _ORIGINAL_CONFIG_KEYS
    from mdpilot.observables import ObservableSpec

    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        max_rounds=1,
        # Every optional parameter set, so the recorded config is the widest
        # one run_campaign can produce.
        task_expectation="fold it",
        state_thresholds=(1.0, 2.0),
        min_recrossings=2,
        cv_upper_wall_nm=0.8,
        bias_pace=200,
        bias_factor=15.0,
        observable=ObservableSpec(
            cv_type="gyration", selections=("name CA",), name="rg_nm"
        ),
        **_run_kwargs(),
    )

    recorded = set(store.get_campaign_config(tmp_path))
    uncovered = recorded - _ORIGINAL_CONFIG_KEYS - set(_LEGACY_CONFIG_DEFAULTS)
    assert not uncovered, (
        f"config key(s) {sorted(uncovered)} have no compatibility entry in "
        f"store._LEGACY_CONFIG_DEFAULTS; campaigns already on disk would be "
        f"stranded by them"
    )


def test_the_real_task_file_drives_the_loop_and_reopens_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """The seam between `task_file` and `run_campaign`, which nothing else
    covers: the two were built separately and only met in a benchmark script.

    Also the round trip that matters operationally — the same task file must
    reopen its own campaign, which is only true if the rendered
    `task_expectation` is deterministic.
    """
    from mdpilot.memory import store
    from mdpilot.task_file import load_task_file

    task = load_task_file(Path("benchmarks/tasks/cln025_folding.yaml"))
    kwargs = task.run_kwargs(
        max_extra_ns=2.0, seed=42, initial_steps=100,
        report_interval_steps=50, equilibration_steps=0, max_rounds=1,
    )

    _stub_collaborators(monkeypatch, [_stop()])
    result = run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=task.spec),
        **kwargs,
    )
    assert result.stop_reason == "scientist_said_stop"

    config = store.get_campaign_config(tmp_path)
    assert config["state_thresholds"] == [1.5, 4.0]
    assert config["min_recrossings"] == 2
    assert config["task_expectation"] == kwargs["task_expectation"]

    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=task.spec),
        **kwargs,
    )
