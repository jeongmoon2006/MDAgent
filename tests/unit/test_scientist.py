"""Wiring tests for scientist.decide() — request shape + response extraction.

The real Claude call is exercised by the integration test in
tests/integration/test_scientist_live.py (skipped when ANTHROPIC_API_KEY is
not set).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, get_args, get_type_hints

import pytest

from mdpilot.orchestrator.scientist import Decision, MetadProposal, decide


class _FakeClient:
    """Stand-in for anthropic.Anthropic that records the request and replays a fixed tool_use response."""

    def __init__(self, tool_input: dict[str, Any]) -> None:
        self._tool_input = tool_input
        self.last_request: dict[str, Any] | None = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        block = SimpleNamespace(type="tool_use", name="record_decision", input=self._tool_input)
        return SimpleNamespace(content=[block], stop_reason="tool_use")


def test_decide_extracts_tool_use_block() -> None:
    fake = _FakeClient(
        tool_input={"decision": "extend", "reason": "ess=8 < 50", "extra_ns": 0.5}
    )
    result = decide({"plateau_reached": False, "ess": 8}, client=fake)
    assert result == Decision(decision="extend", reason="ess=8 < 50", extra_ns=0.5)


def test_decide_handles_stop_with_null_extra_ns() -> None:
    fake = _FakeClient(
        tool_input={"decision": "stop", "reason": "plateau + ess=320", "extra_ns": None}
    )
    result = decide({"plateau_reached": True, "ess": 320}, client=fake)
    assert result.decision == "stop"
    assert result.extra_ns is None


def test_decide_request_shape() -> None:
    fake = _FakeClient(
        tool_input={"decision": "extend", "reason": "stub", "extra_ns": 1.0}
    )
    decide(
        {"plateau_reached": False, "ess": 3},
        hypothesis_ledger=["traj is heating from minimized state"],
        prior_round_summaries=[{"round": 1, "decision": "extend"}],
        client=fake,
    )
    req = fake.last_request
    assert req is not None
    assert req["model"] == "claude-sonnet-4-6"  # M4 bump from Haiku
    assert req["tool_choice"] == {"type": "tool", "name": "record_decision"}
    assert req["tools"][0]["name"] == "record_decision"
    assert req["tools"][0]["strict"] is True
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_text = req["messages"][0]["content"]
    assert "round_index" in user_text
    assert '"round_index": 2' in user_text  # 1 prior summary → this is round 2


def test_decide_raises_when_no_tool_use_in_response() -> None:
    class _BrokenClient:
        messages = SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="oops")],
                stop_reason="end_turn",
            )
        )

    with pytest.raises(RuntimeError, match="no record_decision tool_use"):
        decide({}, client=_BrokenClient())


def test_decide_extracts_ledger_note_when_present() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "extend",
            "reason": "ess=3",
            "extra_ns": 1.0,
            "ledger_note": "trajectory appears stuck in metastable basin",
        }
    )
    result = decide({"ess": 3}, client=fake)
    assert result.ledger_note == "trajectory appears stuck in metastable basin"


def test_decide_handles_null_ledger_note() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "extend",
            "reason": "ess=8",
            "extra_ns": 0.5,
            "ledger_note": None,
        }
    )
    result = decide({"ess": 8}, client=fake)
    assert result.ledger_note is None


def test_decision_tool_schema_includes_ledger_note() -> None:
    from mdpilot.orchestrator.scientist import _DECISION_TOOL

    props = _DECISION_TOOL["input_schema"]["properties"]
    assert "ledger_note" in props
    assert props["ledger_note"]["type"] == ["string", "null"]
    assert "ledger_note" in _DECISION_TOOL["input_schema"]["required"]


def test_decide_passes_hypothesis_ledger_in_user_message() -> None:
    fake = _FakeClient(
        tool_input={"decision": "extend", "reason": "x", "extra_ns": 0.5, "ledger_note": None}
    )
    decide(
        {"ess": 3},
        hypothesis_ledger=[
            "R1: trajectory stuck on initial basin",
            "R2: low ESS expected if torsion is slow",
        ],
        client=fake,
    )
    user_text = fake.last_request["messages"][0]["content"]
    assert "R1: trajectory stuck on initial basin" in user_text
    assert "R2: low ESS expected if torsion is slow" in user_text


# ---------- M4: switch_to_metad + task_expectation ----------

def test_decide_extracts_switch_to_metad_with_proposal() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "switch_to_metad",
            "reason": "pinned, exploring=False; task wants fold/unfold; budget can't reach µs",
            "extra_ns": None,
            "ledger_note": "Rg is the natural folding coordinate for this hairpin",
            "metad_proposal": {
                "cv_type": "gyration",
                "selections": ["backbone and resSeq 1 to 10"],
                "label": "rg_back",
            },
        }
    )
    result = decide({"exploring": False, "n_basins": 1}, client=fake)
    assert result.decision == "switch_to_metad"
    assert result.extra_ns is None
    assert result.metad_proposal == MetadProposal(
        cv_type="gyration",
        selections=("backbone and resSeq 1 to 10",),
        label="rg_back",
    )


def test_decide_task_expectation_appears_in_user_message() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "extend",
            "reason": "x",
            "extra_ns": 0.5,
            "ledger_note": None,
            "metad_proposal": None,
        }
    )
    decide(
        {"exploring": True},
        task_expectation="expect fold/unfold transition; ~µs; budget 50 ns",
        client=fake,
    )
    user_text = fake.last_request["messages"][0]["content"]
    assert "fold/unfold transition" in user_text
    assert "budget 50 ns" in user_text


def test_decide_rejects_switch_without_proposal() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "switch_to_metad",
            "reason": "vanilla inadequate",
            "extra_ns": None,
            "ledger_note": None,
            "metad_proposal": None,
        }
    )
    with pytest.raises(RuntimeError, match="metad_proposal is null"):
        decide({}, client=fake)


def test_decide_rejects_proposal_with_non_switch_decision() -> None:
    fake = _FakeClient(
        tool_input={
            "decision": "extend",
            "reason": "ess low",
            "extra_ns": 1.0,
            "ledger_note": None,
            "metad_proposal": {
                "cv_type": "distance",
                "selections": ["name CA and resSeq 1", "name CA and resSeq 10"],
                "label": "d",
            },
        }
    )
    with pytest.raises(RuntimeError, match="must be null"):
        decide({}, client=fake)


def test_decision_tool_schema_includes_metad_proposal() -> None:
    from mdpilot.orchestrator.scientist import _DECISION_TOOL

    props = _DECISION_TOOL["input_schema"]["properties"]
    assert "metad_proposal" in props
    assert props["metad_proposal"]["type"] == ["object", "null"]
    assert props["decision"]["enum"] == ["extend", "stop", "switch_to_metad"]
    assert "metad_proposal" in _DECISION_TOOL["input_schema"]["required"]
    sub = props["metad_proposal"]["properties"]
    # Pinned against cv_designer's vocabulary rather than a literal list: the
    # schema is what the model may emit and `_CV_TYPES` is what the resolver
    # accepts, so a type added to one and not the other is a runtime ValueError
    # on a live campaign — exactly the kind of drift a mocked test would miss.
    from mdpilot.sampling.cv_designer import _CV_TYPES

    assert sub["cv_type"]["enum"] == list(_CV_TYPES)
    # Third copy of the same vocabulary. It annotates the dataclass the parsed
    # decision lands in, so it documents the contract without constraining it
    # at runtime — which is exactly why it silently fell a type behind the
    # other two when `contacts` was added.
    # `get_type_hints`, not `__annotations__`: this module uses
    # `from __future__ import annotations`, so the raw annotation is a string.
    hints = get_type_hints(MetadProposal)
    assert set(get_args(hints["cv_type"])) == set(_CV_TYPES)


def test_metad_proposal_round_trips_through_dict() -> None:
    mp = MetadProposal(
        cv_type="distance",
        selections=("name CA and resSeq 1", "name CA and resSeq 10"),
        label="d_term",
    )
    assert MetadProposal.from_dict(mp.to_dict()) == mp


# ---------- tool schema validity under strict mode ----------


def test_tool_schemas_avoid_keywords_strict_mode_rejects() -> None:
    """`strict: true` restricts the usable JSON Schema vocabulary.

    `minItems`/`maxItems` on an array return a 400 ("For 'array' type,
    property 'maxItems' is not supported"). Both were present on
    `metad_proposal.selections` from the M4 action-space refactor (f0e4b8e)
    until the first live campaign hit the API — every vanilla decide() call in
    between would have failed, and every unit test stayed green because they
    all mock the client. This is the cheap guard; the authoritative check is
    still a live call.
    """
    from mdpilot.orchestrator.scientist import (
        _DECISION_TOOL,
        _METAD_DECISION_TOOL,
    )

    rejected = {"minItems", "maxItems"}

    def walk(node, path="input_schema"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in rejected, f"{path}.{key} is rejected under strict mode"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    for tool in (_DECISION_TOOL, _METAD_DECISION_TOOL):
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False
        walk(tool["input_schema"])


# ---------- switch_to_metad is gated on task_expectation ----------

def test_a_campaign_with_no_expectation_cannot_express_a_pivot() -> None:
    """`task_expectation` is the only input the switch_to_metad rule compares
    against — the required transition and its timescale live in that string.
    With none supplied there is nothing to judge "the budget cannot reach it"
    on, so the action leaves the enum entirely.

    Load-bearing beyond tidiness: `run_campaign` only requires
    `state_thresholds` when `task_expectation` is set, so a pivot reached from
    a campaign without one lands in a biased phase that counts recrossings
    between whichever two basins are currently deepest (F9).
    """
    fake = _FakeClient(
        tool_input={"decision": "extend", "reason": "ess=8", "extra_ns": 0.5,
                    "ledger_note": None}
    )
    decide({"phase": "vanilla", "ess": 8}, client=fake, task_expectation=None)

    schema = fake.last_request["tools"][0]["input_schema"]
    assert schema["properties"]["decision"]["enum"] == ["extend", "stop"]
    assert "metad_proposal" not in schema["properties"]
    assert "metad_proposal" not in schema["required"]


def test_a_campaign_with_an_expectation_keeps_the_pivot() -> None:
    fake = _FakeClient(
        tool_input={"decision": "extend", "reason": "ess=8", "extra_ns": 0.5,
                    "ledger_note": None, "metad_proposal": None}
    )
    decide(
        {"phase": "vanilla", "ess": 8},
        client=fake,
        task_expectation="fold the hairpin; 20 ns budget",
    )

    schema = fake.last_request["tools"][0]["input_schema"]
    assert schema["properties"]["decision"]["enum"] == [
        "extend", "stop", "switch_to_metad",
    ]
    assert schema["properties"]["metad_proposal"]["type"] == ["object", "null"]


def test_the_convergence_tool_keeps_every_non_pivot_field() -> None:
    """Dropping the pivot must not quietly drop `ledger_note` or `extra_ns`
    with it — strict mode requires every property to be listed as required, so
    a missing one is a 400 rather than a silently narrower action."""
    from mdpilot.orchestrator.scientist import _CONVERGENCE_TOOL, _DECISION_TOOL

    convergence = _CONVERGENCE_TOOL["input_schema"]
    full = _DECISION_TOOL["input_schema"]

    assert set(convergence["properties"]) == set(full["properties"]) - {
        "metad_proposal"
    }
    assert set(convergence["required"]) == set(convergence["properties"])
    assert _CONVERGENCE_TOOL["strict"] is True
    assert _CONVERGENCE_TOOL["name"] == "record_decision"


# ---------- switch_cv action space ----------

def test_plain_biased_tool_cannot_express_a_cv_switch() -> None:
    """The default biased schema is the one used once the allowance is spent.
    `switch_cv` must be absent from it, not merely rejected downstream."""
    from mdpilot.orchestrator.scientist import _METAD_DECISION_TOOL

    schema = _METAD_DECISION_TOOL["input_schema"]
    assert schema["properties"]["decision"]["enum"] == ["extend", "stop"]
    assert "metad_proposal" not in schema["properties"]


def test_switch_enabled_biased_tool_requires_a_proposal_slot() -> None:
    from mdpilot.orchestrator.scientist import _METAD_SWITCH_TOOL

    schema = _METAD_SWITCH_TOOL["input_schema"]
    assert schema["properties"]["decision"]["enum"] == ["extend", "stop", "switch_cv"]
    # strict mode requires every property to be listed as required; the
    # nullable type is what makes "no proposal this round" expressible.
    assert "metad_proposal" in schema["required"]
    assert schema["properties"]["metad_proposal"]["type"] == ["object", "null"]


def test_switch_cv_must_carry_a_proposal() -> None:
    from mdpilot.orchestrator.scientist import _parse_decision

    with pytest.raises(RuntimeError, match="metad_proposal is null"):
        _parse_decision({
            "decision": "switch_cv", "reason": "cv is wrong",
            "extra_ns": None, "ledger_note": None, "metad_proposal": None,
        })


def test_extend_must_not_carry_a_proposal() -> None:
    """The cross-field invariant has to hold for the new action too, or an
    `extend` carrying a stale proposal would look like a silent switch."""
    from mdpilot.orchestrator.scientist import _parse_decision

    with pytest.raises(RuntimeError, match="must be null unless"):
        _parse_decision({
            "decision": "extend", "reason": "still filling",
            "extra_ns": 1.0, "ledger_note": None,
            "metad_proposal": {
                "cv_type": "contacts", "selections": ["name CA"], "label": "q",
            },
        })


def test_switch_cv_round_trips_with_its_proposal() -> None:
    from mdpilot.orchestrator.scientist import _parse_decision

    decision = _parse_decision({
        "decision": "switch_cv",
        "reason": "recrossings counted entirely inside the unfolded state",
        "extra_ns": None,
        "ledger_note": "rmsd_ca is one-way; trying native contacts",
        "metad_proposal": {
            "cv_type": "contacts", "selections": ["name CA"], "label": "q_native",
        },
    })

    assert decision.decision == "switch_cv"
    assert decision.metad_proposal is not None
    assert decision.metad_proposal.cv_type == "contacts"


# ---------- knowledge-base retrieval ----------
#
# The prompt is assembled per round from `mdpilot/knowledge/*.md`. What matters
# is not that a chunk *can* be loaded but that guidance the round cannot act on
# is absent, and that guidance it needs is never dropped silently.

def _system_text(fake: _FakeClient) -> str:
    assert fake.last_request is not None
    return fake.last_request["system"][0]["text"]


def _stub(**overrides: Any) -> _FakeClient:
    payload: dict[str, Any] = {"decision": "extend", "reason": "stub", "extra_ns": 0.5}
    payload.update(overrides)
    return _FakeClient(tool_input=payload)


def test_chunks_are_readable_and_unknown_keys_raise() -> None:
    from mdpilot.orchestrator.scientist import _chunk

    assert "`cv_type` — one of" in _chunk("cv_vocabulary")
    with pytest.raises(FileNotFoundError, match="no knowledge chunk 'nope'"):
        _chunk("nope")


def test_keys_are_ordered_with_the_always_on_chunks_first() -> None:
    """Assembly order is the shared cache prefix; role must lead every variant."""
    from mdpilot.orchestrator.scientist import knowledge_keys

    for phase in ("vanilla", "metad"):
        for propose in (True, False):
            for switch in (True, False):
                keys = knowledge_keys(
                    phase, can_propose_cv=propose, allow_cv_switch=switch
                )
                assert keys[0] == "role"
                assert keys[1] == f"phase_{phase}"
                assert keys[-1] == "output_contract"


def test_a_biased_round_is_not_shown_the_vanilla_rubric() -> None:
    """The equilibrium rubric names ess/plateau_reached, which the biased report
    omits on purpose. Sending it invites exactly the category error the phase
    split exists to prevent."""
    fake = _stub()
    decide({"phase": "metad", "fes_converged": False}, phase="metad", client=fake)
    text = _system_text(fake)
    assert "PHASE `metad`" in text
    assert "PHASE `vanilla`" not in text
    assert "plateau_reached AND well_sampled" not in text


def test_a_vanilla_round_is_not_shown_the_free_energy_rubric() -> None:
    fake = _stub()
    decide({"ess": 8}, task_expectation="fold it", client=fake)
    text = _system_text(fake)
    assert "PHASE `vanilla`" in text
    assert "PHASE `metad`" not in text
    assert "fes_drift_kj_per_mol" not in text


def test_cv_vocabulary_travels_with_the_metad_proposal_field() -> None:
    """The vocabulary is present exactly when the tool can carry a proposal —
    the invariant `decide` derives the key from. Checked against the schema
    rather than restated, so the two cannot drift."""
    for kwargs in (
        {"task_expectation": "fold it"},                       # vanilla, may pivot
        {},                                                    # vanilla, convergence only
        {"phase": "metad"},                                    # biased, no switch left
        {"phase": "metad", "allow_cv_switch": True},           # biased, switch offered
    ):
        fake = _stub()
        decide({"stub": True}, client=fake, **kwargs)
        req = fake.last_request
        assert req is not None
        carries_proposal = "metad_proposal" in req["tools"][0]["input_schema"]["properties"]
        has_vocabulary = "`cv_type` — one of" in req["system"][0]["text"]
        assert carries_proposal == has_vocabulary, kwargs


def test_switch_cv_guidance_appears_only_while_the_allowance_lasts() -> None:
    spent = _stub()
    decide({"stub": True}, phase="metad", client=spent)
    assert "`switch_cv` replaces the biased" not in _system_text(spent)

    offered = _stub()
    decide({"stub": True}, phase="metad", allow_cv_switch=True, client=offered)
    assert "`switch_cv` replaces the biased" in _system_text(offered)


def test_the_unbounded_cv_warning_survives_every_round_that_can_propose_one() -> None:
    """F6: RMSD-to-native is unbounded above and cost two campaigns. Whenever a
    CV can be proposed, the round must carry that warning."""
    for kwargs in (
        {"task_expectation": "fold it"},
        {"phase": "metad", "allow_cv_switch": True},
    ):
        fake = _stub()
        decide({"stub": True}, client=fake, **kwargs)
        assert "unbounded above" in _system_text(fake), kwargs


def test_retrieval_shortens_the_prompt_against_sending_everything() -> None:
    from mdpilot.orchestrator.scientist import build_system_prompt

    widest = build_system_prompt("metad", can_propose_cv=True, allow_cv_switch=True)
    for phase, propose, switch in (
        ("vanilla", True, False),
        ("vanilla", False, False),
        ("metad", False, False),
    ):
        got = build_system_prompt(
            phase, can_propose_cv=propose, allow_cv_switch=switch
        )
        assert len(got) < len(widest)


# ---------- the prompt names only actions the round's enum offers ----------
#
# Chunks are shared across variants, so a rule written for one round's action
# space travels into rounds that do not have it. Both tests below read the
# enum off the tool actually sent and check the prose against it, rather than
# restating which actions each variant has.

def _paragraph_containing(text: str, needle: str) -> str:
    """The blank-line block stating one rule. Matched case-insensitively so the
    test survives a rewording of the sentence and still checks the rule."""
    for block in text.split("\n\n"):
        if needle.lower() in block.lower():
            return block
    raise AssertionError(f"no paragraph containing {needle!r} in:\n{text}")


def _decision_enum(fake: _FakeClient) -> list[str]:
    assert fake.last_request is not None
    schema = fake.last_request["tools"][0]["input_schema"]
    return schema["properties"]["decision"]["enum"]


def test_the_proposal_instruction_names_the_action_the_round_can_take() -> None:
    """`cv_vocabulary` is shared by both proposal-carrying rounds. Naming only
    `switch_to_metad` told a biased round to populate `metad_proposal` on an
    action `phase_metad` had just called permanently unavailable."""
    for kwargs, action in (
        ({"task_expectation": "fold it"}, "switch_to_metad"),
        ({"phase": "metad", "allow_cv_switch": True}, "switch_cv"),
    ):
        fake = _stub()
        decide({"stub": True}, client=fake, **kwargs)
        assert action in _decision_enum(fake), kwargs
        rule = _paragraph_containing(_system_text(fake), "populate `metad_proposal`")
        assert f"`{action}`" in rule, (kwargs, rule)


def test_every_non_extending_action_is_told_to_null_extra_ns() -> None:
    """`extra_ns` is meaningful only on an extend, and the loop ignores it
    otherwise — but the rule listed `stop` and `switch_to_metad` only, so a
    `switch_cv` round was given no instruction and could persist a number that
    means nothing."""
    for kwargs in (
        {"task_expectation": "fold it"},
        {},
        {"phase": "metad"},
        {"phase": "metad", "allow_cv_switch": True},
    ):
        fake = _stub()
        decide({"stub": True}, client=fake, **kwargs)
        rule = _paragraph_containing(_system_text(fake), "`extra_ns` must be null")
        for action in _decision_enum(fake):
            if action == "extend":
                continue
            assert f"`{action}`" in rule, (kwargs, action, rule)
