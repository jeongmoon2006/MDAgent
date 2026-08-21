"""Wiring tests for scientist.decide() — request shape + response extraction.

The real Claude call is exercised by the integration test in
tests/integration/test_scientist_live.py (skipped when ANTHROPIC_API_KEY is
not set).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
