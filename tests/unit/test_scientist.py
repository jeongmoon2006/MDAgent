"""Wiring tests for scientist.decide() — request shape + response extraction.

The real Claude call is exercised by the integration test in
tests/integration/test_scientist_live.py (skipped when ANTHROPIC_API_KEY is
not set).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mdpilot.orchestrator.scientist import Decision, decide


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
    assert req["model"] == "claude-haiku-4-5"
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
