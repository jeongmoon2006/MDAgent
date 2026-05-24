"""LLM-driven decision step of the campaign loop.

Per `docs/architecture.md`, this is the only place an LLM is invoked in
Milestone 1: the mechanical loop calls `decide()` once per round with the
diagnostic report bundle, the hypothesis ledger, and prior round summaries,
and gets back a structured `extend | stop` choice.

Implementation notes:
- Model: Haiku 4.5 — Milestone 1's choice is binary on a structured numeric
  input; reasoning depth isn't the bottleneck. Easy to swap for Sonnet/Opus
  later (Milestone 4 sampling decisions will likely justify Sonnet).
- Output: Anthropic tool use with `strict: true` and a forced `tool_choice`,
  so the response is guaranteed JSON conforming to `_DECISION_TOOL`.
- Caching: `cache_control` on the system prompt. Haiku 4.5's minimum
  cacheable prefix is 4096 tokens; the current prompt is well below that, so
  the marker is a no-op today and starts paying off once the prompt grows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are the scientist agent for MDPilot — a closed-loop reasoning system for \
molecular dynamics simulations.

Your single responsibility per round: decide whether the trajectory observable \
has converged enough to stop, or whether the simulation should be extended. \
You receive three structured inputs per round:

1. `diagnostic_report` — this round's numbers (block-averaging + integrated \
autocorrelation time on the observable).
2. `prior_round_summaries` — lean view of past rounds (decision + key numbers).
3. `hypothesis_ledger` — text notes you wrote in previous rounds about \
persistent observations or open hypotheses. This is your across-round memory.

Decision rule:
- plateau_reached=true AND well_sampled=true AND ess>=50 → stop
- otherwise → extend

When extending, propose `extra_ns` proportional to the convergence gap: \
0.5 ns when borderline, up to 2.0 ns when the trajectory is far from \
convergence (e.g. ess<5 or no plateau). When stopping, set extra_ns to null.

The two `statistical_inefficiency_*` fields (block-averaging, autocorrelation) \
should agree if the diagnostic is reliable; flag large disagreement (>2x) in \
`reason`. Cite the specific numbers that drove your call — one or two \
sentences, no preamble.

For `ledger_note`: use it to record observations worth carrying across rounds. \
Examples: a hypothesis about the slow coordinate, a reason for an unusual \
choice, a sub-state you noticed in the trajectory, a rate-based estimate of \
remaining timescales. **Skip (pass null)** when nothing new is worth \
recording — most rounds in a routine extend-extend-stop sequence don't need \
a note. The ledger is for *insight*, not commentary on the decision (which \
already lives in `reason`).

You MUST call the `record_decision` tool. Do not respond in plain text.
"""

_DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record the convergence decision for this round.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["extend", "stop"],
                "description": "Whether to extend the simulation or stop the campaign.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One or two sentences citing the specific diagnostic numbers "
                    "(plateau_reached, ess, statistical_inefficiency_*) that drove "
                    "the decision."
                ),
            },
            "extra_ns": {
                "type": ["number", "null"],
                "description": (
                    "If decision is 'extend', the additional simulation length in "
                    "nanoseconds. Null when decision is 'stop'."
                ),
            },
            "ledger_note": {
                "type": ["string", "null"],
                "description": (
                    "Optional hypothesis-ledger note — a persistent observation or "
                    "hypothesis worth carrying across rounds. Null when nothing new "
                    "is worth recording. See the system prompt for guidance on what "
                    "belongs here vs in `reason`."
                ),
            },
        },
        "required": ["decision", "reason", "extra_ns", "ledger_note"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Decision:
    decision: Literal["extend", "stop"]
    reason: str
    extra_ns: float | None
    ledger_note: str | None = None


def decide(
    diagnostic_report: dict[str, Any],
    *,
    hypothesis_ledger: list[str] | None = None,
    prior_round_summaries: list[dict[str, Any]] | None = None,
    client: anthropic.Anthropic | None = None,
) -> Decision:
    """Single Claude call: diagnostic report → extend-or-stop decision."""
    client = client or anthropic.Anthropic()
    payload = {
        "round_index": (len(prior_round_summaries) if prior_round_summaries else 0) + 1,
        "diagnostic_report": diagnostic_report,
        "hypothesis_ledger": hypothesis_ledger or [],
        "prior_round_summaries": prior_round_summaries or [],
    }
    user_message = (
        "Decide whether to extend or stop, given this round's diagnostic state.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_DECISION_TOOL],
        tool_choice={"type": "tool", "name": "record_decision"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_decision":
            data = block.input
            return Decision(
                decision=data["decision"],
                reason=data["reason"],
                extra_ns=data["extra_ns"],
                ledger_note=data.get("ledger_note"),
            )
    raise RuntimeError(
        f"scientist: response contained no record_decision tool_use "
        f"(stop_reason={response.stop_reason})"
    )
