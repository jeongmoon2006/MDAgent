"""LLM-driven decision step of the campaign loop.

Per `docs/architecture.md`, this is the only place an LLM is invoked. The
mechanical loop calls `decide()` once per round with the diagnostic report
bundle, hypothesis ledger, prior round summaries, and (M4) a free-form
task-encoded expectation, and gets back a structured ternary choice.

Implementation notes:
- Action space (M4): `extend | stop | switch_to_metad`. On `switch_to_metad`
  the model also returns a `MetadProposal` — a structured CV proposal in the
  same shape `cv_designer.CVProposal` consumes (cv_type, MDTraj selection
  strings, label). Bias parameters (sigma, height, pace) are *not* in the
  model's output: they are physics-unit numbers, derivable deterministically
  from the prior trajectory and from rule-of-thumb constants; a small helper
  in `sampling/` will fill them at the point of use (step 4 of the M4 plan).
- Model: Sonnet 4.6 — bumped from M1's Haiku 4.5 because the action is now
  ternary with structured chemistry reasoning (CV selection, transition
  timescale judgment). The wrong CV ends a campaign in wasted compute; cost
  per round is small next to that. `model` is parameter-overridable so unit
  tests stay cheap (mocked) and live tests can pin a tier.
- Output: Anthropic tool use with `strict: true` and forced `tool_choice`.
  `metad_proposal` is a nullable object so the discriminated union stays in
  one tool call; the cross-field invariant (`metad_proposal` non-null iff
  decision == switch_to_metad) is enforced in `decide()`'s parser.
- Caching: `cache_control` on the system prompt. Sonnet 4.6's minimum
  cacheable prefix is 1024 tokens; the current prompt is below that today
  and starts paying off as the prompt grows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048

_SYSTEM_PROMPT = """\
You are the scientist agent for MDPilot — a closed-loop reasoning system for \
molecular dynamics simulations.

Your responsibility per round: decide what to do next. The action space is \
three-way:

- `extend` — run more vanilla MD; the trajectory needs more time.
- `stop` — the observable has converged for the task at hand; the campaign \
ends.
- `switch_to_metad` — vanilla MD is *inadequate*: the system is pinned in a \
single basin AND the task requires a transition the budget cannot reach. \
Propose a collective variable for metadynamics.

You receive four structured inputs per round:

1. `diagnostic_report` — this round's numbers. Convergence: \
`plateau_reached`, `ess`, `tau_int_frames`, `statistical_inefficiency_*`. \
Exploration: `bimodality_coefficient`, `n_basins`, `minor_basin_occupancy`, \
`exploring`. The two statistical-inefficiency fields (block-averaging, \
autocorrelation) should agree if the diagnostic is reliable; flag >2× \
disagreement in `reason`.

2. `prior_round_summaries` — lean view of past rounds (decision + key \
numbers).

3. `hypothesis_ledger` — text notes you wrote in previous rounds about \
persistent observations. Your across-round memory.

4. `task_expectation` — campaign-level expectation: what the trajectory must \
accomplish, characteristic timescale, compute budget. Free text, may be null \
for pure convergence tasks (a single-basin equilibration with no required \
transition).

Decision rule:

- `exploring=true` (n_basins >= 2): the system has visited multiple states; \
vanilla MD is reaching them. Decide between `extend` and `stop` on \
convergence numbers — `plateau_reached AND well_sampled AND ess>=50` → \
`stop`, else `extend`.

- `exploring=false` (pinned, n_basins == 1) AND `task_expectation` does not \
require a transition (or is null): single-basin convergence. Same rule — \
`plateau_reached AND well_sampled AND ess>=50` → `stop`, else `extend`.

- `exploring=false` AND `task_expectation` explicitly requires a transition \
that the budget cannot reach (compare cumulative simulation time to the \
characteristic timescale in the expectation): vanilla is inadequate. Decide \
`switch_to_metad` and propose a CV.

Sizing `extra_ns` when extending: proportional to the convergence gap — \
0.5 ns when borderline, up to 2.0 ns when far (ess<5 or no plateau). When \
`stop` or `switch_to_metad`, `extra_ns` must be null.

When `switch_to_metad`, populate `metad_proposal`:

- `cv_type` — one of `distance`, `torsion`, `gyration`. Pick the type that \
matches the physical coordinate you believe is slow.
- `selections` — MDTraj selection strings. Arity is type-specific:
  - `distance`: 2 selections, each must resolve to exactly 1 atom. \
Example: `["name CA and resSeq 1", "name CA and resSeq 10"]`.
  - `torsion`: 4 selections, each must resolve to exactly 1 atom. \
Example: `["resSeq 2 and name N", "resSeq 2 and name CA", \
"resSeq 2 and name C", "resSeq 3 and name N"]`.
  - `gyration`: 1 selection, must resolve to ≥2 atoms. \
Example: `["backbone and resSeq 1 to 10"]`.
- `label` — short snake_case identifier (e.g. `rg_back`, `d_term`).

Bias parameters (sigma, height, pace) are *not* your concern — a \
deterministic helper computes them from the prior trajectory.

When not switching, `metad_proposal` must be null.

For `ledger_note`: record insights worth carrying across rounds — a \
hypothesis about the slow coordinate, the reason for an unusual CV choice, a \
rate estimate. Pass null when nothing new is worth recording.

For `reason`: cite the specific numbers that drove the call \
(`plateau_reached`, `ess`, `exploring`, `n_basins`, `bimodality_coefficient`) \
and, if switching, briefly justify the CV in physical terms. One to three \
sentences.

You MUST call the `record_decision` tool. Do not respond in plain text.
"""

_METAD_PROPOSAL_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "cv_type": {
            "type": "string",
            "enum": ["distance", "torsion", "gyration"],
            "description": "Which type of CV to bias with metadynamics.",
        },
        "selections": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
            "description": (
                "MDTraj selection strings, one per logical position. Arity "
                "depends on cv_type — see system prompt."
            ),
        },
        "label": {
            "type": "string",
            "description": "Short snake_case identifier for the CV.",
        },
    },
    "required": ["cv_type", "selections", "label"],
    "additionalProperties": False,
    "description": (
        "Structured CV proposal. Non-null iff decision is 'switch_to_metad'; "
        "null otherwise."
    ),
}

_DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record the decision (extend, stop, or switch_to_metad) for this round.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["extend", "stop", "switch_to_metad"],
                "description": "What to do next with the trajectory.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One to three sentences citing the specific diagnostic "
                    "numbers that drove the decision, plus a brief CV "
                    "justification when switching."
                ),
            },
            "extra_ns": {
                "type": ["number", "null"],
                "description": (
                    "Additional simulation length in nanoseconds when extending. "
                    "Null when stopping or switching."
                ),
            },
            "ledger_note": {
                "type": ["string", "null"],
                "description": (
                    "Optional hypothesis-ledger note — a persistent observation "
                    "worth carrying across rounds. Null when nothing new is "
                    "worth recording."
                ),
            },
            "metad_proposal": _METAD_PROPOSAL_SCHEMA,
        },
        "required": ["decision", "reason", "extra_ns", "ledger_note", "metad_proposal"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class MetadProposal:
    """Structured metaD CV proposal. Mirrors `sampling.cv_designer.CVProposal`."""

    cv_type: Literal["distance", "torsion", "gyration"]
    selections: tuple[str, ...]
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_type": self.cv_type,
            "selections": list(self.selections),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadProposal":
        return cls(
            cv_type=data["cv_type"],
            selections=tuple(data["selections"]),
            label=data["label"],
        )


@dataclass(frozen=True)
class Decision:
    decision: Literal["extend", "stop", "switch_to_metad"]
    reason: str
    extra_ns: float | None
    ledger_note: str | None = None
    metad_proposal: MetadProposal | None = None


def decide(
    diagnostic_report: dict[str, Any],
    *,
    hypothesis_ledger: list[str] | None = None,
    prior_round_summaries: list[dict[str, Any]] | None = None,
    task_expectation: str | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = _MODEL,
) -> Decision:
    """Single Claude call: diagnostic + context → extend / stop / switch decision."""
    client = client or anthropic.Anthropic()
    payload = {
        "round_index": (len(prior_round_summaries) if prior_round_summaries else 0) + 1,
        "diagnostic_report": diagnostic_report,
        "hypothesis_ledger": hypothesis_ledger or [],
        "prior_round_summaries": prior_round_summaries or [],
        "task_expectation": task_expectation,
    }
    user_message = (
        "Decide what to do next, given this round's state.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )

    response = client.messages.create(
        model=model,
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
            return _parse_decision(block.input)
    raise RuntimeError(
        f"scientist: response contained no record_decision tool_use "
        f"(stop_reason={response.stop_reason})"
    )


def _parse_decision(data: dict[str, Any]) -> Decision:
    decision = data["decision"]
    metad_raw = data.get("metad_proposal")
    metad = MetadProposal.from_dict(metad_raw) if metad_raw else None

    if decision == "switch_to_metad" and metad is None:
        raise RuntimeError(
            "scientist: decision='switch_to_metad' but metad_proposal is null"
        )
    if decision != "switch_to_metad" and metad is not None:
        raise RuntimeError(
            f"scientist: metad_proposal present with decision={decision!r}; "
            f"must be null unless decision=='switch_to_metad'"
        )

    return Decision(
        decision=decision,
        reason=data["reason"],
        extra_ns=data["extra_ns"],
        ledger_note=data.get("ledger_note"),
        metad_proposal=metad,
    )
