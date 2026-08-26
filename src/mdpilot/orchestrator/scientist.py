"""LLM-driven decision step of the campaign loop.

Per `docs/architecture.md`, this is the only place an LLM is invoked. The
mechanical loop calls `decide()` once per round with the diagnostic report
bundle, hypothesis ledger, prior round summaries, and (M4) a free-form
task-encoded expectation, and gets back a structured choice whose action
space depends on the campaign phase (see below).

Implementation notes:
- Phase-dependent contract (M4). A campaign has two phases and the scientist
  sees a different report and a different action space in each:
    * `vanilla` — the equilibrium convergence bundle from `diagnostics.report`,
      action space `extend | stop | switch_to_metad`.
    * `metad` — the free-energy bundle from `diagnostics.free_energy`, action
      space `extend | stop`. The equilibrium verdicts are *absent*, not
      relabelled: a biased trajectory is not an equilibrium ensemble, so
      `plateau_reached` / `ess` / `exploring` do not describe convergence
      there. Omitting them makes the category error unrepresentable rather
      than merely discouraged. A second `switch_to_metad` is out of scope (the
      loop ends the campaign for human review), so it is dropped from the
      biased tool schema instead of being emitted and rejected.
- Action space (vanilla): `extend | stop`, plus `switch_to_metad` when a
  `task_expectation` was supplied — that string is the only thing the pivot
  rule compares against, so without it the campaign is a pure convergence task
  and the action is dropped from the enum rather than left available and
  discouraged in prose. On `switch_to_metad` the model also returns a
  `MetadProposal` — a structured CV proposal in the same shape
  `cv_designer.CVProposal` consumes (cv_type, MDTraj selection strings,
  label). Bias parameters (sigma, height, pace) are *not* in the
  model's output: they are physics-unit numbers, derivable deterministically
  from the prior trajectory and from rule-of-thumb constants; a small helper
  in `sampling/` will fill them at the point of use (step 4 of the M4 plan).
- Model: Sonnet 4.6 — bumped from M1's Haiku 4.5 because the action is now
  ternary with structured chemistry reasoning (CV selection, transition
  timescale judgment). The wrong CV ends a campaign in wasted compute; cost
  per round is small next to that. `model` is parameter-overridable so unit
  tests stay cheap (mocked) and live tests can pin a tier.
- Output: Anthropic tool use with `strict: true` and forced `tool_choice`.
  Strict mode restricts the usable JSON Schema vocabulary — `minItems` /
  `maxItems` on an array are rejected with a 400, so array arity is validated
  downstream in `cv_designer` rather than declared here.
  `metad_proposal` is a nullable object so the discriminated union stays in
  one tool call; the cross-field invariant (`metad_proposal` non-null iff
  decision == switch_to_metad) is enforced in `decide()`'s parser.
- Prompt assembly: the system prompt is retrieved per round from the
  Markdown knowledge base in `mdpilot/knowledge/`, keyed on the same facts
  that select the tool schema (phase, whether a CV proposal is possible,
  whether a CV switch is offered). A round is never shown rules it cannot
  act on — a biased round gets no equilibrium rubric, a pure-convergence
  campaign gets no CV vocabulary. See `build_system_prompt`.
- Caching: `cache_control` on the assembled system prompt. There are four
  possible assemblies and the prompt is constant within a campaign phase, so
  each variant caches after its first round. The render order is tools →
  system → messages, so a breakpoint on the system block caches the tool
  schema *with* it: the ~715-token tool sits inside the cached prefix, not
  before it. That is what keeps even the smallest assembly clear of Sonnet
  4.6's 1024-token minimum — a pure-convergence vanilla round is a
  1,014-token system block but a 1,729-token prefix, and it does cache
  (measured `cache_read_input_tokens` 1,413 on the second call). Retrieval
  therefore never trims a round out of the cache; an earlier note here
  claiming the smallest assembly might miss it compared the system block
  against the minimum instead of the prefix. That same render order is why
  swapping the tool at the pivot invalidates the prefix for one round — the
  same round the assembly changes anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048

# --- Prompt knowledge base -------------------------------------------------
#
# The rubrics, CV vocabulary and physics constraints live as Markdown under
# `mdpilot/knowledge/`, one file per retrievable chunk, and are assembled per
# round by `build_system_prompt`. Moved out of this module because a single
# static prompt sent every round carried both phases' rules and the whole CV
# vocabulary regardless of which were usable — ~2.9k tokens, of which roughly
# half was unreachable guidance for the round being decided.
#
# Retrieval is by key, not by similarity: the keys are exactly the facts that
# already select the tool schema, so the prose the model reads and the actions
# it can emit cannot drift apart. There is deliberately no retrieval *within*
# `cv_vocabulary` — the scientist is choosing among the CV types, so showing it
# a filtered subset would pre-decide the science it exists to decide (the same
# argument `benchmarks/tasks/cln025_folding.yaml` makes about not pre-selecting
# an RMSD CV; run 1 and run 3 differed only in the model picking `contacts`).

Phase = Literal["vanilla", "metad"]

_KNOWLEDGE_PACKAGE = "mdpilot.knowledge"


@lru_cache(maxsize=None)
def _chunk(name: str) -> str:
    """Read one knowledge chunk by key (its filename stem), cached per process.

    A missing key raises rather than resolving to empty. Silently dropping a
    chunk would ship a prompt with, say, the bounded-vs-unbounded CV warning
    absent — which is F6 exactly, and it would show up as a bad campaign weeks
    later rather than as an error here.
    """
    try:
        return (
            resources.files(_KNOWLEDGE_PACKAGE)
            .joinpath(f"{name}.md")
            .read_text(encoding="utf-8")
            .strip()
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        available = sorted(
            f.name.removesuffix(".md")
            for f in resources.files(_KNOWLEDGE_PACKAGE).iterdir()
            if f.name.endswith(".md")
        )
        raise FileNotFoundError(
            f"scientist: no knowledge chunk {name!r} in {_KNOWLEDGE_PACKAGE}; "
            f"available: {available}"
        ) from e


def knowledge_keys(
    phase: Phase, *, can_propose_cv: bool, allow_cv_switch: bool
) -> tuple[str, ...]:
    """Which knowledge chunks this round needs, in assembly order.

    `can_propose_cv` is whether the tool this round carries a `metad_proposal`
    field at all — true for a vanilla round that may pivot, and for a biased
    round with a CV switch still in its allowance. It is the caller's job to
    keep it in step with the tool selected in `decide`; `test_scientist` pins
    the two together.

    Always-on chunks come first so the four variants share the longest possible
    cache prefix.
    """
    keys = ["role", f"phase_{phase}"]
    if allow_cv_switch:
        keys.append("action_switch_cv")
    if can_propose_cv:
        keys.append("cv_vocabulary")
    keys.append("output_contract")
    return tuple(keys)


def build_system_prompt(
    phase: Phase, *, can_propose_cv: bool, allow_cv_switch: bool
) -> str:
    """Assemble the round's system prompt from the knowledge base."""
    return "\n\n".join(
        _chunk(k)
        for k in knowledge_keys(
            phase, can_propose_cv=can_propose_cv, allow_cv_switch=allow_cv_switch
        )
    )

_METAD_PROPOSAL_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "cv_type": {
            "type": "string",
            "enum": ["distance", "torsion", "gyration", "rmsd", "contacts"],
            "description": "Which type of CV to bias with metadynamics.",
        },
        "selections": {
            "type": "array",
            "items": {"type": "string"},
            # No minItems/maxItems: under `strict: true` the API rejects both
            # ("For 'array' type, property 'maxItems' is not supported"), so
            # arity cannot be constrained here. It is enforced deterministically
            # in `cv_designer._build_{distance,torsion,gyration}`, which raise a
            # ValueError naming the expected count — a better error than a
            # schema violation anyway, and the only enforcement that also checks
            # each selection resolves to the right number of *atoms*.
            "description": (
                "MDTraj selection strings, one per logical position. Arity "
                "depends on cv_type and is validated on resolution: distance "
                "takes 2 single-atom selections, torsion 4 single-atom "
                "selections, gyration 1 selection of >=2 atoms."
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


# Biased rounds get their own tool rather than a runtime check on the ternary
# one. A second `switch_to_metad` ends the campaign for human review, so the
# action is not available to the model; dropping it from the enum makes that
# unrepresentable instead of emitted-then-rejected. `metad_proposal` goes with
# it — there is no proposal to make.
_METAD_DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record the decision (extend or stop) for this biased round.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["extend", "stop"],
                "description": "What to do next with the biased run.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One to three sentences citing the specific free-energy "
                    "numbers that drove the decision (fes_drift_kj_per_mol, "
                    "recrossings, fes_converged)."
                ),
            },
            "extra_ns": {
                "type": ["number", "null"],
                "description": (
                    "Additional biased simulation length in nanoseconds when "
                    "extending. Null when stopping."
                ),
            },
            "ledger_note": {
                "type": ["string", "null"],
                "description": (
                    "Optional hypothesis-ledger note — a persistent observation "
                    "worth carrying across rounds, e.g. a suspicion that the "
                    "biased CV is not the slow coordinate. Null when nothing "
                    "new is worth recording."
                ),
            },
        },
        "required": ["decision", "reason", "extra_ns", "ledger_note"],
        "additionalProperties": False,
    },
}

# The biased action space with CV revision re-opened. Offered only while the
# campaign has a switch left in its budget: past that, `switch_cv` is dropped
# from the enum so a further revision is unrepresentable rather than
# emitted-and-rejected — the same reason a second `switch_to_metad` is absent
# from `_METAD_DECISION_TOOL`.
_METAD_SWITCH_TOOL = {
    "name": "record_decision",
    "description": (
        "Record the decision (extend, stop, or switch_cv) for this biased round."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            **_METAD_DECISION_TOOL["input_schema"]["properties"],
            "decision": {
                "type": "string",
                "enum": ["extend", "stop", "switch_cv"],
                "description": "What to do next with the biased run.",
            },
            "metad_proposal": _METAD_PROPOSAL_SCHEMA,
        },
        "required": [
            *_METAD_DECISION_TOOL["input_schema"]["required"],
            "metad_proposal",
        ],
        "additionalProperties": False,
    },
}

# The vanilla action space for a campaign that cannot pivot. `task_expectation`
# is the sole input the `switch_to_metad` rule compares against — it is where
# the required transition and the characteristic timescale are stated — so with
# no expectation there is nothing to judge "the budget cannot reach it" on.
# Dropping the action from the enum makes the pivot unrepresentable rather than
# merely discouraged by the prompt, the same argument as `_METAD_DECISION_TOOL`
# for a second pivot and `_METAD_SWITCH_TOOL` for a spent CV-revision budget.
#
# This is load-bearing beyond tidiness: a pivot from a campaign with no
# expectation also has no `state_thresholds` (`run_campaign` only requires them
# when `task_expectation` is set), so the biased phase would fall back to
# counting recrossings between the two deepest basins of the current surface —
# the F9 behaviour that fallback exists to keep unreachable.
_CONVERGENCE_TOOL = {
    "name": "record_decision",
    "description": "Record the decision (extend or stop) for this round.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            **{
                k: v
                for k, v in _DECISION_TOOL["input_schema"]["properties"].items()
                if k != "metad_proposal"
            },
            "decision": {
                "type": "string",
                "enum": ["extend", "stop"],
                "description": "What to do next with the trajectory.",
            },
        },
        "required": [
            k
            for k in _DECISION_TOOL["input_schema"]["required"]
            if k != "metad_proposal"
        ],
        "additionalProperties": False,
    },
}

_TOOL_FOR_PHASE: dict[str, dict[str, Any]] = {
    "vanilla": _DECISION_TOOL,
    "metad": _METAD_DECISION_TOOL,
}


@dataclass(frozen=True)
class MetadProposal:
    """Structured metaD CV proposal. Mirrors `sampling.cv_designer.CVProposal`."""

    cv_type: Literal["distance", "torsion", "gyration", "rmsd", "contacts"]
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
    decision: Literal["extend", "stop", "switch_to_metad", "switch_cv"]
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
    phase: Phase = "vanilla",
    allow_cv_switch: bool = False,
    client: anthropic.Anthropic | None = None,
    model: str = _MODEL,
) -> Decision:
    """Single Claude call: diagnostic + context → next-action decision.

    `phase` selects the action space: `vanilla` allows extend / stop /
    switch_to_metad, `metad` allows extend / stop. It must match the report
    being passed — an equilibrium bundle with phase="metad" would offer the
    right actions against the wrong numbers.

    `task_expectation` also gates the pivot. It is the only input the
    switch_to_metad rule has to compare against, so with none supplied the
    campaign is a pure convergence task and `switch_to_metad` is dropped from
    the vanilla enum. Not merely tidier: a pivot from such a campaign also has
    no `state_thresholds`, and the biased phase would then count recrossings
    against whichever two basins are currently deepest (F9).

    `allow_cv_switch` adds `switch_cv` to the biased action space. The caller
    owns that budget: once the campaign has spent its allowance the action is
    dropped from the enum rather than refused after the fact, so the model
    never emits a decision the loop will not honour.
    """
    client = client or anthropic.Anthropic()
    if phase not in _TOOL_FOR_PHASE:
        raise ValueError(
            f"scientist: unknown phase {phase!r}; expected one of "
            f"{sorted(_TOOL_FOR_PHASE)}"
        )
    tool = _TOOL_FOR_PHASE[phase]
    if phase == "vanilla" and task_expectation is None:
        tool = _CONVERGENCE_TOOL
    elif phase == "metad" and allow_cv_switch:
        tool = _METAD_SWITCH_TOOL
    # One source of truth for "can the model propose a CV this round": the tool
    # it is actually given. Deriving the prompt key from the schema rather than
    # re-deriving it from (phase, task_expectation, allow_cv_switch) means the
    # vocabulary is present exactly when the field that consumes it is, with no
    # second condition to keep in step.
    can_propose_cv = "metad_proposal" in tool["input_schema"]["properties"]
    system_prompt = build_system_prompt(
        phase, can_propose_cv=can_propose_cv, allow_cv_switch=allow_cv_switch
    )
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
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[tool],
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


_PROPOSAL_ACTIONS = frozenset({"switch_to_metad", "switch_cv"})


def _parse_decision(data: dict[str, Any]) -> Decision:
    decision = data["decision"]
    metad_raw = data.get("metad_proposal")
    metad = MetadProposal.from_dict(metad_raw) if metad_raw else None

    # Both actions that (re)define the biased coordinate carry a proposal:
    # `switch_to_metad` opens the biased phase, `switch_cv` replaces the CV
    # inside it. Everything else must leave it null.
    if decision in _PROPOSAL_ACTIONS and metad is None:
        raise RuntimeError(
            f"scientist: decision={decision!r} but metad_proposal is null"
        )
    if decision not in _PROPOSAL_ACTIONS and metad is not None:
        raise RuntimeError(
            f"scientist: metad_proposal present with decision={decision!r}; "
            f"must be null unless decision is one of "
            f"{sorted(_PROPOSAL_ACTIONS)}"
        )

    return Decision(
        decision=decision,
        reason=data["reason"],
        extra_ns=data["extra_ns"],
        ledger_note=data.get("ledger_note"),
        metad_proposal=metad,
    )
