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
- Action space (vanilla): `extend | stop | switch_to_metad`. On `switch_to_metad`
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
  Strict mode restricts the usable JSON Schema vocabulary — `minItems` /
  `maxItems` on an array are rejected with a 400, so array arity is validated
  downstream in `cv_designer` rather than declared here.
  `metad_proposal` is a nullable object so the discriminated union stays in
  one tool call; the cross-field invariant (`metad_proposal` non-null iff
  decision == switch_to_metad) is enforced in `decide()`'s parser.
- Caching: `cache_control` on the system prompt. Sonnet 4.6's minimum
  cacheable prefix is 1024 tokens; the two-phase prompt is around that size,
  so caching may now actually engage. Note the render order is tools →
  system → messages, so swapping the tool at the pivot invalidates the
  system prefix for one round. That happens once per campaign and is not
  worth designing around.
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

Your responsibility per round: decide what to do next.

A campaign runs in one of two phases, named by `diagnostic_report.phase`. \
The report you receive and the actions available to you both depend on it. \
Read `phase` first.

You receive four structured inputs per round:

1. `diagnostic_report` — this round's numbers. Phase-dependent; see below.

2. `prior_round_summaries` — lean view of past rounds (decision + key \
numbers). Each carries its own `phase`, so rounds from before a pivot are \
not comparable to rounds after one.

3. `hypothesis_ledger` — text notes you wrote in previous rounds about \
persistent observations. Your across-round memory.

4. `task_expectation` — campaign-level expectation: what the trajectory must \
accomplish, characteristic timescale, compute budget. Free text, may be null \
for pure convergence tasks (a single-basin equilibration with no required \
transition).

=== PHASE `vanilla` — unbiased MD ===

Action space is three-way:

- `extend` — run more vanilla MD; the trajectory needs more time.
- `stop` — the observable has converged for the task at hand; the campaign \
ends.
- `switch_to_metad` — vanilla MD is *inadequate*: the system is pinned in a \
single basin AND the task requires a transition the budget cannot reach. \
Propose a collective variable for metadynamics.

Report fields. Convergence: `plateau_reached`, `ess`, `tau_int_frames`, \
`statistical_inefficiency_*`. Exploration: `bimodality_coefficient`, \
`n_basins`, `minor_basin_occupancy`, `exploring`. The two \
statistical-inefficiency fields (block-averaging, autocorrelation) should \
agree if the diagnostic is reliable; flag >2× disagreement in `reason`.

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

=== PHASE `metad` — well-tempered metadynamics ===

Action space is two-way: `extend` or `stop`. The campaign has already \
pivoted; proposing another pivot is not available to you.

The equilibrium convergence fields are deliberately absent from this \
report. A biased trajectory is not an equilibrium ensemble — the bias drives \
the observable — so a long autocorrelation would mean the bias is still \
filling and a bimodal marginal would mean the bias worked, not that the \
system is sampling freely. Do not ask for those numbers or reason as if you \
had them.

Report fields, all derived from the deposited bias (HILLS) integrated into a \
free-energy surface:

- `fes_drift_kj_per_mol` — how much the surface changed between the last two \
cumulative estimates. The standard well-tempered convergence test.
- `recrossings` — barrier crossings between the two deepest basins, counted \
with hysteresis. `barrier_crossed` is `recrossings >= 1`. The two basins are \
re-derived from the *current* surface every round, so these boundaries move \
as the bias fills. `recrossing_low` and `recrossing_high` are the CV values \
the count was actually taken between; `cv_start` is where the walker began.
- `fes_converged` — true only when drift is below kT (≈2.5 kJ/mol at 300 K) \
AND `recrossings >= 1`. Low drift *alone* is not convergence: a walker that \
never left its starting basin produces a surface that stops changing \
immediately, because nothing new is being sampled.
- `n_basins_fes`, `barrier_kj_per_mol`, `fes_depth_kj_per_mol`, \
`n_fes_estimates` — shape of the surface recovered so far. `cv_min` and \
`cv_max` are the range the walker actually visited, and `fes_depth` is \
measured over that range only, not over the wider grid `sum_hills` writes.

Decision rule:

- `fes_converged=true` → `stop`. The surface has stopped moving and the \
walker has crossed the barrier at least once.
- otherwise → `extend`. This includes `fes_converged=null` (not enough \
estimates or no COLVAR yet) and the low-drift/zero-recrossing case, which is \
an under-filled basin, not a converged surface.
- If many rounds have passed with `recrossings=0` and a large \
`fes_depth_kj_per_mol`, say so in `reason` and record it in `ledger_note` — \
that pattern suggests the biased CV is not the slow coordinate. You cannot \
act on it, but a human reading the ledger can.
- Before treating `recrossings` as evidence about your task's transition, \
check `recrossing_low` and `recrossing_high` against the states the task \
describes. If both boundaries sit on the same side of those states, the count \
is measuring motion *within* one state rather than the transition you were \
asked for, and a non-zero count is then not evidence that the CV is working. \
Say so in `reason` and record it in `ledger_note`.

=== BOTH PHASES ===

Sizing `extra_ns` when extending: proportional to the gap — 0.5 ns when \
borderline, up to 2.0 ns when far (vanilla: ess<5 or no plateau; metad: \
drift well above kT or zero recrossings). When `stop` or `switch_to_metad`, \
`extra_ns` must be null.

When `switch_to_metad`, populate `metad_proposal`:

- `cv_type` — one of `distance`, `torsion`, `gyration`, `rmsd`. Pick the type \
that matches the physical coordinate you believe is slow.
- `selections` — MDTraj selection strings. Arity is type-specific:
  - `distance`: 2 selections, each must resolve to exactly 1 atom. \
Example: `["name CA and resSeq 1", "name CA and resSeq 10"]`.
  - `torsion`: 4 selections, each must resolve to exactly 1 atom. \
Example: `["resSeq 2 and name N", "resSeq 2 and name CA", \
"resSeq 2 and name C", "resSeq 3 and name N"]`.
  - `gyration`: 1 selection, must resolve to ≥2 atoms. \
Example: `["backbone and resSeq 1 to 10"]`.
  - `rmsd`: 1 selection, must resolve to ≥3 atoms. RMSD to the campaign's \
reference structure after optimal superposition — the usual folding order \
parameter. Example: `["name CA"]`.
- `label` — short snake_case identifier (e.g. `rg_back`, `d_term`). It MUST \
name the coordinate you are actually biasing. Do not name it after a \
coordinate you would have preferred but did not select: the label is written \
into plumed.dat, HILLS, COLVAR and every downstream report, and a label that \
misdescribes the CV makes the run unreadable afterwards. If you want RMSD, \
choose `cv_type: rmsd` — do not call a `distance` an rmsd.

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
            "enum": ["distance", "torsion", "gyration", "rmsd"],
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

Phase = Literal["vanilla", "metad"]

_TOOL_FOR_PHASE: dict[str, dict[str, Any]] = {
    "vanilla": _DECISION_TOOL,
    "metad": _METAD_DECISION_TOOL,
}


@dataclass(frozen=True)
class MetadProposal:
    """Structured metaD CV proposal. Mirrors `sampling.cv_designer.CVProposal`."""

    cv_type: Literal["distance", "torsion", "gyration", "rmsd"]
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
    phase: Phase = "vanilla",
    client: anthropic.Anthropic | None = None,
    model: str = _MODEL,
) -> Decision:
    """Single Claude call: diagnostic + context → next-action decision.

    `phase` selects the action space: `vanilla` allows extend / stop /
    switch_to_metad, `metad` allows extend / stop only. It must match the
    report being passed — an equilibrium bundle with phase="metad" would
    offer the right actions against the wrong numbers.
    """
    client = client or anthropic.Anthropic()
    if phase not in _TOOL_FOR_PHASE:
        raise ValueError(
            f"scientist: unknown phase {phase!r}; expected one of "
            f"{sorted(_TOOL_FOR_PHASE)}"
        )
    tool = _TOOL_FOR_PHASE[phase]
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
