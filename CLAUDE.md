# CLAUDE.md — MDPilot

Behavioral guidelines for working on MDPilot. These bias toward caution over speed; for trivial tasks, use judgment.

For *what* MDPilot is and why, see `docs/architecture.md`.
For *what to build next*, see `ROADMAP.md`.

---

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

For MDPilot specifically: when a request touches the agent's reasoning loop, memory architecture, or subagent boundaries, stop and verify against `docs/architecture.md` before coding. These are load-bearing decisions.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that *your* changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Add convergence diagnostic" → "Write a test against a known-converged and a known-under-converged trajectory; the diagnostic must distinguish them"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

For MDPilot research code: a passing unit test is necessary but not sufficient. Also verify the science — does this convergence rubric actually identify the under-converged case in `benchmarks/tasks/`? Code that runs cleanly but answers the wrong question is a failure mode this project specifically exists to prevent.

## 5. Subagent Test

When unsure whether a subtask should be its own subagent:
- Bounded input + structured output → yes, spawn a fresh-context subagent.
- Needs the campaign narrative or hypothesis context to reason about → no, stays with the scientist.

Never use natural language as an API between components. Subagents return structured artifacts (JSON, files), not prose summaries. The scientist talks to subagents the way it talks to any other tool.

---

## Project-specific anti-goals

These encode design decisions, not aesthetic preferences. Override only with written justification in the commit message.

- **Do not rebuild MDCrow's *agent layer*.** What MDCrow contributes is an LLM that orchestrates setup tools (PDBFixer, OpenMM Modeller, `gmx` commands, etc.) from natural-language prompts. *That* is the thing not to rebuild. Calling those underlying libraries deterministically from an adapter — given a structured system spec, not a prompt — is fine and is how `adapters/` works today (see `openmm_adapter.py`, `gromacs_adapter.py`). If a future use case needs natural-language setup, integrating MDCrow then is a deliberate decision, not a default. Recorded as D5 in `docs/activity-log.md` (2026-05-23).
- **Do not build a multi-agent system** with persistent agents communicating in natural language. The scientist is one agent; subagents are ephemeral function calls.
- **Do not put raw simulation logs, trajectories, or large files into the agent's context.** Tools return compact structured summaries plus filesystem paths.
- **Do not lock to one MD engine.** Adapters exist so the scientist doesn't care whether OpenMM or GROMACS ran the simulation.
- **Do not store campaign state in the conversation.** Hypothesis ledger, findings log, trajectory metadata go to disk via `memory/`.

---

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and architectural anti-goals stay intact across sessions.
