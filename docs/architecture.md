# MDPilot Architecture

A closed-loop scientific reasoning agent for molecular dynamics simulations.

> An MD operator runs the simulation you describe.
> An MD scientist decides what to simulate next.
> MDPilot is the latter.

---

## Thesis

Existing MD agents (MDCrow, DynaMate, NAMD-Agent, the various MDAgent projects, PolyJarvis, MatSciAgent) all focus on the inner loop: **setup → submit → analyze**. They are MD *operators*. They run the simulation you describe and return the result.

The outer loop — looking at results, judging whether they're adequate for the question, and deciding what to simulate next — is largely unsolved. MDPilot is a specialist for this outer loop.

## Position in the field

Setup, parameterization, execution, and basic analysis are well-handled by existing agents. MDPilot delegates these via adapters and focuses entirely on:

1. **Convergence and adequacy judgment** — when is a trajectory "done"? Does it answer the question that was asked?
2. **Enhanced sampling decisions** — when vanilla MD is inadequate, pick metadynamics CVs, REMD ladders, umbrella windows.
3. **Hypothesis-driven follow-up** — given what we just saw, what should we run next?
4. **Ensemble reasoning** — coherent inference across multiple replicas, temperatures, conditions.

If a contribution can be summarized as "MDCrow with X," it doesn't belong in MDPilot.

For positioning relative to specific systems, see `related_work.md`.

---

## Architecture principles

### One persistent agent + ephemeral subagents

MDPilot is a *single persistent agent* (the scientist) that owns the campaign narrative end-to-end. Bounded subtasks are dispatched to *ephemeral fresh-context subagents* that return structured artifacts and die.

This is **not** a multi-agent design pattern. There are no multiple persistent agents communicating in natural language. Subagents are function calls with their own scratch context, not collaborators. Lifecycle: spawn → process bounded input → return structured output → terminate.

The test for spawning a subagent: does the subtask have a clean input and a structured output? If yes, subagent. If it needs the campaign narrative to make sense, it stays with the scientist.

### Externalized state

The hypothesis ledger, findings log, and trajectory store live on disk (SQLite + filesystem) — not in the agent's context window. The scientist reads what it needs each round and writes back what it learned. Context is for active reasoning, not memory.

### Structured tool returns

Tools never dump raw logs, trajectories, or large files into context. Returns are compact JSON summaries plus filesystem paths. If the agent needs detail, it re-fetches a specific slice.

### Hierarchical round summaries

After each round (simulate → analyze → decide), a structured round summary is generated and persisted. The active context becomes:

- Current round in detail
- Structured summaries of all prior rounds
- Pinned hypothesis ledger and current plan

Full traces stay on disk for provenance and reproducibility.

---

## File structure

```
mdpilot/
├── README.md
├── CLAUDE.md                          # behavioral guidelines for coding sessions
├── ROADMAP.md                         # milestones
├── pyproject.toml
├── .env.example
│
├── src/mdpilot/
│   │
│   ├── orchestrator/                  # outer loop: plan → run → analyze → decide
│   │   ├── scientist.py               # top-level LLM agent (THE agent)
│   │   ├── planner.py                 # initial campaign plan
│   │   ├── replanner.py               # revises plan after each round
│   │   └── state.py                   # campaign state machine
│   │
│   ├── reasoning/                     # the scientific judgment core
│   │   ├── hypothesis.py              # open hypotheses + evidence ledger
│   │   ├── decision_policy.py         # extend? switch method? add replicas? stop?
│   │   ├── rubrics.py                 # structured criteria per decision type
│   │   └── prompts/
│   │       ├── convergence_judge.txt
│   │       ├── sampling_judge.txt
│   │       └── followup_planner.txt
│   │
│   ├── diagnostics/                   # mechanical convergence + adequacy (no LLM)
│   │   ├── block_averaging.py
│   │   ├── autocorrelation.py
│   │   ├── effective_sample_size.py
│   │   ├── pca_drift.py               # is the system still exploring?
│   │   ├── replica_agreement.py       # cross-replica convergence
│   │   └── report.py                  # structured diagnostic bundle
│   │
│   ├── sampling/                      # enhanced sampling decisions
│   │   ├── strategy_selector.py       # vanilla / REMD / metaD / umbrella
│   │   ├── cv_designer.py             # propose collective variables
│   │   ├── ladder_designer.py         # REMD temperature ladder
│   │   ├── window_designer.py         # umbrella sampling windows
│   │   └── metad_config.py            # bias height, width, deposition rate
│   │
│   ├── ensemble/                      # multi-trajectory reasoning
│   │   ├── replica_manager.py
│   │   ├── cross_replica_stats.py
│   │   ├── markov_state.py            # MSM construction from ensemble
│   │   └── pose_clustering.py
│   │
│   ├── adapters/                      # MD engines behind the MDAdapter Protocol
│   │   ├── base.py                    # MDAdapter Protocol (engine contract)
│   │   ├── openmm_adapter.py          # OpenMM implementation
│   │   ├── gromacs_adapter.py         # GROMACS implementation (M3)
│   │   ├── mdcrow_adapter.py          # optional natural-language setup (deferred)
│   │   └── plumed_writer.py           # generate PLUMED input files (M4)
│   │
│   ├── execution/                     # HPC-aware run management
│   │   ├── slurm.py
│   │   ├── job_monitor.py             # async polling, partial failures
│   │   ├── checkpoint.py
│   │   └── walltime_planner.py
│   │
│   ├── memory/                        # campaign-level persistence
│   │   ├── trajectory_store.py        # what was run, where, with what params
│   │   ├── findings_log.py            # what we learned per round
│   │   └── provenance.py              # reproducibility metadata
│   │
│   └── tools/                         # exposed to the LLM as tool calls
│       ├── analyze_trajectory.py
│       ├── propose_followup.py
│       ├── extend_simulation.py
│       ├── switch_method.py
│       └── stop_campaign.py
│
├── configs/
│   ├── defaults.yaml
│   ├── policies/
│   │   ├── conservative.yaml          # extend before switching methods
│   │   └── exploratory.yaml           # try enhanced sampling sooner
│   └── budgets/
│       └── gpu_hours.yaml
│
├── benchmarks/
│   ├── tasks/
│   │   ├── villin_convergence.yaml
│   │   ├── trpcage_folding.yaml
│   │   └── alchemical_fep_check.yaml
│   ├── runners.py
│   └── scoring.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── notebooks/
│   ├── 01_diagnostics_demo.ipynb
│   ├── 02_sampling_decisions.ipynb
│   └── 03_full_campaign.ipynb
│
└── docs/
    ├── architecture.md                # this file
    ├── related_work.md
    ├── decision_policies.md
    └── extending.md
```

---

## Tech choices

- **LLM:** Claude via the Anthropic SDK
- **MD engines:** OpenMM (direct), GROMACS (direct), MDCrow (delegation), DynaMate (delegation, optional)
- **Memory:** SQLite + filesystem (start simple, swap later if justified)
- **HPC:** Slurm-aware execution layer (Milestone 5)
- **Analysis:** MDAnalysis, MDTraj
- **Enhanced sampling:** PLUMED

---

## See also

- `../CLAUDE.md` — behavioral guidelines for coding sessions
- `../ROADMAP.md` — milestones and what to build next
- `related_work.md` — references and detailed positioning
