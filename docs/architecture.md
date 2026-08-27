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

`setup_agent.py` is the worked example: bounded input (a sentence plus a fixed corpus), structured output (a task file), no need for campaign narrative, and it dies when the file is written. It runs *before* the loop, so the scientist still sees exactly one LLM call per round.

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

Two things are shown together below: what exists today, and the shape the
roadmap is heading toward. **Lines marked `[planned]` are not in the tree** —
they are milestones, not modules. Check the repository before assuming a path
here is importable; this section has drifted from reality before, and
`CLAUDE.md` sends coding sessions here to verify load-bearing decisions.

```
mdpilot/
├── README.md
├── CLAUDE.md                          # behavioral guidelines for coding sessions
├── ROADMAP.md                         # milestones
├── pyproject.toml
├── .env.example
│
├── app.py                             # Streamlit control surface (optional extra)
│
├── src/mdpilot/
│   │
│   ├── task_file.py                   # task YAML -> SystemSpec + run_campaign kwargs
│   │                                  #   (checks declared-but-fixed fields)
│   ├── run.py                         # headless campaign runner (server-side)
│   ├── forcefields.py                 # validated protein+water pairs, per engine
│   ├── preflight.py                   # checks run on the built structure, before MD
│   ├── observables.py                 # the coordinate every round is judged on,
│   │                                  #   declared as a CV (default: CA-RMSD)
│   ├── setup_agent.py                 # request -> reviewable task file (the one
│   │                                  #   LLM call outside the round loop)
│   │
│   ├── orchestrator/                  # outer loop: plan → run → analyze → decide
│   │   ├── scientist.py               # the single LLM call per round
│   │   └── loop.py                    # the mechanical campaign state machine
│   │
│   ├── knowledge/                     # the scientist's prompt, as retrievable prose
│   │   ├── role.md                    # who it is + the four structured inputs
│   │   ├── phase_vanilla.md           # equilibrium rubric + pivot rule
│   │   ├── phase_metad.md             # free-energy rubric
│   │   ├── action_switch_cv.md        # when to revise the biased coordinate
│   │   ├── cv_vocabulary.md           # CV types, arities, bounded-vs-unbounded
│   │   ├── output_contract.md         # extra_ns / ledger_note / reason
│   │   ├── setup_role.md              # [setup agent] what it emits, and for whom
│   │   └── forcefield_guide.md        # [setup agent] how to choose a force field
│   │
│   ├── diagnostics/                   # mechanical convergence + adequacy (no LLM)
│   │   ├── block_averaging.py         # Flyvbjerg-Petersen SEM plateau
│   │   ├── autocorrelation.py         # Geyer IMPS tau_int + ESS
│   │   ├── exploration.py             # pinned in one basin, or visiting several?
│   │   ├── free_energy.py             # HILLS -> FES, well-tempered convergence
│   │   └── report.py                  # structured diagnostic bundle
│   │
│   ├── sampling/                      # enhanced sampling decisions
│   │   ├── cv_designer.py             # CV proposal -> resolved atom indices
│   │   └── bias_designer.py           # SIGMA / HEIGHT / PACE from the trajectory
│   │
│   ├── adapters/                      # MD engines behind the MDAdapter Protocol
│   │   ├── base.py                    # MDAdapter Protocol (engine contract)
│   │   ├── system_spec.py             # engine-agnostic "what to simulate"
│   │   ├── openmm_adapter.py          # OpenMM implementation
│   │   ├── gromacs_adapter.py         # GROMACS implementation (M3)
│   │   └── plumed_writer.py           # generate PLUMED input files (M4)
│   │
│   └── memory/                        # campaign-level persistence
│       └── store.py                   # SQLite: campaign / rounds / ledger
│
├── benchmarks/
│   ├── tasks/
│   │   ├── trpcage_convergence.yaml   # M1 convergence-judgment task
│   │   └── cln025_folding.yaml        # M4 done-criterion task
│   ├── generate_trpcage_planted.py    # planted reference trajectories
│   └── run_cln025.py                  # M4 campaign runner + done-criterion check
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── docs/
    ├── architecture.md                # this file
    ├── activity-log.md                # decisions, findings, session journal
    └── related_work.md
```

Planned, per `ROADMAP.md` — none of these exist yet:

```
src/mdpilot/
├── reasoning/                         # [planned] judgment core split out of scientist.py
│   ├── hypothesis.py                  # [planned] ledger currently lives in memory/store.py
│   ├── decision_policy.py             # [planned] currently the prompt + loop branches
│   └── rubrics.py                     # [planned]
│
├── sampling/
│   ├── strategy_selector.py           # [planned] vanilla / REMD / metaD / umbrella
│   ├── ladder_designer.py             # [planned] REMD temperature ladder
│   └── window_designer.py             # [planned] umbrella sampling windows
│
├── ensemble/                          # [planned] M6-era multi-trajectory reasoning
│   ├── replica_manager.py             # [planned]
│   ├── cross_replica_stats.py         # [planned]
│   └── markov_state.py                # [planned]
│
├── execution/                         # [planned] M5 HPC-aware run management
│   ├── slurm.py                       # [planned]
│   ├── job_monitor.py                 # [planned]
│   └── walltime_planner.py            # [planned]
│
└── adapters/
    └── mdcrow_adapter.py              # [planned, deferred indefinitely — see D5]

configs/                               # [planned] policy + budget YAML
notebooks/                             # [planned] demo notebooks
```

**Deliberately not built, though earlier drafts of this file listed them.**
`tools/` — the scientist has no tool-dispatch layer; the loop calls
`decide()` directly and everything else is deterministic Python. That is what
"the LLM is called once per round" means in practice — plus one call per
*campaign*, in `setup_agent.py`, before the loop starts.
`diagnostics/effective_sample_size.py` — ESS is returned by
`autocorrelation.py` rather than living alone. `orchestrator/planner.py`,
`replanner.py` and `state.py` — `loop.py` is the state machine and there is no
separate plan artifact yet.

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
