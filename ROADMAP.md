# MDPilot Roadmap

Each milestone is shippable on its own. Reach a working end-to-end state before adding complexity. Don't skip ahead — the closed loop has to actually run before any of the upper milestones make sense.

For architecture context, see `docs/architecture.md`.

---

## Milestone 1 — Day-1 skeleton

**Goal:** one working closed loop on Trp-cage convergence.

- `orchestrator/scientist.py` — minimal Claude-powered agent loop
- `diagnostics/block_averaging.py`, `autocorrelation.py`
- `adapters/openmm_runner.py` — direct execution of a hardcoded OpenMM script
- `benchmarks/tasks/trpcage_convergence.yaml`
- One end-to-end run: simulate → diagnose → decide-extend-or-stop

**Done when:** the scientist takes a Trp-cage convergence task, runs OpenMM, computes block-averaged RMSD with autocorrelation analysis, decides "extend" or "stop", and the decision is correct on a planted under-converged trajectory.

## Milestone 2 — Memory layer + resume from disk

**Goal:** the scientist remembers across rounds and survives restarts.

- SQLite memory layer (`memory/store.py`) — per-campaign DB, `campaigns` + `rounds` tables
- OpenMM checkpoint save/load wired through the adapter
- `run_campaign` resume-aware: detects existing rounds, validates config, loads the last checkpoint, continues

**Done when:** a multi-round campaign survives a between-round kill — restart on the same `work_dir` picks up at round k+1 without redoing rounds 1..k, and the findings log (per-round diagnostic report + decision) is intact.

*Deferred from this milestone:* hypothesis ledger persistence. Pushed to M4 when `reasoning/` lands and there's something to write to it. The `scientist.decide` signature already accepts an inert `hypothesis_ledger` arg.

## Milestone 3 — Adapter integration

**Goal:** stop hardcoding setup; delegate to existing tooling.

- `adapters/mdcrow_adapter.py` — delegate protein setup
- `adapters/gromacs_runner.py` — cross-engine execution

**Done when:** the same Trp-cage benchmark task runs through either MDCrow→OpenMM or direct→GROMACS without touching the scientist code.

## Milestone 4 — Sampling strategy decisions

**Goal:** the scientist knows when vanilla MD isn't enough.

- `sampling/strategy_selector.py` (vanilla / REMD / metaD / umbrella)
- `sampling/cv_designer.py` for metadynamics
- `adapters/plumed_writer.py`

**Done when:** on a benchmark designed to require enhanced sampling (e.g., a small protein with a known slow conformational transition), the scientist (a) detects that vanilla MD is inadequate, (b) selects metadynamics with a reasonable CV, and (c) the resulting run actually crosses the barrier.

## Milestone 5 — HPC execution

**Goal:** real production runs, not laptop demos.

- `execution/slurm.py`
- `execution/job_monitor.py` — async polling, partial-failure recovery
- `execution/walltime_planner.py` — checkpoint-aware planning

**Done when:** a multi-day Slurm campaign survives at least one node failure, one walltime overrun, and one walltime extension decided autonomously by the scientist based on convergence diagnostics.

## Milestone 6 — Benchmark release

**Goal:** the field has a shared way to measure progress on outer-loop reasoning.

- Curated graded eval tasks with reference solutions
- Documented scoring methodology
- Public leaderboard
- Paper

---

## Deliberately out of scope

These are not in the roadmap, on purpose. Add only with strong justification.

- **Web UI.** Terminal + notebooks only until at least Milestone 6.
- **Free energy methods beyond MM/PB(GB)SA.** Out of scope until enhanced sampling is solid.
- **ML potentials (MACE, Allegro, ANI).** Future work; classical force fields are sufficient to validate the closed-loop reasoning, which is the actual contribution.
- **Multi-user / shared deployment.** Single-researcher tool until adoption justifies the complexity.
- **Custom force field development.** Use existing parameter sets only.
- **Coarse-grained MD.** All-atom only for v1.
