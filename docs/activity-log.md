# MDPilot Activity Log

A running record of work on MDPilot. Two sections:

1. **Decisions & findings** — load-bearing choices and empirical results that future sessions should respect. Stable; append rarely.
2. **Session journal** — chronological notes per working session. Newest on top.

For *what* MDPilot is, see `architecture.md`. For *what's next*, see `../ROADMAP.md`.

---

## 1. Decisions & findings

### D1 — Scientist loop is a mechanical Python state machine (2026-05-06)
The Milestone 1 loop is plain Python: `run → diagnostics → summary → persist → scientist.decide → apply`. The LLM is invoked **only** at the `decide` step (one `messages.create` per round). Full Anthropic tool-use was considered and rejected for M1 (too much plumbing nondeterminism); a hybrid with read-only tools at decide-time is deferred to Milestone 4 when the action space gets richer.

Lives in `src/mdpilot/orchestrator/loop.py`; LLM call in `orchestrator/scientist.py`.

### D2 — Dev environment baseline (2026-05-06)
OpenMM local install, GPU available, Anthropic API key configured — all confirmed working end-to-end.

### F1 — 5 ns Trp-cage RMSD-from-first is NOT converged (2026-05-06)
Generated `benchmarks/data/trpcage/converged_ref.dcd` (5 ns NPT, TIP3P / AMBER14):
- mean RMSD-from-min ≈ 1.29 Å
- τ_int ≈ 542 ps; ESS ≈ 4.6 over 5000 1-ps frames
- `plateau_reached = False`, `well_sampled = False`

This is real physics (slow sub-state interconversion on the native basin), not a diagnostic bug. Used as a **negative** fixture (`test_scientist_extends_on_5ns_trpcage_due_to_long_autocorrelation`). The stop-path is covered by a synthetic iid series instead. If a future diagnostic change (e.g. RMSD-from-mean, equilibration discard) makes this trajectory pass, that test must flip and this finding is stale.

### D4 — M2 memory layer: per-campaign SQLite + OpenMM checkpoint (2026-05-22)
Persistence is one SQLite file per campaign at `<work_dir>/state.db` (singleton `campaign` row + `rounds` table). Trajectories and checkpoints stay on the filesystem; the DB only stores their paths plus the diagnostic report and decision. Commit order per round is **`save_checkpoint` THEN `store.append_round`** — a crash between leaves a dangling checkpoint (harmless) and an absent row, so restart re-runs the round. Resume guarantee scope: survives kills **between** completed rounds; mid-round crash recovery is out of scope.

Locked-on-init config fields: `seed`, `initial_steps`, `report_interval_steps`, `equilibration_steps`. Mismatch on resume raises `ValueError` before any OpenMM work. `max_rounds` and `max_extra_ns` are loop-control bounds and may change between invocations.

Hypothesis ledger was **deferred** from M2 to M4. The `scientist.decide` signature already accepts (and ignores) a `hypothesis_ledger` arg — left inert until `reasoning/` exists and there's something to write to it.

Lives in `src/mdpilot/memory/store.py`; resume wiring in `orchestrator/loop.py`; checkpoint helpers in `adapters/openmm_runner.py`.

### F2 — Resume currently pays the full setup tax (2026-05-23)
The M2 live resume test passed but took **3h 26m** wall time for 2 rounds × 5000 steps on this machine. Bottleneck: `build_simulation` (solvate + energy-minimize the ~few-thousand-atom system) runs on every `run_campaign` invocation, including resume — even though `load_checkpoint` immediately overwrites the resulting positions/velocities/RNG state. Minimization on resume is therefore wasted work.

Not a correctness issue; M2's between-round-kill guarantee holds. Becomes painful when resume frequency rises — most likely M5 (Slurm walltime overruns, autonomous extension). Fix at that point: cache the solvated System + serialized starting state on first init; on resume, rebuild only the `Simulation` shell and `load_checkpoint`, skip solvate+minimize.

### D3 — Anti-goals (from CLAUDE.md, recorded here for searchability)
- Do not rebuild MDCrow setup tooling — delegate via `adapters/`.
- Do not build a persistent multi-agent system; subagents are ephemeral function calls returning structured artifacts, not prose.
- Do not put raw trajectories / logs into agent context — only compact structured summaries + filesystem paths.
- Do not lock to one MD engine.
- Do not store campaign state in conversation; persist via `memory/`.

---

## 2. Session journal

### 2026-05-23 — M2 live test green, perf observation captured
- `tests/integration/test_resume_live.py` passed: round-1 DCD byte-hash unchanged across the two invocations, round 2 ran, both rounds in SQLite. M2 is verified end-to-end.
- Wall time was 3h 26m — far above the "few minutes" expectation. Root cause + deferred fix captured as F2 above (skip redundant solvate+minimize on resume, address in M5).
- M2 work still sits as one local commit `c30c5db`; push remains the user's manual step.
- **Open / next:** push c30c5db; then scope M3 (MDCrow adapter for protein setup + GROMACS runner).

### 2026-05-22 — Milestone 2 landed (SQLite memory + checkpoint resume)
- Scoped M2 with the user: SQLite + filesystem (not pure-JSON), hypothesis ledger deferred to M4 (decisions captured as D4 above).
- New module `src/mdpilot/memory/` with `store.py`: `init_campaign`, `append_round`, `list_rounds`, `get_last_round`, `get_campaign_config`. Singleton campaign row, CHECK constraint on `decision ∈ {extend, stop}`, idempotent init rejects config mismatch. 9 unit tests in `tests/unit/test_store.py`.
- Added `save_checkpoint` / `load_checkpoint` to `adapters/openmm_runner.py`. Round-trip verified on a toy 1-particle Verlet system (`tests/unit/test_checkpoint.py`, runs in ~3s — no Trp-cage setup needed for the property under test).
- Rewrote `orchestrator/loop.py` for resume: at start, validates config + loads prior rounds; if last decision was "stop", returns before any OpenMM work; otherwise rebuilds the Simulation, loads the last checkpoint, and continues. Commit order per round: checkpoint → JSON → SQLite row.
- Tests: full unit suite 30/30 green. Live resume test `tests/integration/test_resume_live.py` exists but not run this session (requires API key + ~3 min CPU).
- **M2 status:** complete to the new, honest done-criterion (survives between-round kills, hypothesis ledger deferred). Live resume test should be run once before considering M2 sealed.
- **Open / next:** run the live resume integration test; then commit; then start M3 (adapter integration — MDCrow adapter for protein setup + GROMACS runner).

### 2026-05-11 — orientation, activity log, close out M1 tests
- Recovered context from auto-memory (loop shape, Trp-cage 5 ns finding, env baseline).
- Created `docs/activity-log.md` to track decisions + per-session journal going forward.
- Walked the full repo structure vs the architecture target. M1 modules (`orchestrator/loop.py`, `orchestrator/scientist.py`, `diagnostics/*`, `adapters/openmm_runner.py`) are in place; reasoning/sampling/ensemble/execution/memory/tools subtrees are still empty per ROADMAP M2–M5.
- Reviewed the uncommitted `tests/integration/test_milestone1_live.py` diff. It replaces the unreachable "stop on real 5 ns Trp-cage" assertion with two more honest tests: extend-on-5ns (negative guard tied to F1) + stop-on-synthetic (stop-path coverage via `_summarize` on iid Gaussian noise). Kept as-is.
- Committed in two pieces:
  - `1d27c6d` tests: refine Milestone 1 done-criterion to match real Trp-cage behavior
  - `5a083a6` docs: add activity log
- `git pull --rebase` was a no-op; `git push` failed (HTTPS remote, no creds, no `gh` installed). Push left for the user to run manually.
- `.claude/settings.local.json` has new permission entries this session (git fetch/add/commit/pull/stash/push, gh auth) — left uncommitted by user choice.
- **M1 status:** the closed-loop scientist runs end-to-end on Trp-cage convergence and the done-criterion tests now match real physics. M1 is effectively complete pending the push landing.
- **Open / next:** push the two local commits; then start M2 (SQLite memory layer + resume-from-disk).

### 2026-05-06/07 — Milestone 1 skeleton landed (`c95e66d`)
Large foundational commit. New layout:
- `src/mdpilot/orchestrator/` — `loop.py` (state machine), `scientist.py` (LLM call).
- `src/mdpilot/adapters/openmm_runner.py` — OpenMM execution behind an adapter boundary.
- `src/mdpilot/diagnostics/` — `autocorrelation.py`, `block_averaging.py`, `report.py`.
- `benchmarks/` — `generate_trpcage_planted.py`, `tasks/trpcage_convergence.yaml`.
- `tests/unit/` — autocorrelation, block averaging.
- `tests/integration/` — `test_loop_live.py`, `test_scientist_live.py`, `test_milestone1_live.py`.
- Docs: `architecture.md`, `related_work.md`, `ROADMAP.md`, `README.md`.
- CLAUDE.md rewritten from a 1799-line draft down to the current concise behavioral spec.

Decisions D1, D2 and finding F1 (above) were settled in this session.
