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

### D6 — Strategic plan: ambitious framework, chignolin as M4 forcing function, ice as showcase (2026-05-24)
User goal: MDPilot as a general research tool, with **ice-philicity** (heterogeneous nucleation on surfaces) as one of multiple intended showcases — first one funded at ~$100 of AWS compute. Decision was to build M4+M5 thoroughly *before* applying to the ice campaign, so that the showcase has real capability behind it.

**Architectural risk surfaced:** building M4 without a forcing function (a real second use case) repeats the design-from-abstraction trap we explicitly avoided with `MDAdapter` (which we designed *after* implementing two engines). Ice nucleation is brute-force vanilla MD with seeded clusters and barely exercises enhanced sampling — using it as M4's forcing function would yield M4 abstractions that fit neither problem.

**Resolution:** M4 development uses **chignolin** as the forcing function (10-residue mini-protein, ~5 μs vanilla folding, ~tens of ns with metaD; also serves as an M5 forcing function since it's beyond local CPU). Ice campaign is the *showcase* that uses M2+M3+M5 + a new ice-diagnostic module; M4 capabilities are available in the toolbox but only weakly exercised by the ice problem itself.

**Ordered plan replacing the literal roadmap order:**
1. F2 perf fix — required regardless. (Done in this commit.)
2. SystemSpec generalization — adapters take arbitrary structure/FF, not Trp-cage-hardcoded.
3. Hypothesis ledger activation — minimum-viable structured findings store.
4. M4 full build with chignolin as forcing function: PLUMED install, writer, scientist multi-tool refactor, strategy selector, CV designer, ledger writes.
5. M5 lite AWS launcher: EC2 + S3 sync; verify on M4 benchmark.
6. Ice campaign showcase: ice diagnostic module, seeded-nucleation setup, real $100 AWS campaign.

Total estimated scope ~15-20 sessions. Core principle: core MDPilot stays general; problem-specific work (ice diagnostics, ice setup) lives in `science/<problem>/` modules that *use* the core rather than modify it.

### D5 — M3 adapter design + MDCrow anti-goal refined (2026-05-23)
**Adapter Protocol** (`src/mdpilot/adapters/base.py`): the loop talks to engines exclusively through `MDAdapter`, a 5-method + 2-property contract (`prepare`, `start`, `run_steps`, `save_checkpoint`, `load_checkpoint`; `topology_path`, `trajectory_extension`). Adapters are stateful per-campaign objects constructed with `work_dir` and `seed`; lifecycle is `prepare()` → `start()` → repeat[`load_checkpoint`?, `run_steps`, `save_checkpoint`]. The trajectory extension is engine-owned because mdtraj infers the reader from the file suffix (OpenMM = `.dcd`, GROMACS = `.xtc`).

**GROMACS adapter pinned to:** amber99sb-ildn force field (closest GROMACS-distributed analog to OpenMM's amber14-all; small side-chain torsion differences accepted for cross-engine convergence-judgment work), TIP3P via spc216 seed lattice, 1.0 nm cubic padding, 0.15 M NaCl (neutralized), 300 K via stochastic-dynamics integrator with `tau-t = 1.0 ps`, 2 fs timestep, H-bonds constrained (LINCS), PME with 1.0 nm cutoff. Subprocess pattern: every `gmx` call goes through `_run_gmx()` which captures stderr and raises informatively on non-zero exit.

**MDCrow anti-goal refined to the *functional* reading** (CLAUDE.md updated): the thing not to rebuild is MDCrow's LLM-orchestrated **agent layer**, not the underlying setup libraries themselves. Adapters that call PDBFixer / `gmx pdb2gmx` / etc. deterministically from a structured system spec do not violate the spirit of the anti-goal. MDCrow integration is deferred indefinitely; it would re-introduce multi-agent natural-language dialogue, which violates the "no persistent agents communicating in natural language" anti-goal. Revisit only when a campaign genuinely needs setup-from-natural-language that the scientist can't generate structured itself.

**SystemSpec extraction deferred** (was task #7). Both adapters currently hold their own engine-specific constants. They're *aligned* by inspection (300 K, 2 fs, 1 nm, 0.15 M etc.) but not *shared*. If a user wants to change e.g. temperature for a campaign they have to edit both files. Acceptable cost until a real campaign forces parameter variation; address then.

**Cross-engine done-criterion met:** `tests/integration/test_cross_engine_live.py` runs the loop + scientist end-to-end via the GROMACS adapter and gets a well-formed diagnostic report + valid `extend` decision. OpenMM-through-loop was already covered by M1 tests. Engine independence proven.

### F2 — Resume currently pays the full setup tax (2026-05-23) — RESOLVED 2026-05-24
The M2 live resume test passed but took **3h 26m** wall time for 2 rounds × 5000 steps on this machine. Bottleneck: `build_simulation` (solvate + energy-minimize the ~few-thousand-atom system) runs on every `run_campaign` invocation, including resume — even though `load_checkpoint` immediately overwrites the resulting positions/velocities/RNG state. Minimization on resume is therefore wasted work.

Not a correctness issue; M2's between-round-kill guarantee holds. Becomes painful when resume frequency rises — most likely M5 (Slurm walltime overruns, autonomous extension). Fix at that point: cache the solvated System + serialized starting state on first init; on resume, rebuild only the `Simulation` shell and `load_checkpoint`, skip solvate+minimize.

**Resolution (2026-05-24):** addressed earlier than planned because D6 reframes the deployment target to AWS (where every spot-instance restart costs money, not just wall time). Both adapters now have idempotent `start()`:
- `OpenMMAdapter`: first call caches `system.xml` + `initial_state.xml` + `topology.pdb` to `<work_dir>/cache/`; subsequent calls deserialize the System and post-minimization State directly, skipping Modeller/solvate/createSystem/minimize entirely.
- `GROMACSAdapter`: first call runs the full pdb2gmx → solvate → genion → minimize pipeline; subsequent calls detect `setup/em.gro` + `setup/topol.top` + the topology PDB and short-circuit.

GROMACS side verified live (`test_gromacs_adapter_live.py` extended with em.gro mtime assertion; second `start()` adds ~ms not ~minutes). OpenMM side verified by code review; the next OpenMM live-resume run will be the real measurement (expected: ~3.5h → ~1.5h, since only the first `run_campaign` invocation pays the full setup tax).

### F3 — Planted reference trajectories are stale as of the equilibration fix (2026-08-11)
`benchmarks/data/trpcage/converged_ref.dcd` and `under_converged.dcd` were generated by the pre-equilibration pipeline: **NVT throughout, no barostat, and velocities left at zero after minimization** (the OpenMM adapter never called `setVelocitiesToTemperature`, so both trajectories open with a 0 K → 300 K thermalization transient driven only by the Langevin thermostat). `under_converged.dcd` starts *directly* from the minimized state, so that 50 ps fixture is mostly transient.

Production is now NPT from an equilibrated start, so **new runs are not drawn from the same ensemble as these fixtures**. Consequences:
- Tests that *read* the planted DCDs still pass — the files are unchanged and the diagnostics are unchanged. Nothing is broken today.
- Any comparison between a freshly generated trajectory and these references is invalid. Regenerate before drawing that comparison: `--traj under` ≈ 6 min, `--traj converged` ≈ 9 h CPU.
- F1's numbers (τ_int ≈ 542 ps, ESS ≈ 4.6, `plateau_reached=False`) describe the *old* trajectory. They were the basis for the M1 negative fixture; whether an NPT, properly-equilibrated 5 ns Trp-cage still fails the convergence rubric is **unverified**. F1's own escape clause applies — if regeneration makes it pass, that test flips and F1 is stale.

Deliberately not regenerated in this session: 9 h of CPU is a poor trade before the M4 live work, which will want chignolin fixtures anyway.

### F4 — `SIGMA = spread/3` needs a trajectory that has actually sampled the basin (2026-08-11) — GUARDED 2026-08-11
Measured on the first live metaD run: sizing the Rg hill width off a **10 ps** vanilla Trp-cage trajectory gives `SIGMA ≈ 0.0018 nm`. Backbone Rg barely moves in 10 ps, so spread/3 measures thermal jitter, not basin width. Hills that narrow never overlap → the accumulated bias at each new deposit stays ≈ 0 → well-tempering has nothing to damp and heights never decay. WT-MetaD silently degenerates into plain metaD that fills nothing.

At `SIGMA = 0.02 nm` (a physically sensible Rg width for a 20-residue protein) the same run behaves correctly: heights decay 1.386 → 0.577 kJ/mol over 30 hills while the bias accumulates to 19.7 kJ/mol.

The heuristic is not wrong — σ ≈ intrabasin fluctuation / 3 is standard — but it is only valid once the vanilla phase has sampled the basin. The `switch_to_metad` decision fires precisely when the CV looks *pinned*, which is also when its measured spread is least trustworthy.

**Guard (2026-08-11):** physical per-CV-type floors replace the old `_SIGMA_FLOOR = 1e-3`, which was a PLUMED-validity epsilon (keep SIGMA > 0) and did nothing for this. Floors are the narrowest hill worth depositing for each coordinate: **0.02 nm** for `distance` and `gyration` (~5 deposits across a 0.1 nm basin, enough overlap to fill), **0.15 rad** for `torsion` (torsions span 2π; the conventional dihedral range is 0.1–0.35 rad, and sharing the linear floor would have mis-sized every dihedral bias by an order of magnitude). `size_sigma()` returns `(sigma, floored)`; the flag rides on `MetadynamicsBias.sigma_floored` and makes `PlumedInput.render()` emit a `# NOTE:` block, so a substituted width is never indistinguishable from a measured one when the plumed.dat is read back later.

Chosen over the τ_int precondition considered alongside it: a low-ESS check correctly identifies that the spread is untrustworthy but yields no better number, and its only honest response — refuse the pivot and extend — would need a new path through the loop. The floor gives a defensible width now; the reliability signal is the `floored` flag.

Verified against the original input: the same `smoke.dcd` + backbone-Rg proposal that measured 0.00166 nm now yields SIGMA = 0.02 nm with the note in the rendered file. **The floor is a floor, not an override** — a CV measuring wider keeps its measured value (unit-tested both sides).

Still open on sizing: there is no *upper* guard. A CV that drifted monotonically gives an inflated spread and hills wider than the features being resolved. Not observed yet, so not guarded.

### F5 — PLUMED resolves FILE= against the process CWD, and buffers output (2026-08-11)
Two defects found by the first live run, both fixed in the same session:
- A bare `FILE=HILLS` wrote the campaign's deposited bias into whatever directory python was started from — the repo root, in that first run, complete with `bck.0.*` backups. Concurrent campaigns would collide and resume could never find the previous HILLS. `PlumedInput` now requires an **absolute** `output_dir` and prefixes it onto every file PLUMED writes; a relative one raises.
- PLUMED buffers HILLS/COLVAR and flushes only when the context is finalized, so a campaign killed mid-round loses every deposited hill and a resume reads what looks like an empty file. `PlumedInput.render()` now emits `FLUSH STRIDE=<print_stride>`.

Also worth knowing when reading HILLS: PLUMED records well-tempered hill heights pre-scaled by **γ/(γ-1)**, so the first hill reads `W0·γ/(γ-1)` (1.2472 × 10/9 = 1.3857), not `W0`.

### D3 — Anti-goals (from CLAUDE.md, recorded here for searchability)
- Do not rebuild MDCrow setup tooling — delegate via `adapters/`.
- Do not build a persistent multi-agent system; subagents are ephemeral function calls returning structured artifacts, not prose.
- Do not put raw trajectories / logs into agent context — only compact structured summaries + filesystem paths.
- Do not lock to one MD engine.
- Do not store campaign state in conversation; persist via `memory/`.

---

## 2. Session journal

### 2026-08-11 — F4 guarded; HILLS read back into a free-energy surface
Two of the three blockers on the M4 done-criterion run. Both were cases of a statistic reporting success on a run that did nothing.

- **F4 σ guard** (details in the finding). Physical per-CV-type floors — 0.02 nm for `distance`/`gyration`, 0.15 rad for `torsion` — replacing the old `1e-3` PLUMED-validity epsilon. `size_sigma()` returns `(sigma, floored)`; the flag rides on `MetadynamicsBias.sigma_floored` and makes `PlumedInput.render()` emit a `# NOTE:` block, so a substituted width can never be mistaken for a measured one when the plumed.dat is read back. Verified against the original F4 input: `smoke.dcd` + backbone Rg went 0.00166 nm → 0.02 nm with the note present.
- **`diagnostics/free_energy.py`** — the deposited bias is no longer write-only. `sum_hills()` shells out to `plumed sum_hills` (delegated, not reimplemented: PLUMED pre-scales WT heights by γ/(γ-1) and uses stretched Gaussians, so a hand-rolled sum would silently disagree with the file it integrates). `load_fes` / `load_colvar` parse the results; `FreeEnergySurface` yields basins, barrier height and depth; `fes_drift_kj_per_mol` implements the standard WT convergence test over the overlapping grid region; `count_recrossings` measures barrier crossings with hysteresis. `metad_report()` returns the same JSON-serializable scalars-and-paths contract as `make_report`.
- **`fes_converged` requires low drift AND at least one recrossing.** Drift alone is trivially satisfied: a walker that never leaves its basin produces a surface that stops changing immediately, because nothing new is being sampled. Reporting that as converged is the metastable-basin error one level up — the same shape as the problem the exploration diagnostic exists to solve. Confirmed on the real Trp-cage HILLS, which reports drift 1.83 kJ/mol (under kT) with zero recrossings and is now correctly *not* converged.
- **Two bugs the tests caught, both in prominence/threshold logic:**
  1. Basin prominence was measured against the *global* maximum on each side, so any ripple at the bottom of a deep well inherited the prominence of the whole well and counted as a basin. Now measured against the enclosing local maximum (`_uphill_peak`). Follow-on: the filter then discarded the global minimum when it sat inside a ripple, leaving a surface with no basins at all — the deepest point is now always a basin by definition.
  2. Recrossings were counted against the basin *minima* as thresholds, so a walker oscillating around a basin without landing exactly on the minimum scored zero. `basin_thresholds()` now places the dividing surfaces halfway between the crest and each minimum, committor-style.
- **PLUMED naming gotcha:** with `--stride`, sum_hills appends the index *and* another `.dat` to `--outfile`, giving `fes.dat0.dat`, not `fes_0.dat`. A glob written for the latter silently finds nothing and falls back to a single surface, which would have disabled the drift check without failing.
- Tests: 163 unit green (was 137; +21 free energy, +5 σ floor), 162 + 1 skipped in `.venv`. New `tests/integration/test_free_energy_live.py` (7 tests) runs real `sum_hills` against synthesized two-basin HILLS — clustered hills integrate to two basins with a positive barrier, a shuttling walker reports ≥3 recrossings, a pinned one is refused a convergence verdict.
- **Not wired into the loop yet.** `metad_report` exists but nothing calls it: biased rounds still receive `make_report`'s equilibrium statistics. Doing so needs a decision on whether the equilibrium verdicts (`plateau_reached`, `ess`, `well_sampled`, `exploring`) are suppressed or merely relabelled on a biased round, and a matching change to the scientist's system prompt — which is the agent's reasoning loop, so it wants sign-off per CLAUDE.md §1.
- **Open / next:** wire `metad_report` into `loop.py` for biased rounds + update the scientist prompt; then chignolin (CLN025 / 5AWL) and the done-criterion campaign.

### 2026-08-11 (latest) — GPU by default, README rewrite (commit `70a4b23`)
- **Platform selection.** `resolve_platform()` walks `CUDA → HIP → OpenCL → CPU` and returns the first that is *usable*, not merely registered — it builds a throwaway two-particle Context with a real `NonbondedForce` and takes a step. `lru_cache`d; `OpenMMAdapter(platform=...)` pins it explicitly. Both hardcoded `Platform.getPlatformByName("CPU")` sites are gone.
- **Why the probe exists.** The conda env registered CUDA and then died at kernel load with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: conda-forge had installed `cuda-nvrtc 13.3.33` while the driver supports CUDA 13.2, so OpenMM's JIT emitted PTX the driver could not consume. Selecting on registration alone would have handed back a platform that fails *mid-campaign, after setup is already paid for*. Fixed the env with `micromamba install -n mdpilot -c conda-forge "cuda-version=13.2"`; the pin is documented in README because a fresh `micromamba create` reproduces the mismatch.
- **Mixed precision on GPU.** `{"Precision": "mixed"}` for CUDA/HIP/OpenCL. OpenMM's GPU default is single, which would make a GPU run quietly *less* accurate than the CPU run it replaces — a silent quality regression is worse than a slow one.
- **Measured** on the real 4810-atom Trp-cage system: **11.4 ns/day (CPU) → 298.9 ns/day (CUDA)**, 26×. Through the real adapter, the metaD + equilibration live suites went from ~10.5 min to **61 s** for 10 tests.
- Tests: +6 unit for platform selection, including the registered-but-broken fallback and a bogus-platform probe. 130 unit green in conda, 129 + 1 skipped in `.venv`.
- README rewritten around demonstrated capability, with the M4 done-criterion explicitly marked unmet. `.gitignore` extended for PLUMED output (defence-in-depth against the F5 CWD bug), GROMACS derived files, checkpoints, and presentation artifacts.
- **Not done:** the GROMACS adapter still shells out to the Ubuntu `gmx`, which is a CPU-only build. GPU applies to the OpenMM path only.
- **Planning consequence.** At ~300 ns/day the tens-of-ns metaD run chignolin needs is a few hours locally, not the multi-day CPU job that made D6 step 5 (AWS launcher) look like a prerequisite. **The M4 done-criterion now looks reachable on this machine**, so M5 stays deferred.
- **Open / next:** F4 σ guard (blocking — narrow hills mean WT-MetaD fills nothing); read HILLS back via `plumed sum_hills` so post-pivot rounds stop being judged by the unbiased convergence rubric; chignolin (CLN025 / 5AWL) `SystemSpec` + `task_expectation`; then the done-criterion campaign with the LLM in the loop.

### 2026-08-11 (later) — M4 step 5a: PLUMED runtime live, first metadynamics ever executed
Closes the "written but never run" gap on the whole M4 bias path. Prior to this session no hill had ever been deposited by this codebase.

- **Environment.** `openmm-plumed` is not on PyPI and PLUMED is not in the Ubuntu repos, so the plain `.venv` cannot host it. User chose conda-forge over the two alternatives (OpenMM's native `app.metadynamics`, which would have meant dropping PLUMED for the OpenMM path and refactoring the Protocol's `plumed_input: str` into a structured bias spec; or a source build). Created via micromamba: env `mdpilot` with `python=3.12 openmm openmm-plumed plumed` + conda-forge deps + `pip install -e .`. PLUMED 2.9, OpenMM 8.4.0. Note `~/micromamba` is an unrelated pre-existing binary occupying the conventional root prefix, so `MAMBA_ROOT_PREFIX=$HOME/.micromamba` is required.
- **Two environments now exist and they are not equivalent:** `.venv` (pip OpenMM 8.5.1, no PLUMED) and the conda env (OpenMM 8.4.0, PLUMED). The unit suite is green in both — 124 in conda, 123 + 1 skipped in `.venv`, where the real-PlumedForce test skips. Anything touching metaD must run in the conda env.
- **Test-suite bug surfaced by the install.** Three tests in `test_openmm_plumed_hook.py` asserted `openmmplumed` was genuinely absent, so they stopped testing anything the moment the runtime appeared. Reworked to *simulate* absence via `sys.modules[...] = None`, plus a new test that a real runtime returns a real `PlumedForce` (the stub cannot catch signature drift).
- **Two real defects found by running it** — see F5: PLUMED resolving `FILE=` against the process CWD (campaign output landed in the repo root), and PLUMED buffering output until finalization (a killed round loses all hills). Both fixed; `PlumedInput` now takes a required absolute `output_dir` and emits `FLUSH`.
- **One scientific finding** — see F4: `SIGMA = spread/3` off a short vanilla run underestimates the basin width badly enough that WT-MetaD degenerates to plain metaD. Blocks the done-criterion run until guarded.
- **Verified live** (`tests/integration/test_metad_live.py`, 6 tests, ~9 min, no API key): full mechanical pivot vanilla → `design_cv` → `design_bias` → plumed.dat → biased run. HILLS lands in the campaign dir, 30 hills at PACE=500 over 30 ps, `biasf=10` echoed by PLUMED, heights decay 1.386 → 0.577 kJ/mol, `metad.bias` climbs 0 → 19.7 kJ/mol. The decay and the bias growth together are the evidence that well-tempering is live and the force is genuinely coupled to the dynamics, not merely rendered into a file.
- **M4 done-criterion still unmet.** What is proven is that the machinery executes and WT-MetaD behaves correctly. What is not: the scientist (LLM) detecting inadequacy and proposing the CV on a system that genuinely requires enhanced sampling, and a real barrier crossing. That needs chignolin, the σ guard from F4, and materially more compute — 30 ps of Trp-cage at a folded Rg of ~0.70 nm crosses nothing, nor should it.
- **Open / next:** F4 σ guard; chignolin `SystemSpec` + `task_expectation`; GPU platform (both OpenMM code paths still hardcode `Platform.getPlatformByName("CPU")`, which will not do for a folding-timescale run); then the done-criterion campaign.

### 2026-08-11 — Equilibration, velocity-init parity, and well-tempered metaD (pre-M4 correctness pass)
Three physics defects fixed before resuming M4. All three were code that ran cleanly and produced wrong physics — the failure mode CLAUDE.md §4 exists to catch.

- **Proper equilibration in both adapters.** Setup was minimize → production. Now minimize → staged NVT heating (50 K → 300 K) → NPT density relaxation → cache. OpenMM ramps a 6-stage staircase via `integrator.setTemperature` between `step()` calls; GROMACS uses `annealing = single` over the same span (linear interpolation vs staircase is the one accepted asymmetry — same endpoints, same step budget). Stage lengths default to 100 ps + 100 ps and are constructor-overridable (`nvt_steps` / `npt_steps`) so the test suite doesn't pay 200 ps of CPU MD. Setting both to 0 is an explicit test-only opt-out.
- **Velocity-initialization parity.** OpenMM never called `setVelocitiesToTemperature`, so the cached state had *exactly zero* velocities and every production run began at 0 K, warming under the thermostat — while GROMACS had always seeded velocities via `gen-vel`. The two engines were physically inequivalent at t=0, which quietly undercut the M3 cross-engine claim. OpenMM now seeds velocities (seeded RNG) at the bottom of the ramp; GROMACS's `gen-vel` moved into the NVT stage. After equilibration GROMACS leaves an `npt.cpt` behind, so the `run_steps` cold-start `gen-vel=yes` override is now reachable only on the equilibration-disabled path.
- **Production is NPT.** User decision, taken with the fixture cost understood (see F3). `MonteCarloBarostat` (1 bar, 300 K, interval 25, seeded) is added to the System *before* serialization so it survives into production; GROMACS gets `pcoupl = C-rescale` in the npt and production mdps. Density is no longer frozen at whatever `addSolvent` produced.
- **Well-tempered metadynamics.** `MetadynamicsBias` gained `bias_factor` (γ, PLUMED `BIASFACTOR`) and `temperature_k` (`TEMP`), both validated (γ > 1, T > 0). γ defaults to 10 → ΔT = 2700 K, flattening barriers to ~γk_BT ≈ 25 kJ/mol. Plain metaD is no longer constructible: it overfills basins and never converges to the FES, so there is no reason to keep it reachable. `design_bias` passes the thermostat temperature through rather than letting PLUMED assume one. HEIGHT is now explicitly the *initial* hill height W0.
- **Equilibration is auditable.** Both engines write per-stage traces — OpenMM `cache/equilibration_{nvt,npt}.csv` via `StateDataReporter`, GROMACS `nvt.log`/`npt.log` + `.edr` (the equilibration mdps had `nstenergy = 0`, which would have made the stages uninspectable).
- **Verified live, not just unit-tested.** OpenMM Trp-cage (4810 atoms, 6 ps NVT + 6 ps NPT): velocities nonzero (was exactly 0), 295.0 K against σ_T ≈ 3.4 K, box 52.7 → 48.5 nm³, density 0.927 → 1.009 g/mL, one barostat in the cached System. NVT volume constant throughout, confirming the barostat is genuinely off during heating. GROMACS full chain pdb2gmx → … → em → nvt → npt in 38.9 s, all five stage outputs present, `_last_cpt` = `npt.cpt`. Real WT plumed.dat rendered through `design_cv` + `design_bias` + `plumed_writer`: `BIASFACTOR=10 TEMP=300`.
- Tests: 121 unit green (was 105; +16 across equilibration wiring and WT-MetaD validation) plus a new `tests/integration/test_equilibration_live.py` (4 tests, ~2 min, no API key — it pins the zero-velocity regression directly, computing T from velocities and masses because the cached State is serialized without energies).
- **Known gaps left standing** (all pre-existing, none in this work order): metaD phase still starts from the cached equilibrated state rather than the vanilla endpoint; no `RESTART` in `plumed_writer`, so a resumed biased phase drops deposited HILLS; HILLS is written but never read back, so biased rounds are still judged by the unbiased convergence rubric; `PACE` is still a hardcoded 500 rather than derived from the CV's own correlation time; nothing checks the proposed CV is the slow coordinate. Diagnostics still do no burn-in discard — less severe now that production starts equilibrated, but not gone.
- **Open / next:** M4. `openmmplumed` is still not installed and no metaD has ever executed, so the D6 step-4/5 boundary is unchanged.

### 2026-07-12 — M4 step 4: in-place metaD pivot wired (loop + bias_designer + persistence)
- Closes the loop-wiring half of M4. When `decide()` returns `switch_to_metad`, `run_campaign` no longer terminates — it **pivots in place** and runs the biased phase in the same call. This was the first of the two step-4 design judgments the step-3 handoff deferred; **decided: in-place pivot** (vs two-phase reinvocation), because the campaign is one logical thing and a two-phase seam would leak into the user API (the whole thesis is "I called `run_campaign`, MDPilot decided what to simulate"). User signed off.
- Second deferred judgment — per-round phase tracking — **decided: nullable `plumed_dat_path` column** (vs a `phase` enum or a directory convention). It is both the phase marker (NULL ⇒ vanilla) *and* the audit artifact (reading it back recovers the exact bias that drove the round). Migration is a plain `ALTER TABLE ADD COLUMN` (no CHECK change), added after the step-3 metaD rename-create-copy-drop so it also covers the table that migration recreates.
- New `sampling/bias_designer.py`: resolved CV + prior trajectory → sized `MetadynamicsBias`. SIGMA = spread/3 (floored to 1e-3 so a pinned CV can't yield SIGMA=0, which PLUMED rejects); HEIGHT = 0.5·k_B·T (~1.25 kJ/mol at 300 K, matches the handoff's conservative number); PACE = 500. Torsions use circular std (linear stddev is wrong across the ±π wrap); distance/gyration use ordinary stddev. Same LLM-emits-intent / code-resolves-physics boundary as `cv_designer`.
- Loop flow: `_build_plumed_input` (proposal → `design_cv` → `design_bias` → `PlumedInput.render`) and `_pivot_to_metad` (write `plumed.dat`, construct + start the biased adapter). Biased adapter built via an injectable `biased_adapter_factory` (defaults to `OpenMMAdapter(..., plumed_input=...)` over the same `SystemSpec`) — keeps the loop engine-agnostic and lets unit tests drive the pivot with fakes. Resume handles both the pivot-resume (last row = switch → build bias from stored proposal + trajectory, start fresh) and mid-metaD-phase resume (last row has `plumed_dat_path` → rebuild biased adapter from the persisted plumed.dat, load that round's biased checkpoint) cases.
- Tests: 105 unit green (was 95: +6 bias_designer, +3 store plumed column/migration, +3 loop pivot orchestration; −1 obsolete "switch is terminal" resume test, whose behavior is now the pivot test). Loop pivot tests use fake adapters + stubbed `decide`/`make_report`/`_build_plumed_input` so they need neither OpenMM nor PLUMED. Separately verified the *real* render chain (`design_cv`+`design_bias`+`plumed_writer`) against `benchmarks/data/trpcage/smoke.dcd` — produces valid PLUMED syntax (gyration over the backbone atom set, SIGMA from the real Rg fluctuation, HEIGHT 1.247 kJ/mol). No PLUMED runtime is needed for text generation.
- **Two known scope boundaries (honest, deferred — both need the PLUMED environment to fix/verify):**
  1. **MetaD phase starts from the cached minimized state, not the vanilla endpoint.** The vanilla checkpoint is not portable across the added `PlumedForce`, so the biased phase restarts from the minimized structure and re-equilibrates under bias. SIGMA is still sized from the vanilla basin sampling (the relevant width), so this is acceptable for the milestone. Follow-up: warm-start the metaD phase from the vanilla endpoint via a portable State handoff.
  2. **MetaD-phase *resume* does not yet reload PLUMED HILLS.** `plumed_writer` emits no `RESTART`, so resuming a biased phase would not continue the deposited bias correctly. Beyond step-4 wiring scope and untestable without a PLUMED runtime; fix alongside the AWS/PLUMED live work (D6 step 5).
- **Open / next:** the M4 done-criterion ("the biased run actually crosses the barrier") is still unmet — it requires a PLUMED runtime + real compute and is the D6-step-5 AWS-lite work. The remaining verification is a live `test_metad_pivot` on chignolin under PLUMED: force a `switch_to_metad`, confirm the biased trajectory crosses a barrier vanilla could not. Nothing else in the pivot is unverified locally.

### 2026-06-08 — M4 step 3: scientist action-space refactor (extend | stop | switch_to_metad)
- Conceptually richest piece of M4. Scientist now picks among three actions per round; on `switch_to_metad` it emits a structured `metad_proposal` in the same shape `cv_designer.CVProposal` consumes (cv_type ∈ {distance, torsion, gyration}, MDTraj selection strings, label). Bias parameters (σ, height, pace) stay out of the LLM's output — derivable deterministically from the prior trajectory and rule-of-thumb constants; the `bias_designer` helper is scheduled for step 4. Same boundary as cv_designer: LLM emits *intent*, code resolves *physics*.
- Model: Sonnet 4.6 (was Haiku 4.5 from D1, with user sign-off). Bumped because the action is now ternary and requires chemistry reasoning (CV selection, transition-timescale judgment vs budget). `model` is parameter-overridable so unit tests stay mockable and live tests can pin a tier explicitly.
- Tool schema: single `record_decision` tool retained per the design memo. `decision` enum extended; `metad_proposal` is a nullable object (`type: ["object", "null"]` — matches the existing nullable pattern used for `extra_ns` and `ledger_note`, so strict mode is happy). Cross-field invariant (`metad_proposal` non-null iff `decision == "switch_to_metad"`) enforced in `_parse_decision` with clear `RuntimeError` messages.
- System prompt rewritten: four inputs (added `task_expectation`), three-way decision rule wired to the new exploration fields (`exploring`, `n_basins`) and the task expectation, CV-proposal guidance with arity-specific examples per CV type, preserved ledger-note and reason guidance.
- Persistence (M2 invariants kept): `rounds` table grew a nullable `metad_proposal_json` column; CHECK constraint widened to three values. `_migrate_rounds_for_metad` rename-create-copy-drops to update the CHECK on pre-M4 DBs (SQLite cannot ALTER a CHECK in place). Idempotent: no-op once the column exists. Migration tested against a synthetic pre-M4 DB.
- Loop: `run_campaign` grows `task_expectation: str | None`, threaded into every `decide()` call. New `StopReason` value `switch_to_metad_requested`; on a switch decision the round is persisted (with `metad_proposal` round-tripping through SQLite and the per-round JSON) and the campaign terminates cleanly. Actual adapter reconfig with `plumed_input` is step 4.
- 95 unit tests green (was 85: +6 scientist switch-path + schema, +3 store metad column + migration, +1 loop switch termination). All 10 integration tests still collect cleanly post-refactor.
- **Open / next:** step 4 — loop wiring. Deterministic parts (no user input needed): `sampling/bias_designer.py` evaluates the proposed CV per frame on the prior vanilla trajectory (mdtraj `compute_distances` / `compute_rg` / `compute_dihedrals` chosen by `cv_type`); σ ≈ stddev along that CV / 3; HEIGHT default ≈ k_B T (~1.2 kJ/mol conservative, 2.5 kJ/mol = kT at 310 K); PACE ≈ 500 steps. Resolve the `MetadProposal` through `cv_designer` → `MetadynamicsBias` → `PlumedInput` → write `plumed.dat`. Construct a fresh adapter from the same `SystemSpec` with `plumed_input` set; the OpenMM adapter already rebuilds the runtime sim each `start()` per the M4 hook commit, so this is mostly wiring.
- **Two design judgments for step 4, surfaced this session but not yet decided (next session should decide before writing code):**
  1. **In-place pivot vs two-phase.** When `decide()` returns `switch_to_metad`, does `run_campaign` continue past the switch in the same call (call bias_designer → build new adapter with `plumed_input` → continue rounds; the campaign internally has a vanilla phase then a metaD phase), or does it terminate as today and require a separate `run_campaign` reinvocation that detects the switch row and starts the metaD phase? **Recommendation: in-place pivot** — the campaign is one logical thing; two-phase leaks an implementation seam into the user API (the whole MDPilot thesis is "I called `run_campaign`, MDPilot decided what to simulate").
  2. **Per-round phase tracking in persistence.** Options: (a) new `phase` column on `rounds` (`"vanilla" | "metad"`); (b) a nullable `plumed_dat_path` column whose presence implies the round was biased; (c) directory-naming convention only (`rounds/metad/round_NNN.dcd`). **Recommendation: (b) `plumed_dat_path` column** — it is both the marker *and* the audit artifact (you can read back exactly which bias config drove that round), avoids inventing a new enum, NULL is unambiguous for vanilla, migration mirrors the M4 step 3 rename-create-copy-drop.
- Step 5 (chignolin metaD live smoke) follows step 4. Cost estimate for the AWS chignolin run was sketched conversationally (T4 spot + step-5 scope ≈ $5–10; reasonable converged FES on A10G ≈ $50–150) but not captured here in detail because no MDPilot-specific chignolin throughput has been measured yet — that's part of what step 5 produces.

### 2026-05-28 — M4 steps 1+2: exploration diagnostic + cv_designer
- Planning conversation on the M4 scientist multi-tool refactor. Surfaced the crux: the existing report (RMSD-to-first + block-averaging + integrated autocorrelation) cannot distinguish a *metastable single basin* (which reads as a converged plateau, low variance, settled mean) from genuine convergence — so "vanilla MD is inadequate" is not detectable from the current diagnostic alone. Locked-in plan: scope to vanilla ↔ metaD only (REMD/umbrella need ensemble/multi-window loop machinery that doesn't exist; M3 lesson); detect inadequacy via an exploration diagnostic *plus* a task-encoded expectation (sequenced diagnostic-first because the report would otherwise lie by omission); CV spec is scientist-proposed-physical-idea → deterministic resolver, never a pre-curated per-system menu (initial framing of "Rg + end-to-end as the chignolin menu" was correctly rejected by the user as an autonomy regression — a menu per system defeats the whole outer-loop thesis).
- **Step 1 — exploration diagnostic.** `src/mdpilot/diagnostics/exploration.py` reports `bimodality_coefficient`, `n_basins`, `minor_basin_occupancy`, `exploring`. Method is Sarle's bimodality coefficient on the observable's marginal — deliberately chosen *over* a k-means cluster-mean-separation criterion because k-means manufactures a split even on unimodal data (splitting a standard normal at its mean yields cluster means ~2.6σ apart, which would false-positive bimodality on every single-basin trajectory). When BC clears the 5/9 cutoff, a 1D 2-means is used only to compute minor-basin occupancy as a populated-second-state safety gate. Wired into `report.py`; both branches (n<8 small-frame and n≥8) carry the keys (`None` in small-frame). Synthetic-fixture margins: single-basin BC 0.32–0.36 (iid normal, AR(1) phi 0.95/0.99), two-basin telegraph 0.77–0.96, cutoff 0.555 cleanly between. Documented limitation: a single barrier crossing reads as `exploring` — correct, since vanilla *did* cross; ESS catches whether it crossed often enough.
- **Step 2 — cv_designer.** `src/mdpilot/sampling/` created. `cv_designer.py` exposes the system-agnostic CV-type vocabulary that `plumed_writer` can render (`distance`, `torsion`, `gyration`) — not a per-system menu. The scientist proposes `CVProposal(cv_type, selections, label)` where `selections` are MDTraj selection strings (`"backbone and resSeq 1 to 10"`, `"name CA and resSeq 1"`); the resolver validates against the actual topology, looks up atom indices, and returns a typed CV object. Wrong arity, multi-atom selection where one is required, empty selection, or unknown type → `ValueError` with a message naming the failure (so the scientist's tool-use loop can recover deterministically in step 3 rather than guessing). Added `GyrationCV` to `plumed_writer` (PLUMED `GYRATION TYPE=RADIUS`, ≥2 atoms enforced); RMSD-to-reference, contact-count, and order-parameter CVs deferred until a real campaign needs them (M3 lesson).
- 85 unit tests green (was 67 at the start of this session: +6 exploration, +2 plumed_writer for GyrationCV, +10 cv_designer; assertions added to existing report tests).
- **Open / next:** push commit; then step 3 — scientist action-space refactor (`extend | stop | switch_to_metad`; on switch emit a `CVProposal` + bias spec; add task-encoded expectation to the prompt; likely Haiku → Sonnet). This is the conceptually richest piece of M4 — will re-check in on the tool-use schema and prompt shape before writing.

### 2026-05-25 — OpenMM adapter PLUMED hook (M4 sub-step)
- `OpenMMAdapter.__init__` grows an optional `plumed_input: str | None`. When non-None, a `PlumedForce(plumed_input)` is attached to the runtime System inside `start()` via a guarded `from openmmplumed import PlumedForce`; absence raises a clear `RuntimeError` naming `openmmplumed` and the pip install command. `plumed.dat` is written to `<work_dir>/plumed.dat` *before* the import attempt so the audit artifact survives the error.
- Refactored `start()` to collapse the two prior paths (`_start_from_cache` / `_start_fresh_and_cache`) into `_setup_and_cache_vanilla` + `_build_runtime_simulation`. The cache stays **bias-agnostic**: vanilla System + post-minimization State only. The runtime Simulation is rebuilt from the cache on every `start()` and the plumed force is freshly injected each time. Consequence: re-supplying or changing `plumed_input` on resume just works — no cache invalidation needed, no openmmplumed serialization dependency.
- Cost of the refactor on the vanilla (no-plumed) path: first `start()` now does an extra serialize-and-deserialize round trip after minimization (~ms on a few-thousand-atom system). Identical behavior on resume.
- 4 new unit tests in `tests/unit/test_openmm_plumed_hook.py`: (1) plumed.dat written even when `openmmplumed` is missing, (2) error message names `openmmplumed` and `pip install`, (3) success path via injected stub module returns the right force, (4) deep work_dir is created.
- 67 unit tests green (was 63). Live "bias actually acts on dynamics" is still deferred to AWS (D6 step 5).
- **Open / next:** scientist multi-tool refactor — the scientist needs a way to choose between vanilla and biased rounds (and, when biased, propose CVs + bias params for `plumed_writer`). This is the conceptually richest part of M4; will likely need a planning conversation first.

### 2026-05-24 (late night) — PLUMED writer (M4 sub-step, PLUMED-agnostic build)
- Tried `sudo apt install plumed plumed-dev`; package not found in this WSL Ubuntu's apt sources (likely universe repo not enabled). Decided **not** to fight apt: instead, build the PLUMED-using code in a PLUMED-runtime-agnostic way and defer the bias-actually-acts smoke test to AWS (D6 step 5), where the environment is controlled.
- New `src/mdpilot/adapters/plumed_writer.py`: pure text generation, zero runtime PLUMED required. Typed `DistanceCV` / `TorsionCV` (0-based atom indices in code, 1-based in output — matches the rest of the codebase). Bias dataclasses `MetadynamicsBias` and `HarmonicRestraint`. `PlumedInput` composite validates CV-label references and unique CV labels, then renders the full plumed.dat text including header comments and a PRINT directive.
- 17 unit tests cover index conversion, multi-CV metaD, parameter validation, restraint, undefined-CV-reference rejection, duplicate-label rejection.
- 63 unit tests total (was 46).
- **Open / next:** OpenMM adapter PLUMED hook — accept an optional `plumed_input: str | None` at construction, attach a `PlumedForce` via guarded import of `openmmplumed` so absence doesn't break the codebase. Then scientist multi-tool refactor; then chignolin metaD pilot.

### 2026-05-24 (night) — Chignolin runs vanilla through the loop, no code changes
- New `tests/integration/test_chignolin_vanilla_live.py`. Constructs `GROMACSAdapter(spec=SystemSpec(pdb_id="1UAO"))`, calls `run_campaign` with `initial_steps=500, max_rounds=1`. The full pipeline (PDBFixer download → pdb2gmx → editconf → solvate → genion → minimize → MD → diagnostic → scientist) succeeds end-to-end and the scientist correctly says "extend" on 1 ps of chignolin.
- Passes in **48 s** wall time (vs ~28 s for the analogous Trp-cage adapter test — chignolin's setup is similar, the test runs one full loop iteration including an Anthropic API call).
- This is the empirical validation that D6 step 2 (SystemSpec generalization) is sound. No engine code touched between Trp-cage and chignolin — only the constructor argument changed. The M4 forcing function is ready to use.
- **Open / next:** PLUMED install + smoke test. Will need `sudo apt install plumed plumed-dev` (system) and `pip install openmmplumed` (in venv). System install is an environment change worth confirming with the user before running.

### 2026-05-24 (evening) — Hypothesis ledger activation (D6 step 3)
- New `ledger` table in the per-campaign SQLite (autoincrementing PK, multiple notes per round allowed, no FK to `rounds` so notes can refer to future rounds or be added without a corresponding round row). Helpers `append_ledger_note(work_dir, round_index, text)` and `list_ledger_notes(work_dir) -> list[LedgerNote]` in `memory/store.py`.
- `Decision` dataclass + `record_decision` tool schema both grow an optional `ledger_note: str | None` field; with strict mode it's required to be present in every response but may be null (skip). System prompt now distinguishes three inputs (diagnostic_report, prior_round_summaries, hypothesis_ledger) with explicit guidance on what belongs in `ledger_note` vs `reason`: ledger is for *insight that should outlive this round*, reason is for *the specific numbers that drove this round's decision*.
- Loop loads existing ledger notes via `store.list_ledger_notes` before the round loop, formats each as `"R{idx}: {text}"`, passes the list into `decide(hypothesis_ledger=...)`, and after each round persists `decision.ledger_note` if non-null (commit order: checkpoint → JSON → append_round → append_ledger_note; if we crash after append_round but before the ledger insert, the round is still recoverable, just without that round's note).
- Cross-campaign ledger access (ice surface comparison) is a later concern; today the ledger is per-campaign and lives entirely in the per-campaign SQLite DB.
- 46 unit tests green (+7 from 39: 4 scientist ledger tests + 3 store ledger tests).
- **Open / next:** push commit; then D6 step 4 — M4 full build with chignolin as forcing function. That's a multi-session block: PLUMED install, plumed_writer adapter integration, scientist multi-tool refactor, strategy selector, CV designer. Most of the conceptually interesting work for MDPilot starts here.

### 2026-05-24 (later) — SystemSpec generalization (D6 step 2)
- New `src/mdpilot/adapters/system_spec.py`: minimal `SystemSpec` dataclass with `pdb_id` XOR `structure_path`, a `trpcage()` factory for the M1-era default, and `to_dict()` for SQLite serialization.
- `MDAdapter` Protocol grows a `spec` property; both `OpenMMAdapter` and `GROMACSAdapter` accept `spec` at construction (defaulting to `SystemSpec.trpcage()` for backward compat) and use it in `prepare()` instead of the hardcoded `_PDB_ID` constant. Cached PDB filenames now include the spec tag so two different campaigns in the same `work_dir` don't collide.
- Loop includes `adapter.spec.to_dict()` in the SQLite campaign config dict — resuming with a different spec (Trp-cage → chignolin) now hits the existing config-mismatch guard and raises before any engine setup. Verified by new unit test `test_system_spec_mismatch_on_resume_is_rejected`.
- Intentionally minimal: SystemSpec carries only the structure source. Other parameters (force field family, water model, temperature, integrator) remain hardcoded per adapter. They get added when a real campaign (ice nucleation: TIP4P/Ice + below-273-K + possibly CLAYFF) forces them. Following the M3 lesson: discover the right abstraction from two real implementations, not from speculation.
- 39 unit tests green (+9 from yesterday's 30: 8 SystemSpec + 1 spec-mismatch).
- **Open / next:** push commit; then start D6 step 3 — hypothesis ledger activation (minimum-viable structured findings store, scientist appends one ledger note per decision).

### 2026-05-24 — F2 resolved + strategic plan locked
- Long planning conversation (see D6 above): user wants MDPilot as a general research tool with ice nucleation as one of multiple showcases. Agreed to build M4+M5 thoroughly with **chignolin** as the M4 forcing function (avoids designing enhanced-sampling abstractions from zero use cases — the same trap we sidestepped with `MDAdapter` in M3).
- Replaced the literal roadmap order with the D6 order: F2 → SystemSpec generalization → hypothesis ledger → M4 (chignolin) → M5 lite (AWS launcher) → ice campaign showcase.
- **F2 fix landed** in this commit. Both adapter `start()` methods are now idempotent. GROMACS short-circuit verified live (27.5s, em.gro mtime unchanged after second start). OpenMM short-circuit verified by code review.
- 30 unit tests still green.
- **Open / next:** push commit; then SystemSpec generalization (step 2 of D6 plan) — adapters take arbitrary system specs, no more Trp-cage hardcode. This unblocks chignolin (M4 forcing function) and eventually ice (showcase).

### 2026-05-23 (afternoon) — Milestone 3 landed (adapter Protocol + GROMACS + cross-engine)
- Scoped M3 with the user: GROMACS only, MDCrow deferred indefinitely (decision and refined anti-goal captured as D5 above).
- Two commits worth of work landed:
  1. `8f070f3` — `MDAdapter` Protocol + OpenMM refactor; `openmm_runner.py` deleted; loop holds an adapter instance.
  2. (this commit) — GROMACS adapter + cross-engine integration test + CLAUDE.md anti-goal refinement.
- Added `trajectory_extension` to the Protocol so engine-specific suffixes (`.dcd` vs `.xtc`) flow correctly through the loop without the loop needing to know which engine is in use.
- Live tests: `test_gromacs_adapter_live.py` (setup + run + checkpoint round-trip on Trp-cage, 28s) and `test_cross_engine_live.py` (one campaign round through GROMACS + scientist, 42s). Both green.
- Note on GROMACS reproducibility: even with `-ntmpi 1` + fixed `ld-seed`, OpenMP force-summation order is non-deterministic; resumed trajectories track continuous ones at the CA-RMSD < 1 Å level, not bit-identically. Adapter contract is "physically equivalent resume," not "bit-identical resume." Captured in the test comment.
- 30 unit tests still green throughout.
- **M3 status:** complete to the amended done-criterion (engine independence via OpenMM OR GROMACS, MDCrow deferred).
- **Open / next:** push pending local commits; start M4 (sampling-strategy decisions — when vanilla MD is inadequate, scientist picks metaD CV / REMD ladder / umbrella windows; activates the hypothesis ledger deferred from M2).

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
