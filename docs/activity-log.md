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

**Upper tail closed (2026-08-21):** the floors only ever bound from below, which left the opposite pathology unguarded — a CV sized on an *already-biased* trajectory measures a spread inflated by the bias that drove the walker across it, and a hill wider than the features it should resolve flattens the surface by construction. `_SIGMA_CEILINGS` adds the matching bound (0.2 nm for `distance`/`gyration`/`rmsd`, 1.0 rad for `torsion`, 2.0 for a contact count), `size_sigma` returns `(sigma, floored, ceiled)`, and a ceiled width gets its own `# NOTE:` block in the rendered input. This became load-bearing with `switch_cv`, which sizes a replacement CV on exactly such a trajectory.

### F5 — PLUMED resolves FILE= against the process CWD, and buffers output (2026-08-11)
Two defects found by the first live run, both fixed in the same session:
- A bare `FILE=HILLS` wrote the campaign's deposited bias into whatever directory python was started from — the repo root, in that first run, complete with `bck.0.*` backups. Concurrent campaigns would collide and resume could never find the previous HILLS. `PlumedInput` now requires an **absolute** `output_dir` and prefixes it onto every file PLUMED writes; a relative one raises.
- PLUMED buffers HILLS/COLVAR and flushes only when the context is finalized, so a campaign killed mid-round loses every deposited hill and a resume reads what looks like an empty file. `PlumedInput.render()` now emits `FLUSH STRIDE=<print_stride>`.

Also worth knowing when reading HILLS: PLUMED records well-tempered hill heights pre-scaled by **γ/(γ-1)**, so the first hill reads `W0·γ/(γ-1)` (1.2472 × 10/9 = 1.3857), not `W0`.

### F6 — RMSD-to-native is a one-way CV for folding (2026-08-20)
Two independent CLN025 metadynamics campaigns biased CA-RMSD to the native structure. Both reached the extended state, neither came back.

| | run 1 | run 2 |
|---|---|---|
| Biased sampling | 4 ns (stopped early) | **20 ns** (budget exhausted) |
| Upper wall | none | 0.8 nm |
| CA-RMSD reached | 5.56 Å | **9.54 Å** |
| Returned to native | no | no |
| A→B→A recrossings (CA-RMSD, 1.5/4.0 Å) | 1 | **1** — needed 2 |
| Done criterion | ❌ | ❌ |

Run 2's trace is unambiguous: the peptide left the folded state at **t = 582 ps** and never returned in the remaining 19.4 ns. From 12 ns onward it was **100 % above 4.0 Å** and still drifting outward. The recovered surface is not merely unconverged, it is *inverted* — the free-energy minimum sits at 3.5 Å with the native state ~72 kJ/mol above it, because nothing deposited in the native basin after 582 ps to correct the over-filling.

**Mechanism.** RMSD-to-native lumps an enormous unfolded conformational space into one number and is unbounded above. Well-tempered metaD fills the native basin, pushes the walker out, and the return trip is entropically hopeless. The `UPPER_WALLS` restraint added between the runs bounded the *coordinate* (`cv_max` pinned at 1.067 nm from round 3, vs run 1's runaway) but not the problem — the walker simply sat against the wall.

**Resolution:** a `contacts` CV type (PLUMED `CONTACTMAP ... SUM`) was added to the vocabulary, bounded on `[0, n_pairs]` by construction, and the scientist's prompt now states which CV types are unbounded above and what that failure mode is. Do not read run 1 or run 2 as evidence that metadynamics or the pivot logic failed — both worked. The coordinate was wrong.

### F7 — Biased-phase diagnostics were measuring regions the physics does not occupy (2026-08-20) — FIXED `aa85f6b`
Three separate defects, found by auditing run 2's own output rather than by a test.

1. **`fes_depth_kj_per_mol` measured grid padding.** `sum_hills` grids a few SIGMA past the outermost hill; out there the surface is extrapolation no hill ever touched, it is the highest point on the grid, and it never changes. Run 2's grid ran −0.174 … 1.130 nm while the walker visited 0.003 … 0.954 nm. Reported depth **143.01 kJ/mol → 76.58** once restricted to the sampled range. The unphysical negative `cv_min` the scientist flagged in its own ledger at rounds 6 and 8 was the same artifact; it was right, and nothing downstream acted on it.
2. **`recrossings` was counted between boundaries that migrate.** `basin_thresholds` derives the dividing surfaces from the two deepest basins of the *current* FES. Once the native basin filled and was abandoned, both boundaries drifted into the unfolded ensemble — **4.52 Å and 6.10 Å at round 12**, against a task that defines native as < 1.5 Å. The loop reported up to 7 "barrier recrossings" while the peptide had not refolded once. The count was measuring oscillation *within* the unfolded state.
3. **The CV-failure detector could not fire.** The ledger note built to flag a wrong CV keys on `recrossings == 0`; the count was non-zero for the wrong reason, so it stayed silent for the whole campaign.

**Fix.** Depth (only) is restricted to the COLVAR-sampled range, and `cv_min`/`cv_max` now report that range rather than the grid. `recrossing_low`, `recrossing_high` and `cv_start` are published in the report and carried into the compact prior, so a count is never readable without the boundaries it was taken against — the same principle as F4's SIGMA-floor note. The scientist's prompt gained a rule to check those boundaries against the task's states before treating a count as evidence.

**Deliberately *not* changed:** `fes_drift_kj_per_mol`. Restricting it to the sampled range moves it by 0.06 kJ/mol (79.84 → 79.78), and to the 1st–99th percentile by 0.09. Drift ≈ 80 was an honest report that the surface had not converged. Basins, barrier and drift stay on the full grid: cropping at the sampled extremes strands a basin minimum on the boundary where `minima()`'s interior-only scan cannot see it, which dropped `n_basins_fes` from 2 to 1 in an existing test.

### F8 — `seed` does not make system construction reproducible (2026-08-20)
`Modeller.addSolvent` takes no seed and places ions by "randomly selecting a water molecule". Two campaigns built from the same `SystemSpec` and the same `seed=42` produced **4874 vs 4889 atoms**. The campaign-local cache makes this invisible *within* a run, so nothing is broken, but `seed` covers only the integrator, barostat and initial velocities — not solvation. Any comparison between two campaigns is a comparison between two different systems unless the cached system is reused explicitly (as run 2 did, reusing run 1's).

### F9 — `recrossings` is not comparable across rounds, and conflates "none" with "unmeasurable" (2026-08-21) — FIXED 2026-08-21
Run 3 exercised the full failure range of this diagnostic inside a single campaign, on a coordinate the walker was demonstrably diffusing across:

| round | 2 | 3-6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| `recrossings` | 1 | 2 | **0** | **0** | **30** | **0** | 22 | 26 |
| `n_basins_fes` | 3 | 3-2 | 1 | 1 | 2 | 1 | 2 | 2 |
| band width | 3.61 | ~1.2-3.2 | — | — | 1.66 | — | 1.31 | 1.31 |

Two distinct defects, both rooted in the band being re-derived from the current FES every round.

1. **Unmeasurable reads as zero.** When fewer than two basins resolve, `basin_thresholds` returns None and the fallback sets `recrossings = 0`, `barrier_crossed = False`. The missing-COLVAR path correctly reports `None` for the same situation — "I cannot measure this" — so the two disagree about how to say the same thing. In rounds 7, 8 and 10 the walker's own `cv_min`/`cv_max` spanned 0.59-10.56 of 12 contacts; it had crossed repeatedly and the field said it had not. Same shape as run 1's `barrier_crossed` bug, through a different door.
2. **A collapsed band inflates the count.** As the barrier fills, the two deepest minima become shallow ripples close together. At round 9 they sat 1.66 contacts apart and the walker diffusing freely across that gap scored **30**; at rounds 11-12 the band was 1.31 wide and *inside the folded basin* (7.3-8.6), scoring 22 and 26. Those are not folding events.

**Consequence beyond reporting.** `_fes_converged` requires `drift < kT AND recrossings >= min_recrossings`. Defect 1 makes convergence unreachable while the surface reads single-basin regardless of drift; defect 2 holds the recrossing gate wide open on diffusive noise, so if drift ever dipped below kT with a collapsed band the loop would declare convergence on a count that means nothing. Combined with `_refuse_premature_stop`, both roads lead to budget exhaustion.

**Fix — the band is anchored to the task's states.** User chose task-supplied states over freezing the first round's basins. The wrinkle: the task defines its states in CA-RMSD Angstrom while the biased CV may be in any units at all (run 3's was a contact count), so anchoring cannot mean translating the thresholds into CV space. It means **counting on the coordinate the task defines its states on**, whatever is being biased — effectively promoting `verify_done_criterion` from a post-hoc check into the loop, so the number the scientist sees each round is the number that decides the criterion.

- `report.campaign_observable()` is split out of `make_report` so the vanilla report, the biased report and the post-hoc check share one definition of CA-RMSD-to-campaign-reference and cannot drift apart.
- `metad_report` takes `(observable, observable_name, state_thresholds)`; when present it counts there and sets `recrossing_basis="task_states"`. Absent, it falls back to the surface-derived band and says `recrossing_basis="fes_basins"`, so a count is never readable without knowing which basis produced it.
- The unresolvable case now returns `None`, not `0`. `_fes_converged` already withholds a verdict on a null count, so a single-basin round extends rather than reading as "never crossed".
- The count stays **cumulative** across biased rounds, matching the HILLS/COLVAR semantics it replaces. Each round's series is persisted to `rounds/round_NNN.obs.npy` (~16 KB against a ~117 MB DCD) and concatenated, rather than re-deriving from the trajectories every round — which by the end of a 20 ns campaign would mean re-reading over a gigabyte.
- Computed only when thresholds are supplied, so a campaign without them behaves exactly as before.

**Verified by replaying run 3's stored rounds.** Anchored: `1,1,1,2,3,3,3,3,3,3,3` — monotone, no unmeasurable rounds, ending at 3 and agreeing exactly with `done_criterion.json`. Reported at the time: `1,2,2,2,2,None,None,30,None,22,26`. That series *decreases*, which is arithmetically impossible for a cumulative transition count and is the clearest evidence it was not measuring one quantity. The anchored count crosses `min_recrossings=2` at round 5, matching the mid-run hand computation.

**Made strict (same day).** The fallback is no longer reachable by accident. `run_campaign` refuses at entry when `task_expectation` is set — the sole input gating `switch_to_metad`, so exactly the predicate for "this campaign can reach a biased phase" — and `state_thresholds` is None. Argument-only validation, so it raises before any filesystem or engine work rather than at the pivot, which would have thrown away the whole vanilla phase. A pure convergence task (no `task_expectation`, hence no possible pivot) still needs nothing. Inverted thresholds are refused too: `count_recrossings` returns 0 rather than raising for a band with `high <= low`, so a swapped pair would have read as a campaign that never crossed. `state_thresholds` joins the locked config alongside `task_expectation` and `cv_upper_wall_nm` — resuming with a different band would splice two definitions of "a transition" into one campaign's history.

*Consequence:* campaigns created before this (runs 1-3) carry no `state_thresholds` key and will fail the config guard on resume. All three are finished, so this costs nothing, but a resume of any of them now needs the key added to its `campaign` row.

**The "strict" claim was half true (2026-08-24) — CLOSED.** The entry-time refusal covers `task_expectation` set + `state_thresholds` None. It said nothing about the reverse, and nothing enforced the premise it rested on: `decide()` selected its tool by *phase alone*, so `_DECISION_TOOL` offered `switch_to_metad` on every vanilla round whether or not a `task_expectation` existed. A campaign with neither key passed the entry guard, and the model could still emit a pivot — landing in a biased phase with no band to count recrossings against, i.e. the F9 fallback with all its behaviour intact. `test_a_pure_convergence_campaign_needs_no_state_thresholds` asserted the false premise in its own docstring. Closed by giving that case its own tool: `_CONVERGENCE_TOOL` (`extend | stop`, no `metad_proposal`), selected when `task_expectation is None`. Same "unrepresentable beats emitted-then-rejected" rule as `_METAD_DECISION_TOOL` and `_METAD_SWITCH_TOOL`; the loop's own `switch_to_metad` branch is unchanged and still reachable by a caller that stubs `decide`, which is how the pivot orchestration tests drive it.

**Also locked (2026-08-24):** `min_recrossings`. It is the other half of the same definition — `state_thresholds` says *where* the states are, `min_recrossings` says *how many* transitions between them count as done. It is the threshold `_fes_converged` compares against and `_refuse_premature_stop` reads to decide whether a `stop` is allowed, so resuming with a different value re-judges rounds already decided under the old one. Same migration consequence as above.

### F10 — the well-tempered drift test was vacuous at exactly two FES estimates (2026-08-24) — FIXED

`metad_report` measured drift against `surfaces[len(surfaces) // 2]` — the half-way estimate, deliberately not the previous one, because with `--stride 10` over thousands of hills consecutive estimates are ~0.25% of the run apart and their difference is near zero however unconverged the surface is. But at exactly **two** estimates `n // 2 == 1 == -1`: the baseline *is* the final surface, and drift came back `0.0` by construction.

This is the same defect as the trailing-duplicate surface `sum_hills` already pops, arriving by a different route — and it is worse than merely uninformative. `_fes_converged` requires only `drift < kT AND recrossings >= min_recrossings`, so a first biased round short enough to yield two estimates (~10 ps at `PACE=500`, `stride=10` — which is where `run_cln025 --dry-run` lives) could report `fes_converged=true` on a surface still moving by tens of kJ/mol. `_refuse_premature_stop` explicitly passes a `stop` through when `fes_converged is True`, so the single guard against declaring victory against the diagnostic would have stood aside.

**Reproduced** against two analytic double wells differing by 6.06 kJ/mol: reported drift `0.0`, `recrossings=1`, `fes_converged=True`. After the fix: drift `6.06`, `fes_converged=False`.

**Fix.** `_baseline_index(n)` returns `min(n // 2, n - 2)` — the half-way surface, clamped to stay strictly below the last index. At two estimates that is index 0, which is also the most widely separated pair available, so the comparison is not merely non-vacuous but the right one. No test had covered `n_fes_estimates == 2`; there are now two, one on the index arithmetic and one end-to-end through `metad_report`.

*Second defect, same session:* `sum_hills` never cleared `out_dir`. A round re-run after a crash resumes from the previous round's restored (shorter) HILLS and writes fewer indexed surfaces, while leftovers from the aborted attempt survive — `indexed.sort()` then puts a stale surface last and `surfaces[-1]`, the profile every statistic is taken on, is from the previous attempt. Stale outputs are now removed before plumed is invoked. The unit test that fabricated sum_hills' output beforehand was rewritten so the fake subprocess writes it *during* the call, which is what the real binary does.

### D7 — The task file is the campaign contract, and `task_expectation` is rendered from it (2026-08-26)
**Goal context:** MDPilot is aimed at being a *general* MD agent — chignolin folding is the current forcing function, protein-ligand binding/unbinding is the next, and others follow. Interfaces should therefore be named for mechanics, not for the first problem that used them.

**One file owns what a campaign is.** `benchmarks/tasks/*.yaml` was decorative: `run_cln025.py` read four fields and every other declaration — force field, water model, padding, ionic strength, timestep, constraints, observable, target ESS — was documentation the adapters were free to contradict. `mdpilot/task_file.py` makes it load-bearing, in three modes per field: **mapped** (becomes a `SystemSpec`/`Ensemble` or a `run_campaign` keyword), **verified** (not yet tunable, so checked against the constant that really governs it — a mismatch raises), **informational** (prose, carried not interpreted). Unknown keys raise, which is how a generated file's typos surface.

**`task_expectation` is rendered, not authored.** It is the only input gating `switch_to_metad`, and it was the one load-bearing input that escaped structuring — free prose that restated the state thresholds, the round-trip requirement and the compute budget in words. Three of its four decision-driving numbers existed twice with nothing linking the copies. It is now a *view* of `expectation:` + `done_criterion:`, so drift is unrepresentable. Only the objective and the characteristic timescale are genuinely free, and the timescale carries a source because it is the one decision-driving number that exists nowhere else. The budget-vs-timescale comparison the pivot rule asks for is computed rather than left as arithmetic on prose.

**States are positions, not roles.** `done_criterion.states.{low,high}` each carry a `name` and a `threshold` on the campaign observable, replacing `folded_state_rmsd_angstrom` / `extended_state_rmsd_angstrom`. `count_recrossings` only ever wanted a low band and a high band; a binding campaign names them `bound`/`unbound` on a distance and nothing in the mechanics changes. `run_campaign`'s `state_thresholds` documentation and error messages were reworded the same way.

**Known blocker for the next system class**, now tested rather than latent: `diagnostics.report.campaign_observable` selects `protein and name CA` and raises on an empty selection, so CA-RMSD is the only observable the loop can compute. `task_file` refuses a file declaring any other `observable.name` — see `test_a_non_protein_observable_is_refused_until_it_is_tunable`. A binding campaign needs that generalized first, and per the M3 lesson the abstraction should be discovered from the second real observable, not invented ahead of it.

### F11 — CLN025 campaigns sampled configurations where the peptide interacts with its own periodic image (2026-08-26)
Measured on the completed campaigns, not inferred. Box is cubic, L ≈ 3.66 nm (1.0 nm padding applied to the *folded* structure, then shrunk by NPT). Minimum heavy-atom distance between the solute and its nearest periodic image, per round:

| campaign | biased CV | rounds within the 1.0 nm cutoff of their own image | closest approach |
|---|---|---|---|
| `cln025_metad` | `rmsd_ca` + 0.8 nm wall | **6 / 12** | **0.27 nm** |
| `cln025_metad_run3` (criterion met) | `native_contacts_ca`, no wall | 2 / 12 | 0.63 nm |
| `cln025_metad_run1` | (4 rounds, failed) | 0 / 4 | 1.54 nm |

0.27 nm is contact distance — a hydrogen bond. The molecule is whole in the stored frames (max bonded heavy-atom distance 0.167 nm), so this is not a wrapping artifact. In `round_011` of `cln025_metad`, 100% of frames have a solute span exceeding L/2 = 1.83 nm (mean 3.34 nm, max 4.02 nm), meaning each chain end is nearer a periodic image of the other end than the end itself.

**Cause.** Padding is applied to the starting (folded) structure, but metadynamics deliberately drives the chain to extended states. `_PADDING_NM = 1.0` on a folded 10-residue hairpin gives L/2 barely larger than the *folded* peptide (heavy-atom extent 1.87 nm vs L/2 = 1.83 nm), and nowhere near enough for the unfolded ensemble the campaign exists to sample. The 0.8 nm RMSD wall (added for F6) bounds the CV but not the chain's spatial extent, and is not tight enough for this box.

**This is a deepening of F6, not a separate story.** RMSD-to-native being unbounded above did not only produce one-way excursions — it drove the chain into configurations the box cannot support. The `contacts` CV, chosen for F6 reasons, mitigated this too: run 3 is the least affected, by a wide margin.

**What it does and does not invalidate.** The M4 done criterion is about *agent behaviour* — did the scientist detect vanilla inadequacy, pivot, choose a defensible CV, and recross. That result stands; run 3 met it, and its contamination is the mildest of the three. What is not trustworthy is the *physical* free-energy surface on the extended side: solute self-interaction perturbs the unfolded ensemble, so barrier heights and ΔG from these runs should not be quoted.

**Resolved 2026-08-26 by raising the default box.** `padding_nm` is now a `SystemSpec` field, default **1.5 nm** (was a hardcoded 1.0). Sized against this campaign's own sampled spans rather than a rule of thumb — the box actually built at p=1.0 was L=3.78 nm (NPT shrink factor 0.976 on folded extent 1.87 nm), and the sampled heavy-atom span had median 2.24, p95 3.20, max 3.87 nm:

| padding | L | atoms vs 1.0 | worst image gap | frames under 1.0 nm |
|---|---|---|---|---|
| 1.0 | 3.78 | 1.0x | −0.09 nm | 20.20% |
| 1.2 | 4.17 | 1.3x | +0.30 nm | 5.69% |
| **1.5** | **4.75** | **2.0x** | **+0.89 nm** | **0.18%** |
| 2.0 | 5.73 | 3.5x | +1.86 nm | 0.00% |

1.5 nm is a better default, not a guarantee: it leaves 0.18% of that ensemble inside the cutoff, and a campaign that drives its system further apart than CLN025 did should raise it. Both shipped task files now declare it, and `system.padding_nm` moved from the loader's verified-but-fixed table to a mapped field.

**The default was *raised*, which the config-compatibility layer had to be taught.** `SystemSpec.to_dict` omits a default `ensemble` because that default equals what pre-ensemble campaigns ran under; padding cannot use the same trick, because absent (a pre-F11 campaign, built at 1.0) and default (1.5) mean *different boxes*. It is always serialized, and `store._LEGACY_SYSTEM_SPEC_DEFAULTS` supplies 1.0 for configs recorded before the field existed — so a pre-F11 campaign is correctly refused at the new default and still resumable at its own 1.0.

**Still open:** the two cheaper mitigations remain available and independent — a tighter `cv_upper_wall_nm` (already tunable), and preferring bounded CVs, which run 3's numbers already argue for. Nothing yet checks box-versus-extent at pivot time, which would catch this class of problem before compute is spent rather than after.

### D8 — The campaign observable is a declared CV, and setup gets one LLM call outside the loop (2026-08-26)
**Observable.** `campaign_observable` was hardcoded to `protein and name CA`, which made every non-protein campaign impossible at round one. It is now an `ObservableSpec` — a `CVProposal` plus a display name and a unit `scale` — declared in the task file's `observable:` block and computed through `bias_designer.cv_series`, the same engine that sizes the bias. That reuse is the point: "size the bias from this coordinate" and "judge the campaign on this coordinate" must be one computation or they disagree with nothing to catch it. `rmsd` is the one type computed directly, because PLUMED's RMSD action needs a reference *file* while the campaign observable measures against the topology already in memory. Not a new abstraction: an observable is a collective variable, and `sampling/` already knew how to resolve and compute five.

**Setup agent (`setup_agent.py`).** The only LLM call outside the round loop — once per campaign, before any compute. It emits a structured proposal via strict tool use; this module renders the YAML, so a syntactically invalid task file is unrepresentable and the only failures left are semantic ones `load_task_file` can explain. A refused proposal is retried with the loader's own message as a `tool_result`; the loader, not a rubric, is the judge. `out_path` only ever holds a file the loader accepted.

**Why this does not violate D5.** The MDCrow anti-goal forbids rebuilding an LLM *agent layer that orchestrates setup tools*. Nothing here runs: the model proposes, `task_file` validates, a human reviews, deterministic Python builds. Same boundary `cv_designer` draws for the biased CV. D5's closing line — "revisit only when a campaign genuinely needs setup-from-natural-language" — is the trigger being pulled, deliberately and recorded here.

**The tool schema cannot express the biased CV.** There is no field for it. Choosing which coordinate to bias is the scientist's judgment at the moment it decides unbiased MD is inadequate; a field here would answer the question the campaign asks. Unrepresentable rather than discouraged, the same argument the decision tool's enums make.

### D9 — Force fields are a closed vocabulary of validated pairs, and engines refuse rather than substitute (2026-08-26)
`mdpilot/forcefields.py` holds six protein+water combinations, each verified on **two** axes against the installed OpenMM: the XML set loads, *and* its water model is one `Modeller.addSolvent` can build. Both are necessary — `amber14/opc` passes the first and fails the second (`addSolvent` supports only tip3p, spce, tip4pew, tip5p, swm4ndp), and is left out rather than offered untested. `SystemSpec.forcefield` is a key into this table, so the spec stays engine-agnostic and each adapter resolves it; `tests/unit/test_forcefields.py` re-checks the whole table so an entry cannot rot.

**Closed vocabulary, for the same reason `cv_type` is an enum.** Force-field choice is high-stakes and silent when wrong: a mismatched pairing builds, runs, and reports plausible numbers while sampling the wrong ensemble. A combination that is not in the table is unrepresentable rather than discouraged.

**Engine coverage is uneven and is not hidden.** The stock GROMACS install ships amber94/96/99/99sb/99sb-ildn/03, charmm27, gromos and oplsaa — no ff14SB, no charmm36. `for_gromacs` therefore *raises* on an unmapped key. `amber99sb-ildn` is a different force field, not a translation of `amber14-all.xml`, and substituting it would turn a cross-engine comparison into a comparison of two force fields with nothing recording that it happened. `amber99sbildn/tip3p` is the only entry both engines can build, which makes it the only combination a genuine cross-engine study can use — and note the two adapters had until now *defaulted to different force fields* with nothing marking the difference.

**Known gaps, stated in the guide rather than worked around:** no TIP4P/Ice ships with OpenMM (so the D6 ice showcase needs a parameter file this project does not carry), no OPC solvation, and no small-molecule force field (GAFF/OpenFF), so protein-ligand systems still cannot be parameterised. The setup agent is instructed to propose the closest listed combination and name the gap.

### F12 — A task file's system spec was silently discarded, and the wrong molecule ran for 40 minutes (2026-08-26, **corrected 2026-08-26**)
**This entry originally blamed the setup agent for proposing the wrong PDB. That was wrong, and the correction is the more useful finding.**

First campaign launched from the Streamlit UI. The task file said chignolin, `starting_pdb: 2RVD`. What ran was 1L2Y — Trp-cage, 20 residues, `NLYIQWLKDGGPSSGRPPPS`.

**`2RVD` is chignolin.** Fetched directly: `TYR-TYR-ASP-PRO-GLU-THR-GLY-THR-TRP-TYR` = YYDPETGTWY, 10 residues, CLN025. The agent's identifier was correct.

**The cause was `app.py` calling `run_campaign` with no adapter.** `run_kwargs` deliberately excludes the system spec — the spec belongs to the adapter, which the caller constructs. `run_campaign(adapter=None)` then falls back to `OpenMMAdapter(work_dir, seed)`, whose default is `SystemSpec.trpcage()`. So the *entire* `system:` and `integrator:` block was discarded — PDB, force field, padding, and temperature (340 K requested, 300 K run) — with nothing anywhere recording that the file had been ignored. The stored config proves it: `system_spec: {'pdb_id': '1L2Y', ...}`, and `inputs/1L2Y_fixed.pdb` sits in the work directory. `benchmarks/run_cln025.py` had always done it correctly (`spec=task.spec`); the app did not.

**Fixed by removing the footgun, not just the instance.** `TaskFile.build_adapter(work_dir, seed=...)` constructs the adapter from the file's own spec; both the app and the benchmark runner use it, so the two halves of a task file — what to simulate and how to judge it — are no longer separable by accident. A test asserts the spec reaches the engine.

**What was genuine agent drift, and still stands.** The observable was named `native_contacts_fraction` with `scale: 1` — an absolute count, 938-2140, against thresholds of 0.3 and 0.7. Every frame read as the same state, so `recrossings` was fixed at 0, `fes_converged` could never be true, and `_refuse_premature_stop` converted every `stop` into an `extend`. That half of the original entry is correct.

**And part of *that* was also ours.** `ObservableSpec.normalize` was added to the spec and the loader but not to the setup agent's tool schema, so the agent could not express a fraction at all and could only reach for `scale`. Fixed; with `normalize` available it now emits `normalize: true` on the first attempt.

**A third, separate agent error, now caught at load.** A later draft wrote two selections for a `contacts` observable — one per hairpin strand. `contacts` forms native pairs *within* one group. Arity is pure schema and needs no topology, so it moved from `cv_designer` (which only finds out after a structure has been fetched and solvated) into `ObservableSpec.__post_init__`, with a message that explains the specific mistake.

**What the pre-flight checks are actually worth.** `check_residue_count` did fire on this campaign — the description said 10 residues, the built structure had 20. It caught a real inconsistency at round zero. What it could not tell anyone is *why*, and the obvious reading — "the agent picked the wrong PDB" — was the wrong one. A check that compares two artifacts tells you they disagree; it does not tell you which is at fault, and the first explanation that fits is not evidence.

### F13 — A bounded CV trapped the walker anyway, and the cumulative range hid it (2026-08-26)
`campaigns/ui_campaign_chignolin`, biased on native-contact count. The walker left the folded state in round 2 and never returned:

| round | observable this round | folded | unfolded | fes_depth |
|---|---|---|---|---|
| 2 | [0.110, 0.781] | 10.8% | 25.9% | 51 kJ/mol |
| 3 | [0.038, 0.436] | 0.0% | 92.5% | 92 kJ/mol |
| 4 | [0.030, 0.126] | 0.0% | 100% | 116 kJ/mol |

`recrossings` stuck at 1 against a criterion of 2. 116 kJ/mol over the *sampled* range is an order of magnitude past chignolin's real folding free energy.

**Bounded is not reversible.** `cv_vocabulary.md` claimed `torsion` and `contacts` "do not have that failure mode" — true for *range*, false for *return*. A contact count maps every disordered conformation onto roughly the same value, so once the chain is disordered the coordinate cannot separate those states and the bias fills one degenerate bin forever. Guidance corrected.

**The scientist could not see it.** Its round-4 reasoning quoted `cv_min=0.39 to cv_max=10.01` and concluded "the full CV range is being explored" — correct, but cumulative: COLVAR appends across the whole biased phase under RESTART, so those fields keep reporting the widest excursion the campaign ever made long after the walker stopped. Nothing gave a per-round view.

**Fixed with computed signals rather than more prose.** `observable_min_this_round` / `observable_max_this_round`, plus `confined_to_state` and `rounds_confined` from `_confinement`, which walks back through the per-round observable files and stops at the first round that was not confined (or was confined to the other state). Replayed on the real campaign it reports `('low', 1)` at round 4 — a threshold of 2 fires at round 5. `cv_switches_used` / `cv_switches_remaining` are in the report too, so a switch can be weighed against what is left instead of the action simply vanishing.

**Escape verified against the live model.** Given the trapped report it chose `switch_cv` to `rmsd` on `name CA`, citing `rounds_confined=2`, the per-round range "entirely in [0.03, 0.126]", and the 131 kJ/mol depth. Given a healthy unconverged round it chose `extend` — the check does not over-trigger.

**And the escape exposed one more gap.** The model justified RMSD as "bounded", which it is not (F6), and `cln025_contacts.yaml` carried no `cv_upper_wall_nm` — so the escape would have landed straight back in the runaway F6 came from. A campaign that allows a CV switch must carry a wall in case the replacement is length-dimensioned; `design_upper_wall` ignores it for types that do not need one. Wall added, guidance corrected to say what actually makes `rmsd` usable is the wall, not the coordinate.

**Also fixed:** the viewer plotted the raw `sum_hills` grid, which extends a few SIGMA past the outermost hill — so the axis ran to *negative contacts* and the frozen extrapolated shelf inflated the apparent depth (139 vs 116 kJ/mol). `metad_report` had always cropped; the plot now does too, using the round's own COLVAR snapshot.

### D3 — Anti-goals (from CLAUDE.md, recorded here for searchability)
- Do not rebuild MDCrow setup tooling — delegate via `adapters/`.
- Do not build a persistent multi-agent system; subagents are ephemeral function calls returning structured artifacts, not prose.
- Do not put raw trajectories / logs into agent context — only compact structured summaries + filesystem paths.
- Do not lock to one MD engine.
- Do not store campaign state in conversation; persist via `memory/`.

---

## 2. Session journal

### 2026-08-26 (very late) — Contacts CVs are fractions; viewer aligned; headless runner
Branch `test`. Three fixes from reading a real campaign's free-energy plot.

**`ContactsCV` now renders a fraction, always.** The raw `CONTACTMAP ... SUM` is an intermediate (`<label>_count`) and the CV exposed under the label is that count divided by the pair count, via `COMBINE`. A campaign was judging its states on a normalised fraction (0.3 / 0.7) while biasing the raw count, with both carrying the name `native_contacts_fraction` — so the free-energy axis ran to 10 where every threshold in the campaign lived below 1. A count and a fraction of the same contacts are the same coordinate scaled; only one of them can be compared with the rest of the campaign. `cv_series` divides to match, and the contacts SIGMA floor/ceiling are still stated in *contacts* but divided by the pair count, so "half a contact" keeps meaning that whatever the map size.

**The viewer superposes and centres.** `trajectory_pdb` now calls `superpose(traj, 0)` and `center_coordinates()`. Centre-of-mass drift across frames is 0.0000 nm where it was free to wander, and the residual RMSD (0-0.544 nm on a real round) is conformational change rather than tumbling.

**`python -m mdpilot.run <task.yaml> <work_dir>`** — the headless counterpart to the app, and the piece missing for server use: the only campaign CLI was `run_cln025.py`, hardcoded to one task file. Unlike the app, omitting `--biased-cap-ns` keeps the budget the file asked for rather than overriding it.

401 unit tests pass (4 new).

### 2026-08-26 (night, last) — The upper wall is measured from the simulation
Branch `test`. `bias_designer.box_limited_wall` fits the solute's widest extent against the CV over the source trajectory and solves for the extent at which only the nonbonded cutoff of solvent would remain between the solute and its periodic image. `design_upper_wall` uses it as the position when the campaign configured none, and records it either way so a configured wall beyond what the box can hold is flagged rather than silently honoured.

**It reproduces F11 as a pre-run check.** On `cln025_metad`'s own trajectory the derived limit is 0.55 nm; that campaign ran with a hand-chosen 0.8 nm wall. On `ui_campaign_chignolin` the limit is 0.38 nm. Both would have been refused-with-warning before a single biased round.

**It declines rather than guessing.** A 0.05 ns folded vanilla round has too little extent variation to extrapolate from, and returns None — the caller then emits the F6 warning ("no wall could be set") instead of inventing a ceiling.

**Three warnings, routed to the ledger** rather than only into plumed.dat, which the scientist never reads: an unbounded CV with no wall (F6), a wall beyond the box limit (F11), and a wall the campaign did not choose (say so, and note that if the task's unfolded state lies beyond it the box is too small for the question).

**Measured along the way, worth recording:** padding is applied correctly but not as `span + 2*padding`. Building CLN025 with equilibration disabled gives L = 3.76 / 4.26 / 4.78 nm for padding 1.0 / 1.5 / 2.0, i.e. clearances of 0.88 / 1.10 / 1.38 nm around the *folded* structure. The box grows by roughly the padding increment, not twice it. This is exactly why the wall is better measured than derived from the padding a campaign asked for.

399 unit tests pass (8 new).

### 2026-08-26 (night, later) — CV-switch escape hatch for trapped walkers
Branch `test`. Recorded as F13. Trap detection is computed, not inferred: `_confinement` reports `confined_to_state` and `rounds_confined` from the per-round observable files, alongside `observable_min/max_this_round` and the remaining CV-switch allowance. `action_switch_cv.md` names `rounds_confined >= 2` with a growing `fes_depth_kj_per_mol` as the clearest trap signal, and says why the cumulative `cv_min`/`cv_max` cannot show it.

`max_cv_switches` is now a task-file field (`sampling.max_cv_switches`) and is reported to the scientist as used/remaining.

Tested three ways: `_confinement` replayed against the real campaign's data; an end-to-end fake-adapter campaign that replays the trap and asserts the loop offers the switch only while the allowance lasts, honours it, and rebuilds on a different coordinate; and two live tests against the real model — one that must escape a trap, one that must *not* mistake a healthy unconverged round for one.

391 unit tests pass (4 new), plus 2 live.

### 2026-08-26 (latest) — Correcting F12: the app was dropping the system spec
Branch `test`. Traced the "wrong molecule" campaign properly instead of accepting the first explanation that fit. `2RVD` is chignolin; the agent was right. `app.py` called `run_campaign` without an adapter, so the default `SystemSpec.trpcage()` (1L2Y) was used and the task file's whole `system:`/`integrator:` block — PDB, force field, padding, 340 K — was discarded silently. F12 rewritten.

`TaskFile.build_adapter` now owns that construction so it cannot be forgotten; `app.py` and `benchmarks/run_cln025.py` both use it. Two other real defects fixed in the same pass: `observable_normalize` was missing from the setup agent's tool schema (so the agent literally could not express a fraction, which is why it kept emitting raw counts), and observable arity moved into `ObservableSpec.__post_init__` where it needs no topology — a `contacts` observable with one selection per strand now fails at load with a message explaining that contacts form pairs within a single group.

375 unit tests pass (4 new).

### 2026-08-26 (late) — Pre-flight checks, after a campaign ran 40 minutes toward an unreachable criterion
Branch `test`. Recorded as F12. `mdpilot/preflight.py` runs after `adapter.start()` and before the first step: `check_residue_count` compares the description's claim against the fetched structure, `check_observable_scale` compares the observable's value on the starting structure against its own state thresholds. Both raise; a `preflight_ok` event carries the numbers into the UI log.

`ObservableSpec.normalize` (contacts only) divides by the native-pair count, so a task file can say "fraction" and mean it. `description` is threaded into `run_campaign` for the check and deliberately kept *out* of the locked config — editing prose must not stop a resume.

`_FakeAdapter` now writes a real four-residue PDB rather than the string "PDB", so every loop test exercises the pre-flight path a real adapter takes. Two loop tests assert MD never started: `adapter.run_calls == []`.

Test data is the real thing — the drifted description, `2061.0`, and `(0.3, 0.7)` are the values from `campaigns/ui_campaign`. Parametrised cases pin the cases that must *not* fire: a folded start above its band, an unfolded start below it, CA-RMSD reading exactly 0 against its own reference, and an ambiguous description ("a 10-residue core inside a 30-residue construct") which is not treated as a claim at all.

371 unit tests pass (22 new).

### 2026-08-26 (night) — Streamlit control surface (`app.py`)
Branch `test`. **This overrides a stated scope decision**: `ROADMAP.md` lists a web UI under *Deliberately out of scope* — "Terminal + notebooks only until at least Milestone 6". Recorded here rather than left implicit. The justification is that the thing MDPilot is *about* — the scientist deciding, mid-campaign, whether to extend, stop or pivot — reads poorly as a terminal transcript, and a three-column view makes the reasoning legible next to the trajectory and surface it was taken on. It is a view over the existing backend and adds no science: every number it shows is read from `campaigns/<name>/`.

**One core change, deliberately small.** `run_campaign(on_event=...)` — an observer called with `(name, payload)` at `campaign_start`, `round_start`, `simulated`, `report`, `decision`, `override`, `pivot`, `campaign_end`. It cannot influence the run and `_emit` swallows anything it raises: a multi-hour biased run must not die because something watching it threw. Verified by running a campaign with an observer that raises on every event — it completes normally. Every `return` now goes through `_finish`, so the outcome is always reported. This is not a UI-specific seam; a CLI progress bar or M5's HPC monitor wants the same thing, which is why it is a callback rather than the app monkeypatching the loop.

**Structure.** Left: objective → setup agent → editable `task.yaml` → validate → *Lock & Run* (a worker thread; Streamlit forbids `st.*` off the main thread, so the worker only fills a queue the script drains on rerun). Middle: the formatted event stream, plus per-call token accounting from a recording Anthropic client so the setup agent's cache behaviour is visible. Right: campaign and round selectors defaulting to "Latest (agent-driven)", with py3Dmol structure, the free-energy surface, and the raw report.

**Two bugs the build surfaced**, both fixed:
- `mdtraj.save_pdb` takes a path, not a file object. Handing it a `StringIO` wrote the model to *stdout* and killed the calling script.
- The viewer reloaded the full trajectory on every rerun — and the app reruns every 1.5 s while a campaign is running. Round 12 of `cln025_metad_run3` is a 59 MB DCD, so the UI was unusable during exactly the runs it exists to watch. Now strided at load (`md.open` for the frame count first) and memoised on `(path, mtime, size)`: 0.15 s instead of seconds.

Verified headlessly with `streamlit.testing.v1.AppTest` — no exception, all three columns render, both selectboxes populate from real campaigns on disk. `streamlit`, `py3Dmol` and `matplotlib` are a `[ui]` extra, not dependencies; nothing under `src/mdpilot/` imports them. 349 unit tests pass (17 new).

### 2026-08-26 (evening) — Force-field selection: vocabulary, guide, and it is now the agent's to choose
Branch `test`. Recorded as D9. Before this the force field was not chosen by anything: `system.forcefield` sat in the loader's verified-but-fixed table and `setup_role.md` told the agent it was not adjustable. It is now a `SystemSpec` field, a task-file key, and an enum on the setup agent's tool schema generated from the vocabulary so the two cannot drift.

`knowledge/forcefield_guide.md` is the selection guidance — water-model tradeoffs, ff14SB vs ff19SB (and why `amber19/tip3p` is a documented compromise, ff19SB having been parameterised with OPC), CHARMM36 as an independent family for cross-checking, the cross-engine constraint, and the gaps. It ends with "do not change the force field to change a result", which is the failure mode a system that can pick its own parameters invites.

**Verified live on three requests, each with a different correct answer:** a cross-engine comparison chose `amber99sbildn/tip3p` and said why in its own description; a solvent-dynamics question chose `amber14/spce`; an ice-nucleation request chose the closest listed 4-site water and flagged that the campaign "cannot currently be honoured in full", naming the missing surface force field rather than inventing a key.

Also: `padding_nm` joined the setup agent's schema in the same pass, so the F11 box lesson is something it can act on rather than something only a human can set. 332 unit tests pass (13 new).

### 2026-08-26 (later still) — Dry run green; observable generalized; setup agent lands
Branch `test`. Items 1, 2, 3 and 5 of the agreed plan. 312 unit tests pass.

**1. Dry run (`campaigns/cln025_dryrun2`), exit 0.** First end-to-end exercise of the rewired entry path — task file -> `Ensemble` -> both adapters -> `run_campaign` -> OpenMM -> PLUMED -> `sum_hills` -> `decide`. Pivoted, 2 biased rounds, rendered `task_expectation` reached the scientist (its reason quotes "RMSD>4 Å"), new state-named `done_criterion.json` written, no warnings. `passed: false` is correct for a 0.1 ns shakedown.

**3. `mdrun` no longer runs under a wall-clock ceiling.** `_GMX_TIMEOUT_SECONDS = 1800` bounded every `gmx` call including production dynamics; a 2 ns CPU round plausibly exceeds it and would have died mid-round leaving a partial `.gro`/`.cpt`. Setup subcommands keep the ceiling (a hang there means something is stuck on stdin); all four `mdrun` sites pass `timeout=_NO_TIMEOUT`, pinned by a call-site test. Walltime planning is M5's job, not a constant here. The finite-size half of item 3 is recorded separately as F11.

**2. Observable generalized** — see D8. The test that documented the blocker (`test_a_non_protein_observable_is_refused_until_it_is_tunable`) became the test that proves it gone. Verified on real trajectory data: CA-RMSD, an end-to-end distance, Rg and native contacts all compute from `campaigns/cln025_metad/rounds/round_011.dcd`, and `make_report` produces a full bundle on a distance observable. Locked in the campaign config, omitted when it is the default so nothing already on disk is stranded.

**5. Setup agent** — see D8. Two defects surfaced by running it live on "I want to learn about the chignolin folding and unfolding", both now fixed:
- *The model wrote `characteristic_timescale_ns: 1.0` while its own source said "~1 µs".* A unit slip in the one decision-driving number with no second copy, and it inverts the pivot decision — the scientist would read 1 ns, conclude unbiased MD reaches the transition easily, and never switch. `_check_timescale_units` now refuses a timescale more than an order of magnitude from the unit its source names. Tolerance is one order of magnitude either way, so 0.6 µs stated as 600 ns still passes; the first version was tighter and rejected `cln025_folding.yaml` itself.
- *The retry loop was malformed and 400'd on its first retry.* It answered a refused `tool_use` with a bare user turn; the API requires a matching `tool_result` immediately after. So the loop never actually retried, and the failure escaped the `except ValueError`, leaving an invalid task file at `out_path`. Unit tests missed it because the fake client does not enforce the API protocol — confirmed against the real API before fixing. Candidates are now validated from a scratch path so `out_path` only ever holds an accepted file.

With both fixed the agent self-corrects: it now emits `1000` and spells the conversion out in the source ("~1 µs (≈1000 ns)"). It still needs human review — the citation it produced named Shaw as first author where Lindorff-Larsen is correct.

### 2026-08-26 (later) — Campaign config compares by meaning, not by bytes
Branch `test`. `store.init_campaign` compared the serialized config JSON byte-for-byte, so *adding* a config key stranded every campaign already on disk — their stored config could not contain a key that did not exist when they started. This had already happened: all four campaigns under `campaigns/` were unresumable, stranded by the commits that added `state_thresholds` and `min_recrossings`. Nothing surfaced it because all four had finished.

**Fix.** `_LEGACY_CONFIG_DEFAULTS` maps each later-added key to the behaviour in force *before* it existed — deliberately not to the parameter's current default, which is a different thing and would be wrong the moment a default changes. Both configs are normalized through it before comparison. A campaign that predates a key resumes when the requested value matches what it actually ran under, and is still refused when it does not. Read-time only; the stored config is never rewritten, so a campaign's record keeps saying what was actually recorded.

**Verified against the real databases** (copies, not the originals): all four now resume under their own parameters. `cln025_metad` still refuses against the current task file, correctly — it ran with `min_recrossings=1` and no task states, so recrossings came from the F9 surface-basin fallback, and reopening it under task-state counting would change when it is allowed to end for rounds already judged. Rendering `task_expectation` also changed that locked string, which is a genuine value change and not something a compatibility table should paper over.

**Forget-proofing.** `test_every_config_key_is_covered_by_the_compatibility_table` runs a campaign with every optional parameter set and asserts each recorded key is either in `_ORIGINAL_CONFIG_KEYS` or has a compatibility entry. Adding a config key without one now fails a test instead of silently stranding in-flight campaigns; confirmed by removing an entry and watching it fire.

**Also:** the refusal now names the differing fields instead of printing two JSON blobs, and marks values that were inferred rather than recorded (`stored=1 (not recorded; the behaviour before this key existed)`).

288 unit tests pass (9 new).

### 2026-08-26 — Task file becomes the campaign contract; `task_expectation` rendered
Branch `test`. Recorded as D7. `src/mdpilot/task_file.py` + `tests/unit/test_task_file.py` (20 tests); `run_cln025.py` rewired to load the file instead of hand-assembling kwargs, producing byte-identical `run_campaign` arguments in both dry-run and full mode.

**Two things found by doing it.** `trpcage_convergence.yaml` never parsed — its `diagnostics:` block mixed a sequence and a mapping key, invalid YAML since it was written, and nothing had ever loaded it. Fixed. And the `observable.name` verification refused a ligand-distance test case, which is correct and surfaced the protein-CA blocker as a tested fact.

**Pre-existing, not caused by this work:** all four on-disk campaigns are already unresumable. `min_recrossings` and `state_thresholds` were added to the config lock after they were created (`git show HEAD` confirms both keys predate this session), so their stored configs are missing keys the loop now computes. `store.py` has migration machinery for the `rounds` *schema* but none for the campaign *config*, so every added key strands every in-flight campaign. Impact today is nil — all four finished — but a 20 ns run interrupted after a future key lands would be lost. Rendering `task_expectation` also changes that locked string, so campaigns started under the authored prose cannot resume under the rendered one either.

279 unit tests pass (9 new here, on top of the day's earlier `Ensemble` work).

### 2026-08-25 (later) — `Ensemble` on the spec: temperature and timestep become campaign parameters
Branch `test`. First slice of making setup tunable. `_TEMPERATURE_K` and `_TIMESTEP_FS` / `_TIMESTEP_PS` were adapter class constants — in the GROMACS case interpolated into the mdp templates *at import*, so no campaign could reach them. They now live on `SystemSpec.ensemble`.

**Why the spec and not an adapter keyword argument.** `run_campaign` already locks `adapter.spec.to_dict()` into the campaign config, so every field added to the spec is covered by the resume guard for free. An adapter keyword would need its own config key, and `run_campaign`'s docstring had been asserting that temperature and timestep were "covered by the engine lock" — true only while they were immutable class constants. That coverage would have vanished silently the moment they became settable, and a campaign could have been resumed at a different temperature with the guard none the wiser. Test: `test_ensemble_mismatch_on_resume_is_rejected`.

**Backward compatibility.** `SystemSpec.to_dict()` emits `ensemble` only when it differs from the default, because `store.init_campaign` compares config JSON byte-for-byte. Verified against the real `campaigns/cln025_metad` and `_run3` databases: both still rebuild an identical `system_spec`. The same rule is used for the new `bias_pace` / `bias_factor` config keys.

**Scope, deliberately two fields.** Water model, force field, padding, ionic strength and pressure stay hardcoded per adapter; they get fields when a campaign needs them, per the rule `SystemSpec` already follows. `timestep_fs > 2.5` raises rather than silently integrating unstably — neither adapter implements HMR and both constrain h-bonds only.

**Also landed:** `bias_pace` / `bias_factor` pass-through on `run_campaign`. `design_bias` already accepted both; the loop never passed them, so no campaign could change the shape of its own bias. `None` means "let `bias_designer` decide", so its defaults stay the single definition. Worth exercising: γ=10 flattens barriers to ~γ·kT ≈ 25 kJ/mol, and the CLN025 surface reached `fes_depth ≈ 41 kJ/mol`.

259 unit tests pass (7 new).

### 2026-08-25 — Prompt knowledge base: the scientist retrieves its rules per round
Branch `test`. `scientist.py` held one static system prompt covering both phases and the whole CV vocabulary, sent unchanged every round: 10,835 chars / ~2,900 tokens, of which roughly half described actions the round could not take. Split into six Markdown chunks under `src/mdpilot/knowledge/`, assembled per round by `build_system_prompt`.

**Retrieval is by key, not similarity.** The keys are the facts that already select the tool schema — phase, whether the tool carries a `metad_proposal` field, whether `switch_cv` is offered. `can_propose_cv` is read off the selected schema rather than re-derived from `(phase, task_expectation, allow_cv_switch)`, so the vocabulary is present exactly when the field that consumes it is and the two cannot drift.

| round type | ~tokens | vs before |
|---|---|---|
| vanilla, pivot possible | 1,650 | −44% |
| vanilla, pure convergence | 1,030 | −65% |
| metad, no switch left | 1,470 | −50% |
| metad, switch offered | 2,455 | −16% |

**No retrieval *within* `cv_vocabulary`, deliberately.** The obvious next filter is "show only the relevant CV types", and it is wrong: the scientist is choosing among them. `cln025_folding.yaml` already says pre-selecting a CV "would decide the science the agent exists to decide", and runs 1 and 3 differed only in the model picking `contacts` over `rmsd` unprompted. The chunk is retrieved whole or not at all.

**Two latent prompt defects surfaced by the split**, both from text that assumed every round saw everything:
- `reason` instructed the model to cite `plateau_reached`, `ess`, `exploring`, `n_basins`, `bimodality_coefficient` — all vanilla-only fields that `phase_metad` explicitly says are absent. Every biased round since the phase split was being asked for numbers it had been told it would not get. Now phase-neutral.
- `role` said the report was "Phase-dependent; see below" and described a two-phase prompt. Reworded to say the round carries one phase's rules only.

**Caching.** Four assemblies, constant within a campaign phase, so each caches after its first round; the pivot invalidates the prefix on the one round the assembly changes anyway. All four clear Sonnet 4.6's 1024-token minimum, including the smallest: `cache_control` sits on the system block but the cached prefix is tools → system, so the ~715-token tool schema counts toward it. A pure-convergence vanilla round measures a 1,014-token system block and a 1,729-token prefix, and caches (`cache_read_input_tokens` 1,413 on the second call, measured 2026-08-25). Retrieval cannot trim a round out of the cache — an earlier draft of this entry said it could, having compared the system block against the minimum rather than the prefix.

Content moved verbatim: each chunk was verified whitespace-normalised-identical to the section it replaced before any edit was applied. `[tool.setuptools.package-data]` added — without it a non-editable install ships the code and none of the prose, and every `decide()` fails on the first chunk read; verified by building a wheel and loading a chunk from a clean install. 247 unit tests pass (8 new).

### 2026-08-24 — External review pass: two convergence-path defects, and the loop stops guessing at physics constants
Branch `test`. A full read of the tree surfaced two defects on the path that decides when a campaign is allowed to end, plus three structural gaps. No campaign was re-run; everything here is unit-covered.

- **F10 — drift was identically 0.0 at exactly two FES estimates.** The one that mattered: it could hand `fes_converged=true` to `_refuse_premature_stop`, which is the single mechanism standing between the scientist and declaring victory against the diagnostic. Reproduced, fixed, and covered. Full write-up in F10.
- **F9's "no longer reachable by accident" was half true.** `decide()` picked its tool by phase alone, so `switch_to_metad` was offered on every vanilla round regardless of whether a `task_expectation` existed to judge it against — and a campaign with neither key passed the entry guard and could still pivot into the F9 fallback. A `_CONVERGENCE_TOOL` (`extend | stop`) now covers that case. Note the shape of the miss: the guard was placed at `run_campaign`'s entry, where the *arguments* are, while the thing that needed guarding was the *action space*, two modules away. The test that should have caught it asserted the false premise in its docstring instead.
- **The loop no longer hardcodes the timestep or the thermostat.** `_TIMESTEP_FS = 2.0` and `_METAD_TEMPERATURE_K = 300.0` sat in `loop.py` as values that merely happened to match both adapters, with nothing connecting them. `MDAdapter` gains `timestep_fs` and `temperature_k`; the loop reads both off the adapter once and threads them through `_extend_steps`, the biased-step budget, `metad_report` and `design_bias`. This is an M3 gap, not a tidiness one — the whole point of the Protocol is that the loop does not know which engine it has, and it was quietly assuming two physics constants of one. A wrong dt makes `extra_ns` stop meaning nanoseconds; a wrong T makes PLUMED compute its well-tempered scaling against a temperature the thermostat is not running at, with no error anywhere. New `tests/unit/test_adapter_contract.py` pins both adapters and their agreement.
- **`min_recrossings` joined the locked config** — see the F9 addendum.
- **`architecture.md`'s file structure now distinguishes what exists from what is planned.** It listed `reasoning/`, `ensemble/`, `execution/`, `tools/`, `configs/`, `notebooks/`, three diagnostics modules and three memory modules that have never existed, alongside benchmark task names that were never used. CLAUDE.md §1 sends coding sessions to that file to verify load-bearing decisions, so the drift was self-inflicted. Every path in the "exists" block was checked against the tree.
- **In-flight contacts-CV work landed first**, as its own commit: PBC-aware native-pair selection, the `MM == 2*NN` import-time invariant, and `MetadProposal.cv_type` catching up to `_CV_TYPES` with a test that pins all three copies of the vocabulary together.
- Tests: **239 unit + 1 skipped** (was 223), ~14 s.
- **Known residual:** `run_campaign` still *accepts* a `switch_to_metad` decision from a campaign with no `task_expectation`. The model can no longer produce one, but a caller stubbing `decide` can — which is exactly how the pivot orchestration tests drive that branch, so closing it would mean reworking fourteen of them. Left open deliberately; the reachable path is closed.
- **Open / next:** unchanged from the previous entry (warm-start handoff across a pivot or switch; `switch_cv` unexercised by a live campaign), plus: `scipy` is declared in `pyproject.toml` and imported nowhere.

### 2026-08-21 (later) — Run 3: M4 done-criterion **met**
`campaigns/cln025_metad_run3/`, `biased_budget_exhausted` at exactly 20.0 ns over 11 biased rounds. `done_criterion.json`: **`passed: true`**, `rmsd_recrossings = 3` against 2 required, CA-RMSD spanning 0.024-8.00 A, both states reached.

- **The scientist chose the CV itself.** From a 1 ns vanilla round (ess=221, `exploring=false`, `n_basins=1`) it judged the folded basin well-sampled but the task unreachable — "the CLN025 folding timescale is 0.6-1.0 us, roughly two orders of magnitude" beyond budget — and proposed `cv_type: contacts`, `label: native_contacts_ca` from the vocabulary widened the day before. No one steered it toward contacts.
- **The coordinate is what changed the outcome, not the compute.** Run 1: 4 ns biased on RMSD, 1 recrossing. Run 2: 20 ns biased on RMSD plus an 0.8 nm wall, 1 recrossing. Run 3: 20 ns on native contacts, **3 recrossings**. Run 2 spent the same budget on the same system and never came back; F6's diagnosis predicted exactly this and it held.
- **The refolding event.** Q went 10.5 (folded) -> 0.43 (57% of frames below Q=2 in the 1-2 ns window) -> back above 8, with **100% of frames in the 6-7 ns window above Q=8** and holding through 7-8 ns. Not a transient touch of the native basin — it repopulated it and stayed. The criterion was already satisfiable by round 5, at 7 of 20 ns.
- **No wall was used or needed.** `design_upper_wall` returns None for a contact count, and the bounded coordinate removed the runaway mechanism by construction rather than by restraint.
- **The F7 fix earned itself mid-campaign.** At round 7 the scientist was handed `recrossings=0, barrier_crossed=false` and did not believe it: *"n_basins_fes=1 suggests the bias has not yet resolved the two-basin landscape in this estimate window. The walker's CV range (0.59-10.56) still spans both the extended and folded sides, confirming the coordinate is physically correct."* It cross-checked a broken field against `cv_min`/`cv_max` — fields that only report the *sampled* range because of the F7 work two days earlier — and reached the right conclusion.
- **`switch_cv` was never triggered.** Implemented, tested, and unused: the scientist picked a working coordinate first time, so nothing asked it to revise. Insurance that did not have to be claimed. Its resume and cap behaviour remain covered by unit tests only, not by a live campaign.
- **Drift never converged** — it oscillated 16.6 to 55.0 across all eleven biased rounds with no trend, so `fes_converged` stayed false throughout and `_refuse_premature_stop` had nothing to refuse. The campaign ended on budget, not on convergence. A 20 ns well-tempered surface on this system is a barrier-crossing demonstration, not a converged free-energy profile, and should not be quoted as one.
- **New finding F9** on the `recrossings` diagnostic, which this run broke in both directions.
- **Infrastructure.** Two GPU context deaths (`CUDA_ERROR_LAUNCH_FAILED` via cuFFT) — one killing run 2 at round 10, one killing run 3's first attempt at round 2. The GPU is in WDDM mode and shared with the Windows desktop, so a driver reset is the leading hypothesis; unproven. Run 3 was relaunched under a resume-on-crash wrapper with a no-progress guard (retry only if the attempt completed a round, so a deterministic bug cannot be masked by looping). It was not needed after the relaunch — one attempt ran the full ~4.5 h. Throughput ~96 ns/day biased, consistent with the ~104 measured earlier.
- **Open / next:** fix F9 (needs the same anchoring decision F7 left open); `switch_cv` still unexercised by a live campaign; warm-start handoff across a pivot or switch still deferred; README test count and CRLF churn still outstanding.

### 2026-08-21 — `switch_cv`: the scientist can revise a CV it got wrong
Closes the gap the previous entry left open. The scientist could already *see* that its coordinate had failed (F7's published boundaries); it had no legal move but `extend`.

- **Architecture read first**, per CLAUDE.md §1. CV revision is in scope by the thesis — `architecture.md`'s position #2 names picking metaD CVs as MDPilot's job, and a scientist that cannot revise one is running the inner loop with extra steps. D1 anticipated the action space growing at M4. The 2026-06-08 in-place-pivot decision ("the campaign is one logical thing; a two-phase seam would leak into the user API") argues against making CV failure a human-arbitrated stop, so `switch_cv` continues the campaign in the same call rather than terminating for a restart. Per §5 the decision stays with the scientist, not a subagent: "why did this CV fail and what replaces it" is narrative-dependent by definition.
- **Prerequisite closed first:** the upper SIGMA guard (see F4). A replacement CV is sized on the outgoing bias's trajectory, which is precisely the inflated-spread case the floors could not catch.
- **Action space is offered per round, not per phase.** `decide(..., allow_cv_switch=)` selects between `_METAD_DECISION_TOOL` (`extend | stop`) and `_METAD_SWITCH_TOOL` (`extend | stop | switch_cv`, with `metad_proposal` back in the schema). Once `max_cv_switches` is spent the action is **dropped from the enum** rather than refused at runtime — the same "unrepresentable beats emitted-then-rejected" reasoning that keeps a second `switch_to_metad` out of the biased tool. Default allowance is 1; recomputed from the persisted rounds so a restart cannot buy a second one, exactly as the biased-step meter does.
- **The outgoing hills are deleted, not archived.** They describe a different coordinate and PLUMED would read them back as though they did not. `rounds/round_NNN.hills` already preserves them via `_snapshot_bias_state`, so deletion loses no evidence — and `unlink(missing_ok=True)` is idempotent, so a resume re-entering the path cannot accumulate half-written archives of an interrupted round.
- **Resume branch ordering is the subtle part.** A `switch_cv` round is itself biased, so the generic `plumed_dat_path is not None` branch matches it and would resume the campaign on the CV the scientist just rejected, with `RESTART` reading its hills back. The new branch is ordered ahead of it — the same class of bug as the second-pivot resume ordering found in the 2026-08-19 review. There is a regression test for exactly this.
- **Known limitation, inherited deliberately:** a switch rebuilds the biased adapter the way `_pivot_to_metad` does, from the cached equilibrated system rather than the previous round's checkpoint. An OpenMM checkpoint is only valid for the System it was written from, and the replacement bias changes the `PlumedForce`; carrying positions across would need a coordinate-level handoff the `MDAdapter` Protocol does not expose. So a switch discards the conformation reached under the old bias. This is the same warm-start gap the original pivot has, now on a second code path.
- **Budget is cumulative across CVs** — the 20 ns is a compute budget, not a per-coordinate allowance, so a late switch buys little. The prompt says so explicitly.
- **SQLite migration.** The `rounds.decision` CHECK constraint had to widen. SQLite cannot ALTER a CHECK in place, so `_migrate_rounds_for_cv_switch` does rename-create-copy-drop like the M4 migration; unlike the other two it adds no column, so idempotency is detected from the stored DDL rather than `PRAGMA table_info`. Tested for both the widening and the copy not dropping rows.
- **`_compact_prior` now carries `cv_label`** on biased rounds. Across a switch the campaign history holds recrossing counts from two different coordinates, and a bare list of them invites exactly the comparison F7 was about.
- **Verified against real PLUMED, not just rendered text.** Building the replacement bias through `_build_plumed_input` on run 2's own trajectories: sizing on round 2 (which spans folded→unfolded) gives a measured SIGMA of 0.948; sizing on round 12 (fully unfolded, no variance on the new coordinate) correctly **floors to 0.5 and says so**. The composed input — CONTACTMAP continuation block, then METAD, then PRINT — parses, deposits hills, and writes COLVAR at `q_native ≈ 9.85`. No `UPPER_WALLS`, since a contact count needs none.
- Tests: **214 unit + 1 skipped** (was 199), 227 with the fast PLUMED integration files.
- **Open / next:** a fresh CLN025 campaign with `contacts` in the vocabulary and `switch_cv` available. Note the scientist may still choose `rmsd` — the prompt now explains the bounded/unbounded distinction but the choice is genuinely its own, and either outcome is informative. Warm-start handoff across a pivot or switch remains deferred.

### 2026-08-19/20 — Two CLN025 campaigns, biased-diagnostics audit, `contacts` CV
The M4 done-criterion run, twice, and the defects both runs exposed. Findings F6-F8 carry the detail; this is the work record.

- **Run 1** (`campaigns/cln025_metad_run1/`) stopped itself after 4 ns of a 20 ns budget. Three causes, in descending order of severity: `fes_drift` was structurally **always 0.0** — when the hill count is a multiple of `--stride`, `sum_hills` writes the final surface *and* an identical last strided one, and drift compared that pair, so the drift half of the convergence gate had been vacuous in every biased round ever computed (recomputed on run 1's real data: **0.0 → 91 kJ/mol**, 36× kT); `barrier_crossed` had been gated on `min_recrossings`, contradicting its own name, which the scientist caught live in round 3 and then reasoned badly from; and the agent broke an explicit prompt rule to stop anyway.
- **Fixes between runs:** trailing duplicate surface dropped and drift measured against the half-way estimate; `barrier_crossed` back to `n >= 1`; `_refuse_premature_stop` converts a biased-phase `stop` to an extend while `fes_converged` is false and budget remains, writing the refusal into the ledger rather than swallowing it; `UPPER_WALLS` support (`UpperWall`, `PlumedInput.walls`, `design_upper_wall`) with `cv_upper_wall_nm` locked into the campaign config.
- **Run 2** (`campaigns/cln025_metad/`) used the full 20 ns across 11 biased rounds and ended `biased_budget_exhausted`. The enforced stop rule held — nine consecutive `extend`s with `fes_converged=false`, no early exit. Physics result in F6.
- **Resume worked, and its crash-window guard earned itself.** Run 2 was killed mid-round-10 by a machine event, having deposited 1,206 hills never paired with a checkpoint or a persisted row. Resume restored the round-9 `HILLS`/`COLVAR` snapshot over the live files and enabled `RESTART`: PLUMED read back **exactly 15,000 Gaussians** (= 15.0 ns at PACE 500 × 2 fs), the orphan hills were discarded rather than double-counted, the budget meter continued at 15/20 rather than granting a fresh 20, and the final round was clamped to land on exactly 20.0 ns. First real exercise of the D4 window; without the snapshot pairing this would have produced a locally over-filled surface that integrates cleanly and looks fine.
- **`contacts` CV added** — `ContactsCV` rendering `CONTACTMAP ... SUM`, native CA-CA pairs within 0.75 nm resolved from the campaign reference with sequence separation < 3 excluded, SIGMA floor 0.5 (dimensionless count, not nm), and no wall (`design_upper_wall` already returns None for anything not length-dimensioned — the point of a bounded coordinate). On CLN025 it resolves **12 native contacts**, and they are the actual β-hairpin register (res 1↔9,10; 2↔7-10; 3↔6-10; 4↔7). Driven over run 2's own trajectories: **Q ≈ 7.82/12 folded (round 2) vs 0.69/12 unfolded (round 12)**.
- **`plumed driver --mf_pdb` silently reports every pairwise distance as zero.** It parses and runs, so a contact map comes back saturated at 1.0 per pair — which happened to equal the "≈ number of pairs" assertion in the first live test written, and passed for exactly the wrong reason. The `--mf_dcd --natoms` path is correct: PLUMED **22.670438** vs the Python evaluation's **22.670439**. Documented in `tests/integration/test_cv_designer_live.py`; the cross-validation test is what caught it, not the self-consistency one.
- **Schema/resolver drift pinned.** `test_scientist` now asserts the tool schema's `cv_type` enum equals `cv_designer._CV_TYPES` rather than a literal list. A type added to one and not the other was previously a runtime `ValueError` on a live campaign, and every unit test mocks the client.
- **Measured throughput:** ~**104 ns/day** biased vs ~325 ns/day unbiased on the 1660 SUPER — an RMSD-over-all-CAs `PlumedForce` costs roughly 3×. The earlier 325 figure was unbiased and should not be used to size a biased campaign.
- Tests: **199 unit + 1 skipped** (was 188), plus `tests/integration/test_cv_designer_live.py` (2 tests) and 2 added to `test_free_energy_live.py`.
- **Open / next (done 2026-08-21, see the entry above):** `switch_cv` — the scientist can now *see* that its CV failed (F7's published boundaries) but still cannot act, since the biased action space is `extend | stop`. Architecture read done: CV revision is in scope by the thesis (`architecture.md` position #2 names picking metaD CVs), D1 anticipated the action space growing at M4, and the 2026-06-08 in-place-pivot decision argues against making CV failure a human-arbitrated stop. **Prerequisite:** F4's guard is a lower floor only — sizing a replacement CV from a *biased* trajectory inflates its spread, and there is no upper SIGMA guard to catch it. Then a fresh campaign with `contacts` available.

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
