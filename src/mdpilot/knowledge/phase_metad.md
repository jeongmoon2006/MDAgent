=== PHASE `metad` — well-tempered metadynamics ===

Action space is `extend` or `stop`, and — only when `switch_cv` appears in the
`decision` enum of your tool — `switch_cv`. When it is absent the campaign has
spent its CV-revision allowance and the choice is not yours to make; do not
argue for it. Proposing another *pivot* is never available: the campaign has
already pivoted.

The equilibrium convergence fields are deliberately absent from this report. A
biased trajectory is not an equilibrium ensemble — the bias drives the
observable — so a long autocorrelation would mean the bias is still filling
and a bimodal marginal would mean the bias worked, not that the system is
sampling freely. Do not ask for those numbers or reason as if you had them.

Report fields, all derived from the deposited bias (HILLS) integrated into a
free-energy surface:

- `fes_drift_kj_per_mol` — how much the surface changed between the last two
cumulative estimates. The standard well-tempered convergence test.
- `recrossings` — barrier crossings, counted with hysteresis between
`recrossing_low` and `recrossing_high`. `barrier_crossed` is `recrossings >=
1`. `recrossing_basis` says what those boundaries are:
- `task_states` — the states your task defines, measured on
`recrossing_observable`, which is usually *not* the CV you are biasing. Fixed
for the whole campaign, so the count is comparable across rounds and across a
change of CV. Trust this one.
- `fes_basins` — the two deepest basins of the *current* surface. These move
as the bias fills, so a count on this basis means something different every
round; compare the boundaries against your task's states before reading it.
- `recrossings` may be `null`, meaning the count could not be taken at all
(fewer than two basins resolved on the surface). That is not the same as zero
crossings. Do not treat a null as evidence the walker stayed put — check
`cv_min`/`cv_max` against `cv_start` to see how far it has actually moved.
- `fes_converged` — true only when drift is below kT (≈2.5 kJ/mol at 300 K)
AND `recrossings >= 1`. Low drift *alone* is not convergence: a walker that
never left its starting basin produces a surface that stops changing
immediately, because nothing new is being sampled.
- `observable_min_this_round` / `observable_max_this_round` — the range the
walker covered on the task observable *in this round alone*. Compare them
against `recrossing_low`/`recrossing_high`: a range that sits entirely inside
one state, and shrinks round on round, is a walker that has settled there. That
is the signal `cv_min`/`cv_max` cannot give you — those are cumulative over the
whole biased phase, so they go on reporting the widest excursion the campaign
ever made long after the walker stopped moving. If the per-round range has
collapsed into one state while `fes_depth_kj_per_mol` keeps growing, the bias
is filling a basin the coordinate cannot lead the system out of: say so in
`reason` and record it in `ledger_note`.
- `n_basins_fes`, `barrier_kj_per_mol`, `fes_depth_kj_per_mol`,
`n_fes_estimates` — shape of the surface recovered so far. `cv_min` and
`cv_max` are the range the walker actually visited, and `fes_depth` is
measured over that range only, not over the wider grid `sum_hills` writes.

Decision rule:

- `fes_converged=true` → `stop`. The surface has stopped moving and the walker
has crossed the barrier at least once.
- otherwise → `extend`. This includes `fes_converged=null` (not enough
estimates or no COLVAR yet) and the low-drift/zero-recrossing case, which is
an under-filled basin, not a converged surface.
- If many rounds have passed with `recrossings=0` and a large
`fes_depth_kj_per_mol`, say so in `reason` and record it in `ledger_note` —
that pattern suggests the biased CV is not the slow coordinate. You cannot act
on it, but a human reading the ledger can.
- Before treating `recrossings` as evidence about your task's transition,
check `recrossing_low` and `recrossing_high` against the states the task
describes. If both boundaries sit on the same side of those states, the count
is measuring motion *within* one state rather than the transition you were
asked for, and a non-zero count is then not evidence that the CV is working.
Say so in `reason` and record it in `ledger_note`.
