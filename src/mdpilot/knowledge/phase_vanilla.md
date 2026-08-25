Action space is `extend` or `stop`, and — only when `switch_to_metad` appears
in the `decision` enum of your tool — `switch_to_metad`:

- `extend` — run more vanilla MD; the trajectory needs more time.
- `stop` — the observable has converged for the task at hand; the campaign
ends.
- `switch_to_metad` — vanilla MD is *inadequate*: the system is pinned in a
single basin AND the task requires a transition the budget cannot reach.
Propose a collective variable for metadynamics.

`switch_to_metad` is offered only when `task_expectation` is non-null. A
campaign with no stated expectation is a pure convergence task: there is no
required transition to judge the budget against, so the pivot is not yours to
make and the action is absent from your tool. When it is absent, do not argue
for it.

Report fields. Convergence: `plateau_reached`, `ess`, `tau_int_frames`,
`statistical_inefficiency_*`. Exploration: `bimodality_coefficient`,
`n_basins`, `minor_basin_occupancy`, `exploring`. The two
statistical-inefficiency fields (block-averaging, autocorrelation) should
agree if the diagnostic is reliable; flag >2× disagreement in `reason`.

Decision rule:

- `exploring=true` (n_basins >= 2): the system has visited multiple states;
vanilla MD is reaching them. Decide between `extend` and `stop` on convergence
numbers — `plateau_reached AND well_sampled AND ess>=50` → `stop`, else
`extend`.

- `exploring=false` (pinned, n_basins == 1) AND `task_expectation` does not
require a transition (or is null): single-basin convergence. Same rule —
`plateau_reached AND well_sampled AND ess>=50` → `stop`, else `extend`.

- `exploring=false` AND `task_expectation` explicitly requires a transition
that the budget cannot reach (compare cumulative simulation time to the
characteristic timescale in the expectation): vanilla is inadequate. Decide
`switch_to_metad` and propose a CV.
