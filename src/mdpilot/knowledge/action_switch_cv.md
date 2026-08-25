--- Action `switch_cv` (available this round) ---

`switch_cv` replaces the biased collective variable and starts a fresh bias on
the new coordinate. The deposited hills on the old CV are kept as a record but
are not carried over — they describe a different coordinate. The compute
already spent is *not* refunded: the biased budget is cumulative across CVs,
so a switch late in a campaign buys little. Use it when the evidence says the
coordinate is wrong, not when the surface is merely still filling:

- the boundaries `recrossings` was counted between sit on the same side of the
states your task describes (see the rule below), so the count is not measuring
the transition you were asked for;
- the walker left the region it started in — compare `cv_start` against
`cv_min`/`cv_max` — and many rounds have passed without it returning;
- `recrossings` has stayed at 0 across several rounds while
`fes_depth_kj_per_mol` keeps growing, which is a bias filling a basin it
cannot escape along this coordinate.

Prefer a coordinate that is bounded on both sides when the failure was a
walker that left and did not come back. Justify the replacement in `reason` by
naming what the previous CV failed to do, and record the diagnosis in
`ledger_note` — the next rounds will be judged on a different coordinate and
the history has to explain the discontinuity.
