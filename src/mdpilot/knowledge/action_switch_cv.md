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
cannot escape along this coordinate;
- **the walker is trapped**: `rounds_confined` is 2 or more, meaning this
round's observable range and the previous rounds' all sat entirely inside the
single state named by `confined_to_state`, while `fes_depth_kj_per_mol` kept
rising. This is the clearest trap signal you have, and it is the one the
cumulative `cv_min`/`cv_max` cannot show you — those keep reporting the widest
excursion the campaign ever made. A bounded coordinate does not protect you
here: a contact count collapses every disordered conformation onto roughly the
same value, so once the system is disordered the bias fills one degenerate bin
and cannot lead it back. Prefer a replacement that separates the states on the
side you are stuck in — an `rmsd` with an upper wall, or a `gyration`, both of
which still distinguish disordered structures a contact count cannot.

`cv_switches_remaining` says how many revisions the campaign has left. When it
reaches 0 the action disappears from your tool; spend one on a coordinate you
have evidence against, not on a surface that is merely still filling.

Prefer a coordinate that is bounded on both sides when the failure was a
walker that left and did not come back — but do not mistake `rmsd`, `distance`
or `gyration` for bounded. They are unbounded above; what makes `rmsd` a usable
replacement is the upper wall the campaign configures, not the coordinate
itself. Say which it is in `reason`; claiming a coordinate is bounded when it
is not is how the previous CV was chosen. Justify the replacement in `reason` by
naming what the previous CV failed to do, and record the diagnosis in
`ledger_note` — the next rounds will be judged on a different coordinate and
the history has to explain the discontinuity.
