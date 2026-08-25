Sizing `extra_ns` when extending: proportional to the gap — 0.5 ns when
borderline, up to 2.0 ns when far (vanilla: ess<5 or no plateau; metad: drift
well above kT or zero recrossings). When `stop` or `switch_to_metad`,
`extra_ns` must be null.

For `ledger_note`: record insights worth carrying across rounds — a hypothesis
about the slow coordinate, the reason for an unusual CV choice, a rate
estimate. Pass null when nothing new is worth recording.

For `reason`: cite the specific diagnostic numbers that drove the call, named
in your phase's section above, and if switching briefly justify the CV in
physical terms. One to three sentences.

You MUST call the `record_decision` tool. Do not respond in plain text.
