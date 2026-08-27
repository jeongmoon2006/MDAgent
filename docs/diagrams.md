# MDPilot — architecture at a glance

Three diagrams, in the order you would draw them on a whiteboard. For the
reasoning behind these choices see `architecture.md`; for the empirical
failures most of the guardrails came from, see `activity-log.md`.

> Most MD agents automate the *inner* loop — set up, submit, analyze.
> MDPilot owns the *outer* loop: look at the result, judge whether it answers
> the question, decide what to run next.

---

## A — End to end

One LLM call turns a question into a reviewable spec. A human approves it.
Deterministic Python builds and runs everything after that.

```mermaid
flowchart TD
    Q["researcher asks:<br/>'does chignolin fold, and how fast?'"]
    SA["SETUP AGENT — setup_agent.py<br/>first LLM call, once per campaign<br/>strict tool use, emits structured fields"]
    TF["task file (YAML)"]
    HR["HUMAN REVIEW<br/>nothing runs until a person reads this"]
    V["task_file.py + preflight.py<br/>every declared field checked against the<br/>code that consumes it; unknown key = refuse"]
    LOOP["run_campaign() — the loop<br/>see diagram B"]
    OUT["trajectories · free-energy surface · decision log"]

    Q --> SA --> TF --> HR --> V --> LOOP --> OUT
```

The point to say out loud: **the LLM never touches the simulation.** It writes
a proposal, a human approves it, deterministic Python builds it.

---

## B — Inside the loop

One LLM call per round. Everything else is a mechanical state machine.

```mermaid
flowchart TD
    START(["campaign starts / resumes"]) --> SIM
    SIM["SIMULATE<br/>adapter: OpenMM or GROMACS"]
    CKPT["CHECKPOINT + PERSIST<br/>SQLite state.db + filesystem"]
    DIAG["DIAGNOSE<br/>deterministic — no LLM"]
    DEC{"DECIDE<br/>scientist.py — one LLM call"}
    PIVOT["resolve CV · size bias · write plumed.dat<br/>cv_designer + bias_designer + plumed_writer"]
    DONE(["campaign ends"])

    SIM --> CKPT --> DIAG --> DEC
    DEC -->|extend| SIM
    DEC -->|stop| DONE
    DEC -->|"switch_to_metad (vanilla phase)"| PIVOT
    DEC -->|"switch_cv (biased phase)"| PIVOT
    PIVOT --> SIM
```

### Two phases, and the phase changes what the agent may see

Not the same numbers relabelled — genuinely different report bundles. A biased
trajectory is not an equilibrium ensemble, so effective sample size and
"plateau reached" do not describe convergence there: a long correlation time
means the bias is still filling, and a bimodal marginal means the bias
*worked*. Handing those to the model as convergence evidence is a category
error, so they are absent rather than discouraged.

| | phase `vanilla` | phase `metad` |
|---|---|---|
| what is running | unbiased MD | well-tempered metadynamics on a CV |
| diagnostics | block-averaged SEM<br/>autocorrelation → ESS<br/>bimodality → exploring? | HILLS → free-energy surface<br/>drift between successive estimates<br/>barrier recrossings |
| action space | `extend` · `stop` · `switch_to_metad` | `extend` · `stop` · `switch_cv` |

The action space is enforced by the **tool schema**, not the prompt. If an
action is not valid this round it is not in the enum — unrepresentable rather
than discouraged. A second pivot does not exist in the biased schema;
`switch_to_metad` is dropped entirely from campaigns that declared no
expectation to judge a pivot against.

---

## C — The boundary

If you only draw one thing, draw this. Nothing with a physical unit ever
passes through the model.

```mermaid
flowchart LR
    subgraph LLM["THE LLM DECIDES — judgment, chemistry"]
        L1["extend / stop / pivot"]
        L2["which coordinate is slow<br/>e.g. 'backbone radius of gyration'"]
        L3["why — reason + ledger note"]
    end

    subgraph CODE["CODE DECIDES — numbers, physics"]
        C1["how many steps that actually is"]
        C2["which atom indices it resolves to<br/>cv_designer, against the topology"]
        C3["SIGMA / HEIGHT / PACE<br/>bias_designer, from the trajectory"]
    end

    subgraph GATE["CODE OVERRULES"]
        G1["has it converged?<br/>diagnostics — no model involved"]
        G2["is it allowed to stop?<br/>refuses a stop against the data"]
        G3["budget caps + extension clamps<br/>recounted from disk on resume"]
    end

    L1 --> C1
    L2 --> C2
    L2 --> C3
    GATE -.->|constrains| LLM
```

Everything on the right is deterministic, testable and reproducible. That is
what makes a campaign auditable: `plumed.dat` and the per-round records can be
read back months later and say exactly what ran and why.

### Why the split falls where it does

A quantity with a right answer derivable from the data should be derived, not
generated — however good the model is. The failure mode is not a crash. A
wrong hill width produces a run that completes, emits output, and silently
resolves nothing (`activity-log.md`, F4). A wrong number in this domain looks
exactly like a right one.

### The yardstick the agent does not control

The agent chooses which coordinate to bias. It is *scored* on a coordinate the
task file fixes for the life of the campaign — usually not the one being
biased. Counting success on the agent's own chosen coordinate was tried and
failed in both directions within one campaign: a collapsed surface scored 22
and 26 "crossings" that were thermal jiggle inside a single state, while
rounds that could not resolve two basins reported 0 when the truth was
"cannot measure" (F7, F9). Now "cannot measure" reports as null, and the
convergence check withholds a verdict rather than calling it failure.
