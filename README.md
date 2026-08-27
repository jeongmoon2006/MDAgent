# MDPilot

A closed-loop scientific reasoning agent for molecular dynamics.

> An MD operator runs the simulation you describe.
> An MD scientist decides what to simulate next.
> MDPilot is the latter.

Existing MD agents solve the inner loop — setup, submit, analyze. MDPilot owns
the outer loop: read the diagnostics, judge whether the trajectory answers the
question that was asked, and decide the next run. Extend? Stop? Or is vanilla MD
inadequate and the campaign needs enhanced sampling?


## Setup

Python 3.10+ and an Anthropic API key. A CPU OpenMM build is enough for
vanilla MD.

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env  # then set ANTHROPIC_API_KEY
```

### Metadynamics needs conda

`openmm-plumed` is not on PyPI, so the venv above cannot run biased simulations —
anything reaching `switch_to_metad` will raise.

```sh
export MAMBA_ROOT_PREFIX=$HOME/.micromamba
micromamba create -y -n mdpilot -c conda-forge \
    python=3.12 openmm openmm-plumed plumed mdtraj pdbfixer pyyaml \
    python-dotenv pytest numpy scipy
micromamba run -n mdpilot pip install anthropic
micromamba run -n mdpilot pip install --no-deps -e .
```

**On a CUDA machine, pin the toolkit to your driver.** conda-forge may install an
NVRTC newer than the driver supports, which registers the CUDA platform and then
fails at kernel load with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Check
`nvidia-smi` for the supported CUDA version and match it:

```sh
micromamba install -y -n mdpilot -c conda-forge "cuda-version=13.2"
```

The two environments pin different OpenMM builds and are not interchangeable.

## Quick start

Describe the science, read what it proposes, then run it.

```sh
# 1 — draft a task file from one sentence
python -m mdpilot.setup_agent "study chignolin folding and unfolding" \
    --out campaigns/chignolin/task.yaml

# 2 — read it. Nothing runs until you do.

# 3 — run it (conda env: metadynamics needs PLUMED)
micromamba run -n mdpilot python -m mdpilot.run \
    campaigns/chignolin/task.yaml campaigns/chignolin \
    --opening-ns 1.0 --max-rounds 20 --biased-cap-ns 20
```

Step 2 is not a formality. The loader checks every declared field against the
code that consumes it, and pre-flight checks the built structure — but one
field has no cross-check at all: `characteristic_timescale_ns` is what the
pivot decision is taken against, and it comes from the model's own knowledge.
Read it, and read the source it cites.

## The task file

One file defines the campaign. Every field is in one of three modes:

| mode | meaning |
|---|---|
| **mapped** | a real parameter — system, force field, padding, ensemble, equilibration, seed, observable, thresholds, budgets |
| **verified** | not yet tunable, but checked against the constant that governs it, so the file cannot disagree with the run |
| **informational** | prose, carried not interpreted |

Anything else is refused. See
[`benchmarks/tasks/cln025_contacts.yaml`](benchmarks/tasks/cln025_contacts.yaml)
for a fully commented example.

## Running a campaign

```sh
python -m mdpilot.run <task.yaml> <work_dir> [--opening-ns …] [--max-rounds …]
```

Resumable: re-running the same command after an interruption continues from the
last checkpoint, and a task file that changed what the campaign *is* is refused
rather than silently spliced. The flags own only loop bounds. Line-flushed
output, so `nohup`/`tmux` plus `tail -f` works on a server.

## Web app

A three-column control surface: draft a campaign from a sentence, watch the
scientist decide, inspect what it produced.

```sh
micromamba run -n mdpilot pip install -e ".[ui]"
micromamba run -n mdpilot streamlit run app.py
```

Run it in the conda environment — the app starts real campaigns, so it needs
the same PLUMED-capable environment metadynamics does. The left column calls
the setup agent (needs `ANTHROPIC_API_KEY`); the middle streams
`run_campaign`'s event log; the right renders trajectories and free-energy
surfaces straight out of `campaigns/`. It starts idle and follows whichever
campaign you lock; pick one from its dropdown to inspect an earlier run.

Run bounds in the left column default to shakedown sizes on purpose — a click
should not start a 20 ns campaign.

## Python API

`run_campaign` can be called directly, but it bypasses the task file — and with
it the pre-flight checks, the declared-field verification, and
`TaskFile.build_adapter`. Calling it without an `adapter` silently uses the
default Trp-cage system whatever you meant to simulate. Prefer a task file.

```python
from pathlib import Path

from mdpilot.orchestrator.loop import run_campaign
from mdpilot.task_file import load_task_file

task = load_task_file("benchmarks/tasks/cln025_contacts.yaml")
result = run_campaign(
    work_dir=Path("campaigns/demo"),
    adapter=task.build_adapter("campaigns/demo"),
    **task.run_kwargs(max_rounds=1),
)
```

## Tests

```sh
pytest tests/unit          # 401 tests, ~5 s — no API key, no MD
pytest tests/integration   # live MD; PLUMED and API-key tests skip if unavailable
```

Integration coverage includes equilibration physics (velocities, temperature,
barostat, density) and metadynamics against PLUMED's own HILLS and COLVAR
output. Both run without an API key. Reference trajectories for the
Milestone-1 criterion are generated on demand:

```sh
python -m benchmarks.generate_trpcage_planted --traj under       # 50 ps
python -m benchmarks.generate_trpcage_planted --traj converged   # 5 ns
```

They land in `benchmarks/data/trpcage/` (gitignored; deterministic given the
seed). Note these predate the equilibration work and are NVT — see F3 in
[`docs/activity-log.md`](docs/activity-log.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — what MDPilot is and why
- [`docs/diagrams.md`](docs/diagrams.md) — the same, as three diagrams
- [`ROADMAP.md`](ROADMAP.md) — milestones and what's next
- [`docs/activity-log.md`](docs/activity-log.md) — decisions, findings, session journal
- [`docs/related_work.md`](docs/related_work.md) — positioning
- [`CLAUDE.md`](CLAUDE.md) — how to work on this codebase

## License

See [`LICENSE`](LICENSE).
