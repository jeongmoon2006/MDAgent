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

## Run a campaign

```python
from pathlib import Path
from mdpilot.orchestrator.loop import run_campaign

result = run_campaign(
    work_dir=Path("campaigns/demo"),
    initial_steps=10_000,   # 20 ps at 2 fs
    max_rounds=1,
)
print(result.rounds[0].decision)
```

First call fetches PDB 1L2Y, solvates, minimizes and equilibrates; the state is
cached, so later calls on the same `work_dir` skip straight to dynamics.
Per-round artifacts land in `campaigns/demo/rounds/`.

## Tests

```sh
pytest tests/unit          # 219 tests, ~22 s — no API key, no MD
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
- [`ROADMAP.md`](ROADMAP.md) — milestones and what's next
- [`docs/activity-log.md`](docs/activity-log.md) — decisions, findings, session journal
- [`docs/related_work.md`](docs/related_work.md) — positioning
- [`CLAUDE.md`](CLAUDE.md) — how to work on this codebase

## License

See [`LICENSE`](LICENSE).
