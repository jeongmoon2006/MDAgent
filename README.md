# MDPilot

[![CI](https://github.com/jeongmoon2006/MDPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/jeongmoon2006/MDPilot/actions/workflows/ci.yml)

A closed-loop scientific reasoning agent for molecular dynamics.

> An MD operator runs the simulation you describe.
> An MD scientist decides what to simulate next.
> MDPilot is the latter.

Existing MD agents solve the inner loop — setup, submit, analyze. MDPilot owns
the outer loop: read the diagnostics, judge whether the trajectory answers the
question that was asked, and decide the next run. Extend? Stop? Or is vanilla MD
inadequate and the campaign needs enhanced sampling?

## Install

Python 3.10+ and an Anthropic API key.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

That runs vanilla MD on CPU. **Metadynamics needs conda** — `openmm-plumed` is
not on PyPI, so anything reaching `switch_to_metad` will raise without it:

```sh
micromamba create -y -n mdpilot -c conda-forge \
    python=3.12 openmm openmm-plumed plumed mdtraj pdbfixer pyyaml \
    python-dotenv pytest numpy scipy
micromamba run -n mdpilot python -m pip install anthropic
micromamba run -n mdpilot python -m pip install --no-deps -e .
```

The two environments pin different OpenMM builds and are not interchangeable.
On a CUDA machine, match the toolkit to your driver
(`micromamba install -n mdpilot -c conda-forge "cuda-version=13.2"`) — otherwise
conda-forge may install an NVRTC the driver cannot load, which registers the
CUDA platform and then dies mid-campaign.

## Use it

Describe the science, read what it proposes, then run it.

```sh
# 1 — draft a task file from one sentence
python -m mdpilot.setup_agent "study chignolin folding and unfolding" \
    --out campaigns/chignolin/task.yaml

# 2 — read it. Nothing runs until you do.

# 3 — run it
micromamba run -n mdpilot python -m mdpilot.run \
    campaigns/chignolin/task.yaml campaigns/chignolin \
    --opening-ns 1.0 --max-rounds 20 --biased-cap-ns 20
```

Step 2 is not a formality. Every declared field is checked against the code
that consumes it, and pre-flight checks the built structure — but
`characteristic_timescale_ns` has no cross-check at all, and the decision to
abandon vanilla MD is taken against it. Read it, and read the source it cites.

Re-running step 3 after an interruption resumes from the last checkpoint.

### Watch it decide

```sh
micromamba run -n mdpilot python -m pip install -e ".[ui]"
micromamba run -n mdpilot streamlit run app.py
```

Three columns: draft a campaign, stream the scientist's reasoning round by
round, and inspect trajectories and free-energy surfaces from `campaigns/`.

## Docs

- [`docs/diagrams.md`](docs/diagrams.md) — the whole design in three diagrams
- [`docs/architecture.md`](docs/architecture.md) — what MDPilot is, and why
- [`docs/activity-log.md`](docs/activity-log.md) — decisions and findings, including
  the campaigns that failed and what each one changed
- [`ROADMAP.md`](ROADMAP.md) — milestones
- [`CLAUDE.md`](CLAUDE.md) — working on this codebase, and how to run the tests
- [`docs/related_work.md`](docs/related_work.md) — positioning

## License

See [`LICENSE`](LICENSE).
