"""Adapter Protocol for MD engines.

The campaign loop in `orchestrator/loop.py` talks to engines exclusively
through this Protocol. Adding a new engine (GROMACS, NAMD, etc.) means
writing a new class that implements these methods; no changes to the
loop or to the scientist are required.

The engine also declares the two physics constants the loop reasons in —
`timestep_fs` (to convert nanoseconds to steps) and `temperature_k` (which
PLUMED's well-tempered scaling and the kT convergence threshold are taken
against). Both were previously hardcoded in the loop as values that merely
happened to agree with both adapters.

Lifecycle:
  1. construct(work_dir=..., seed=..., [engine-specific args])
  2. prepare()             # idempotent: cache structures to disk
  3. start()               # build engine state, write topology_path
  4. repeat:
     - load_checkpoint(path)            # optional, for resume
     - run_steps(n, trajectory_path=…)  # advance + record
     - save_checkpoint(path)            # persist resumable state

The topology written by `start()` and any trajectories written by
`run_steps()` must be loadable by mdtraj — diagnostics live one level up
and don't care which engine produced them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from mdpilot.adapters.system_spec import SystemSpec


@runtime_checkable
class MDAdapter(Protocol):
    """Stateful per-campaign handle to an MD engine."""

    @property
    def spec(self) -> SystemSpec:
        """The physical system this adapter was constructed for.

        The loop includes this in the SQLite campaign config so a resume
        attempt with a different system gets rejected by the
        config-mismatch guard.
        """
        ...

    @property
    def timestep_fs(self) -> float:
        """Integrator timestep in femtoseconds.

        The loop sizes every round in nanoseconds — `extra_ns`, `max_biased_ns`
        — and has to convert to steps. It used to do that against a hardcoded
        2 fs that merely happened to match both adapters, so an engine at a
        different timestep would have run rounds of the wrong length with
        `extra_ns` silently no longer meaning nanoseconds.
        """
        ...

    @property
    def temperature_k(self) -> float:
        """Thermostat temperature in kelvin.

        Not cosmetic for the biased phase: PLUMED's `METAD ... TEMP=` must
        match the thermostat or the well-tempered scaling factor is computed
        against the wrong temperature, and the deposited bias converges to
        something other than -(1 - 1/γ)F(s) — with no error anywhere. It is
        also the kT the free-energy convergence threshold is taken against.
        """
        ...

    @property
    def trajectory_extension(self) -> str:
        """File extension (including leading dot) for trajectories this
        adapter writes, e.g. ".dcd" for OpenMM, ".xtc" for GROMACS.

        The loop uses this to build trajectory paths; mdtraj infers the
        reader from the extension.
        """
        ...

    @property
    def topology_path(self) -> Path:
        """PDB topology consistent with trajectories produced by run_steps.

        Populated by `start()`. mdtraj-loadable so downstream diagnostics
        work regardless of which engine produced it.
        """
        ...

    def prepare(self) -> None:
        """Idempotent setup: download/fix structures, cache to disk.

        Safe to call multiple times; only the first call does real I/O.
        """
        ...

    def start(self) -> None:
        """Build engine state (system + integrator + minimization) and
        write `topology_path`. After this, run_steps / save_checkpoint /
        load_checkpoint are valid.
        """
        ...

    def run_steps(
        self,
        n_steps: int,
        *,
        trajectory_path: Path | None = None,
        report_interval_steps: int = 500,
    ) -> Path | None:
        """Advance by n_steps. If trajectory_path is given, write frames
        every report_interval_steps. Mutates internal state.
        """
        ...

    def save_checkpoint(self, path: Path) -> Path:
        """Persist enough state for load_checkpoint to resume bit-for-bit
        on a freshly constructed + started adapter of the same type.
        """
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Restore state previously saved by save_checkpoint. Overwrites
        current state.
        """
        ...
