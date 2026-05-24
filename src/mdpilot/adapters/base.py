"""Adapter Protocol for MD engines.

The campaign loop in `orchestrator/loop.py` talks to engines exclusively
through this Protocol. Adding a new engine (GROMACS, NAMD, etc.) means
writing a new class that implements these methods; no changes to the
loop or to the scientist are required.

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


@runtime_checkable
class MDAdapter(Protocol):
    """Stateful per-campaign handle to an MD engine."""

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
