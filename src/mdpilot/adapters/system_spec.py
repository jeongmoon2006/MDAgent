"""Engine-agnostic description of *what physical system to simulate*.

Adapters take a `SystemSpec` at construction and translate it into engine-
specific setup commands (OpenMM Modeller + ForceField, GROMACS pdb2gmx
chain, etc). The loop reads `adapter.spec` to include the spec in the
SQLite campaign config, so resuming a Trp-cage campaign with a chignolin
spec is rejected by the existing config-mismatch guard rather than
silently producing nonsense.

Current scope is intentionally minimal — just structure source. Other
parameters (force field family, water model, ionic strength, integrator
choice) remain hardcoded in each adapter for now because no current
campaign needs to vary them. Fields get added when a real second
campaign (e.g. ice nucleation, which needs TIP4P/Ice + different
temperature + possibly different force field) forces the contract.

This follows the M3 lesson: the right abstraction is discovered from
two real implementations, not invented in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SystemSpec:
    """Physical system to simulate. Exactly one of pdb_id / structure_path must be set."""

    pdb_id: str | None = None
    structure_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.pdb_id is None) == (self.structure_path is None):
            raise ValueError(
                "SystemSpec requires exactly one of pdb_id or structure_path"
            )

    @classmethod
    def trpcage(cls) -> "SystemSpec":
        """Convenience: the M1-era hardcoded default. Used when callers don't
        pass an adapter explicitly to `run_campaign`."""
        return cls(pdb_id="1L2Y")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for SQLite storage and config-mismatch checks."""
        return {
            "pdb_id": self.pdb_id,
            "structure_path": str(self.structure_path)
            if self.structure_path is not None
            else None,
        }
