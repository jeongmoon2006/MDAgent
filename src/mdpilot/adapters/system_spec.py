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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Neither adapter implements hydrogen mass repartitioning, and both constrain
# only h-bonds, so a timestep past roughly 2.5 fs integrates the fast X-H
# bending modes too coarsely and the run goes unstable — or, worse, stays
# stable and quietly reports the wrong ensemble.
_MAX_TIMESTEP_FS_WITHOUT_HMR = 2.5


@dataclass(frozen=True)
class Ensemble:
    """Thermodynamic state and integration timestep the campaign runs at.

    Deliberately two fields. Water model, force field, box padding, ionic
    strength and pressure are still hardcoded per adapter; they get fields
    here when a campaign actually needs them, which is the same rule
    `SystemSpec` itself follows.

    This lives on `SystemSpec` rather than on adapter constructors so the
    resume guard covers it for free: `run_campaign` already locks
    `adapter.spec.to_dict()` into the campaign config, so a campaign cannot be
    restarted at a different temperature. An adapter keyword argument would
    need its own config key, and the key someone forgets to add is the one
    that silently joins two halves of a campaign sampled under different
    conditions.

    Temperature is not a setup-only knob. It also sets PLUMED's `TEMP=` (and
    so the well-tempered scaling factor), the initial hill height in
    `bias_designer`, and the kT the free-energy convergence threshold is taken
    against — which is why `MDAdapter` already exposes it as a property.
    """

    temperature_k: float = 300.0
    timestep_fs: float = 2.0

    def __post_init__(self) -> None:
        if self.temperature_k <= 0:
            raise ValueError(
                f"Ensemble: temperature_k must be positive (got {self.temperature_k})"
            )
        if self.timestep_fs <= 0:
            raise ValueError(
                f"Ensemble: timestep_fs must be positive (got {self.timestep_fs})"
            )
        if self.timestep_fs > _MAX_TIMESTEP_FS_WITHOUT_HMR:
            raise ValueError(
                f"Ensemble: timestep_fs={self.timestep_fs} exceeds "
                f"{_MAX_TIMESTEP_FS_WITHOUT_HMR} fs, which needs hydrogen mass "
                f"repartitioning. Neither adapter implements HMR and both "
                f"constrain h-bonds only, so this would be integrated "
                f"unstably. Add HMR to the adapters before raising it."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_k": self.temperature_k,
            "timestep_fs": self.timestep_fs,
        }


@dataclass(frozen=True)
class SystemSpec:
    """Physical system to simulate. Exactly one of pdb_id / structure_path must be set."""

    pdb_id: str | None = None
    structure_path: Path | None = None
    ensemble: Ensemble = field(default_factory=Ensemble)

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
        spec: dict[str, Any] = {
            "pdb_id": self.pdb_id,
            "structure_path": str(self.structure_path)
            if self.structure_path is not None
            else None,
        }
        # Emitted only when it differs from the default. `store.init_campaign`
        # compares the serialized config JSON byte-for-byte, so adding a key
        # unconditionally would make every campaign recorded before this field
        # existed refuse to resume. A default ensemble serializes as absent,
        # which is exactly the state those campaigns ran under; anything else
        # locks explicitly. The guard stays correct either way — starting at
        # 240 K and resuming at the default still mismatches.
        if self.ensemble != Ensemble():
            spec["ensemble"] = self.ensemble.to_dict()
        return spec
