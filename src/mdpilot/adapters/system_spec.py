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

from mdpilot import forcefields

# Neither adapter implements hydrogen mass repartitioning, and both constrain
# only h-bonds, so a timestep past roughly 2.5 fs integrates the fast X-H
# bending modes too coarsely and the run goes unstable — or, worse, stays
# stable and quietly reports the wrong ensemble.
_MAX_TIMESTEP_FS_WITHOUT_HMR = 2.5

# Solvent between the solute and the edge of the box, applied to the *starting*
# structure. Raised from 1.0 after F11: padding sized on a folded structure is
# not padding sized for the ensemble a campaign intends to sample, and on
# CLN025 that put the peptide in contact with its own periodic image in half
# the biased rounds. Measured against that campaign's own sampled spans, 1.5 nm
# leaves 0.18% of frames inside the 1.0 nm cutoff where 1.0 nm left 20.2%, at
# roughly 2x the atom count. A campaign that drives its system further apart
# than CLN025 did should raise it again — this is a better default, not a
# guarantee.
_DEFAULT_PADDING_NM = 1.5


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

    Pressure is here for the same reason temperature is: it is thermodynamic
    state, not an engine detail. The *coupling algorithms* are not — OpenMM
    uses LangevinMiddle and MonteCarloBarostat where GROMACS uses `sd` and
    C-rescale, which are idiomatic equivalents rather than interchangeable
    names. Those stay engine-owned and are declared-and-checked in the task
    file instead.

    Temperature is not a setup-only knob. It also sets PLUMED's `TEMP=` (and
    so the well-tempered scaling factor), the initial hill height in
    `bias_designer`, and the kT the free-energy convergence threshold is taken
    against — which is why `MDAdapter` already exposes it as a property.
    """

    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    timestep_fs: float = 2.0

    def __post_init__(self) -> None:
        if self.temperature_k <= 0:
            raise ValueError(
                f"Ensemble: temperature_k must be positive (got {self.temperature_k})"
            )
        if self.pressure_bar <= 0:
            raise ValueError(
                f"Ensemble: pressure_bar must be positive (got "
                f"{self.pressure_bar}). Production is NPT; there is no NVT "
                f"path, so 0 is not a way to switch the barostat off."
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
            "pressure_bar": self.pressure_bar,
            "timestep_fs": self.timestep_fs,
        }


@dataclass(frozen=True)
class Equilibration:
    """How the system reaches the target state before production starts.

    Stated in **picoseconds**, not steps. The adapters convert against their
    own timestep, so halving the timestep no longer silently halves how long
    the system was equilibrated for — which is what a step count does, and
    which nothing would have caught.

    Both stages at 0 skips equilibration entirely. Only useful for tests that
    need the setup pipeline without the MD cost; a run started that way is not
    physically comparable to a production one.
    """

    nvt_ps: float = 100.0        # staged heating, at constant volume
    npt_ps: float = 100.0        # density relaxation at the target pressure
    heat_start_k: float = 50.0   # bottom of the temperature ramp
    heat_stages: int = 6         # equal steps from heat_start_k to the target

    def __post_init__(self) -> None:
        if self.nvt_ps < 0 or self.npt_ps < 0:
            raise ValueError(
                f"Equilibration: stage lengths cannot be negative (got "
                f"nvt_ps={self.nvt_ps}, npt_ps={self.npt_ps})"
            )
        if self.heat_stages < 1:
            raise ValueError(
                f"Equilibration: heat_stages must be >= 1 (got {self.heat_stages})"
            )
        if self.heat_start_k <= 0:
            raise ValueError(
                f"Equilibration: heat_start_k must be positive (got "
                f"{self.heat_start_k})"
            )

    def nvt_steps(self, timestep_fs: float) -> int:
        return int(round(self.nvt_ps * 1000.0 / timestep_fs))

    def npt_steps(self, timestep_fs: float) -> int:
        return int(round(self.npt_ps * 1000.0 / timestep_fs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvt_ps": self.nvt_ps,
            "npt_ps": self.npt_ps,
            "heat_start_k": self.heat_start_k,
            "heat_stages": self.heat_stages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Equilibration":
        return cls(
            nvt_ps=float(data.get("nvt_ps", 100.0)),
            npt_ps=float(data.get("npt_ps", 100.0)),
            heat_start_k=float(data.get("heat_start_k", 50.0)),
            heat_stages=int(data.get("heat_stages", 6)),
        )


@dataclass(frozen=True)
class SystemSpec:
    """Physical system to simulate. Exactly one of pdb_id / structure_path must be set."""

    pdb_id: str | None = None
    structure_path: Path | None = None
    ensemble: Ensemble = field(default_factory=Ensemble)
    padding_nm: float = _DEFAULT_PADDING_NM
    # A key into `mdpilot.forcefields`, not a file list: the combination is
    # engine-agnostic here and each adapter resolves it to its own names.
    forcefield: str = forcefields.DEFAULT_KEY
    equilibration: Equilibration = field(default_factory=Equilibration)

    def __post_init__(self) -> None:
        if (self.pdb_id is None) == (self.structure_path is None):
            raise ValueError(
                "SystemSpec requires exactly one of pdb_id or structure_path"
            )
        forcefields.resolve(self.forcefield)   # raises, listing what exists
        if self.padding_nm <= 0:
            raise ValueError(
                f"SystemSpec: padding_nm must be positive (got {self.padding_nm})"
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
        # `ensemble` is emitted only when it differs from the default, because
        # its default *is* what every campaign recorded before it existed ran
        # under — so absent and default mean the same thing and old campaigns
        # stay resumable.
        #
        # `padding_nm` cannot use that trick: its default was raised from 1.0 to
        # 1.5, so absent (a pre-F11 campaign, built at 1.0) and default (1.5)
        # mean *different* boxes. It is always emitted, and
        # `store._LEGACY_SYSTEM_SPEC_DEFAULTS` supplies 1.0 for configs
        # recorded before the field existed — so resuming one of those at the
        # new default is correctly refused rather than silently accepted.
        if self.ensemble != Ensemble():
            spec["ensemble"] = self.ensemble.to_dict()
        spec["padding_nm"] = self.padding_nm
        # Omit-when-default is safe here, unlike padding: this default *is*
        # what every campaign recorded before the field existed was built with.
        if self.forcefield != forcefields.DEFAULT_KEY:
            spec["forcefield"] = self.forcefield
        # Same omit-when-default rule: this default is what every campaign
        # recorded before the field existed was equilibrated with.
        if self.equilibration != Equilibration():
            spec["equilibration"] = self.equilibration.to_dict()
        return spec
