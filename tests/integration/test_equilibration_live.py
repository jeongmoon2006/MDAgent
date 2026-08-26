"""Live OpenMM equilibration: does the cached state actually come out at 300 K?

This is the test the unit suite cannot stand in for. The defect it pins down —
velocities left at zero after minimization, with the Langevin thermostat
expected to warm the system during production — passes every structural check
and only shows up when you look at the kinetic energy of the state that gets
cached.

Runs against the already-fixed Trp-cage PDB so no network fetch is needed.
Equilibration stages are shortened to a few ps: enough to thermalize
velocities (Langevin relaxation time is ~1 ps at 1/ps friction), far too short
to actually relax the density. Real campaigns use the 100 ps + 100 ps defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openmm import MonteCarloBarostat, XmlSerializer, unit

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.system_spec import Ensemble, SystemSpec

_GAS_CONSTANT_KJ_PER_MOL_K = 0.0083144621
_FIXED_PDB = Path("benchmarks/data/trpcage/1L2Y_fixed.pdb")

pytestmark = pytest.mark.skipif(
    not _FIXED_PDB.exists(),
    reason=f"{_FIXED_PDB} missing; run benchmarks.generate_trpcage_planted first",
)


@pytest.fixture(scope="module")
def equilibrated(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, object]:
    """Run setup once; return the cached (system, state) pair."""
    work_dir = tmp_path_factory.mktemp("equil")
    adapter = OpenMMAdapter(
        work_dir=work_dir,
        spec=SystemSpec(structure_path=_FIXED_PDB),
        nvt_steps=3000,   # 6 ps ramp
        npt_steps=1500,   # 3 ps at 1 bar
    )
    adapter.prepare()
    adapter.start()

    cache = work_dir / "cache"
    system = XmlSerializer.deserialize((cache / "system.xml").read_text())
    state = XmlSerializer.deserialize((cache / "initial_state.xml").read_text())
    return system, state


def _instantaneous_temperature_k(system, state) -> float:
    """T = Σ m·v² / (N_dof · R), with constraints and COM removal subtracted.

    Computed from velocities rather than `state.getKineticEnergy()` because
    the cached state is serialized without energies — and adding them would
    grow the cache for no production benefit.
    """
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        ]
    )
    # kJ/mol: dalton·(nm/ps)² is already kJ/mol in OpenMM's unit system.
    two_ke = float((masses * (velocities**2).sum(axis=1)).sum())
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    return two_ke / (dof * _GAS_CONSTANT_KJ_PER_MOL_K)


def test_cached_state_has_nonzero_velocities(equilibrated) -> None:
    """The regression itself: pre-fix, setVelocitiesToTemperature was never
    called and every cached velocity was exactly zero."""
    _, state = equilibrated
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    assert abs(velocities).max() > 0.0


def test_cached_state_is_at_the_thermostat_temperature(equilibrated) -> None:
    system, state = equilibrated
    temperature = _instantaneous_temperature_k(system, state)
    # Band is deliberately wide; the failure mode being guarded against is a
    # 0 K start, not a few-kelvin fluctuation (σ_T ≈ 3 K at this system size).
    assert 250.0 < temperature < 350.0, f"equilibrated to {temperature:.1f} K"


def test_cached_system_carries_the_barostat_into_production(equilibrated) -> None:
    system, _ = equilibrated
    barostats = [
        f for f in system.getForces() if isinstance(f, MonteCarloBarostat)
    ]
    assert len(barostats) == 1
    assert barostats[0].getDefaultTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(Ensemble().temperature_k)


def test_cached_state_has_a_periodic_box(equilibrated) -> None:
    """NPT changes the box; the cached state must carry the relaxed vectors,
    not fall back to whatever the topology PDB's CRYST1 record said."""
    _, state = equilibrated
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    assert box[0][0] > 0.0 and box[1][1] > 0.0 and box[2][2] > 0.0
