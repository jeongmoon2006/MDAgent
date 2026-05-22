"""OpenMM checkpoint save/load round-trip on a toy 1-particle system.

We don't need Trp-cage to verify the round-trip property: saving the state
of any Simulation and reloading it into a fresh Simulation built from the
same System must produce identical subsequent trajectories. Toy system
keeps this test in the millisecond range, runnable on every commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openmm import Platform, System, VerletIntegrator, Vec3
from openmm.app import Element, Simulation, Topology
from openmm.unit import amu, nanometer, picosecond

from mdpilot.adapters.openmm_runner import load_checkpoint, save_checkpoint


def _toy_simulation() -> Simulation:
    """One free argon-like particle, Verlet integrator. Fully deterministic."""
    top = Topology()
    chain = top.addChain()
    residue = top.addResidue("AR", chain)
    top.addAtom("AR", Element.getBySymbol("Ar"), residue)

    system = System()
    system.addParticle(40.0 * amu)

    integrator = VerletIntegrator(0.001 * picosecond)
    sim = Simulation(top, system, integrator, Platform.getPlatformByName("Reference"))
    sim.context.setPositions([Vec3(0, 0, 0) * nanometer])
    sim.context.setVelocities([Vec3(0.1, 0, 0) / picosecond * nanometer])
    return sim


def _positions(sim: Simulation) -> np.ndarray:
    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    return np.asarray(pos)


def test_checkpoint_round_trip_continues_trajectory(tmp_path: Path) -> None:
    sim_a = _toy_simulation()
    sim_a.step(100)
    ckpt = save_checkpoint(sim_a, tmp_path / "state.chk")
    assert ckpt.exists() and ckpt.stat().st_size > 0

    sim_a.step(100)
    pos_a_after_200 = _positions(sim_a)

    # Fresh simulation with different initial state — must be overwritten by the load.
    sim_b = _toy_simulation()
    sim_b.context.setPositions([Vec3(9, 9, 9) * nanometer])
    load_checkpoint(sim_b, ckpt)
    sim_b.step(100)
    pos_b_after_load_and_100 = _positions(sim_b)

    np.testing.assert_allclose(pos_b_after_load_and_100, pos_a_after_200, atol=1e-6)


def test_checkpoint_preserves_step_count(tmp_path: Path) -> None:
    sim_a = _toy_simulation()
    sim_a.step(57)
    save_checkpoint(sim_a, tmp_path / "s.chk")

    sim_b = _toy_simulation()
    load_checkpoint(sim_b, tmp_path / "s.chk")
    assert sim_b.context.getStepCount() == 57
