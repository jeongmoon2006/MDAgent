"""Hardcoded Trp-cage OpenMM runner for Milestone 1.

Direct execution path. The polymorphic adapter interface gets formalised
in Milestone 3 when the GROMACS runner lands; for now this module just
exports two concrete functions.
"""

from __future__ import annotations

from pathlib import Path

from openmm import LangevinMiddleIntegrator, Platform, app, unit
from pdbfixer import PDBFixer

_PDB_ID = "1L2Y"
_FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3p.xml")
_PADDING_NM = 1.0
_SALT_M = 0.15
_TEMPERATURE_K = 300.0
_FRICTION_PER_PS = 1.0
_TIMESTEP_FS = 2.0
_NONBONDED_CUTOFF_NM = 1.0


def prepare_trpcage_pdb(work_dir: Path) -> Path:
    """Download 1L2Y via PDBFixer, add missing atoms + hydrogens, cache to work_dir."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{_PDB_ID}_fixed.pdb"
    if out.exists():
        return out

    fixer = PDBFixer(pdbid=_PDB_ID)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(out, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    return out


def build_simulation(
    pdb_path: Path,
    *,
    seed: int | None = None,
) -> app.Simulation:
    """Solvate Trp-cage in TIP3P + 0.15 M NaCl and energy-minimize.

    Returns a `Simulation` at the energy minimum, ready for `run_steps`.
    No equilibration is performed here — that is the caller's choice.
    """
    pdb = app.PDBFile(str(pdb_path))
    forcefield = app.ForceField(*_FORCEFIELD_FILES)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=_PADDING_NM * unit.nanometer,
        ionicStrength=_SALT_M * unit.molar,
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=_NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds,
    )

    integrator = LangevinMiddleIntegrator(
        _TEMPERATURE_K * unit.kelvin,
        _FRICTION_PER_PS / unit.picosecond,
        _TIMESTEP_FS * unit.femtosecond,
    )
    if seed is not None:
        integrator.setRandomNumberSeed(seed)

    platform = Platform.getPlatformByName("CPU")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    return simulation


def write_topology_pdb(simulation: app.Simulation, path: Path) -> Path:
    """Write the current solvated topology + positions to a PDB.

    Required so that downstream analysis (mdtraj, MDAnalysis) can match
    a DCD's atom count to a topology — `prepare_trpcage_pdb` returns the
    pre-solvation PDB only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(path, "w") as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)
    return path


def run_steps(
    simulation: app.Simulation,
    n_steps: int,
    dcd_path: Path | None = None,
    *,
    report_interval_steps: int = 500,
) -> Path | None:
    """Advance the simulation by n_steps; optionally write a DCD trajectory.

    With a 2 fs timestep the default report_interval_steps=500 = 1 ps/frame.
    Mutates `simulation` in place; subsequent calls continue from the
    current state.
    """
    reporter: app.DCDReporter | None = None
    if dcd_path is not None:
        dcd_path = Path(dcd_path)
        dcd_path.parent.mkdir(parents=True, exist_ok=True)
        reporter = app.DCDReporter(str(dcd_path), report_interval_steps)
        simulation.reporters.append(reporter)
    try:
        simulation.step(n_steps)
    finally:
        if reporter is not None:
            simulation.reporters.remove(reporter)
    return dcd_path


def save_checkpoint(simulation: app.Simulation, path: Path) -> Path:
    """Write an OpenMM binary checkpoint (positions, velocities, RNG state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(simulation.context.createCheckpoint())
    return path


def load_checkpoint(simulation: app.Simulation, path: Path) -> None:
    """Restore a checkpoint into an existing Simulation built from the same system."""
    with open(path, "rb") as f:
        simulation.context.loadCheckpoint(f.read())
