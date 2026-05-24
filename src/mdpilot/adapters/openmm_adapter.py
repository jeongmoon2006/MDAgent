"""OpenMM implementation of `MDAdapter`.

Trp-cage in TIP3P + 0.15 M NaCl, AMBER14, LangevinMiddle integrator. The
system parameters are hardcoded here for now; engine-agnostic SystemSpec
arrives in a follow-up step of M3 (when GROMACS lands and we need
parameters to flow into both adapters from one place).
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


class OpenMMAdapter:
    """MDAdapter: direct OpenMM execution of Trp-cage in solvent."""

    def __init__(self, *, work_dir: Path, seed: int = 42):
        self._work_dir = Path(work_dir)
        self._seed = seed
        self._pdb_path: Path | None = None
        self._sim: app.Simulation | None = None
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def trajectory_extension(self) -> str:
        return ".dcd"

    @property
    def topology_path(self) -> Path:
        return self._topology_path

    def prepare(self) -> None:
        inputs = self._work_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        out = inputs / f"{_PDB_ID}_fixed.pdb"
        if out.exists():
            self._pdb_path = out
            return
        fixer = PDBFixer(pdbid=_PDB_ID)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        with open(out, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        self._pdb_path = out

    def start(self) -> None:
        if self._pdb_path is None:
            raise RuntimeError("OpenMMAdapter.start() called before prepare()")
        pdb = app.PDBFile(str(self._pdb_path))
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
        integrator.setRandomNumberSeed(self._seed)
        platform = Platform.getPlatformByName("CPU")
        sim = app.Simulation(modeller.topology, system, integrator, platform)
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()
        self._sim = sim

        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        with open(self._topology_path, "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f, keepIds=True)

    def run_steps(
        self,
        n_steps: int,
        *,
        trajectory_path: Path | None = None,
        report_interval_steps: int = 500,
    ) -> Path | None:
        sim = self._require_sim()
        reporter: app.DCDReporter | None = None
        if trajectory_path is not None:
            trajectory_path = Path(trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            reporter = app.DCDReporter(str(trajectory_path), report_interval_steps)
            sim.reporters.append(reporter)
        try:
            sim.step(n_steps)
        finally:
            if reporter is not None:
                sim.reporters.remove(reporter)
        return trajectory_path

    def save_checkpoint(self, path: Path) -> Path:
        return save_checkpoint(self._require_sim(), path)

    def load_checkpoint(self, path: Path) -> None:
        load_checkpoint(self._require_sim(), path)

    def _require_sim(self) -> app.Simulation:
        if self._sim is None:
            raise RuntimeError("OpenMMAdapter not started; call start() first")
        return self._sim
