"""OpenMM implementation of `MDAdapter`.

Trp-cage in TIP3P + 0.15 M NaCl, AMBER14, LangevinMiddle integrator. The
system parameters are hardcoded here for now; engine-agnostic SystemSpec
arrives later when arbitrary-system support lands.

F2 fix: `start()` is idempotent. On the first call we run the full
Modeller / solvate / createSystem / minimize pipeline, then serialize
the resulting `System` and post-minimization `State` to `<work_dir>/cache/`.
On subsequent calls (e.g. process restart for resume) we detect the
cache and skip straight to constructing a `Simulation` from it. The
expensive part (minimization on a few-thousand-atom solvated system) no
longer runs every time `run_campaign` is invoked.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, app, unit
from pdbfixer import PDBFixer

from mdpilot.adapters.system_spec import SystemSpec

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
    """MDAdapter: direct OpenMM execution. System chosen by SystemSpec; the
    integrator/forcefield/box/ions choices are still hardcoded (AMBER14,
    TIP3P, 1 nm padding, 0.15 M NaCl, LangevinMiddle at 300 K, 2 fs)."""

    def __init__(
        self, *, work_dir: Path, seed: int = 42, spec: SystemSpec | None = None
    ):
        self._work_dir = Path(work_dir)
        self._seed = seed
        self._spec = spec if spec is not None else SystemSpec.trpcage()
        self._pdb_path: Path | None = None
        self._sim: app.Simulation | None = None
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def trajectory_extension(self) -> str:
        return ".dcd"

    @property
    def topology_path(self) -> Path:
        return self._topology_path

    def prepare(self) -> None:
        inputs = self._work_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        out = inputs / f"{self._spec_tag()}_fixed.pdb"
        if out.exists():
            self._pdb_path = out
            return
        if self._spec.pdb_id is not None:
            fixer = PDBFixer(pdbid=self._spec.pdb_id)
        else:
            assert self._spec.structure_path is not None
            fixer = PDBFixer(filename=str(self._spec.structure_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        with open(out, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        self._pdb_path = out

    def _spec_tag(self) -> str:
        """Filename-safe identifier for the spec, used to namespace cached files."""
        if self._spec.pdb_id is not None:
            return self._spec.pdb_id
        assert self._spec.structure_path is not None
        return self._spec.structure_path.stem

    def start(self) -> None:
        if self._pdb_path is None:
            raise RuntimeError("OpenMMAdapter.start() called before prepare()")
        cache_dir = self._work_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        system_xml = cache_dir / "system.xml"
        state_xml = cache_dir / "initial_state.xml"
        cached_topology = cache_dir / "topology.pdb"

        if system_xml.exists() and state_xml.exists() and cached_topology.exists():
            self._start_from_cache(system_xml, state_xml, cached_topology)
            return

        self._start_fresh_and_cache(system_xml, state_xml, cached_topology)

    def _start_from_cache(
        self, system_xml: Path, state_xml: Path, cached_topology: Path
    ) -> None:
        pdb = app.PDBFile(str(cached_topology))
        system = XmlSerializer.deserialize(system_xml.read_text())
        integrator = self._make_integrator()
        platform = Platform.getPlatformByName("CPU")
        sim = app.Simulation(pdb.topology, system, integrator, platform)
        state = XmlSerializer.deserialize(state_xml.read_text())
        sim.context.setState(state)
        self._sim = sim
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached_topology, self._topology_path)

    def _start_fresh_and_cache(
        self, system_xml: Path, state_xml: Path, cached_topology: Path
    ) -> None:
        assert self._pdb_path is not None
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
        integrator = self._make_integrator()
        platform = Platform.getPlatformByName("CPU")
        sim = app.Simulation(modeller.topology, system, integrator, platform)
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()
        self._sim = sim

        # Cache for next time
        system_xml.write_text(XmlSerializer.serialize(system))
        state = sim.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=True
        )
        state_xml.write_text(XmlSerializer.serialize(state))
        self._write_topology(sim, cached_topology)
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached_topology, self._topology_path)

    def _make_integrator(self) -> LangevinMiddleIntegrator:
        integrator = LangevinMiddleIntegrator(
            _TEMPERATURE_K * unit.kelvin,
            _FRICTION_PER_PS / unit.picosecond,
            _TIMESTEP_FS * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(self._seed)
        return integrator

    @staticmethod
    def _write_topology(sim: app.Simulation, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        with open(path, "w") as f:
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
