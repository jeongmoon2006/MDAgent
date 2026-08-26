"""OpenMM implementation of `MDAdapter`.

Trp-cage in TIP3P + 0.15 M NaCl, AMBER14, LangevinMiddle integrator. The
system parameters are hardcoded here for now; engine-agnostic SystemSpec
arrives later when arbitrary-system support lands.

Equilibration: setup runs minimize → staged NVT heating → NPT density
relaxation before anything is cached, so production never starts from a
minimized structure. Velocities are seeded explicitly with
`setVelocitiesToTemperature` at the first heating stage rather than left at
zero for the thermostat to warm up — that zero-velocity start was a real
physical mismatch with the GROMACS adapter, which has always generated
velocities via `gen-vel`. Both engines reach the ensemble temperature over
an equivalent ramp from 50 K under the same stochastic (Langevin / `sd`)
thermostat.

The barostat added for the NPT stage stays in the cached `System`, so
production runs NPT as well; the box is therefore free to fluctuate rather
than being frozen at whatever density `addSolvent` happened to produce.

F2 fix: `start()` is idempotent. On the first call we run the full
Modeller / solvate / createSystem / minimize / equilibrate pipeline, then
serialize the resulting `System` and post-equilibration `State` to
`<work_dir>/cache/`. On subsequent calls (e.g. process restart for resume)
we detect the cache and skip straight to constructing a `Simulation` from
it. The expensive part (minimization plus equilibration on a few-thousand-
atom solvated system) no longer runs every time `run_campaign` is invoked.

PLUMED hook: `plumed_input` (str) is intentionally *not* part of the
cache. The cache stays bias-agnostic; the runtime Simulation is built
from the cached vanilla System plus a freshly-attached PlumedForce
each `start()`. Re-supplying or changing `plumed_input` on resume
therefore just works — no cache invalidation. `openmmplumed` is a
guarded import: required only when `plumed_input` is set.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from openmm import (
    Context,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
    Platform,
    System,
    VerletIntegrator,
    XmlSerializer,
    app,
    unit,
)
from pdbfixer import PDBFixer

from mdpilot.adapters.system_spec import SystemSpec

_FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3p.xml")
_PADDING_NM = 1.0
_SALT_M = 0.15
_FRICTION_PER_PS = 1.0
_NONBONDED_CUTOFF_NM = 1.0
# Thermostat temperature and integrator timestep are NOT here: they come from
# `spec.ensemble`, so they are locked by the campaign config on resume. The
# rest of this block is still fixed per adapter.

# Equilibration. Heating ramps from _HEAT_START_K to the ensemble temperature
# in _HEAT_STAGES equal steps under the production thermostat; NPT then relaxes
# the density at 1 bar. Both stage lengths are constructor-overridable so the
# test suite can run the pipeline without paying 200 ps of CPU MD.
_HEAT_START_K = 50.0
_HEAT_STAGES = 6
_NVT_EQUIL_STEPS = 50_000   # 100 ps at 2 fs
_NPT_EQUIL_STEPS = 50_000   # 100 ps at 2 fs
_PRESSURE_BAR = 1.0
_BAROSTAT_INTERVAL = 25     # steps between volume-move attempts
_EQUIL_REPORT_INTERVAL = 500

# Platform selection, fastest first. GPU platforms run production MD in
# 'mixed' precision: forces in single, accumulation in double. Plain 'single'
# (the OpenMM default) degrades energy conservation relative to the CPU
# platform's double, which would make a GPU run quietly less trustworthy than
# the CPU run it replaces.
_PLATFORM_PREFERENCE = ("CUDA", "HIP", "OpenCL", "CPU")
_MIXED_PRECISION_PLATFORMS = frozenset({"CUDA", "HIP", "OpenCL"})


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


def _platform_is_usable(name: str) -> bool:
    """Can this platform actually build a Context and take a step?

    Registration is not usability. A conda OpenMM whose NVRTC is newer than
    the installed driver registers CUDA quite happily and then dies at kernel
    load with CUDA_ERROR_UNSUPPORTED_PTX_VERSION — mid-campaign, after setup
    has already been paid for. A throwaway two-particle system with a real
    nonbonded force compiles the same kernel machinery, for microseconds.
    """
    try:
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        force = NonbondedForce()
        force.addParticle(0.0, 1.0, 0.0)
        force.addParticle(0.0, 1.0, 0.0)
        system.addForce(force)
        integrator = VerletIntegrator(0.001)
        context = Context(system, integrator, Platform.getPlatformByName(name))
        context.setPositions([(0, 0, 0), (0, 0, 1)])
        integrator.step(1)
        del context
        return True
    except Exception:
        return False


def _registered_platforms() -> set[str]:
    return {
        Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
    }


@lru_cache(maxsize=1)
def resolve_platform() -> str:
    """Name of the fastest usable platform. GPU when there is one, else CPU.

    Cached: the probe is cheap but not free, and the answer cannot change
    within a process.
    """
    registered = _registered_platforms()
    for name in _PLATFORM_PREFERENCE:
        if name in registered and _platform_is_usable(name):
            return name
    return "CPU"


def _precision_properties(name: str) -> dict[str, str]:
    return {"Precision": "mixed"} if name in _MIXED_PRECISION_PLATFORMS else {}


def _platform_and_properties(name: str) -> tuple[Platform, dict[str, str]]:
    return Platform.getPlatformByName(name), _precision_properties(name)


def _attach_equilibration_reporter(sim: app.Simulation, path: Path) -> None:
    """Log temperature/density/volume through an equilibration stage.

    An equilibration you cannot inspect is only marginally better than none —
    this is the OpenMM counterpart of the `.log`/`.edr` pair GROMACS writes for
    its `nvt`/`npt` stages, so both engines leave the same audit trail.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sim.reporters.append(
        app.StateDataReporter(
            str(path),
            _EQUIL_REPORT_INTERVAL,
            step=True,
            temperature=True,
            volume=True,
            density=True,
            potentialEnergy=True,
        )
    )


def _prepare_plumed_force(plumed_input: str, work_dir: Path):
    """Write `plumed.dat` to ``work_dir`` and return an ``openmmplumed.PlumedForce``.

    The disk write happens *before* the import attempt so the audit
    artifact survives on disk even when ``openmmplumed`` is missing —
    useful for inspecting what we were about to run.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "plumed.dat").write_text(plumed_input)
    try:
        from openmmplumed import PlumedForce
    except ImportError as e:
        raise RuntimeError(
            "OpenMMAdapter was constructed with plumed_input, but "
            "`openmmplumed` is not importable. Install it (pip install "
            "openmmplumed) and ensure a PLUMED runtime is on PATH."
        ) from e
    return PlumedForce(plumed_input)


class OpenMMAdapter:
    """MDAdapter: direct OpenMM execution. System chosen by SystemSpec; the
    forcefield/box/ions choices are still hardcoded (AMBER14, TIP3P, 1 nm
    padding, 0.15 M NaCl, LangevinMiddle, MonteCarloBarostat at 1 bar).
    Thermostat temperature and timestep come from ``spec.ensemble``.

    ``platform`` pins the OpenMM platform by name; left at None the adapter
    picks the fastest one that actually works (GPU when present, CPU
    otherwise) via ``resolve_platform()``.

    ``nvt_steps`` / ``npt_steps`` size the two equilibration stages. Set both
    to 0 to skip equilibration entirely — only useful for tests that need the
    setup pipeline without the MD cost; a run started that way is *not*
    physically comparable to a production one.

    ``plumed_input`` (optional): a plumed.dat-format string. When set, a
    `PlumedForce` is attached to the runtime System on every ``start()``;
    the cache stays vanilla so resume with a different bias just works.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        seed: int = 42,
        spec: SystemSpec | None = None,
        plumed_input: str | None = None,
        nvt_steps: int = _NVT_EQUIL_STEPS,
        npt_steps: int = _NPT_EQUIL_STEPS,
        platform: str | None = None,
    ):
        self._work_dir = Path(work_dir)
        self._seed = seed
        self._spec = spec if spec is not None else SystemSpec.trpcage()
        self._plumed_input = plumed_input
        self._nvt_steps = nvt_steps
        self._npt_steps = npt_steps
        self._platform_name = platform
        self._pdb_path: Path | None = None
        self._sim: app.Simulation | None = None
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def timestep_fs(self) -> float:
        return self._spec.ensemble.timestep_fs

    @property
    def temperature_k(self) -> float:
        return self._spec.ensemble.temperature_k

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

        if not (system_xml.exists() and state_xml.exists() and cached_topology.exists()):
            self._setup_and_cache_vanilla(system_xml, state_xml, cached_topology)

        self._build_runtime_simulation(system_xml, state_xml, cached_topology)

    def _setup_and_cache_vanilla(
        self, system_xml: Path, state_xml: Path, cached_topology: Path
    ) -> None:
        """First-time setup: build the vanilla (no-bias) System, minimize,
        equilibrate (NVT heating then NPT density relaxation), cache. The
        throwaway setup Simulations are discarded; the runtime Simulation is
        rebuilt from cache in `_build_runtime_simulation` so the
        bias-injection path is the same on first call and on every resume.

        What lands in the cache is the *post-equilibration* System (barostat
        included, so production is NPT) and State (positions, velocities and
        box vectors at 300 K / 1 bar)."""
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
        platform, platform_props = self._platform()
        cache_dir = system_xml.parent

        # NVT: minimize, seed velocities at the bottom of the ramp, heat.
        integrator = self._make_integrator(_HEAT_START_K)
        sim = app.Simulation(
            modeller.topology, system, integrator, platform, platform_props
        )
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(
            _HEAT_START_K * unit.kelvin, self._seed
        )
        _attach_equilibration_reporter(sim, cache_dir / "equilibration_nvt.csv")
        self._heat_nvt(sim, integrator)

        # NPT: the barostat is added to the System itself, so it survives
        # serialization and production inherits it. Mutating `system` here is
        # safe — the NVT context holds its own copy.
        system.addForce(self._make_barostat())
        sim = self._relax_npt(
            modeller.topology,
            system,
            sim,
            platform,
            platform_props,
            log_path=cache_dir / "equilibration_npt.csv",
        )

        system_xml.write_text(XmlSerializer.serialize(system))
        state = sim.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=True
        )
        state_xml.write_text(XmlSerializer.serialize(state))
        self._write_topology(sim, cached_topology)

    def _heat_nvt(
        self, sim: app.Simulation, integrator: LangevinMiddleIntegrator
    ) -> None:
        """Ramp the thermostat from _HEAT_START_K to the ensemble temperature in equal
        stages. A staircase rather than a continuous ramp: the temperature can
        only change between `step()` calls, and _HEAT_STAGES steps is fine
        enough for a small solvated protein. No positional restraints — at
        Trp-cage scale a staged ramp from a minimized structure is gentle
        enough without them."""
        if self._nvt_steps <= 0:
            return
        per_stage = max(self._nvt_steps // _HEAT_STAGES, 1)
        for stage in range(1, _HEAT_STAGES + 1):
            target = _HEAT_START_K + (self.temperature_k - _HEAT_START_K) * (
                stage / _HEAT_STAGES
            )
            integrator.setTemperature(target * unit.kelvin)
            sim.step(per_stage)

    def _relax_npt(
        self,
        topology: app.Topology,
        system: object,
        nvt_sim: app.Simulation,
        platform: Platform,
        platform_props: dict[str, str],
        *,
        log_path: Path,
    ) -> app.Simulation:
        """Rebuild the Simulation against the barostat-bearing System, carrying
        positions/velocities/box across, and relax the density at 1 bar."""
        state = nvt_sim.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=True
        )
        sim = app.Simulation(
            topology, system, self._make_integrator(), platform, platform_props
        )
        sim.context.setState(state)
        _attach_equilibration_reporter(sim, log_path)
        if self._npt_steps > 0:
            sim.step(self._npt_steps)
        return sim

    @property
    def platform_name(self) -> str:
        """Platform this adapter runs on. Auto-resolved (GPU if usable, else
        CPU) unless pinned at construction."""
        if self._platform_name is None:
            self._platform_name = resolve_platform()
        return self._platform_name

    def _platform(self) -> tuple[Platform, dict[str, str]]:
        return _platform_and_properties(self.platform_name)

    def _make_barostat(self) -> MonteCarloBarostat:
        barostat = MonteCarloBarostat(
            _PRESSURE_BAR * unit.bar,
            self.temperature_k * unit.kelvin,
            _BAROSTAT_INTERVAL,
        )
        # Volume moves are Monte Carlo; seed them so runs stay reproducible.
        barostat.setRandomNumberSeed(self._seed)
        return barostat

    def _build_runtime_simulation(
        self, system_xml: Path, state_xml: Path, cached_topology: Path
    ) -> None:
        pdb = app.PDBFile(str(cached_topology))
        system = XmlSerializer.deserialize(system_xml.read_text())
        if self._plumed_input is not None:
            force = _prepare_plumed_force(self._plumed_input, self._work_dir)
            system.addForce(force)
        integrator = self._make_integrator()
        platform, platform_props = self._platform()
        sim = app.Simulation(
            pdb.topology, system, integrator, platform, platform_props
        )
        state = XmlSerializer.deserialize(state_xml.read_text())
        sim.context.setState(state)
        self._sim = sim
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached_topology, self._topology_path)

    def _make_integrator(
        self, temperature_k: float | None = None
    ) -> LangevinMiddleIntegrator:
        """Production integrator, or one pinned to a heating-stage temperature.

        `temperature_k=None` means the ensemble's own temperature. The timestep
        is never overridden: it defines what a step is worth, and the loop
        converts nanoseconds to steps against `adapter.timestep_fs`.
        """
        integrator = LangevinMiddleIntegrator(
            (self.temperature_k if temperature_k is None else temperature_k)
            * unit.kelvin,
            _FRICTION_PER_PS / unit.picosecond,
            self._spec.ensemble.timestep_fs * unit.femtosecond,
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
