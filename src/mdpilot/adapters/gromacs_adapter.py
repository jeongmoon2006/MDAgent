"""GROMACS implementation of `MDAdapter`.

Trp-cage in TIP3P + 0.15 M NaCl, amber99sb-ildn force field, stochastic-
dynamics integrator (Langevin equivalent) at 300 K. The system-level
parameters intentionally mirror those in `openmm_adapter` so that the
same convergence rubric can be applied to trajectories produced by
either engine; small force-field-level differences (amber99sb-ildn vs
amber14-all) are accepted.

Subprocess pattern: every `gmx` invocation goes through `_run_gmx()`,
which captures stderr and raises a RuntimeError with the tail of the
log on non-zero exit. No interactive prompts — all subcommands run with
explicit flags or piped stdin.

State across `run_steps()` calls is carried by the latest GROMACS
checkpoint (`.cpt`) plus the latest structure file (`.gro`); grompp uses
`-t .cpt` for velocities/RNG/step-counter and `-c .gro` for atom ordering.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from openmm import app
from pdbfixer import PDBFixer

from mdpilot.adapters.system_spec import SystemSpec

_FORCEFIELD = "amber99sb-ildn"
_WATER_MODEL = "tip3p"
_BOX_TYPE = "cubic"
_PADDING_NM = 1.0
_SALT_CONC_M = 0.15
_TEMPERATURE_K = 300.0
_TAU_T_PS = 1.0          # inverse friction; 1/friction_per_ps with friction = 1 ps^-1
_TIMESTEP_PS = 0.002     # 2 fs
_CUTOFF_NM = 1.0
_EM_MAX_STEPS = 50_000
_EM_TOLERANCE = 100.0
_GMX_TIMEOUT_SECONDS = 1800.0


_EM_MDP = f"""\
; energy minimization
integrator    = steep
emtol         = {_EM_TOLERANCE}
emstep        = 0.01
nsteps        = {_EM_MAX_STEPS}

cutoff-scheme = Verlet
nstlist       = 10
ns_type       = grid
coulombtype   = PME
rcoulomb      = {_CUTOFF_NM}
rvdw          = {_CUTOFF_NM}
pbc           = xyz
"""

_IONS_MDP = f"""\
; trivial mdp used only to grompp the ion-placement tpr
integrator    = steep
nsteps        = 0
cutoff-scheme = Verlet
nstlist       = 10
coulombtype   = PME
rcoulomb      = {_CUTOFF_NM}
rvdw          = {_CUTOFF_NM}
pbc           = xyz
"""

_MD_MDP_TEMPLATE = f"""\
; production MD, NVT, Langevin (sd) integrator
integrator           = sd
dt                   = {_TIMESTEP_PS}
nsteps               = {{nsteps}}

nstxout-compressed   = {{report_interval}}
compressed-x-grps    = System
nstlog               = {{report_interval}}
nstenergy            = {{report_interval}}
nstxout              = 0
nstvout              = 0
nstfout              = 0

tc-grps              = System
tau-t                = {_TAU_T_PS}
ref-t                = {_TEMPERATURE_K}

constraints          = h-bonds
constraint-algorithm = LINCS
lincs-iter           = 1
lincs-order          = 4

cutoff-scheme        = Verlet
nstlist              = 10
ns_type              = grid
coulombtype          = PME
rcoulomb             = {_CUTOFF_NM}
rvdw                 = {_CUTOFF_NM}
pbc                  = xyz

gen-vel              = no
ld-seed              = {{seed}}
continuation         = yes
"""


def _run_gmx(
    *args: str,
    cwd: Path,
    input_text: str | None = None,
    timeout: float = _GMX_TIMEOUT_SECONDS,
) -> str:
    """Invoke `gmx <args>` in cwd; raise with stderr tail on non-zero exit."""
    cmd = ["gmx", *args]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"`gmx {' '.join(args)}` failed (exit {result.returncode}) in {cwd}\n"
            f"--- stderr tail ---\n{tail}"
        )
    return result.stdout


class GROMACSAdapter:
    """MDAdapter: subprocess-driven GROMACS execution. System chosen by
    SystemSpec; the forcefield/water/integrator choices stay hardcoded
    (amber99sb-ildn, TIP3P, sd at 300 K, 2 fs)."""

    def __init__(
        self, *, work_dir: Path, seed: int = 42, spec: SystemSpec | None = None
    ):
        self._work_dir = Path(work_dir)
        self._seed = seed
        self._spec = spec if spec is not None else SystemSpec.trpcage()
        self._setup_dir = self._work_dir / "setup"
        self._inputs_dir = self._work_dir / "inputs"
        self._topology_path = self._work_dir / "topology.pdb"
        self._pdb_path: Path | None = None
        self._top_path: Path | None = None       # topol.top
        self._last_gro: Path | None = None       # current state structure
        self._last_cpt: Path | None = None       # current checkpoint (or None pre-MD)
        self._started = False

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def trajectory_extension(self) -> str:
        return ".xtc"

    @property
    def topology_path(self) -> Path:
        return self._topology_path

    def prepare(self) -> None:
        """Fetch/fix structure via PDBFixer, cache to inputs/. Idempotent."""
        self._inputs_dir.mkdir(parents=True, exist_ok=True)
        out = self._inputs_dir / f"{self._spec_tag()}_fixed.pdb"
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
        """Run pdb2gmx → editconf → solvate → genion → minimize. Write topology.pdb.

        Idempotent (F2): if the post-minimization outputs already exist in
        `setup/`, skip the whole pipeline and just rebind to them. This is
        what makes resume cheap — the per-restart cost drops from
        "rerun minimization" to "stat a few files."
        """
        if self._pdb_path is None:
            raise RuntimeError("GROMACSAdapter.start() called before prepare()")
        self._setup_dir.mkdir(parents=True, exist_ok=True)

        if self._is_setup_cached():
            self._top_path = self._setup_dir / "topol.top"
            self._last_gro = self._setup_dir / "em.gro"
            self._last_cpt = None
            self._started = True
            return

        # pdb2gmx: parameterize + add hydrogens consistent with chosen FF
        _run_gmx(
            "pdb2gmx",
            "-f", str(self._pdb_path),
            "-o", "processed.gro",
            "-p", "topol.top",
            "-i", "posre.itp",
            "-ff", _FORCEFIELD,
            "-water", _WATER_MODEL,
            "-ignh",  # ignore hydrogens from input; let pdb2gmx add the right ones
            cwd=self._setup_dir,
        )
        # editconf: define cubic box with 1 nm padding
        _run_gmx(
            "editconf",
            "-f", "processed.gro",
            "-o", "boxed.gro",
            "-c",
            "-d", str(_PADDING_NM),
            "-bt", _BOX_TYPE,
            cwd=self._setup_dir,
        )
        # solvate: fill with TIP3P water
        _run_gmx(
            "solvate",
            "-cp", "boxed.gro",
            "-cs", "spc216.gro",
            "-o", "solv.gro",
            "-p", "topol.top",
            cwd=self._setup_dir,
        )
        # grompp for genion (needs a tpr)
        (self._setup_dir / "ions.mdp").write_text(_IONS_MDP)
        _run_gmx(
            "grompp",
            "-f", "ions.mdp",
            "-c", "solv.gro",
            "-p", "topol.top",
            "-o", "ions.tpr",
            "-maxwarn", "1",
            cwd=self._setup_dir,
        )
        # genion: add 0.15 M NaCl, neutralize; replace SOL group
        _run_gmx(
            "genion",
            "-s", "ions.tpr",
            "-o", "ions.gro",
            "-p", "topol.top",
            "-pname", "NA",
            "-nname", "CL",
            "-conc", str(_SALT_CONC_M),
            "-neutral",
            cwd=self._setup_dir,
            input_text="SOL\n",
        )
        # energy minimization
        (self._setup_dir / "em.mdp").write_text(_EM_MDP)
        _run_gmx(
            "grompp",
            "-f", "em.mdp",
            "-c", "ions.gro",
            "-p", "topol.top",
            "-o", "em.tpr",
            cwd=self._setup_dir,
        )
        _run_gmx(
            "mdrun",
            "-deffnm", "em",
            "-ntmpi", "1",
            cwd=self._setup_dir,
        )

        self._top_path = self._setup_dir / "topol.top"
        self._last_gro = self._setup_dir / "em.gro"
        self._last_cpt = None  # no checkpoint yet; first MD round seeds velocities itself

        # Topology snapshot for diagnostics: convert em.gro to a PDB at the canonical path
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        _run_gmx(
            "editconf",
            "-f", str(self._last_gro),
            "-o", str(self._topology_path),
            cwd=self._setup_dir,
        )

        self._started = True

    def run_steps(
        self,
        n_steps: int,
        *,
        trajectory_path: Path | None = None,
        report_interval_steps: int = 500,
    ) -> Path | None:
        if not self._started:
            raise RuntimeError("GROMACSAdapter not started; call start() first")
        assert self._top_path is not None and self._last_gro is not None

        # Each call writes to a fresh per-call basename in setup/, then moves XTC out.
        # Using setup/ keeps tprs/cpts/logs together for forensic inspection.
        tag = f"round_{self._next_tag()}"
        run_dir = self._setup_dir
        mdp_path = run_dir / f"{tag}.mdp"
        mdp_path.write_text(
            _MD_MDP_TEMPLATE.format(
                nsteps=n_steps,
                report_interval=(report_interval_steps if trajectory_path is not None else 0),
                seed=self._seed,
            )
        )

        # On the very first round, no prior checkpoint exists. grompp without -t will
        # generate fresh velocities from the structure, which we don't want (gen-vel=no
        # in mdp). For round 0 we generate velocities once via gen-vel=yes; for
        # subsequent rounds the cpt carries them forward. We side-step this by
        # always running grompp without -t on the first round but injecting a
        # temporary gen-vel=yes mdp override.
        grompp_args = [
            "grompp",
            "-f", mdp_path.name,
            "-c", self._last_gro.name if self._last_gro.parent == run_dir else str(self._last_gro),
            "-p", self._top_path.name if self._top_path.parent == run_dir else str(self._top_path),
            "-o", f"{tag}.tpr",
            "-maxwarn", "1",
        ]
        if self._last_cpt is not None:
            grompp_args.extend(["-t", self._last_cpt.name if self._last_cpt.parent == run_dir else str(self._last_cpt)])
        else:
            # Cold-start round: rewrite mdp with gen-vel=yes for this call only
            cold_mdp = _MD_MDP_TEMPLATE.format(
                nsteps=n_steps,
                report_interval=(report_interval_steps if trajectory_path is not None else 0),
                seed=self._seed,
            ).replace("gen-vel              = no", f"gen-vel              = yes\ngen-temp             = {_TEMPERATURE_K}\ngen-seed             = {self._seed}") \
             .replace("continuation         = yes", "continuation         = no")
            mdp_path.write_text(cold_mdp)
        _run_gmx(*grompp_args, cwd=run_dir)

        # mdrun
        mdrun_args = [
            "mdrun",
            "-deffnm", tag,
            "-ntmpi", "1",
        ]
        _run_gmx(*mdrun_args, cwd=run_dir)

        produced_xtc = run_dir / f"{tag}.xtc"
        if trajectory_path is not None:
            trajectory_path = Path(trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced_xtc), str(trajectory_path))
        elif produced_xtc.exists():
            produced_xtc.unlink()

        self._last_gro = run_dir / f"{tag}.gro"
        self._last_cpt = run_dir / f"{tag}.cpt"
        return trajectory_path

    def save_checkpoint(self, path: Path) -> Path:
        if self._last_cpt is None:
            raise RuntimeError(
                "GROMACSAdapter.save_checkpoint() called before any run_steps()"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._last_cpt, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        if not self._started:
            raise RuntimeError("GROMACSAdapter not started; call start() first")
        # Copy external cpt into setup/ so subsequent grompp -t resolves relatively.
        target = self._setup_dir / "resume.cpt"
        shutil.copy2(path, target)
        self._last_cpt = target

    def _next_tag(self) -> str:
        existing = sorted(self._setup_dir.glob("round_*.tpr"))
        return f"{len(existing) + 1:03d}"

    def _is_setup_cached(self) -> bool:
        return all(
            p.exists()
            for p in (
                self._setup_dir / "topol.top",
                self._setup_dir / "em.gro",
                self._topology_path,
            )
        )
