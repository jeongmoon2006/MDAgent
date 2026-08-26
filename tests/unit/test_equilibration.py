"""Equilibration wiring in both adapters, without running any MD.

The physics these stages produce needs a live run to verify (see
`tests/integration/test_equilibration_live.py`); what is checkable cheaply is
the control logic: the heating ramp hits its endpoints, the barostat carries
the intended pressure/temperature, and the GROMACS mdps declare the coupling
they are supposed to declare.

The heating helper is driven with fakes rather than a real Simulation — the
property under test is ramp arithmetic, which does not need 6500 solvated
atoms to exercise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openmm import unit

import mdpilot.adapters.openmm_adapter as oa
from mdpilot.adapters.gromacs_adapter import (
    _MD_MDP_TEMPLATE,
    _NPT_MDP_TEMPLATE,
    _NVT_MDP_TEMPLATE,
    GROMACSAdapter,
)
from mdpilot.adapters.openmm_adapter import (
    _BAROSTAT_INTERVAL,
    _HEAT_STAGES,
    _HEAT_START_K,
    _PRESSURE_BAR,
    OpenMMAdapter,
)
from mdpilot.adapters.system_spec import Ensemble, SystemSpec


class _FakeIntegrator:
    def __init__(self) -> None:
        self.temperatures_k: list[float] = []

    def setTemperature(self, quantity: unit.Quantity) -> None:  # noqa: N802
        self.temperatures_k.append(quantity.value_in_unit(unit.kelvin))


class _FakeSim:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def step(self, n: int) -> None:
        self.steps.append(n)


# ---------- OpenMM: staged heating ----------

def test_heating_ramp_ends_at_target_temperature(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, nvt_steps=6000, npt_steps=0)
    integrator, sim = _FakeIntegrator(), _FakeSim()

    adapter._heat_nvt(sim, integrator)  # type: ignore[arg-type]

    assert len(integrator.temperatures_k) == _HEAT_STAGES
    # The last stage must be the production temperature exactly — an NPT stage
    # that inherits 290 K would relax the density at the wrong state point.
    assert integrator.temperatures_k[-1] == pytest.approx(adapter.temperature_k)
    assert integrator.temperatures_k[0] > _HEAT_START_K
    assert integrator.temperatures_k[0] < adapter.temperature_k


def test_heating_ramp_is_monotonic(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, nvt_steps=6000, npt_steps=0)
    integrator, sim = _FakeIntegrator(), _FakeSim()

    adapter._heat_nvt(sim, integrator)  # type: ignore[arg-type]

    temps = integrator.temperatures_k
    assert all(b > a for a, b in zip(temps, temps[1:]))


def test_heating_spends_the_requested_step_budget(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, nvt_steps=6000, npt_steps=0)
    integrator, sim = _FakeIntegrator(), _FakeSim()

    adapter._heat_nvt(sim, integrator)  # type: ignore[arg-type]

    assert sum(sim.steps) == 6000
    assert len(sim.steps) == _HEAT_STAGES


def test_heating_is_skipped_when_disabled(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, nvt_steps=0, npt_steps=0)
    integrator, sim = _FakeIntegrator(), _FakeSim()

    adapter._heat_nvt(sim, integrator)  # type: ignore[arg-type]

    assert integrator.temperatures_k == []
    assert sim.steps == []


# ---------- OpenMM: barostat ----------

def test_barostat_carries_pressure_temperature_and_seed(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, seed=1234)

    barostat = adapter._make_barostat()

    assert barostat.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(
        _PRESSURE_BAR
    )
    assert barostat.getDefaultTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(adapter.temperature_k)
    assert barostat.getFrequency() == _BAROSTAT_INTERVAL
    # Volume moves are Monte Carlo; an unseeded barostat would break the
    # project's seeded-determinism contract.
    assert barostat.getRandomNumberSeed() == 1234


# ---------- GROMACS: mdp content ----------

def test_nvt_mdp_anneals_from_start_to_target_and_generates_velocities() -> None:
    mdp = _NVT_MDP_TEMPLATE.format(
        nsteps=50_000, anneal_ps="100", seed=42, ref_t=300.0, dt_ps=0.002
    )

    assert "annealing            = single" in mdp
    assert f"annealing-temp       = {_HEAT_START_K} 300.0" in mdp
    assert "annealing-time       = 0 100" in mdp
    assert "gen-vel              = yes" in mdp
    assert f"gen-temp             = {_HEAT_START_K}" in mdp
    # Heating is NVT — the barostat only comes on for the density stage.
    assert "pcoupl" not in mdp


def test_npt_mdp_couples_pressure_and_continues_from_nvt() -> None:
    mdp = _NPT_MDP_TEMPLATE.format(
        nsteps=50_000, seed=42, ref_t=300.0, dt_ps=0.002
    )

    assert "pcoupl               = C-rescale" in mdp
    assert "ref-p                = 1.0" in mdp
    # Velocities come from the NVT checkpoint, never regenerated here.
    assert "gen-vel              = no" in mdp
    assert "continuation         = yes" in mdp


def test_production_mdp_keeps_the_barostat_on() -> None:
    mdp = _MD_MDP_TEMPLATE.format(
        nsteps=1000, report_interval=500, seed=42, ref_t=300.0, dt_ps=0.002
    )

    assert "pcoupl               = C-rescale" in mdp
    assert "ref-p                = 1.0" in mdp


def test_production_mdp_cold_start_override_still_matches() -> None:
    """`run_steps` rewrites two exact lines for the equilibration-disabled
    path. If the template's spacing drifts, that surgery silently no-ops."""
    mdp = _MD_MDP_TEMPLATE.format(
        nsteps=1000, report_interval=500, seed=42, ref_t=300.0, dt_ps=0.002
    )

    assert "gen-vel              = no" in mdp
    assert "continuation         = yes" in mdp


# ---------- OpenMM: platform selection ----------

def test_resolve_platform_prefers_gpu_over_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oa, "_registered_platforms", lambda: {"CPU", "CUDA"})
    monkeypatch.setattr(oa, "_platform_is_usable", lambda name: True)
    assert oa.resolve_platform.__wrapped__() == "CUDA"


def test_resolve_platform_falls_back_when_gpu_registered_but_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA_ERROR_UNSUPPORTED_PTX_VERSION case: CUDA registers fine and
    then fails at kernel load. Selecting on registration alone would hand back
    a platform that dies mid-campaign."""
    monkeypatch.setattr(oa, "_registered_platforms", lambda: {"CPU", "CUDA"})
    monkeypatch.setattr(oa, "_platform_is_usable", lambda name: name != "CUDA")
    assert oa.resolve_platform.__wrapped__() == "CPU"


def test_resolve_platform_returns_cpu_when_nothing_probes_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oa, "_registered_platforms", lambda: {"CPU", "CUDA"})
    monkeypatch.setattr(oa, "_platform_is_usable", lambda name: False)
    assert oa.resolve_platform.__wrapped__() == "CPU"


def test_explicit_platform_is_not_overridden(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(work_dir=tmp_path, platform="Reference")
    assert adapter.platform_name == "Reference"


def test_gpu_platforms_request_mixed_precision() -> None:
    """Single precision (the OpenMM GPU default) would make a GPU run quietly
    less accurate than the CPU run it replaces."""
    assert oa._precision_properties("CUDA") == {"Precision": "mixed"}
    assert oa._precision_properties("OpenCL") == {"Precision": "mixed"}
    assert oa._precision_properties("CPU") == {}


def test_bogus_platform_probes_as_unusable() -> None:
    assert oa._platform_is_usable("NoSuchPlatform") is False


# ---------- GROMACS: setup staging ----------

def test_final_setup_tag_is_npt_when_equilibrating(tmp_path: Path) -> None:
    adapter = GROMACSAdapter(work_dir=tmp_path)
    assert adapter._final_setup_tag() == "npt"


def test_final_setup_tag_falls_back_to_em_when_equilibration_disabled(
    tmp_path: Path,
) -> None:
    adapter = GROMACSAdapter(work_dir=tmp_path, nvt_steps=0, npt_steps=0)
    assert adapter._final_setup_tag() == "em"


# ---------- the ensemble actually reaches the engine ----------

def test_openmm_heating_ramp_targets_a_non_default_ensemble(tmp_path: Path) -> None:
    """Temperature used to be a module constant, so the ramp always ended at
    300 K whatever the campaign asked for."""
    adapter = OpenMMAdapter(
        work_dir=tmp_path,
        spec=SystemSpec(pdb_id="1L2Y", ensemble=Ensemble(temperature_k=240.0)),
        nvt_steps=6000,
        npt_steps=0,
    )
    integrator, sim = _FakeIntegrator(), _FakeSim()

    adapter._heat_nvt(sim, integrator)  # type: ignore[arg-type]

    assert adapter.temperature_k == 240.0
    assert integrator.temperatures_k[-1] == pytest.approx(240.0)
    assert adapter._make_barostat().getDefaultTemperature().value_in_unit(
        unit.kelvin
    ) == pytest.approx(240.0)


def test_openmm_integrator_uses_the_ensemble_timestep(tmp_path: Path) -> None:
    adapter = OpenMMAdapter(
        work_dir=tmp_path,
        spec=SystemSpec(pdb_id="1L2Y", ensemble=Ensemble(timestep_fs=1.0)),
    )

    assert adapter.timestep_fs == 1.0
    assert adapter._make_integrator().getStepSize().value_in_unit(
        unit.femtosecond
    ) == pytest.approx(1.0)


def test_gromacs_mdp_renders_the_ensemble_not_a_baked_constant() -> None:
    """`ref-t` and `dt` were f-string-interpolated at import, so no campaign
    could change them. They are format placeholders now."""
    mdp = _MD_MDP_TEMPLATE.format(
        nsteps=1000, report_interval=500, seed=42, ref_t=240.0, dt_ps=0.001
    )

    assert "ref-t                = 240.0" in mdp
    assert "dt                   = 0.001" in mdp


# ---------- mdrun must never be killed on a clock ----------

def test_no_mdrun_invocation_carries_a_wall_clock_timeout() -> None:
    """A production round legitimately takes hours on CPU. A ceiling that fires
    kills MD already paid for, mid-round, leaving a partial .gro/.cpt — the
    opposite of what a safety timeout is for. Setup subcommands keep theirs,
    where a hang really does mean something is stuck.

    Checked at the call sites because that is where the invariant lives: adding
    a fifth `mdrun` without the override is the regression this catches.
    """
    import re
    from pathlib import Path

    source = Path("src/mdpilot/adapters/gromacs_adapter.py").read_text()
    calls = re.findall(r"_run_gmx\((?:[^()]|\([^()]*\))*\)", source, re.S)
    mdruns = [c for c in calls if "mdrun" in c]

    assert mdruns, "no mdrun call sites found — did the parse break?"
    for call in mdruns:
        assert "timeout=_NO_TIMEOUT" in call, call[:120]


def test_setup_subcommands_keep_their_ceiling() -> None:
    import inspect

    from mdpilot.adapters.gromacs_adapter import (
        _GMX_SETUP_TIMEOUT_SECONDS,
        _run_gmx,
    )

    default = inspect.signature(_run_gmx).parameters["timeout"].default
    assert default == _GMX_SETUP_TIMEOUT_SECONDS


def test_both_adapters_take_the_box_from_the_spec() -> None:
    """Padding used to be a module constant in each adapter, so a campaign
    could not change its own box. Checked at the call site, which is where the
    regression would be — reintroducing a constant and passing that instead.
    """
    from pathlib import Path

    openmm_src = Path("src/mdpilot/adapters/openmm_adapter.py").read_text()
    gromacs_src = Path("src/mdpilot/adapters/gromacs_adapter.py").read_text()

    assert "padding=self._spec.padding_nm * unit.nanometer" in openmm_src
    assert '"-d", str(self._spec.padding_nm),' in gromacs_src
    for src in (openmm_src, gromacs_src):
        assert "_PADDING_NM" not in src
