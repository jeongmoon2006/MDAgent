"""The `MDAdapter` physics properties, across both engines.

Constructing an adapter runs no MD and shells out to nothing — `prepare()` and
`start()` are where the cost is — so the contract itself can be pinned without
OpenMM kernels or a `gmx` binary.

These two properties exist because the loop used to hardcode both: a 2 fs
timestep and a 300 K thermostat that merely *happened* to match what each
adapter was configured with. Nothing connected the two, so an engine at a
different dt would have run every round at the wrong length (`extra_ns` no
longer meaning nanoseconds), and an engine at a different temperature would
have had PLUMED compute its well-tempered scaling against the wrong T with no
error raised anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.gromacs_adapter import GROMACSAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.system_spec import SystemSpec


def _adapters(work_dir: Path) -> list[MDAdapter]:
    spec = SystemSpec.trpcage()
    return [
        OpenMMAdapter(work_dir=work_dir, spec=spec),
        GROMACSAdapter(work_dir=work_dir, spec=spec),
    ]


@pytest.mark.parametrize("index", [0, 1], ids=["openmm", "gromacs"])
def test_adapters_declare_their_timestep_in_femtoseconds(
    tmp_path: Path, index: int
) -> None:
    adapter = _adapters(tmp_path)[index]
    # 2 fs, not 0.002 — GROMACS states dt in picoseconds internally and has to
    # convert. A unit slip here is a 1000x error in every round length.
    assert adapter.timestep_fs == pytest.approx(2.0)


@pytest.mark.parametrize("index", [0, 1], ids=["openmm", "gromacs"])
def test_adapters_declare_their_thermostat_temperature(
    tmp_path: Path, index: int
) -> None:
    adapter = _adapters(tmp_path)[index]
    assert adapter.temperature_k == pytest.approx(300.0)


def test_both_engines_agree_on_the_physics_they_share(tmp_path: Path) -> None:
    """Cross-engine convergence judgment (M3) assumes the two adapters are
    running comparable physics. These are the two constants the loop reasons
    in, so a divergence would make rounds from the two engines non-comparable
    while every diagnostic still reported cleanly."""
    openmm, gromacs = _adapters(tmp_path)
    assert openmm.timestep_fs == gromacs.timestep_fs
    assert openmm.temperature_k == gromacs.temperature_k


@pytest.mark.parametrize("index", [0, 1], ids=["openmm", "gromacs"])
def test_adapters_satisfy_the_runtime_checkable_protocol(
    tmp_path: Path, index: int
) -> None:
    assert isinstance(_adapters(tmp_path)[index], MDAdapter)
