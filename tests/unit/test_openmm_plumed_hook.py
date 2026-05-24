"""PLUMED prep helper: writes plumed.dat and guard-imports openmmplumed.

Live "force actually attaches to the System and biases dynamics" is
deferred to AWS (D6 step 5) where the PLUMED runtime is controlled.
These unit tests cover the parts that don't need a working PLUMED:
the disk write (which must happen even when the import fails so the
audit artifact survives) and the import-failure error path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mdpilot.adapters.openmm_adapter import _prepare_plumed_force


def test_writes_plumed_dat_even_when_openmmplumed_missing(tmp_path: Path) -> None:
    """The disk write happens before the import attempt. Even on the
    error path, plumed.dat is on disk for inspection."""
    assert "openmmplumed" not in sys.modules
    script = "d1: DISTANCE ATOMS=1,2\nrestraint: RESTRAINT ARG=d1 AT=0.5 KAPPA=1000\n"
    with pytest.raises(RuntimeError, match="openmmplumed"):
        _prepare_plumed_force(script, tmp_path)
    written = (tmp_path / "plumed.dat").read_text()
    assert written == script


def test_raises_runtimeerror_mentioning_openmmplumed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        _prepare_plumed_force("x: DISTANCE ATOMS=1,2\n", tmp_path)
    msg = str(exc_info.value)
    assert "openmmplumed" in msg
    assert "pip install openmmplumed" in msg


def test_returns_plumed_force_when_import_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a stub `openmmplumed` module so the success path is
    exercisable without a real install."""
    calls: list[str] = []

    class _StubPlumedForce:
        def __init__(self, script: str):
            calls.append(script)
            self.script = script

    stub = types.ModuleType("openmmplumed")
    stub.PlumedForce = _StubPlumedForce  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openmmplumed", stub)

    script = "d1: DISTANCE ATOMS=1,2\n"
    force = _prepare_plumed_force(script, tmp_path)

    assert isinstance(force, _StubPlumedForce)
    assert force.script == script
    assert calls == [script]
    assert (tmp_path / "plumed.dat").read_text() == script


def test_creates_work_dir_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "campaign"
    assert not target.exists()
    with pytest.raises(RuntimeError):
        _prepare_plumed_force("d: DISTANCE ATOMS=1,2\n", target)
    assert (target / "plumed.dat").exists()
