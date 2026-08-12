"""PLUMED prep helper: writes plumed.dat and guard-imports openmmplumed.

Two properties, both of which must hold whether or not a PLUMED runtime is
installed on the machine running the suite:

1. plumed.dat lands on disk *before* the import is attempted, so the audit
   artifact survives even when the import fails.
2. A missing `openmmplumed` produces an actionable RuntimeError rather than a
   bare ImportError.

The absence of `openmmplumed` is therefore simulated rather than assumed —
these tests originally asserted it was genuinely missing, which silently
stopped testing anything the moment the runtime was installed.

Live "the force attaches and actually biases dynamics" lives in
`tests/integration/test_metad_live.py`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mdpilot.adapters.openmm_adapter import _prepare_plumed_force


@pytest.fixture
def openmmplumed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `from openmmplumed import PlumedForce` to raise ImportError.

    Binding the name to None in sys.modules is the documented way to make an
    import fail without touching the filesystem.
    """
    monkeypatch.setitem(sys.modules, "openmmplumed", None)


def test_writes_plumed_dat_even_when_openmmplumed_missing(
    tmp_path: Path, openmmplumed_missing: None
) -> None:
    """The disk write happens before the import attempt. Even on the
    error path, plumed.dat is on disk for inspection."""
    script = "d1: DISTANCE ATOMS=1,2\nrestraint: RESTRAINT ARG=d1 AT=0.5 KAPPA=1000\n"
    with pytest.raises(RuntimeError, match="openmmplumed"):
        _prepare_plumed_force(script, tmp_path)
    assert (tmp_path / "plumed.dat").read_text() == script


def test_raises_runtimeerror_mentioning_openmmplumed(
    tmp_path: Path, openmmplumed_missing: None
) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        _prepare_plumed_force("x: DISTANCE ATOMS=1,2\n", tmp_path)
    msg = str(exc_info.value)
    assert "openmmplumed" in msg
    assert "pip install openmmplumed" in msg


def test_creates_work_dir_if_missing(
    tmp_path: Path, openmmplumed_missing: None
) -> None:
    target = tmp_path / "deep" / "nested" / "campaign"
    assert not target.exists()
    with pytest.raises(RuntimeError):
        _prepare_plumed_force("d: DISTANCE ATOMS=1,2\n", target)
    assert (target / "plumed.dat").exists()


def test_returns_plumed_force_when_import_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a stub `openmmplumed` module so the success path is exercisable
    on machines without a PLUMED runtime."""
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


@pytest.mark.skipif(
    "openmmplumed" not in sys.modules
    and __import__("importlib.util", fromlist=["util"]).find_spec("openmmplumed")
    is None,
    reason="no PLUMED runtime installed",
)
def test_returns_a_real_plumed_force_when_runtime_is_present(tmp_path: Path) -> None:
    """With a real runtime, the helper must hand back an actual PlumedForce —
    the stub test above cannot catch a signature drift in openmmplumed."""
    from openmmplumed import PlumedForce

    script = "d1: DISTANCE ATOMS=1,2\n"
    force = _prepare_plumed_force(script, tmp_path)

    assert isinstance(force, PlumedForce)
    assert (tmp_path / "plumed.dat").read_text() == script
