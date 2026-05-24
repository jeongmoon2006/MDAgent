"""SystemSpec: structure source XOR rule, factory, serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot.adapters.system_spec import SystemSpec


def test_pdb_id_only_is_valid() -> None:
    spec = SystemSpec(pdb_id="1L2Y")
    assert spec.pdb_id == "1L2Y"
    assert spec.structure_path is None


def test_structure_path_only_is_valid(tmp_path: Path) -> None:
    p = tmp_path / "chignolin.pdb"
    spec = SystemSpec(structure_path=p)
    assert spec.structure_path == p
    assert spec.pdb_id is None


def test_neither_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SystemSpec()


def test_both_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SystemSpec(pdb_id="1L2Y", structure_path=tmp_path / "x.pdb")


def test_trpcage_factory_uses_1l2y() -> None:
    spec = SystemSpec.trpcage()
    assert spec.pdb_id == "1L2Y"
    assert spec.structure_path is None


def test_to_dict_for_pdb_id() -> None:
    spec = SystemSpec(pdb_id="1UAO")
    assert spec.to_dict() == {"pdb_id": "1UAO", "structure_path": None}


def test_to_dict_for_structure_path(tmp_path: Path) -> None:
    p = tmp_path / "chig.pdb"
    spec = SystemSpec(structure_path=p)
    assert spec.to_dict() == {"pdb_id": None, "structure_path": str(p)}


def test_is_hashable_and_frozen() -> None:
    a = SystemSpec(pdb_id="1L2Y")
    b = SystemSpec(pdb_id="1L2Y")
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(Exception):
        a.pdb_id = "1UAO"  # frozen dataclass
