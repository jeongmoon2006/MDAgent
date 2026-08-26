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
    assert spec.to_dict() == {
        "pdb_id": "1UAO", "structure_path": None, "padding_nm": 1.5,
    }


def test_to_dict_for_structure_path(tmp_path: Path) -> None:
    p = tmp_path / "chig.pdb"
    spec = SystemSpec(structure_path=p)
    assert spec.to_dict() == {
        "pdb_id": None, "structure_path": str(p), "padding_nm": 1.5,
    }


def test_is_hashable_and_frozen() -> None:
    a = SystemSpec(pdb_id="1L2Y")
    b = SystemSpec(pdb_id="1L2Y")
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(Exception):
        a.pdb_id = "1UAO"  # frozen dataclass


# ---------- Ensemble ----------
#
# Temperature and timestep moved off the adapters onto the spec so the campaign
# config lock covers them. These pin the two properties that makes safe: the
# serialization stays backward-compatible, and a changed ensemble is caught.

def test_a_default_ensemble_serializes_as_absent() -> None:
    """`store.init_campaign` compares config JSON byte-for-byte, so an
    unconditional new key would make every campaign recorded before this field
    existed refuse to resume. Verified against the real on-disk campaigns."""
    from mdpilot.adapters.system_spec import Ensemble, SystemSpec

    assert SystemSpec(pdb_id="5AWL").to_dict() == {
        "pdb_id": "5AWL",
        "structure_path": None,
        "padding_nm": 1.5,
    }
    assert "ensemble" in SystemSpec(
        pdb_id="5AWL", ensemble=Ensemble(temperature_k=240.0)
    ).to_dict()


def test_a_changed_ensemble_changes_the_locked_config() -> None:
    """The whole reason this lives on the spec rather than on an adapter
    keyword: resuming at a different temperature must not be representable."""
    from mdpilot.adapters.system_spec import Ensemble, SystemSpec

    base = SystemSpec(pdb_id="5AWL").to_dict()
    for changed in (
        Ensemble(temperature_k=240.0),
        Ensemble(timestep_fs=1.0),
    ):
        assert SystemSpec(pdb_id="5AWL", ensemble=changed).to_dict() != base


def test_a_timestep_needing_hmr_is_refused() -> None:
    """Neither adapter repartitions hydrogen mass and both constrain h-bonds
    only, so 4 fs would integrate unstably — or stay stable and report the
    wrong ensemble."""
    import pytest

    from mdpilot.adapters.system_spec import Ensemble

    with pytest.raises(ValueError, match="hydrogen mass"):
        Ensemble(timestep_fs=4.0)
    with pytest.raises(ValueError, match="temperature_k must be positive"):
        Ensemble(temperature_k=0.0)


# ---------- box padding ----------
#
# Raised from 1.0 to 1.5 after F11: padding sized on a folded starting
# structure is not padding sized for the ensemble a campaign intends to sample,
# and on CLN025 that left the peptide in contact with its own periodic image in
# half the biased rounds.

def test_padding_defaults_to_the_post_f11_value() -> None:
    from mdpilot.adapters.system_spec import SystemSpec

    assert SystemSpec(pdb_id="1L2Y").padding_nm == 1.5


def test_padding_is_always_serialized_unlike_the_ensemble() -> None:
    """The default was *raised*, so absent (a pre-F11 campaign, built at 1.0)
    and default (1.5) mean different boxes. Omitting it when default — the
    trick `ensemble` uses — would let a 1.0 nm campaign resume at 1.5 with the
    guard none the wiser."""
    from mdpilot.adapters.system_spec import Ensemble, SystemSpec

    spec = SystemSpec(pdb_id="1L2Y")

    assert spec.to_dict()["padding_nm"] == 1.5
    assert "ensemble" not in spec.to_dict()          # default ensemble == history
    assert spec.ensemble == Ensemble()


def test_a_pre_f11_campaign_is_refused_at_the_new_default_and_kept_at_its_own() -> None:
    from mdpilot.adapters.system_spec import SystemSpec
    from mdpilot.memory.store import config_differences

    stored = {"system_spec": {"pdb_id": "5AWL", "structure_path": None}}
    at_new_default = {"system_spec": SystemSpec(pdb_id="5AWL").to_dict()}
    at_its_own = {"system_spec": SystemSpec(pdb_id="5AWL", padding_nm=1.0).to_dict()}

    assert config_differences(stored, at_new_default)   # refused: different box
    assert not config_differences(stored, at_its_own)   # its own box: resumable


def test_a_nonsensical_padding_is_refused() -> None:
    import pytest

    from mdpilot.adapters.system_spec import SystemSpec

    with pytest.raises(ValueError, match="padding_nm must be positive"):
        SystemSpec(pdb_id="1L2Y", padding_nm=0.0)
