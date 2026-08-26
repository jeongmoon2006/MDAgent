"""The setup agent: a request in, a reviewable task file out.

The contract that matters is not that the model produces something plausible
but that nothing leaves this module unless `load_task_file` accepts it. These
run against a fake client — no API key, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mdpilot.setup_agent import _TASK_FILE_TOOL, propose_task_file, to_document


def _valid(**overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "cln025_folding",
        "description": "Chignolin variant CLN025 in TIP3P water.",
        "starting_pdb": "5AWL",
        "structure_path": None,
        "forcefield": "amber14/tip3p",
        "padding_nm": 1.5,
        "temperature_K": 300.0,
        "timestep_fs": 2.0,
        "observable_cv_type": "contacts",
        "observable_selections": ["protein and name CA"],
        "observable_name": "q_native_contacts",
        "observable_scale": 1.0,
        "pressure_bar": 1.0,
        "observable_normalize": True,
        "objective": "Sample folding and unfolding in both directions.",
        "characteristic_timescale_ns": 800.0,
        "timescale_source": "Lindorff-Larsen et al. 2011, Science 334:517",
        "low_state": {"name": "unfolded", "threshold": 1.0},
        "high_state": {"name": "native hairpin", "threshold": 9.0},
        "min_recrossings": 2,
        "max_biased_ns": 20.0,
        "cv_upper_wall_nm": None,
    }
    payload.update(overrides)
    return payload


class _Block:
    type = "tool_use"
    name = "record_task_file"

    def __init__(self, payload: dict[str, Any], index: int) -> None:
        self.input = payload
        self.id = f"toolu_fake_{index}"


class _Response:
    stop_reason = "tool_use"

    def __init__(self, payload: dict[str, Any], index: int) -> None:
        self.content = [_Block(payload, index)]


class _FakeClient:
    """Returns each queued payload in turn; records what it was sent."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self._queue = list(payloads)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> _Response:
        self.requests.append(kwargs)
        return _Response(self._queue.pop(0), len(self.requests))


# ---------- the happy path ----------

def test_a_valid_proposal_becomes_a_loadable_task_file(tmp_path: Path) -> None:
    out = tmp_path / "task.yaml"

    task = propose_task_file("fold chignolin", out, client=_FakeClient(_valid()))

    assert out.exists()
    assert task.name == "cln025_folding"
    assert task.observable_name == "q_native_contacts"
    assert task.campaign["state_thresholds"] == (1.0, 9.0)
    assert task.campaign["min_recrossings"] == 2
    # The expectation is rendered from the fields, carrying the source through.
    assert "Lindorff-Larsen" in task.campaign["task_expectation"]


def test_the_written_file_is_yaml_a_human_can_edit(tmp_path: Path) -> None:
    out = tmp_path / "task.yaml"
    propose_task_file("fold chignolin", out, client=_FakeClient(_valid()))

    doc = yaml.safe_load(out.read_text())

    assert doc["system"]["starting_pdb"] == "5AWL"
    assert doc["observable"]["cv_type"] == "contacts"
    assert doc["expectation"]["timescale_source"]
    assert "sampling" not in doc          # null wall is dropped, not written as null


# ---------- the loader is the judge ----------

def test_a_refused_proposal_is_retried_with_the_loaders_own_message(
    tmp_path: Path,
) -> None:
    """The loader names the field and the constraint, which is a better signal
    than any rubric this module could restate."""
    inverted = _valid(
        low_state={"name": "native", "threshold": 9.0},
        high_state={"name": "unfolded", "threshold": 1.0},
    )
    client = _FakeClient(inverted, _valid())

    task = propose_task_file("fold chignolin", tmp_path / "t.yaml", client=client)

    assert task.campaign["state_thresholds"] == (1.0, 9.0)
    assert len(client.requests) == 2
    # A refused tool call must be answered with a matching `tool_result`; the
    # API rejects an assistant `tool_use` followed by a bare user turn, so the
    # plain-text form 400s and the loop never actually retries.
    followup = client.requests[1]["messages"][-1]
    assert followup["role"] == "user"
    (result,) = followup["content"]
    assert result["type"] == "tool_result"
    assert result["is_error"] is True
    assert result["tool_use_id"] == client.requests[1]["messages"][-2]["content"][0].id
    assert "high.threshold > low.threshold" in result["content"]


def test_exhausting_the_attempts_raises_and_leaves_no_file(tmp_path: Path) -> None:
    """A task file that cannot be loaded is not a partial result worth keeping —
    leaving one on disk invites someone to run it."""
    bad = _valid(timestep_fs=4.0)          # no HMR exists; Ensemble refuses it
    out = tmp_path / "t.yaml"

    with pytest.raises(RuntimeError, match="no valid task file after 2"):
        propose_task_file(
            "fold chignolin", out, client=_FakeClient(bad, bad), max_attempts=2
        )
    assert not out.exists()
    assert not list(out.parent.glob(".*.candidate"))   # no scratch left behind


# ---------- what the schema makes impossible ----------

def test_the_schema_cannot_express_the_biased_cv() -> None:
    """Choosing which coordinate to bias is the scientist's judgment at pivot
    time. A field for it here would answer the question the campaign asks."""
    fields = set(_TASK_FILE_TOOL["input_schema"]["properties"])

    assert not any(
        f.startswith("metad") or f in {"cv_type", "bias_cv", "selections"}
        for f in fields
    )
    assert "observable_cv_type" in fields   # the observable is not the biased CV


def test_every_schema_field_is_required_and_closed() -> None:
    """Strict tool use only guarantees a shape if the shape is fully specified;
    an optional field is one the model can silently omit."""
    schema = _TASK_FILE_TOOL["input_schema"]

    assert _TASK_FILE_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_a_local_structure_is_expressible(tmp_path: Path) -> None:
    doc = to_document(_valid(starting_pdb=None, structure_path="seeded_ice.pdb"))

    assert doc["system"] == {
        "structure_path": "seeded_ice.pdb",
        "forcefield": "amber14/tip3p",
        "padding_nm": 1.5,
    }


# ---------- force field: a closed vocabulary, not free text ----------

def test_the_forcefield_enum_is_generated_from_the_vocabulary() -> None:
    """A hand-copied enum drifts, and a combination the code cannot build is
    exactly what must not be offerable."""
    from mdpilot import forcefields

    enum = _TASK_FILE_TOOL["input_schema"]["properties"]["forcefield"]["enum"]

    assert enum == list(forcefields.available())
    assert forcefields.DEFAULT_KEY in enum


def test_the_corpus_carries_the_selection_guide() -> None:
    from mdpilot.setup_agent import build_system_prompt

    prompt = build_system_prompt()

    assert "Choosing a force field" in prompt
    # The two facts most likely to be invented if they were not stated.
    assert "amber99sbildn/tip3p" in prompt          # the only cross-engine pair
    assert "TIP4P/Ice" in prompt                    # not available at all


def test_the_forcefield_reaches_the_spec(tmp_path: Path) -> None:
    task = propose_task_file(
        "study this with CHARMM",
        tmp_path / "t.yaml",
        client=_FakeClient(_valid(forcefield="charmm36/tip3p")),
    )

    assert task.spec.forcefield == "charmm36/tip3p"
    assert task.spec.to_dict()["forcefield"] == "charmm36/tip3p"   # locked
