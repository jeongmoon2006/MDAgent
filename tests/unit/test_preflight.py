"""Pre-flight checks, written from a campaign that ran forty minutes wrong.

A setup agent asked for chignolin and produced a task file whose prose said
"Chignolin (CLN025) … 10-residue beta-hairpin" and whose fields said
`starting_pdb: 2RVD` — a 20-residue Trp-cage — with an observable named
`native_contacts_fraction` returning 938-2140 against thresholds of 0.3 and
0.7. Nothing downstream drifted; the disagreement was inside the proposal,
between its prose and its own fields. These compare across that seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpilot import preflight

# The real numbers from `campaigns/ui_campaign`.
_DRIFTED_DESCRIPTION = (
    "Chignolin (CLN025) is a designed 10-residue beta-hairpin peptide … the "
    "observable is the fraction of native contacts, bounded between 0 and 1."
)
_DRIFTED_FIRST_VALUE = 2061.0
_DRIFTED_BANDS = (0.3, 0.7)


def _pdb(tmp_path: Path, n_residues: int) -> Path:
    """A minimal poly-alanine topology of a given length."""
    lines = []
    for i in range(n_residues):
        lines.append(
            f"ATOM  {i + 1:>5d}  CA  ALA A{i + 1:>4d}    "
            f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    path = tmp_path / f"top{n_residues}.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


# ---------- the description against the structure ----------

def test_the_actual_drift_is_caught(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="10-residue system, but the structure"):
        preflight.check_residue_count(_pdb(tmp_path, 20), _DRIFTED_DESCRIPTION)


def test_a_matching_length_passes(tmp_path: Path) -> None:
    preflight.check_residue_count(_pdb(tmp_path, 10), _DRIFTED_DESCRIPTION)


def test_the_message_names_the_sequence_that_was_actually_fetched(
    tmp_path: Path,
) -> None:
    """So the reader can see *what* they got, not only that it was the wrong
    size — that is what identified it as Trp-cage."""
    with pytest.raises(ValueError) as excinfo:
        preflight.check_residue_count(_pdb(tmp_path, 4), "a 10-residue peptide")

    assert "AAAA" in str(excinfo.value)


@pytest.mark.parametrize(
    "description,expected",
    [
        ("a 10-residue beta-hairpin", 10),
        ("a 20 residue construct", 20),
        ("spanning 12 residues", 12),
        # Not a claim about size — a selection.
        ("contacts between residues 1-10", None),
        ("no claim at all", None),
        (None, None),
        # Two different claims: ambiguous, so not evidence. Guessing which was
        # meant would turn a safeguard into a source of false refusals.
        ("a 10-residue core inside a 20-residue construct", None),
        # The same claim twice is still one claim.
        ("a 10-residue peptide; this 10-residue system folds fast", 10),
    ],
)
def test_only_an_unambiguous_claim_is_treated_as_one(description, expected) -> None:
    assert preflight.declared_residue_count(description) == expected


def test_an_ambiguous_description_does_not_block_a_campaign(tmp_path: Path) -> None:
    preflight.check_residue_count(
        _pdb(tmp_path, 20), "a 10-residue core inside a 30-residue construct"
    )


# ---------- the observable against its own bands ----------

def test_a_count_reported_where_a_fraction_was_intended_is_caught() -> None:
    with pytest.raises(ValueError, match="not on the same scale"):
        preflight.check_observable_scale(
            _DRIFTED_FIRST_VALUE, _DRIFTED_BANDS, "native_contacts_fraction"
        )


def test_the_message_says_how_to_fix_it() -> None:
    with pytest.raises(ValueError) as excinfo:
        preflight.check_observable_scale(2061.0, (0.3, 0.7), "q")

    message = str(excinfo.value)
    assert "normalize: true" in message
    assert "recrossings` will stay at 0" in message


@pytest.mark.parametrize(
    "first_value,bands,why",
    [
        (1.0, (0.3, 0.7), "folded start: fully formed contacts, above the band"),
        (0.0, (0.3, 0.7), "unfolded start: no contacts, below the band"),
        (0.0, (1.5, 4.0), "CA-RMSD against its own reference is exactly 0"),
        (0.5, (0.3, 0.7), "inside the band"),
        (5.0, (1.5, 4.0), "extended start, one band-width out"),
    ],
)
def test_legitimate_starting_values_are_not_refused(first_value, bands, why) -> None:
    """A campaign legitimately starts wholly on one side of its bands — that is
    the normal case, not an error. This must fire on a mismatch of *kind*."""
    preflight.check_observable_scale(first_value, bands, "obs")


def test_no_thresholds_means_nothing_to_check() -> None:
    preflight.check_observable_scale(2061.0, None, "obs")


# ---------- through the loop, before any dynamics ----------

def test_the_loop_refuses_before_integrating_a_single_step(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole point: this should cost seconds, not the forty minutes it did."""
    import tests.unit.test_loop_metad_pivot as P
    from mdpilot.adapters.system_spec import SystemSpec
    from mdpilot.observables import ObservableSpec
    from mdpilot.orchestrator.loop import run_campaign

    P._stub_collaborators(monkeypatch, [P._stop()])
    adapter = P._FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(ValueError, match="not on the same scale"):
        run_campaign(
            work_dir=tmp_path,
            adapter=adapter,
            max_rounds=1,
            # A contact *count* on a 4-residue topology against fraction bands.
            observable=ObservableSpec(
                cv_type="gyration", selections=("name CA",), name="rg_nm"
            ),
            state_thresholds=(0.001, 0.002),
            task_expectation="fold it",
            **P._run_kwargs(),
        )

    assert adapter.run_calls == [], "MD ran despite a failed pre-flight"


def test_a_description_mismatch_also_stops_the_loop(tmp_path: Path, monkeypatch) -> None:
    import tests.unit.test_loop_metad_pivot as P
    from mdpilot.adapters.system_spec import SystemSpec
    from mdpilot.orchestrator.loop import run_campaign

    P._stub_collaborators(monkeypatch, [P._stop()])
    adapter = P._FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(ValueError, match="10-residue system"):
        run_campaign(
            work_dir=tmp_path, adapter=adapter, max_rounds=1,
            description="a 10-residue beta-hairpin", **P._run_kwargs(),
        )
    assert adapter.run_calls == []
