"""Live metadynamics: does the rendered plumed.dat actually bias dynamics?

Everything downstream of `switch_to_metad` has until now been verified against
rendered *text* only — no PLUMED runtime existed, so `_prepare_plumed_force`
had never returned a real force and no hill had ever been deposited. This
exercises the full mechanical pivot path minus the LLM:

    vanilla run -> design_cv -> design_bias -> plumed.dat -> biased run

and checks PLUMED's own output, which is the only evidence that the bias was
applied rather than merely written to a file.

Requires a PLUMED runtime (conda-forge `openmm-plumed` + `plumed`); skipped
otherwise. ~3 min on CPU.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import PlumedInput
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.sampling.bias_designer import design_bias
from mdpilot.sampling.cv_designer import CVProposal, design_cv

_FIXED_PDB = Path("benchmarks/data/trpcage/1L2Y_fixed.pdb")
_VANILLA_STEPS = 5_000   # 10 ps
_BIASED_STEPS = 15_000   # 30 ps -> 30 hills at PACE=500
_SIGMA_NM = 0.02         # Rg hill width; see the note in the fixture

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("openmmplumed") is None,
        reason="no PLUMED runtime installed",
    ),
    pytest.mark.skipif(not _FIXED_PDB.exists(), reason=f"{_FIXED_PDB} missing"),
]


@pytest.fixture(scope="module")
def biased_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run vanilla, size a bias off it, then run biased. Return the artifacts."""
    root = tmp_path_factory.mktemp("metad")
    spec = SystemSpec(structure_path=_FIXED_PDB)

    vanilla = OpenMMAdapter(
        work_dir=root / "vanilla", spec=spec, nvt_steps=3000, npt_steps=1500
    )
    vanilla.prepare()
    vanilla.start()
    vanilla_dcd = root / "vanilla.dcd"
    vanilla.run_steps(_VANILLA_STEPS, trajectory_path=vanilla_dcd)

    topology = md.load_topology(str(vanilla.topology_path))
    cv = design_cv(
        CVProposal(cv_type="gyration", selections=("backbone",), label="rg"),
        topology,
    )
    designed = design_bias(cv, vanilla_dcd, vanilla.topology_path)
    # `designed.sigma` off a 10 ps vanilla run comes out at ~0.0018 nm — the
    # backbone Rg barely moves in 10 ps, so spread/3 is a gross underestimate
    # of the basin width. Hills that narrow never overlap, the accumulated
    # bias at each new deposit stays ~0, and well-tempering has nothing to
    # damp. Widening to a physically sensible Rg width here keeps this test
    # about "is the bias coupled and is WT live" rather than about the sizing
    # heuristic; the sizing weakness itself is recorded as a finding.
    bias = replace(designed, sigma=(_SIGMA_NM,))
    biased_dir = root / "biased"
    biased_dir.mkdir(parents=True, exist_ok=True)
    plumed_input = PlumedInput(
        cvs=(cv,), bias=bias, output_dir=biased_dir.resolve()
    ).render()

    biased = OpenMMAdapter(
        work_dir=biased_dir,
        spec=spec,
        plumed_input=plumed_input,
        nvt_steps=3000,
        npt_steps=1500,
    )
    biased.prepare()
    biased.start()
    biased_dcd = root / "biased.dcd"
    biased.run_steps(_BIASED_STEPS, trajectory_path=biased_dcd)

    return {
        "dir": biased_dir,
        "bias": bias,
        "cv": cv,
        "vanilla_dcd": vanilla_dcd,
        "biased_dcd": biased_dcd,
        "topology": vanilla.topology_path,
    }


def test_hills_file_is_written(biased_run: dict) -> None:
    assert (biased_run["dir"] / "HILLS").exists()


def test_hills_were_actually_deposited(biased_run: dict) -> None:
    """A HILLS file with only a header means PLUMED loaded the input but the
    bias never fired — the exact failure a text-only test cannot see."""
    lines = [
        ln
        for ln in (biased_run["dir"] / "HILLS").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    expected = _BIASED_STEPS // biased_run["bias"].pace
    assert len(lines) == pytest.approx(expected, abs=1)


def test_deposited_hills_carry_the_well_tempered_bias_factor(biased_run: dict) -> None:
    """PLUMED echoes biasf into HILLS. A plain-metaD run writes 1 or -1 there,
    so this is the check that well-tempering is live and not just rendered."""
    text = (biased_run["dir"] / "HILLS").read_text()
    fields = next(ln for ln in text.splitlines() if ln.startswith("#! FIELDS"))
    assert "biasf" in fields

    columns = fields.split()[2:]
    biasf_index = columns.index("biasf")
    rows = [
        ln.split()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert float(rows[0][biasf_index]) == pytest.approx(
        biased_run["bias"].bias_factor
    )


def test_hill_heights_decay_under_well_tempering(biased_run: dict) -> None:
    """The defining property of WT-MetaD: deposition decays as the bias fills
    the basin. Plain metaD would hold every height at W0."""
    text = (biased_run["dir"] / "HILLS").read_text()
    fields = next(ln for ln in text.splitlines() if ln.startswith("#! FIELDS"))
    height_index = fields.split()[2:].index("height")
    heights = np.array(
        [
            float(ln.split()[height_index])
            for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    )
    # PLUMED records well-tempered heights pre-scaled by γ/(γ-1) so that
    # summing HILLS reconstructs the bias directly — the first hill is
    # W0·γ/(γ-1), not W0.
    gamma = biased_run["bias"].bias_factor
    expected_first = biased_run["bias"].height * gamma / (gamma - 1.0)
    assert heights[0] == pytest.approx(expected_first, rel=1e-3)
    assert heights[-1] < heights[0], f"heights did not decay: {heights}"


def test_colvar_tracks_the_designed_cv(biased_run: dict) -> None:
    colvar = biased_run["dir"] / "COLVAR"
    assert colvar.exists()
    fields = next(
        ln for ln in colvar.read_text().splitlines() if ln.startswith("#! FIELDS")
    )
    assert biased_run["cv"].label in fields.split()
    assert "metad.bias" in fields


def test_bias_energy_grows_from_zero(biased_run: dict) -> None:
    """metad.bias starts at 0 (no hills yet) and must be positive once hills
    have accumulated — direct evidence the force is coupled to the dynamics."""
    colvar = biased_run["dir"] / "COLVAR"
    fields = next(
        ln for ln in colvar.read_text().splitlines() if ln.startswith("#! FIELDS")
    )
    bias_index = fields.split()[2:].index("metad.bias")
    values = np.array(
        [
            float(ln.split()[bias_index])
            for ln in colvar.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    )
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values.max() > 0.0
