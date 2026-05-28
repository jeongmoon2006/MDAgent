"""Compose the per-round diagnostic report bundle.

The report is the structured artifact handed to `scientist.decide()`. It must
stay compact (no raw trajectories, no per-level arrays) so that round summaries
fit in the LLM's context. Trajectory bytes stay on disk; the bundle holds
paths instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np

from mdpilot.diagnostics.autocorrelation import autocorrelation
from mdpilot.diagnostics.block_averaging import block_average
from mdpilot.diagnostics.exploration import exploration

_MIN_FRAMES_FOR_STATISTICS = 8


def make_report(dcd_path: Path, top_path: Path) -> dict[str, Any]:
    """Load a DCD, compute CA-RMSD-to-first-frame, summarize convergence."""
    dcd_path = Path(dcd_path)
    top_path = Path(top_path)
    traj = md.load(str(dcd_path), top=str(top_path))
    ca = traj.topology.select("protein and name CA")
    if ca.size == 0:
        raise ValueError("no protein CA atoms found in topology")
    protein = traj.atom_slice(ca)
    rmsd_angstrom = md.rmsd(protein, protein, frame=0) * 10.0  # nm -> A

    if traj.n_frames > 1:
        frame_dt_ps = float(np.diff(traj.time).mean())
        length_ns = float(traj.time[-1] - traj.time[0]) / 1000.0
    else:
        frame_dt_ps = float("nan")
        length_ns = 0.0

    return _summarize(
        observable=rmsd_angstrom,
        observable_name="rmsd_ca_to_first_angstrom",
        dcd_path=dcd_path,
        top_path=top_path,
        frame_dt_ps=frame_dt_ps,
        length_ns=length_ns,
    )


def _summarize(
    *,
    observable: np.ndarray,
    observable_name: str,
    dcd_path: Path,
    top_path: Path,
    frame_dt_ps: float,
    length_ns: float,
) -> dict[str, Any]:
    n = int(observable.size)
    base: dict[str, Any] = {
        "trajectory_path": str(dcd_path),
        "topology_path": str(top_path),
        "n_frames": n,
        "frame_dt_ps": frame_dt_ps,
        "trajectory_length_ns": length_ns,
        "observable_name": observable_name,
    }
    if n < _MIN_FRAMES_FOR_STATISTICS:
        base.update(
            mean=float(observable.mean()) if n > 0 else None,
            plateau_reached=False,
            well_sampled=False,
            ess=float(n),
            statistical_inefficiency_block=None,
            statistical_inefficiency_autocorr=None,
            tau_int_frames=None,
            bimodality_coefficient=None,
            n_basins=None,
            minor_basin_occupancy=None,
            exploring=None,
            note=f"too few frames (n={n} < {_MIN_FRAMES_FOR_STATISTICS}) for statistics",
        )
        return base

    block = block_average(observable)
    autocorr = autocorrelation(observable)
    explore = exploration(observable)
    base.update(
        mean=block.mean,
        sem_blocked=block.sem,
        sem_naive=block.sem_naive,
        plateau_reached=block.plateau_reached,
        statistical_inefficiency_block=block.statistical_inefficiency,
        statistical_inefficiency_autocorr=autocorr.statistical_inefficiency,
        tau_int_frames=autocorr.tau_int,
        ess=autocorr.ess,
        well_sampled=autocorr.well_sampled,
        bimodality_coefficient=explore.bimodality_coefficient,
        n_basins=explore.n_basins,
        minor_basin_occupancy=explore.minor_basin_occupancy,
        exploring=explore.exploring,
    )
    return base


def to_json(report: dict[str, Any]) -> str:
    """Serialize a report to compact JSON (deterministic key order)."""
    return json.dumps(report, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")
