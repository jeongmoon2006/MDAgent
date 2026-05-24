"""Generate the planted Milestone-1 reference trajectories.

Produces:
- benchmarks/data/trpcage/under_converged.dcd  (50 ps from minimised state)
- benchmarks/data/trpcage/converged_ref.dcd    (5 ns after a 100 ps NVT discard)

Both are deterministic via seed. Topology PDB is shared.

Usage:
    python -m benchmarks.generate_trpcage_planted --traj under
    python -m benchmarks.generate_trpcage_planted --traj converged
    python -m benchmarks.generate_trpcage_planted --traj both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mdpilot.adapters.openmm_adapter import OpenMMAdapter

_DATA_DIR = Path("benchmarks/data/trpcage")

# 2 fs timestep ⇒ 500 steps/ps, 500_000 steps/ns
_UNDER_STEPS = 25_000          # 50 ps
_CONVERGED_EQ_STEPS = 50_000   # 100 ps NVT discard
_CONVERGED_PROD_STEPS = 2_500_000  # 5 ns


def _build(seed: int) -> OpenMMAdapter:
    adapter = OpenMMAdapter(work_dir=_DATA_DIR, seed=seed)
    adapter.prepare()
    adapter.start()
    return adapter


def gen_under(seed: int = 100) -> Path:
    out = _DATA_DIR / "under_converged.dcd"
    if out.exists():
        print(f"[under] {out} already exists, skipping")
        return out
    t0 = time.time()
    adapter = _build(seed)
    print(f"[under] built in {time.time()-t0:.1f}s; running {_UNDER_STEPS} steps (50 ps)...")
    t0 = time.time()
    adapter.run_steps(_UNDER_STEPS, trajectory_path=out, report_interval_steps=500)
    print(f"[under] done in {time.time()-t0:.1f}s -> {out}")
    return out


def gen_converged(seed: int = 200) -> Path:
    out = _DATA_DIR / "converged_ref.dcd"
    if out.exists():
        print(f"[converged] {out} already exists, skipping")
        return out
    t0 = time.time()
    adapter = _build(seed)
    print(f"[converged] built in {time.time()-t0:.1f}s; equilibrating {_CONVERGED_EQ_STEPS} steps (100 ps)...")
    t0 = time.time()
    adapter.run_steps(_CONVERGED_EQ_STEPS)
    print(f"[converged] equilibration done in {time.time()-t0:.1f}s")
    print(f"[converged] running production {_CONVERGED_PROD_STEPS} steps (5 ns) -- expect ~9 hours on CPU...")
    t0 = time.time()
    adapter.run_steps(_CONVERGED_PROD_STEPS, trajectory_path=out, report_interval_steps=500)
    print(f"[converged] done in {time.time()-t0:.1f}s -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", choices=["under", "converged", "both"], required=True)
    args = parser.parse_args()
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.traj in ("under", "both"):
        gen_under()
    if args.traj in ("converged", "both"):
        gen_converged()


if __name__ == "__main__":
    main()
