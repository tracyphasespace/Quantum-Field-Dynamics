#!/usr/bin/env python3
"""
Kernel mix sweep for sandbox QFD metric solver.

Iterates over several Gaussian/Ricker blend ratios by reloading the solver
module with different environment variables so we can see how mass errors
respond without touching production code.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

SWEEP_MIXES = [0.0, 0.25, 0.5, 0.75, 1.0]
MODULE_PATH = Path(__file__).with_name("qfd_metric_solver.py")


def load_solver(module_name: str):
    """Dynamically load qfd_metric_solver with current environment."""
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_sweep():
    records = []

    for idx, mix in enumerate(SWEEP_MIXES):
        os.environ["KERNEL_PROFILE"] = "mixed"
        os.environ["KERNEL_MIX"] = str(mix)

        module_name = f"sandbox_qfd_metric_solver_{idx}"
        solver_module = load_solver(module_name)
        results = solver_module.run_solver(verbose=False)

        def pick(name):
            return next((r for r in results if r["name"] == name), None)

        he4 = pick("He-4")
        c12 = pick("C-12")

        records.append({
            "mix": mix,
            "he4_err": he4["error"] if he4 else None,
            "c12_err": c12["error"] if c12 else None,
        })

    print(f"{'Mix':>6} {'He-4 Error (MeV)':>18} {'C-12 Error (MeV)':>18}")
    print("-" * 46)
    for rec in records:
        mix = rec["mix"]
        he4_err = rec["he4_err"]
        c12_err = rec["c12_err"]
        print(f"{mix:>6.2f} {he4_err:>18.2f} {c12_err:>18.2f}")


if __name__ == "__main__":
    run_sweep()
