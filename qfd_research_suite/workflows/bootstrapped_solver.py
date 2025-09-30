#!/usr/bin/env python3
"""
Bootstrapped_Solver.py

Fast refinement starting from a previously found good vector to speed up calibration.
- Uses your existing GOLDEN_SET and Solve_Nucleus.
- First does a short, local Differential Evolution (optional) in a tight box around the seed.
- Then polishes with L-BFGS-B.
- Writes results to ./workflows/runs_bootstrap/ with a timestamp.

Assumptions:
- This file lives in repo_root/workflows/ alongside calibration_pipeline.py
- qfd_lib/ is a sibling package containing solver_engine.py with Solve_Nucleus

Run examples (from repo root):
  python -m workflows.Bootstrapped_Solver
  # or
  PYTHONPATH=$PWD python workflows/Bootstrapped_Solver.py --window 0.25 --no-de
"""
from __future__ import annotations
from pathlib import Path
import os, sys, json, time, tempfile
import numpy as np
from typing import Dict, Any

# Detect WSL to choose safer multiprocessing defaults
IS_WSL = False
try:
    IS_WSL = "microsoft" in os.uname().release.lower()
except Exception:
    IS_WSL = False

# ──────────────────────────────────────────────────────────────────────────────
# Path shim: add repo root to import path (so qfd_lib and workflows are importable)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to import GOLDEN_SET and (optionally) param names from your existing pipeline
try:
    from workflows.calibration_pipeline import GOLDEN_SET as _GOLDEN_SET  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Could not import GOLDEN_SET from workflows.calibration_pipeline. "
        "Ensure you run from repo root and that calibration_pipeline.py defines GOLDEN_SET."
    ) from e

try:
    from workflows.calibration_pipeline import param_names as _PARAM_NAMES  # type: ignore
    PARAM_NAMES = list(_PARAM_NAMES)
except Exception:
    # Fallback generic names
    PARAM_NAMES = [f"p{i}" for i in range(14)]

# Core solver import
from qfd_lib.solver_engine import Solve_Nucleus  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Seed vector from previous Stage 1 best (user-provided)
SEED_VECTOR = np.array([
    -3.17573595e+00,  2.21839636e+00,  1.11395869e+00,  9.85645374e-01,
     1.72473295e+00,  7.43283377e+00,  6.39773333e-01,  1.04742186e+00,
     5.79191864e-01,  1.31341105e+00,  3.36433796e+00,  3.18392413e+00,
     6.68216766e-31,  9.25878809e-03
], dtype=float)

N_PARAMS = len(SEED_VECTOR)
GOLDEN_SET = _GOLDEN_SET
N_DATA = len(GOLDEN_SET)
DOF = max(1, N_DATA - N_PARAMS)

# Sanity check: parameter name count vs vector length
if len(PARAM_NAMES) != N_PARAMS:
    raise RuntimeError(
        f"PARAM_NAMES length ({len(PARAM_NAMES)}) != SEED_VECTOR length ({N_PARAMS}). "
        "Ensure calibration_pipeline.param_names matches the seed vector ordering."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Cost function mirrors your pipeline's χ² definition

def chi2_from_vector(vec: np.ndarray) -> float:
    total = 0.0
    for nuc, props in GOLDEN_SET.items():
        try:
            E_pred = Solve_Nucleus({k: float(v) for k, v in zip(PARAM_NAMES, vec)}, props)
        except Exception:
            return 1e12  # failure penalty
        if not np.isfinite(E_pred) or abs(E_pred) > 1e6:
            return 1e12
        E_exp = props.get("E_exp", np.nan)
        E_err = props.get("E_err", np.nan)
        if not np.isfinite(E_exp):
            return 1e12
        # Robust σ floor to avoid divide-by-zero / huge weights
        sigma = float(max(1e-9, abs(E_err))) if np.isfinite(E_err) else 1e-9
        resid = (E_pred - E_exp) / sigma
        if not np.isfinite(resid):
            return 1e12
        total += float(resid * resid)
    if not np.isfinite(total):
        return 1e12
    return total

# Top-level objective so SciPy multiprocessing can pickle it
def _de_objective(v: np.ndarray) -> float:
    return chi2_from_vector(v)

# ──────────────────────────────────────────────────────────────────────────────
# Small utilities

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    s = json.dumps(obj, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        f.write(s)
    os.replace(tmpp, path)

# ──────────────────────────────────────────────────────────────────────────────
# Optimization

def local_bounds_around_seed(seed: np.ndarray, window: float) -> list[tuple[float, float]]:
    bounds = []
    for v in seed:
        if abs(v) < 1e-12:
            # absolute window when near zero
            w = max(1e-6, window)  # ensure positive window
            lo, hi = -w, w
        else:
            lo, hi = v * (1.0 - window), v * (1.0 + window)
            if lo > hi:
                lo, hi = hi, lo
        bounds.append((float(lo), float(hi)))
    return bounds


def run_bootstrap(window: float = 0.20, do_de: bool = True, de_iters: int = 60,
                  popsize: int = 10, seed: int = 42, de_workers: int | None = None) -> Dict[str, Any]:
    """
    window: fraction around the seed for bounds (e.g., 0.2 => ±20%).
    do_de: run a short Differential Evolution before L-BFGS-B polish.
    de_iters: max iterations for the DE stage.
    popsize: DE population size (per SciPy definition).
    seed: RNG seed for reproducibility.
    """
    from scipy.optimize import differential_evolution, minimize

    rng = np.random.default_rng(seed)
    bounds = local_bounds_around_seed(SEED_VECTOR, window)
    if de_workers is None:
        de_workers = (1 if IS_WSL else -1)

    # Stage A: optional short DE around seed, initializing population near seed
    x0 = SEED_VECTOR.copy()
    x_de = x0.copy()
    chi2_de = chi2_from_vector(x_de)

    if do_de:
        # Build a tight initial population around the seed
        span = np.array([ub - lb for (lb, ub) in bounds])
        pop_n = popsize * len(x0)       # SciPy expects popsize * dim
        init_pop = rng.normal(loc=x0, scale=span / 12.0, size=(pop_n, len(x0)))
        # clip to bounds
        for i, (lb, ub) in enumerate(bounds):
            init_pop[:, i] = np.clip(init_pop[:, i], lb, ub)

        # Early-stopping if no improvement for N iterations
        improve = {"best": np.inf, "stall": 0}
        def de_callback(xk, convergence):
            val = _de_objective(xk)
            if val + 1e-9 < improve["best"]:
                improve["best"] = val
                improve["stall"] = 0
            else:
                improve["stall"] += 1
            # stop if no improvement for 12 successive iters
            return improve["stall"] >= 12

        res_de = differential_evolution(
            _de_objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=de_iters,
            popsize=popsize,
            seed=seed,
            workers=de_workers,
            updating=("immediate" if de_workers == 1 else "deferred"),
            tol=0.0,
            polish=False,
            init=init_pop,
            callback=de_callback,
        )
        x_de = res_de.x
        chi2_de = float(res_de.fun)
    
    # Stage B: L-BFGS-B polish from the best so far
    # tiny jitter helps escape flat/degenerate corners
    x_start = x_de + rng.normal(0.0, 1e-8, size=x_de.shape)
    res_lbfgs = minimize(
        _de_objective,
        x0=x_start,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=2000, ftol=1e-12, gtol=1e-08)
    )

    x_best = res_lbfgs.x
    chi2_best = float(res_lbfgs.fun)
    return {
        "seed_vector": SEED_VECTOR.tolist(),
        "window": window,
        "did_de": bool(do_de),
        "de_iters": int(de_iters),
        "popsize": int(popsize),
        "seed": int(seed),
        "chi2_after_de": float(chi2_de),
        "chi2_after_lbfgs": float(chi2_best),
        "reduced_chi2": float(chi2_best / DOF),
        "best_vector": x_best.tolist(),
        "decoded": {k: float(v) for k, v in zip(PARAM_NAMES, x_best)},
        "n_data": int(N_DATA),
        "n_params": int(N_PARAMS),
        "dof": int(DOF),
        "success_lbfgs": bool(res_lbfgs.success),
        "message_lbfgs": str(res_lbfgs.message),
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrapped QFD solver refinement")
    parser.add_argument("--window", type=float, default=0.20, help="±fraction bounds around seed (default 0.20)")
    parser.add_argument("--no-de", action="store_true", help="Skip the short DE stage and only run L-BFGS-B")
    parser.add_argument("--de-iters", type=int, default=60, help="Max iterations for DE stage (default 60)")
    parser.add_argument("--popsize", type=int, default=10, help="DE popsize (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--workers", type=int, default=None, help="DE workers (-1=all cores; default: 1 on WSL, else -1)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Override output directory (default: workflows/runs_bootstrap)")

    args = parser.parse_args()

    print("--- Bootstrapped QFD Calibration (refinement) ---")
    print(f"Start: {now()}  |  window=±{args.window:.2%}  |  do_de={not args.no_de}")

    out = run_bootstrap(window=args.window, do_de=not args.no_de,
                        de_iters=args.de_iters, popsize=args.popsize,
                        seed=args.seed, de_workers=args.workers)

    # Output directory next to this script
    OUT_DIR = Path(args.outdir) if args.outdir else (Path(__file__).resolve().parent / "runs_bootstrap")
    OUT_DIR.mkdir(exist_ok=True)

    # Timestamped JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"bootstrap_result_{ts}.json"
    atomic_write_json(out_path, out)

    # Snapshot of best decoded couplings
    best_path = OUT_DIR / "Best_QFD_Couplings_bootstrap.json"
    atomic_write_json(best_path, {
        "ts": now(),
        "reduced_chi2": out["reduced_chi2"],
        "decoded": out["decoded"],
    })

    print(f"\nReduced χ² (ν={DOF}): {out['reduced_chi2']:.6g}")
    print(f"Saved results → {out_path}")
    print(f"Best couplings → {best_path}")
