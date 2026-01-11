#!/usr/bin/env python3
"""
QFD Parallel Sweep: Optimized Helicity-locked E ∝ ω derivation

This is an optimized parallel version of derive_hbar_from_topology.py that:
1. Prevents CPU thrashing by limiting threads per worker
2. Reduces default N to 64 (8x faster than 128, sufficient for topology)
3. Regenerates grids in workers to avoid pickle overhead
4. Adds timeouts to prevent hangs

Based on optimization analysis:
- N³ scaling: 64³ is 8x faster than 128³ and usually sufficient
- Pickle trap: Pass settings, not arrays
- OpenMP thrashing: Force OMP_NUM_THREADS=1 per worker
"""

import os

# --- CRITICAL OPTIMIZATION 1: PREVENT CPU THRASHING ---
# NumPy uses all cores by default. When multiprocessing, we must restrict
# individual processes to 1 core so they don't fight each other.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import time
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Import from main module
from derive_hbar_from_topology import (
    make_grid, build_vector_potential, build_abc_flow,
    curl, divergence, integrate_scalar, compute_stats,
    FieldStats
)


def process_scale_worker(scale: float, args_dict: dict) -> Tuple:
    """
    Worker function. Regenerates grid locally to avoid pickle overhead.
    """
    try:
        N = args_dict['N']
        L = args_dict['L']
        R0 = args_dict['R0']
        a0 = args_dict['a0']
        twist = args_dict['twist']
        envelope = args_dict['envelope']
        c = args_dict['c']
        H_target = args_dict['Htarget']
        use_abc = args_dict.get('abc', False)
        abc_kappa = args_dict.get('abc_kappa', 1.0)

        if use_abc:
            # Use ABC flow (exact Beltrami eigenfield)
            X, Y, Z, Bx, By, Bz, dx = build_abc_flow(
                N, L=np.pi, A_coef=1.0, B_coef=1.0, C_coef=1.0, kappa=abc_kappa * scale
            )
            # For ABC flow, B is already the field (no curl needed)
            # Create pseudo A for helicity calculation
            Ax, Ay, Az = Bx / abc_kappa, By / abc_kappa, Bz / abc_kappa
        else:
            # Reconstruct Grid Locally (avoids pickle overhead)
            X, Y, Z, dx = make_grid(N, L)

            R = R0 * scale
            a = a0 * scale

            # Build Initial Potential
            Ax, Ay, Az = build_vector_potential(
                X, Y, Z, R0=R, a=a, amp=1.0, twist=-twist, envelope=envelope
            )
            Bx, By, Bz = curl(Ax, Ay, Az, dx)

        # Compute statistics
        st = compute_stats(Ax, Ay, Az, Bx, By, Bz, dx, c)

        # Derived Constants
        if use_abc:
            k_geom = abc_kappa * scale
        else:
            a = a0 * scale
            k_geom = 1.0 / max(a, 1e-300)

        omega_geom = c * k_geom
        hbar_geom = st.E / max(omega_geom, 1e-300)

        return (scale, st.E, st.H, st.k_eff, st.hbar_eff, st.beltrami_resid, k_geom, hbar_geom)

    except Exception as e:
        import traceback
        return (scale, f"ERROR: {e}\n{traceback.format_exc()}")


def run_parallel_sweep(args):
    """Run parameter sweep in parallel with optimized settings."""
    # Convert args to dict for pickle-safe passing
    args_dict = {
        'N': args.N,
        'L': args.L,
        'R0': args.R0,
        'a0': args.a0,
        'twist': args.twist,
        'envelope': args.envelope,
        'c': args.c,
        'Htarget': args.Htarget,
        'abc': args.abc,
        'abc_kappa': args.abc_kappa,
    }

    print("=" * 70)
    print("QFD PARALLEL SWEEP: Optimized Helicity Derivation")
    print("=" * 70)
    print(f"\nGrid: {args.N}³ ({args.N**3:,} points)")
    print(f"Method: {'ABC Flow (exact Beltrami)' if args.abc else 'Toroidal Ansatz'}")
    print(f"Workers: {args.workers} (OMP_NUM_THREADS=1 per worker)")
    print(f"Scales: {args.scales}")

    start_time = time.time()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_scale_worker, s, args_dict): s
            for s in args.scales
        }

        for future in concurrent.futures.as_completed(future_map):
            scale_val = future_map[future]
            try:
                res = future.result(timeout=args.timeout)

                if isinstance(res[1], str) and res[1].startswith("ERROR"):
                    print(f"  [FAIL] scale={scale_val}: {res[1][:80]}...")
                else:
                    results.append(res)
                    print(f"  [OK] scale={scale_val:.2f}: E={res[1]:.4e}, ħ_eff={res[4]:.4f}, beltrami={res[5]:.4f}")

            except concurrent.futures.TimeoutError:
                print(f"  [TIMEOUT] scale={scale_val} (>{args.timeout}s)")
            except Exception as e:
                print(f"  [ERROR] scale={scale_val}: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")

    # Sort and analyze results
    results.sort(key=lambda x: x[0])

    if results:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"{'Scale':>8} {'Energy':>12} {'Helicity':>12} {'k_eff':>10} {'ħ_eff':>10} {'Beltrami':>10}")
        print("-" * 70)

        hbar_values = []
        for r in results:
            scale, E, H, k_eff, hbar_eff, beltrami, k_geom, hbar_geom = r
            print(f"{scale:>8.2f} {E:>12.4e} {H:>12.4e} {k_eff:>10.4f} {hbar_eff:>10.4f} {beltrami:>10.4f}")
            hbar_values.append(hbar_eff)

        # Statistics
        if len(hbar_values) > 1:
            mean_hbar = np.mean(hbar_values)
            std_hbar = np.std(hbar_values)
            cv = std_hbar / mean_hbar * 100 if mean_hbar > 0 else 0

            print("-" * 70)
            print(f"ħ_eff: mean={mean_hbar:.4f}, std={std_hbar:.4f}, CV={cv:.2f}%")

            if cv < 5:
                print("✓ EXCELLENT: ħ_eff is scale-invariant (CV < 5%)")
            elif cv < 15:
                print("⚠ MODERATE: ħ_eff shows some scale dependence")
            else:
                print("✗ WEAK: ħ_eff varies significantly with scale")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Optimized parallel sweep for QFD helicity derivation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast test with ABC flow (exact Beltrami)
  python derive_hbar_parallel.py --abc --N 32

  # Full sweep with toroidal ansatz
  python derive_hbar_parallel.py --N 64 --scales 0.5 0.75 1.0 1.25 1.5 2.0

  # High resolution (slower)
  python derive_hbar_parallel.py --N 128 --workers 4
        """
    )

    # Grid settings (default N=64 for speed)
    ap.add_argument("--N", type=int, default=64,
                    help="grid points per axis (default 64, use 128 for high-res)")
    ap.add_argument("--L", type=float, default=8.0,
                    help="half-width of cubic domain")

    # Toroidal ansatz params
    ap.add_argument("--R0", type=float, default=1.6, help="base major radius")
    ap.add_argument("--a0", type=float, default=0.35, help="base minor radius")
    ap.add_argument("--twist", type=float, default=0.6, help="poloidal twist")
    ap.add_argument("--envelope", type=str, default="bump",
                    choices=["gaussian", "bump"], help="tube envelope profile")

    # Physics
    ap.add_argument("--c", type=float, default=1.0, help="wave speed")
    ap.add_argument("--Htarget", type=float, default=1.0, help="target helicity")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.8, 1.0, 1.25, 1.5, 2.0], help="scale factors")

    # ABC flow (exact Beltrami)
    ap.add_argument("--abc", action="store_true",
                    help="use ABC flow (exact Beltrami eigenfield)")
    ap.add_argument("--abc_kappa", type=float, default=1.0,
                    help="ABC flow eigenvalue κ")

    # Parallel settings
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8),
                    help="number of parallel workers")
    ap.add_argument("--timeout", type=int, default=600,
                    help="timeout per task in seconds")

    args = ap.parse_args()
    results = run_parallel_sweep(args)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
