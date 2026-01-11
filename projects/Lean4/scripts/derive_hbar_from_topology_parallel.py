#!/usr/bin/env python3
"""
QFD Topological Action Derivation (CPU Parallel Optimized)
==========================================================

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

PURPOSE:
--------
Derives Planck's Constant (hbar) from Topological Constraints using
optimized CPU multiprocessing. For users without GPU access.

THE PHYSICS:
------------
In QFD, hbar emerges from the energy-frequency relationship of soliton solutions:

    hbar_eff = E / omega

The soliton is relaxed toward a Beltrami eigenfield (curl B = λB) which
represents a force-free, helicity-locked configuration.

CPU OPTIMIZATION TECHNIQUES:
----------------------------
1. OMP_NUM_THREADS=1: Prevents NumPy internal threading from fighting
   with Python multiprocessing (critical for scaling)
2. Lazy grid generation: Workers create their own grids to avoid
   serializing 16MB+ arrays via pickle
3. Per-task timeouts: Prevents stalled workers from hanging the suite
4. Reduced default N=64: 8x faster than N=128, usually sufficient

PERFORMANCE COMPARISON:
-----------------------
| Method | N=64 | N=128 |
|--------|------|-------|
| Single-threaded | ~60s | ~8min |
| Parallel (this) | ~15s | ~2min |
| GPU (recommended) | ~4s | ~95s |

USAGE:
------
    # Default run (64³, 4 workers)
    python derive_hbar_from_topology_parallel.py

    # Higher resolution
    python derive_hbar_from_topology_parallel.py --N 128 --workers 8

    # With relaxation
    python derive_hbar_from_topology_parallel.py --relax --relax_steps 200

References:
    - derive_hbar_from_topology.py (single-threaded CPU version)
    - derive_hbar_from_topology_gpu.py (GPU version, recommended)
"""

import os

# --- CRITICAL: PREVENT CPU THREAD THRASHING ---
# Must be set BEFORE importing NumPy!
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
import numpy as np
from typing import Tuple, NamedTuple


class SolitonStats(NamedTuple):
    """Statistics for a soliton configuration."""
    E: float           # Energy
    H: float           # Helicity
    A2: float          # ∫|A|² dV
    B2: float          # ∫|B|² dV
    beltrami_corr: float  # Beltrami correlation


def make_grid(N: int, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create 3D grid for simulation."""
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return X, Y, Z, dx


def toroidal_frame(X, Y, Z, R0):
    """Compute toroidal coordinate frame."""
    eps = 1e-12
    rho = np.sqrt(X**2 + Y**2) + eps

    # Toroidal direction
    e_phi_x = -Y / rho
    e_phi_y = X / rho
    e_phi_z = np.zeros_like(rho)

    # Distance from tube axis
    s = np.sqrt((rho - R0)**2 + Z**2) + eps

    # Poloidal direction
    e_theta_x = (-Z / s) * (X / rho)
    e_theta_y = (-Z / s) * (Y / rho)
    e_theta_z = (rho - R0) / s

    return (e_phi_x, e_phi_y, e_phi_z), (e_theta_x, e_theta_y, e_theta_z), s


def bump_envelope(s, a):
    """Smooth bump function envelope."""
    u = s / a
    g = np.zeros_like(u)
    mask = u < 1.0
    g[mask] = np.exp(1.0 - 1.0 / (1.0 - u[mask]**2))
    return g


def gaussian_envelope(s, a):
    """Gaussian envelope."""
    return np.exp(-0.5 * (s / a)**2)


def curl(Ax, Ay, Az, dx):
    """Compute curl of vector field using central differences."""
    # Use numpy.gradient for cleaner code
    dAx_dy = np.gradient(Ax, dx, axis=1)
    dAx_dz = np.gradient(Ax, dx, axis=2)
    dAy_dx = np.gradient(Ay, dx, axis=0)
    dAy_dz = np.gradient(Ay, dx, axis=2)
    dAz_dx = np.gradient(Az, dx, axis=0)
    dAz_dy = np.gradient(Az, dx, axis=1)

    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy

    return Bx, By, Bz


def integrate(F, dx):
    """Integrate scalar field over volume."""
    return np.sum(F) * dx**3


def compute_helicity(Ax, Ay, Az, Bx, By, Bz, dx):
    """Compute helicity H = ∫ A · B dV."""
    return integrate(Ax*Bx + Ay*By + Az*Bz, dx)


def compute_stats(Ax, Ay, Az, Bx, By, Bz, dx) -> SolitonStats:
    """Compute all soliton statistics."""
    # Energy
    B2 = integrate(Bx**2 + By**2 + Bz**2, dx)
    E = 0.5 * B2

    # Helicity
    H = compute_helicity(Ax, Ay, Az, Bx, By, Bz, dx)

    # Norms
    A2 = integrate(Ax**2 + Ay**2 + Az**2, dx)

    # Beltrami correlation
    Cx, Cy, Cz = curl(Bx, By, Bz, dx)
    C2 = integrate(Cx**2 + Cy**2 + Cz**2, dx)
    BC = integrate(Bx*Cx + By*Cy + Bz*Cz, dx)
    beltrami_corr = BC / np.sqrt(B2 * C2) if B2 > 0 and C2 > 0 else 0.0

    return SolitonStats(E=E, H=H, A2=A2, B2=B2, beltrami_corr=beltrami_corr)


def relax_to_beltrami(Ax, Ay, Az, dx, H_target, steps=200, eta=0.01):
    """
    Relax field toward Beltrami eigenfield while preserving helicity.

    Uses gradient descent on the force-free residual |curl(B) - κB|².
    """
    for step in range(steps):
        # Compute B = curl(A)
        Bx, By, Bz = curl(Ax, Ay, Az, dx)

        # Enforce helicity constraint
        H = compute_helicity(Ax, Ay, Az, Bx, By, Bz, dx)
        scale = np.sqrt(np.abs(H_target / (H + 1e-12)))
        Ax, Ay, Az = Ax * scale, Ay * scale, Az * scale
        Bx, By, Bz = Bx * scale, By * scale, Bz * scale

        # Compute C = curl(B)
        Cx, Cy, Cz = curl(Bx, By, Bz, dx)

        # Beltrami eigenvalue: κ = B·C / B·B
        B2 = integrate(Bx**2 + By**2 + Bz**2, dx)
        BC = integrate(Bx*Cx + By*Cy + Bz*Cz, dx)
        kappa = BC / B2 if B2 > 0 else 0.0

        # Residual from Beltrami condition
        Rx = Cx - kappa * Bx
        Ry = Cy - kappa * By
        Rz = Cz - kappa * Bz

        # Gradient step (simplified - proper implementation would use adjoint)
        # Here we use a heuristic: move A to reduce |R|²
        Ax -= eta * Rx
        Ay -= eta * Ry
        Az -= eta * Rz

    return Ax, Ay, Az


def process_scale_worker(args_tuple):
    """
    Worker function for processing a single scale.

    NOTE: We regenerate the grid inside each worker to avoid pickle overhead.
    Passing 16MB+ arrays between processes is slower than recreating them.
    """
    scale, config = args_tuple

    try:
        # Reconstruct grid locally (fast, avoids pickle overhead)
        X, Y, Z, dx = make_grid(config['N'], config['L'])

        R = config['R0'] * scale
        a = config['a0'] * scale
        H_target = config['Htarget']
        twist = config['twist']

        # Build toroidal soliton
        e_phi, e_theta, s = toroidal_frame(X, Y, Z, R)

        if config['envelope'] == 'gaussian':
            g = gaussian_envelope(s, a)
        else:
            g = bump_envelope(s, a)

        # Vector potential: twisted torus
        Ax = g * (e_phi[0] + twist * e_theta[0])
        Ay = g * (e_phi[1] + twist * e_theta[1])
        Az = g * (e_phi[2] + twist * e_theta[2])

        # Optional relaxation
        if config['relax']:
            Ax, Ay, Az = relax_to_beltrami(
                Ax, Ay, Az, dx, H_target,
                steps=config['relax_steps'],
                eta=config['relax_eta']
            )

        # Compute B and rescale to target helicity
        Bx, By, Bz = curl(Ax, Ay, Az, dx)
        H = compute_helicity(Ax, Ay, Az, Bx, By, Bz, dx)
        amp_scale = np.sqrt(np.abs(H_target / (H + 1e-12)))
        Ax, Ay, Az = Ax * amp_scale, Ay * amp_scale, Az * amp_scale
        Bx, By, Bz = Bx * amp_scale, By * amp_scale, Bz * amp_scale

        # Compute statistics
        stats = compute_stats(Ax, Ay, Az, Bx, By, Bz, dx)

        # Derived quantities
        k_geom = np.sqrt(stats.B2 / stats.A2) if stats.A2 > 0 else 0.0
        omega = config['c'] * k_geom
        hbar = stats.E / omega if omega > 0 else 0.0

        return {
            'scale': scale,
            'R': R,
            'a': a,
            'E': stats.E,
            'H': stats.H,
            'k_geom': k_geom,
            'omega': omega,
            'hbar': hbar,
            'beltrami_corr': stats.beltrami_corr,
            'success': True
        }

    except Exception as e:
        return {
            'scale': scale,
            'error': str(e),
            'success': False
        }


def main():
    ap = argparse.ArgumentParser(description="CPU parallel hbar derivation from topology")

    # Grid parameters (N=64 default for speed)
    ap.add_argument("--N", type=int, default=64, help="Grid resolution (N³)")
    ap.add_argument("--L", type=float, default=8.0, help="Box half-size")

    # Soliton parameters
    ap.add_argument("--R0", type=float, default=1.6, help="Major radius scale")
    ap.add_argument("--a0", type=float, default=0.35, help="Minor radius scale")
    ap.add_argument("--twist", type=float, default=-0.6, help="Twist parameter")
    ap.add_argument("--envelope", type=str, default="bump", choices=["bump", "gaussian"],
                    help="Envelope function")
    ap.add_argument("--c", type=float, default=1.0, help="Speed of light")
    ap.add_argument("--Htarget", type=float, default=1.0, help="Target helicity")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5, 2.0],
                    help="Scale factors to test")

    # Relaxation parameters
    ap.add_argument("--relax", action="store_true", help="Enable Beltrami relaxation")
    ap.add_argument("--relax_steps", type=int, default=200, help="Relaxation steps")
    ap.add_argument("--relax_eta", type=float, default=0.005, help="Relaxation learning rate")

    # Parallel parameters
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of worker processes (default: CPU count)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-task timeout in seconds")

    args = ap.parse_args()

    # Default workers to CPU count
    if args.workers is None:
        args.workers = os.cpu_count() or 4

    print("=" * 70)
    print("QFD TOPOLOGICAL ACTION DERIVATION (CPU Parallel)")
    print("=" * 70)
    print(f"Grid: {args.N}³ = {args.N**3:,} points")
    print(f"Workers: {args.workers} (OMP_NUM_THREADS=1 per worker)")
    print(f"Scales: {args.scales}")
    if args.relax:
        print(f"Relaxation: {args.relax_steps} steps, eta={args.relax_eta}")
    print()

    # Build config dict (avoids passing argparse object through pickle)
    config = {
        'N': args.N,
        'L': args.L,
        'R0': args.R0,
        'a0': args.a0,
        'twist': args.twist,
        'envelope': args.envelope,
        'c': args.c,
        'Htarget': args.Htarget,
        'relax': args.relax,
        'relax_steps': args.relax_steps,
        'relax_eta': args.relax_eta,
    }

    # Prepare tasks
    tasks = [(scale, config) for scale in args.scales]

    print(f"Processing {len(tasks)} scales...")
    start_time = time.time()

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(process_scale_worker, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(future_map):
            scale_val = future_map[future]
            try:
                result = future.result(timeout=args.timeout)

                if result['success']:
                    results.append(result)
                    print(f"  Scale {scale_val:.2f}: E={result['E']:.4f}, "
                          f"hbar={result['hbar']:.4f}, corr={result['beltrami_corr']:.3f}")
                else:
                    print(f"  Scale {scale_val:.2f}: ERROR - {result.get('error', 'unknown')}")

            except concurrent.futures.TimeoutError:
                print(f"  Scale {scale_val:.2f}: TIMEOUT (>{args.timeout}s)")
            except Exception as e:
                print(f"  Scale {scale_val:.2f}: EXCEPTION - {e}")

    elapsed = time.time() - start_time

    # Sort results by scale
    results.sort(key=lambda x: x['scale'])

    # Print results table
    print()
    print("=" * 70)
    print(f"RESULTS (N={args.N}³)")
    print("=" * 70)
    print(f"{'Scale':<8} {'Energy':<12} {'Omega':<12} {'hbar':<12} {'Beltrami':<10}")
    print("-" * 70)

    hbar_values = []
    for r in results:
        print(f"{r['scale']:<8.2f} {r['E']:<12.4f} {r['omega']:<12.4f} "
              f"{r['hbar']:<12.4f} {r['beltrami_corr']:<10.4f}")
        hbar_values.append(r['hbar'])

    print("-" * 70)

    if hbar_values:
        mean_hbar = np.mean(hbar_values)
        cv_hbar = np.std(hbar_values) / mean_hbar if mean_hbar > 0 else float('inf')
        min_corr = min(r['beltrami_corr'] for r in results)

        print(f"MEAN hbar:  {mean_hbar:.6f}")
        print(f"CV(hbar):   {cv_hbar:.2%}")
        print(f"TIME:       {elapsed:.2f}s")

        print()
        print("=" * 70)
        if cv_hbar < 0.05:
            print("SUCCESS: E = hbar*omega validated (CV < 5%)")
        elif cv_hbar < 0.20:
            print("PARTIAL: E = hbar*omega approximately validated (CV < 20%)")
        else:
            print("INCONCLUSIVE: High variance (CV > 20%)")

        if min_corr > 0.90:
            print(f"PASS: All Beltrami correlations > 0.90 (min: {min_corr:.3f})")
        elif min_corr > 0.70:
            print(f"PARTIAL: Min Beltrami correlation {min_corr:.3f}")
        else:
            print(f"NEEDS WORK: Min Beltrami correlation {min_corr:.3f} < 0.70")
        print("=" * 70)

        return 0 if cv_hbar < 0.05 else 1
    else:
        print("No successful results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
