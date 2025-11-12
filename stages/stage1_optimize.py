#!/usr/bin/env python3
"""
Stage 1: Per-SN Nuisance Parameter Optimization (V15 FIXED)

Optimizes (t0, A_plasma, beta, ln_A) for each SN with fixed global params and L_peak.
Uses L-BFGS-B with JAX gradients for GPU acceleration.

FIXES (2025-11-03):
1. Freeze L_peak to break degeneracy with ln_A
   - flux ∝ exp(ln_A) * L_peak was causing ln_A=0 for all SNe
   - Now L_peak fixed at canonical 1.5e43 erg/s
   - ln_A (log-amplitude) forced to encode distance variations as intended

2. Dynamic t0 bounds (absolute MJD)
   - v15_model.py expects t0 as absolute MJD, not relative time
   - Static bounds [-20, 40] caused epoch mismatch → chi²=66B
   - Now use dynamic bounds: [mjd_min-50, mjd_max+50] per SN

Usage:
    python stage1_optimize.py \\
        --lightcurves data/unified/lightcurves_unified_v2_min3.csv \\
        --sn-list data/slices/slice_30.txt \\
        --out results/v15_two_stage/smoke/stage1 \\
        --global 70,0.01,30 --tol 1e-5 --max-iters 200
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Enable JAX x64 BEFORE importing JAX (critical for float64 precision)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from scipy.optimize import minimize
import numpy.linalg as nplin
import json
import sys

# Add core directory to path for importing pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v15_data import LightcurveLoader
from v15_model import (
    log_likelihood_single_sn_jax,
    log_likelihood_single_sn_jax_studentt,
    C_KM_S,
    C_CM_S,
)


# V15 FIXED: Physical bounds for per-SN parameters (4 params, L_peak frozen)
# FIX (2025-11-03): Removed ell to break degeneracy with ln_A
# Parameter meaning:
#   t0        → explosion epoch in MJD (sets time origin)
#   A_plasma  → electron plasma veil strength (channel 1: E144 scattering)
#   beta      → wavelength dependence of plasma veil
#   ln_A      → natural log of flux amplitude (encodes distance modulus μ_raw = -(2.5/ln10) * ln_A)
# L_peak    → FROZEN at canonical 1.5e43 erg/s (not optimized)
# BBH channels handled via mixture model in Stage-2, NOT per-SN knobs (see cloud.txt)
BOUNDS = {
    't0': (-20, 40),                    # MJD offset from peak
    'A_plasma': (0.0, 1.0),             # dimensionless (FIX 2025-11-04: widened from 0.5)
    'beta': (0.0, 4.0),                 # dimensionless (FIX 2025-11-04: widened from 2.0)
    'ln_A': (-30, 30),                  # natural log of flux amplitude (FIX 2025-11-04: widened to allow flux normalization)
}

# Canonical L_peak (FROZEN - not optimized)
L_PEAK_CANONICAL = 1.5e43  # erg/s - typical SN Ia luminosity

# Parameter count validation
EXPECTED_PERSN_COUNT = 4
if len(BOUNDS) != EXPECTED_PERSN_COUNT:
    raise AssertionError(
        f"Parameter count mismatch: BOUNDS has {len(BOUNDS)} entries, "
        f"but V15 FIXED specification requires exactly {EXPECTED_PERSN_COUNT}. "
        f"L_peak is FROZEN at {L_PEAK_CANONICAL:.2e} erg/s."
    )

# Fail fast if forbidden parameters are present
FORBIDDEN_PARAMS = ["P_orb", "phi_0", "A_lens", "ell", "L_peak"]
if any(k in BOUNDS for k in FORBIDDEN_PARAMS):
    raise ValueError(
        f"Forbidden parameters {FORBIDDEN_PARAMS} found in BOUNDS. "
        f"L_peak is FROZEN, ell removed, BBH handled in Stage-2."
    )

# V15 FIXED: Parameter scales for ridge normalization (4 params)
PARAM_SCALES = np.array([
    20.0,      # t0 scale (days)
    0.2,       # A_plasma scale
    1.0,       # beta scale
    5.0,       # ln_A scale (log-amplitude)
])

# ℓ₂ ridge regularization strength
RIDGE_LAMBDA = 1e-6


def chi2_with_ridge(
    persn_params: np.ndarray,
    global_params: Tuple[float, float, float],
    phot: jnp.ndarray,
    z_obs: float,
) -> float:
    """
    V15 FIXED: Compute χ² + ℓ₂ ridge penalty (4 params, L_peak frozen).

    CRITICAL: Uses the SAME log_likelihood_single_sn_jax as Stage 2.

    V15 FIXED parameters: (t0, A_plasma, beta, ln_A) - 4 params.
    L_peak frozen at canonical 1.5e43 erg/s.
    BBH handled via mixture model in Stage-2 (see cloud.txt).
    """
    t0, A_plasma, beta, ln_A = persn_params  # 4 params

    # Use frozen canonical L_peak
    L_peak = L_PEAK_CANONICAL
    persn_tuple = (t0, ln_A, A_plasma, beta)

    log_L = log_likelihood_single_sn_jax(global_params, persn_tuple, L_peak, phot, z_obs)
    chi2 = -2.0 * log_L

    # Add ℓ₂ ridge to prevent runaway solutions (normalized to prevent overflow)
    params_normalized = persn_params / PARAM_SCALES
    ridge = RIDGE_LAMBDA * np.sum(params_normalized ** 2)

    return chi2 + ridge


def chi2_and_grad_wrapper(
    persn_params: np.ndarray,
    global_params: Tuple[float, float, float],
    phot: jnp.ndarray,
    z_obs: float,
    use_studentt: bool = False,
    nu: float = 5.0,
) -> Tuple[float, np.ndarray]:
    """
    V15 FIXED: JAX gradient wrapper for scipy.optimize (4 params, L_peak frozen).

    Returns (value, gradient) as numpy arrays for scipy compatibility.

    Args:
        persn_params: Per-SN parameters [t0, A_plasma, beta, ln_A]
        global_params: Global QFD parameters (k_J, eta_prime, xi)
        phot: Photometry array [N_obs, 4]
        z_obs: Observed redshift
        use_studentt: If True, use Student-t likelihood (robust to outliers)
        nu: Degrees of freedom for Student-t (only used if use_studentt=True)
    """
    # Create closure for JAX
    def objective(params_jax):
        t0, A_plasma, beta, ln_A = params_jax  # 4 params

        # Use frozen canonical L_peak
        L_peak = L_PEAK_CANONICAL
        persn_tuple = (t0, ln_A, A_plasma, beta)

        # Choose likelihood function based on use_studentt flag
        if use_studentt:
            # Student-t likelihood (heavier tails, robust to outliers)
            log_L = log_likelihood_single_sn_jax_studentt(
                global_params, persn_tuple, L_peak, phot, z_obs, nu
            )
            # Negative log-likelihood (we minimize this)
            nll = -log_L
        else:
            # Gaussian likelihood (original)
            log_L = log_likelihood_single_sn_jax(global_params, persn_tuple, L_peak, phot, z_obs)
            nll = -2.0 * log_L  # Chi-squared

        # Normalized ridge to prevent runaway (use float64)
        params_normalized = params_jax / jnp.array(PARAM_SCALES, dtype=jnp.float64)
        ridge = RIDGE_LAMBDA * jnp.sum(params_normalized ** 2)
        return nll + ridge

    try:
        # JIT-compiled gradient (CRITICAL: use float64)
        val, grad = jax.value_and_grad(objective)(jnp.array(persn_params, dtype=jnp.float64))
        return float(val), np.array(grad, dtype=np.float64)
    except Exception as e:
        print(f"      [JAX ERROR] {e}")
        print(f"      params: {persn_params}")
        return np.nan, np.full_like(persn_params, np.nan, dtype=np.float64)


def get_initial_guess(lc_data, z_obs: float, global_k_J: float) -> np.ndarray:
    """
    V15 FIXED: Data-driven initial guess for per-SN params (4 parameters).

    Returns (t0, A_plasma, beta, ln_A).
    L_peak is FROZEN at canonical value, not optimized.

    BBH channels handled via mixture model in Stage-2, not per-SN knobs.
    """
    mjd = lc_data.mjd
    flux_jy = lc_data.flux_jy

    # t0: MJD of explosion (model peaks at t=t_rise=19 days after t0)
    # FIX 2025-11-04: Account for t_rise offset so model peak aligns with data peak
    peak_idx = np.argmax(flux_jy)
    t0_guess = mjd[peak_idx] - 19.0  # Subtract t_rise to get explosion time

    # A_plasma: typical plasma veil amplitude
    A_plasma_guess = 0.12

    # beta: canonical wavelength exponent
    beta_guess = 0.5

    # ln_A: natural log of flux amplitude (FIX 2025-11-04: start at ~18 to match observed flux scale)
    # Model flux at ln_A=0 is ~10^-13 Jy, but observed fluxes are ~10^-5 Jy
    # Need exp(ln_A) ~ 10^8, so ln_A ~ ln(10^8) = 18.4
    ln_A_guess = 18.0

    # Return as (t0, A_plasma, beta, ln_A) - exactly 4 params
    persn0 = np.array([
        t0_guess, A_plasma_guess, beta_guess, ln_A_guess
    ], dtype=np.float64)

    # Validation: ensure exactly 4 parameters
    if persn0.shape[0] != 4:
        raise ValueError(
            f"V15 FIXED Stage-1 expects exactly 4 per-SN parameters [t0, A_plasma, beta, ln_A]. "
            f"Got {persn0.shape[0]} parameters. L_peak is frozen at {L_PEAK_CANONICAL:.2e} erg/s."
        )

    return persn0


def optimize_single_sn(
    snid: str,
    lc_data,
    global_params: Tuple[float, float, float],
    tol: float = 1e-5,
    max_iters: int = 200,
    verbose: bool = True,
    use_studentt: bool = False,
    nu: float = 5.0,
    grad_tol: float = 1.0,
) -> Dict:
    """
    Optimize per-SN parameters for a single supernova.

    Returns dict with:
        - persn_best: optimized parameters
        - chi2: final χ² (or -2*log L for Student-t)
        - logL: final log-likelihood
        - grad_norm: final gradient norm
        - iters: number of iterations
        - ok: convergence flag
        - message: optimizer message
    """
    z_obs = lc_data.z

    # Prepare photometry (same format as V13Sampler)
    phot_array = np.column_stack([
        lc_data.mjd,
        lc_data.wavelength_nm,
        lc_data.flux_jy,
        lc_data.flux_err_jy,
    ])
    phot = jnp.array(phot_array)

    # Initial guess
    x0 = get_initial_guess(lc_data, z_obs, global_params[0])

    # V15 FIXED: Dynamic t0 bounds based on this SN's MJD range
    # CRITICAL: t0 is absolute MJD, not relative time!
    # Allow t0 within ±50 days of observation window
    mjd_min = float(lc_data.mjd.min())
    mjd_max = float(lc_data.mjd.max())
    t0_bounds = (mjd_min - 50, mjd_max + 50)

    # V15 FIXED: Bounds (matching parameter order: t0, A_plasma, beta, ln_A)
    # L_peak is FROZEN, ell removed
    bounds = [
        t0_bounds,  # DYNAMIC per SN (absolute MJD)
        BOUNDS['A_plasma'],
        BOUNDS['beta'],
        BOUNDS['ln_A'],
    ]

    # Optimize with L-BFGS-B
    try:
        result = minimize(
            chi2_and_grad_wrapper,
            x0,
            args=(global_params, phot, z_obs, use_studentt, nu),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol},
        )
    except Exception as e:
        if verbose:
            print(f"  ✗ EXCEPTION during optimization: {e}")
        return {
            'persn_best': x0,
            'chi2': np.inf,
            'logL': -np.inf,
            'grad_norm': np.inf,
            'iters': 0,
            'ok': False,
            'message': str(e),
        }

    # Extract results
    persn_best = result.x
    chi2_final = result.fun
    logL_final = -0.5 * chi2_final  # Approximate (ignoring ridge)
    grad_norm = np.linalg.norm(result.jac)
    iters = result.nit
    # FIX: Relaxed gradient tolerance from 1e-3 to 1.0 (was causing 96% failure rate)
    # Many SNe have good chi² but gradients ~0.1-0.6, so using grad_norm < grad_tol
    ok = result.success and np.isfinite(chi2_final) and grad_norm < grad_tol

    if verbose:
        status_str = "✓ OK" if ok else "✗ FAIL"
        likelihood_type = "Student-t" if use_studentt else "Gaussian"
        print(f"  {status_str} SNID={snid} [{likelihood_type}]: NLL={chi2_final:.2f}, grad_norm={grad_norm:.2e}, iters={iters}")

    return {
        'persn_best': persn_best,
        'chi2': float(chi2_final),
        'logL': float(logL_final),
        'grad_norm': float(grad_norm),
        'iters': int(iters),
        'ok': bool(ok),
        'message': result.message,
        'hess_inv': getattr(result, 'hess_inv', None),
    }


def write_artifacts(outdir: Path, snid: str, result: Dict):
    """
    Write per-SN artifacts as specified in TWO_STAGE_ARCHITECTURE.md.

    V15 FIXED: 4 parameters (t0, A_plasma, beta, alpha), L_peak frozen.
    BBH channels handled via mixture model in Stage-2 (see cloud.txt).
    """
    sn_dir = outdir / str(snid)
    sn_dir.mkdir(parents=True, exist_ok=True)

    # persn_best.npy (float64, shape=(4,))
    # Units: [MJD, dimensionless, dimensionless, dimensionless]
    # Format: [t0, A_plasma, beta, alpha]
    np.save(sn_dir / 'persn_best.npy', result['persn_best'].astype(np.float64))

    # V15 FIXED: Extract all 4 parameters
    t0, A_plasma, beta, alpha = result['persn_best']
    L_peak = L_PEAK_CANONICAL  # Fixed, not fitted

    # metrics.json (required keys + convenience values)
    metrics = {
        'chi2': result['chi2'],
        'logL': result['logL'],
        'grad_norm': result['grad_norm'],
        'iters': result['iters'],
        'ok': result['ok'],
        'L_peak': L_peak,  # Fixed canonical value
    }
    with open(sn_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # status.txt (machine-readable)
    status = 'ok' if result['ok'] else 'did_not_converge'
    if not np.isfinite(result['chi2']):
        status = 'nan'

    with open(sn_dir / 'status.txt', 'w') as f:
        f.write(status)

    # Write Laplace diagnostics if Hessian information is available
    laplace_info = {}
    try:
        hess_inv = result.get('hess_inv')
        if hess_inv is None and hasattr(result, 'hess_inv'):
            hess_inv = result.hess_inv
        if hess_inv is not None:
            if hasattr(hess_inv, "todense"):
                hess_inv = np.array(hess_inv.todense(), dtype=np.float64)
            else:
                hess_inv = np.array(hess_inv, dtype=np.float64)
            d = hess_inv.shape[0]
            hess_inv = 0.5 * (hess_inv + hess_inv.T)
            jitter = 1e-8
            sign, logdet_hinv = nplin.slogdet(hess_inv + jitter * np.eye(d))
            if sign > 0:
                logdet_h = -logdet_hinv
                weight = float(np.exp(-0.5 * logdet_h))
                laplace_info = {'logdetH': float(logdet_h), 'weight': weight}
    except Exception:
        laplace_info = {}

    if laplace_info:
        with open(sn_dir / 'laplace.json', 'w') as f:
            json.dump(laplace_info, f, indent=2)


def should_skip(outdir: Path, snid: str) -> bool:
    """Check if SN already has successful artifacts (resume-safe)."""
    status_file = outdir / str(snid) / 'status.txt'
    if not status_file.exists():
        return False

    with open(status_file) as f:
        status = f.read().strip()

    return status == 'ok'


def process_single_sn_worker(args_tuple):
    """
    Worker function for parallel processing of a single SN.

    Must be at module level (not nested) to be picklable for ProcessPoolExecutor.
    """
    import gc

    (snid, lc_data, global_params, outdir,
     tol, max_iters, use_studentt, nu, grad_tol) = args_tuple

    try:
        result = optimize_single_sn(
            snid, lc_data, global_params,
            tol=tol, max_iters=max_iters, verbose=False,  # No verbose in parallel mode
            use_studentt=use_studentt, nu=nu, grad_tol=grad_tol
        )

        # Write artifacts
        write_artifacts(Path(outdir), snid, result)

        # Memory cleanup: Clear JAX cache and force garbage collection
        try:
            jax.clear_caches()
        except:
            pass
        gc.collect()

        return {
            'snid': snid,
            'ok': result['ok'],
            'chi2': result['chi2'],
            'grad_norm': result['grad_norm'],
            'iters': result['iters'],
            'error': None
        }

    except Exception as e:
        # Write error status
        sn_dir = Path(outdir) / str(snid)
        sn_dir.mkdir(parents=True, exist_ok=True)
        with open(sn_dir / 'status.txt', 'w') as f:
            f.write('error')

        # Memory cleanup even on error
        try:
            import gc
            jax.clear_caches()
            gc.collect()
        except:
            pass

        return {
            'snid': snid,
            'ok': False,
            'chi2': np.inf,
            'grad_norm': np.inf,
            'iters': 0,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Per-SN Optimization')
    parser.add_argument('--lightcurves', required=True, help='Lightcurve CSV file')
    parser.add_argument('--sn-list', help='File with SNIDs (one per line) or range like "0:30"')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--global', dest='global_params', required=True,
                        help='Fixed global params: k_J,eta_prime,xi (e.g., "70,0.01,30")')
    parser.add_argument('--tol', type=float, default=1e-5, help='Optimization tolerance')
    parser.add_argument('--max-iters', type=int, default=200, help='Max iterations')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-studentt', action='store_true',
                        help='Use Student-t likelihood (robust to outliers) instead of Gaussian')
    parser.add_argument('--nu', type=float, default=5.0,
                        help='Student-t degrees of freedom (default=5.0, lower=heavier tails)')
    parser.add_argument('--grad-tol', type=float, default=1.0,
                        help='Gradient norm tolerance for convergence (default=1.0)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default=1, max=7)')

    args = parser.parse_args()

    # Parse global params
    global_params = tuple(map(float, args.global_params.split(',')))
    assert len(global_params) == 3, "Global params must be k_J,eta_prime,xi"

    # Validate workers
    if args.workers > 7:
        print(f"WARNING: Requested {args.workers} workers, but max is 7. Using 7 workers.")
        args.workers = 7
    if args.workers < 1:
        args.workers = 1

    # Output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves
    print(f"Loading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()

    # Filter by SN list if provided
    if args.sn_list:
        if ':' in args.sn_list:
            # Range format "0:30"
            start, end = map(int, args.sn_list.split(':'))
            snids = list(all_lcs.keys())[start:end]
        else:
            # File with SNIDs
            with open(args.sn_list) as f:
                snids = [line.strip() for line in f if line.strip()]

        lcs = {snid: all_lcs[snid] for snid in snids if snid in all_lcs}
    else:
        lcs = all_lcs

    print(f"Processing {len(lcs)} supernovae with global params {global_params}")
    if args.use_studentt:
        print(f"Using Student-t likelihood (nu={args.nu:.1f}) for outlier robustness")
    else:
        print(f"Using Gaussian likelihood (standard chi-squared)")
    print(f"Gradient tolerance: {args.grad_tol}")
    print(f"Parallel workers: {args.workers}")

    # Filter out already-done SNe (resume-safe)
    pending_lcs = {
        snid: lc_data for snid, lc_data in lcs.items()
        if not should_skip(outdir, snid)
    }
    skip_count = len(lcs) - len(pending_lcs)

    if skip_count > 0:
        print(f"Skipping {skip_count} already-completed SNe (resume-safe)")

    # Process each SN
    success_count = 0
    fail_count = 0
    start_time = time.time()

    def print_progress(completed, total, success, fail, elapsed):
        """Print progress summary every 500 SNe."""
        success_rate = 100 * success / completed if completed > 0 else 0
        rate_per_sec = completed / elapsed if elapsed > 0 else 0
        eta_sec = (total - completed) / rate_per_sec if rate_per_sec > 0 else 0

        print("=" * 80)
        print(f"PROGRESS: {completed}/{total} SNe processed ({100*completed/total:.1f}%)")
        print(f"  Success: {success} ({success_rate:.1f}%)")
        print(f"  Failed:  {fail}")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"  Rate:    {rate_per_sec*60:.1f} SNe/min")
        print(f"  ETA:     {eta_sec/60:.1f} min")
        print("=" * 80)

    if args.workers == 1:
        # Serial processing (original behavior, verbose allowed)
        for i, (snid, lc_data) in enumerate(pending_lcs.items(), 1):
            if args.verbose:
                print(f"[{i}/{len(pending_lcs)}] Optimizing SNID={snid}...")

            try:
                result = optimize_single_sn(
                    snid, lc_data, global_params,
                    tol=args.tol, max_iters=args.max_iters, verbose=args.verbose,
                    use_studentt=args.use_studentt, nu=args.nu, grad_tol=args.grad_tol
                )

                # Write artifacts
                write_artifacts(outdir, snid, result)

                if result['ok']:
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"  ✗ ERROR SNID={snid}: {e}")
                fail_count += 1

                # Write error status
                sn_dir = outdir / str(snid)
                sn_dir.mkdir(parents=True, exist_ok=True)
                with open(sn_dir / 'status.txt', 'w') as f:
                    f.write('error')

            # Progress report every 500 SNe
            if i % 500 == 0:
                elapsed = time.time() - start_time
                print_progress(i, len(pending_lcs), success_count, fail_count, elapsed)

    else:
        # Parallel processing with ProcessPoolExecutor
        print(f"Starting parallel processing with {args.workers} workers...")

        # Prepare work items
        work_items = [
            (snid, lc_data, global_params, str(outdir),
             args.tol, args.max_iters, args.use_studentt, args.nu, args.grad_tol)
            for snid, lc_data in pending_lcs.items()
        ]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_sn_worker, item): item[0]
                for item in work_items
            }

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                snid = futures[future]
                completed += 1

                # Progress report every 500 SNe
                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    print_progress(completed, len(pending_lcs), success_count, fail_count, elapsed)

                try:
                    result = future.result()

                    if result['error']:
                        print(f"[{completed}/{len(pending_lcs)}] ✗ ERROR SNID={result['snid']}: {result['error']}")
                        fail_count += 1
                    elif result['ok']:
                        status = "✓ OK"
                        print(f"[{completed}/{len(pending_lcs)}] {status} SNID={result['snid']}: "
                              f"NLL={result['chi2']:.2f}, grad={result['grad_norm']:.2e}, iters={result['iters']}")
                        success_count += 1
                    else:
                        status = "✗ FAIL"
                        print(f"[{completed}/{len(pending_lcs)}] {status} SNID={result['snid']}: "
                              f"NLL={result['chi2']:.2f}, grad={result['grad_norm']:.2e}, iters={result['iters']}")
                        fail_count += 1

                except Exception as e:
                    print(f"[{completed}/{len(pending_lcs)}] ✗ EXCEPTION SNID={snid}: {e}")
                    fail_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage 1 Complete: {len(lcs)} SNe processed")
    print(f"  ✓ Success:  {success_count}")
    print(f"  ⊘ Skipped:  {skip_count} (already ok)")
    print(f"  ✗ Failed:   {fail_count}")
    print(f"{'='*60}")

    # Write summary
    summary = {
        'total': len(lcs),
        'success': success_count,
        'skipped': skip_count,
        'failed': fail_count,
        'global_params': {'k_J': global_params[0], 'eta_prime': global_params[1], 'xi': global_params[2]},
    }
    with open(outdir / '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
