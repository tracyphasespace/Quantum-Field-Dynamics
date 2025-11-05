#!/usr/bin/env python3
"""
Stage 2: Global MCMC Sampling (OPTIMIZED)

JAX-vectorized version that processes all walkers in parallel on GPU.
Expected speedup: 10-20× over CPU-only version.

Usage:
    python stage2_mcmc_optimized.py \
        --stage1-results results/v15_stage1_production \
        --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
        --out results/v15_stage2_mcmc \
        --nwalkers 32 \
        --nsteps 5000 \
        --nburn 1000
"""

import argparse
import json
import numpy as np
from pathlib import Path
import emcee
import h5py
from typing import List, Tuple

# Import model
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from v15_model import log_likelihood_single_sn_jax

def load_stage1_results(stage1_dir, lightcurves_dict):
    """Load all Stage 1 results"""
    results = {}
    failed = []

    for result_dir in Path(stage1_dir).iterdir():
        if not result_dir.is_dir():
            continue

        snid = result_dir.name
        metrics_file = result_dir / "metrics.json"
        persn_file = result_dir / "persn_best.npy"

        if not metrics_file.exists() or not persn_file.exists():
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Load per-SN parameters
            persn_best = np.load(persn_file)

            # Get n_obs from lightcurve
            if snid not in lightcurves_dict:
                failed.append(snid)
                continue

            lc = lightcurves_dict[snid]
            n_obs = len(lc.mjd)

            # Quality filter
            chi2 = metrics['chi2']
            chi2_per_obs = chi2 / n_obs if n_obs > 0 else np.inf

            if chi2_per_obs > 100:
                failed.append(snid)
                continue

            if metrics['iters'] < 5:
                failed.append(snid)
                continue

            # Store result
            result = {
                'snid': snid,
                'chi2': chi2,
                'n_obs': n_obs,
                'persn_best': persn_best,
                'L_peak': metrics['L_peak'],
                'iters': metrics['iters']
            }
            results[snid] = result

        except Exception as e:
            print(f"  Warning: Failed to load {snid}: {e}")
            continue

    print(f"  Loaded {len(results)} good SNe")
    print(f"  Excluded {len(failed)} poor fits")

    return results

def log_prior(theta):
    """Uniform priors on global params"""
    k_J, eta_prime, xi = theta

    if not (50 < k_J < 90):
        return -np.inf
    if not (0.001 < eta_prime < 0.1):
        return -np.inf
    if not (10 < xi < 50):
        return -np.inf

    return 0.0

# JIT-compile vectorized likelihood
@jax.jit
def log_likelihood_batch_jax(
    global_params: jnp.ndarray,  # Shape: (3,)
    persn_params_batch: jnp.ndarray,  # Shape: (n_sne, 4)
    L_peaks: jnp.ndarray,  # Shape: (n_sne,)
    photometries: List,  # List of JAX arrays
    redshifts: jnp.ndarray  # Shape: (n_sne,)
) -> jnp.ndarray:
    """
    Vectorized log-likelihood over multiple SNe.

    Returns: Array of shape (n_sne,) with logL for each SN
    """
    def single_sn_logL(persn_params, L_peak, phot, z_obs):
        return log_likelihood_single_sn_jax(
            tuple(global_params), tuple(persn_params), L_peak, phot, z_obs
        )

    # vmap over SNe
    logLs = vmap(single_sn_logL)(
        persn_params_batch, L_peaks, photometries, redshifts
    )

    return logLs

def log_likelihood_vectorized(
    theta: np.ndarray,
    frozen_persn_params: np.ndarray,
    frozen_L_peaks: np.ndarray,
    photometries: List,
    redshifts: np.ndarray
) -> float:
    """
    Compute log-likelihood for a single walker across all SNe.

    Vectorized using JAX vmap for speed.
    """
    global_params = jnp.array(theta)
    persn_params_batch = jnp.array(frozen_persn_params)
    L_peaks = jnp.array(frozen_L_peaks)
    z_arr = jnp.array(redshifts)

    try:
        # Compute logL for all SNe in parallel
        logLs = log_likelihood_batch_jax(
            global_params, persn_params_batch, L_peaks, photometries, z_arr
        )

        # Sum over SNe
        total_logL = float(jnp.sum(logLs))

        if not np.isfinite(total_logL):
            return -np.inf

        return total_logL

    except Exception:
        return -np.inf

def log_probability(theta, frozen_persn_params, frozen_L_peaks, photometries, redshifts):
    """Log-posterior = log-prior + log-likelihood"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_vectorized(
        theta, frozen_persn_params, frozen_L_peaks, photometries, redshifts
    )
    return lp + ll

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Optimized Global MCMC')
    parser.add_argument('--stage1-results', required=True,
                       help='Directory with Stage 1 results')
    parser.add_argument('--lightcurves', required=True,
                       help='Path to lightcurves CSV')
    parser.add_argument('--out', required=True,
                       help='Output directory')
    parser.add_argument('--nwalkers', type=int, default=32,
                       help='Number of MCMC walkers')
    parser.add_argument('--nsteps', type=int, default=5000,
                       help='Number of MCMC steps')
    parser.add_argument('--nburn', type=int, default=1000,
                       help='Burn-in steps')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: OPTIMIZED GLOBAL MCMC (JAX-VECTORIZED)")
    print("=" * 80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Walkers: {args.nwalkers}, Steps: {args.nsteps}, Burn-in: {args.nburn}")
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves first (needed for loading Stage 1 results)
    print("Loading lightcurves...")
    from v15_data import LightcurveLoader
    loader = LightcurveLoader(args.lightcurves)
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(args.stage1_results, all_lcs)

    if len(stage1_results) < 50:
        print(f"ERROR: Only {len(stage1_results)} good SNe, need at least 50!")
        return 1

    print()

    # Prepare data arrays
    print("Preparing frozen parameters...")
    snids = []
    frozen_persn_params = []
    frozen_L_peaks = []
    photometries = []
    redshifts = []
    L_PEAK_CANONICAL = 1.5e43

    for snid, result in stage1_results.items():
        if snid not in all_lcs:
            continue

        lc = all_lcs[snid]

        # Frozen per-SN params from Stage 1
        persn_best = result['persn_best']
        persn_tuple = (persn_best[0], persn_best[3], persn_best[1], persn_best[2])

        # Photometry as JAX array
        phot = jnp.array(np.column_stack([
            lc.mjd, lc.wavelength_nm, lc.flux_jy, lc.flux_err_jy
        ]))

        snids.append(snid)
        frozen_persn_params.append(persn_tuple)
        frozen_L_peaks.append(L_PEAK_CANONICAL)
        photometries.append(phot)
        redshifts.append(lc.z)

    frozen_persn_params = np.array(frozen_persn_params)
    frozen_L_peaks = np.array(frozen_L_peaks)
    redshifts = np.array(redshifts)

    print(f"  Using {len(snids)} SNe for MCMC")
    print()

    # Compile likelihood function (warm start)
    print("Compiling JAX likelihood function...")
    test_theta = np.array([70.0, 0.01, 30.0])
    _ = log_probability(
        test_theta, frozen_persn_params, frozen_L_peaks, photometries, redshifts
    )
    print("  Compilation complete!")
    print()

    # Initial guess
    initial_theta = np.array([70.0, 0.01, 30.0])

    # Initialize walkers
    ndim = 3
    nwalkers = args.nwalkers
    pos = initial_theta + 1e-3 * np.random.randn(nwalkers, ndim)

    print("Initializing MCMC...")
    print(f"  Initial guess: k_J={initial_theta[0]}, eta'={initial_theta[1]}, xi={initial_theta[2]}")
    print()

    # Backend for saving progress
    backend_file = outdir / "chain.h5"
    backend = emcee.backends.HDFBackend(str(backend_file))
    backend.reset(nwalkers, ndim)

    # Create sampler (no multiprocessing - JAX handles parallelism)
    print("Running MCMC...")
    print(f"  Expected time: ~{args.nsteps * 0.75 / 60:.1f} minutes (JAX-optimized)")
    print()

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(frozen_persn_params, frozen_L_peaks, photometries, redshifts),
        backend=backend
    )

    # Run with progress
    for i, state in enumerate(sampler.sample(pos, iterations=args.nsteps, progress=True)):
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{args.nsteps}")
            print(f"    Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

            # Estimate autocorr time
            if i > 100:
                try:
                    tau = sampler.get_autocorr_time(quiet=True)
                    print(f"    Autocorr time: {np.mean(tau):.1f} steps")
                except:
                    pass

    print()
    print("MCMC complete!")
    print()

    # Analyze chain
    print("Analyzing chain...")
    chain = sampler.get_chain(discard=args.nburn, flat=True)
    log_prob = sampler.get_log_prob(discard=args.nburn, flat=True)

    # Summary statistics
    mean_theta = np.mean(chain, axis=0)
    std_theta = np.std(chain, axis=0)

    print(f"  Acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"  k_J = {mean_theta[0]:.2f} ± {std_theta[0]:.2f}")
    print(f"  eta' = {mean_theta[1]:.4f} ± {std_theta[1]:.4f}")
    print(f"  xi = {mean_theta[2]:.2f} ± {std_theta[2]:.2f}")
    print()

    # Autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        print(f"  Autocorrelation time: {np.mean(tau):.1f} steps")
        print(f"  Effective samples: {len(chain) / np.mean(tau):.0f}")
    except:
        print("  Warning: Could not compute autocorrelation time")
    print()

    # Save results
    print("Saving results...")

    # Samples JSON
    samples_file = outdir / "samples.json"
    with open(samples_file, 'w') as f:
        json.dump({
            'params': ['k_J', 'eta_prime', 'xi'],
            'samples': chain.tolist(),
            'log_prob': log_prob.tolist(),
            'mean': mean_theta.tolist(),
            'std': std_theta.tolist(),
            'acceptance_rate': float(np.mean(sampler.acceptance_fraction)),
            'n_snids': len(snids),
            'nwalkers': nwalkers,
            'nsteps': args.nsteps,
            'nburn': args.nburn,
            'optimization': 'JAX-vectorized'
        }, f, indent=2)

    print(f"  Saved samples to: {samples_file}")
    print(f"  Saved chain to: {backend_file}")
    print()

    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
