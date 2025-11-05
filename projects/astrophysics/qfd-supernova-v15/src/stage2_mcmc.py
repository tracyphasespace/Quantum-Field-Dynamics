#!/usr/bin/env python3
"""
Stage 2: Global MCMC Sampling

Samples (k_J, eta_prime, xi) using emcee with frozen per-SN parameters from Stage 1.

Usage:
    python stage2_mcmc.py \
        --stage1-results results/v15_stage1_production \
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
from multiprocessing import Pool

# Import model
import jax
jax.config.update("jax_enable_x64", True)
from v15_model import log_likelihood_single_sn_jax
import jax.numpy as jnp

# Global data storage (for multiprocessing)
LIGHTCURVES = {}
STAGE1_RESULTS = {}

def load_stage1_results(stage1_dir):
    """Load all Stage 1 results"""
    results = {}
    failed = []

    for result_file in Path(stage1_dir).glob("*.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)

            # Quality filter
            if not result.get('ok', False):
                failed.append(result['snid'])
                continue

            n_obs = result['n_obs']
            chi2_per_obs = result['chi2'] / n_obs if n_obs > 0 else np.inf

            if chi2_per_obs > 100:  # Poor fit
                failed.append(result['snid'])
                continue

            if result['iters'] < 3:  # Didn't converge
                failed.append(result['snid'])
                continue

            results[result['snid']] = result

        except Exception as e:
            print(f"  Warning: Failed to load {result_file.name}: {e}")
            continue

    print(f"  Loaded {len(results)} good SNe")
    print(f"  Excluded {len(failed)} poor fits")

    return results

def log_prior(theta):
    """
    Uniform priors on global params

    theta = [k_J, eta_prime, xi]
    """
    k_J, eta_prime, xi = theta

    # Physical bounds
    if not (50 < k_J < 90):
        return -np.inf
    if not (0.001 < eta_prime < 0.1):
        return -np.inf
    if not (10 < xi < 50):
        return -np.inf

    return 0.0  # Uniform prior

def log_likelihood(theta, snids, frozen_persn_params, frozen_L_peaks, photometries, redshifts):
    """
    Log-likelihood summed over all SNe with frozen per-SN params

    theta = [k_J, eta_prime, xi]
    """
    global_params = tuple(theta)

    total_logL = 0.0

    for i, snid in enumerate(snids):
        persn_params = frozen_persn_params[i]
        L_peak = frozen_L_peaks[i]
        phot = photometries[i]
        z_obs = redshifts[i]

        try:
            logL = float(log_likelihood_single_sn_jax(
                global_params, persn_params, L_peak, phot, z_obs
            ))

            if not np.isfinite(logL):
                return -np.inf

            total_logL += logL

        except Exception:
            return -np.inf

    return total_logL

def log_probability(theta, snids, frozen_persn_params, frozen_L_peaks, photometries, redshifts):
    """Log-posterior = log-prior + log-likelihood"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, snids, frozen_persn_params, frozen_L_peaks, photometries, redshifts)
    return lp + ll

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Global MCMC')
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
    parser.add_argument('--ncores', type=int, default=4,
                       help='Number of CPU cores')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: GLOBAL MCMC SAMPLING")
    print("=" * 80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Walkers: {args.nwalkers}, Steps: {args.nsteps}, Burn-in: {args.nburn}")
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(args.stage1_results)

    if len(stage1_results) < 50:
        print(f"ERROR: Only {len(stage1_results)} good SNe, need at least 50!")
        return 1

    print()

    # Load lightcurves
    print("Loading lightcurves...")
    from v15_data import LightcurveLoader
    loader = LightcurveLoader(args.lightcurves)
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")
    print()

    # Prepare data arrays
    print("Preparing frozen parameters...")
    snids = []
    frozen_persn_params = []
    frozen_L_peaks = []
    photometries = []
    redshifts = []
    L_PEAK_CANONICAL = 1.5e43  # Same as Stage 1

    for snid, result in stage1_results.items():
        if snid not in all_lcs:
            continue

        lc = all_lcs[snid]

        # Frozen per-SN params from Stage 1
        persn_best = result['persn_best']
        # Convert to model format: (t0, alpha, A_plasma, beta)
        persn_tuple = (persn_best[0], persn_best[3], persn_best[1], persn_best[2])

        # Photometry
        phot = jnp.array(np.column_stack([
            lc.mjd, lc.wavelength_nm, lc.flux_jy, lc.flux_err_jy
        ]))

        snids.append(snid)
        frozen_persn_params.append(persn_tuple)
        frozen_L_peaks.append(L_PEAK_CANONICAL)
        photometries.append(phot)
        redshifts.append(lc.z)

    print(f"  Using {len(snids)} SNe for MCMC")
    print()

    # Initial guess from Stage 1
    # Use canonical values
    initial_theta = np.array([70.0, 0.01, 30.0])

    # Initialize walkers in small ball around initial guess
    ndim = 3
    nwalkers = args.nwalkers
    pos = initial_theta + 1e-3 * np.random.randn(nwalkers, ndim)

    print("Initializing MCMC...")
    print(f"  Initial guess: k_J={initial_theta[0]}, eta'={initial_theta[1]}, xi={initial_theta[2]}")
    print()

    # Run MCMC
    print("Running MCMC...")
    print(f"  This will take ~{args.nsteps * len(snids) / 1000:.1f} minutes")
    print()

    # Backend for saving progress
    backend_file = outdir / "chain.h5"
    backend = emcee.backends.HDFBackend(str(backend_file))
    backend.reset(nwalkers, ndim)

    # Create sampler
    with Pool(args.ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(snids, frozen_persn_params, frozen_L_peaks, photometries, redshifts),
            pool=pool,
            backend=backend
        )

        # Run with progress bar
        for i, state in enumerate(sampler.sample(pos, iterations=args.nsteps, progress=True)):
            if (i + 1) % 100 == 0:
                print(f"  Step {i+1}/{args.nsteps}")
                print(f"    Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

                # Estimate autocorr time (if enough steps)
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
