#!/usr/bin/env python3
"""
Stage 2: Global MCMC Sampling with NumPyro (FULLY GPU-OPTIMIZED)

Uses NumPyro's NUTS sampler for:
- Fully GPU-native sampling (10-100× faster than emcee)
- Adaptive step size (no tuning required)
- Better convergence diagnostics
- Automatic parallelization across chains

Expected speedup: 50-100× vs emcee version

Usage:
    python stage2_mcmc_numpyro.py \
        --stage1-results results/v15_stage1_production \
        --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
        --out results/v15_stage2_mcmc_numpyro \
        --nchains 4 \
        --nsamples 2000 \
        --nwarmup 1000
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

# JAX and NumPyro
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Import model
from v15_model import log_likelihood_single_sn_jax

def load_stage1_results(stage1_dir, lightcurves_dict, quality_cut=2000):
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
            if chi2 > quality_cut:
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

    print(f"  Loaded {len(results)} good SNe (chi2 < {quality_cut})")
    print(f"  Excluded {len(failed)} poor fits")

    return results

@jax.jit
def log_likelihood_batch_jax(
    k_J: float,
    eta_prime: float,
    xi: float,
    persn_params_batch: jnp.ndarray,  # Shape: (n_sne, 4)
    L_peaks: jnp.ndarray,  # Shape: (n_sne,)
    photometries: List,  # List of JAX arrays
    redshifts: jnp.ndarray  # Shape: (n_sne,)
) -> jnp.ndarray:
    """
    Vectorized log-likelihood over multiple SNe.

    Returns: Array of shape (n_sne,) with logL for each SN
    """
    global_params = (k_J, eta_prime, xi)

    def single_sn_logL(persn_params, L_peak, phot, z_obs):
        return log_likelihood_single_sn_jax(
            global_params, tuple(persn_params), L_peak, phot, z_obs
        )

    # vmap over SNe
    logLs = vmap(single_sn_logL)(
        persn_params_batch, L_peaks, photometries, redshifts
    )

    return logLs

def numpyro_model(persn_params_batch, L_peaks, photometries, redshifts):
    """
    NumPyro model for global parameter inference.

    Uses uniform priors and vectorized likelihood.
    """
    # Priors
    k_J = numpyro.sample('k_J', dist.Uniform(50, 90))
    eta_prime = numpyro.sample('eta_prime', dist.Uniform(0.001, 0.1))
    xi = numpyro.sample('xi', dist.Uniform(10, 50))

    # Compute log-likelihood for all SNe
    logLs = log_likelihood_batch_jax(
        k_J, eta_prime, xi,
        persn_params_batch, L_peaks, photometries, redshifts
    )

    # Sum log-likelihoods (numpyro handles this automatically)
    total_logL = jnp.sum(logLs)

    # Factor in the total log-likelihood
    numpyro.factor('logL', total_logL)

def main():
    parser = argparse.ArgumentParser(description='Stage 2: NumPyro Global MCMC')
    parser.add_argument('--stage1-results', required=True,
                       help='Directory with Stage 1 results')
    parser.add_argument('--lightcurves', required=True,
                       help='Path to lightcurves CSV')
    parser.add_argument('--out', required=True,
                       help='Output directory')
    parser.add_argument('--nchains', type=int, default=4,
                       help='Number of MCMC chains (parallel on GPU)')
    parser.add_argument('--nsamples', type=int, default=2000,
                       help='Number of samples per chain (post-warmup)')
    parser.add_argument('--nwarmup', type=int, default=1000,
                       help='Number of warmup/burn-in steps')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for Stage 1 quality')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: NUMPYRO GLOBAL MCMC (FULLY GPU-OPTIMIZED)")
    print("=" * 80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Chains: {args.nchains}, Samples: {args.nsamples}, Warmup: {args.nwarmup}")
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves
    print("Loading lightcurves...")
    from v15_data import LightcurveLoader
    loader = LightcurveLoader(args.lightcurves)
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(
        args.stage1_results, all_lcs, args.quality_cut
    )

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

    for snid, result in stage1_results.items():
        if snid not in all_lcs:
            continue

        lc = all_lcs[snid]

        # Frozen per-SN params from Stage 1
        persn_best = result['persn_best']
        # Order: (t0, alpha, A_plasma, beta)
        persn_tuple = (persn_best[0], persn_best[3], persn_best[1], persn_best[2])

        # Photometry as JAX array
        phot = jnp.array(np.column_stack([
            lc.mjd, lc.wavelength_nm, lc.flux_jy, lc.flux_err_jy
        ]))

        snids.append(snid)
        frozen_persn_params.append(persn_tuple)
        frozen_L_peaks.append(result['L_peak'])
        photometries.append(phot)
        redshifts.append(lc.z)

    # Convert to JAX arrays
    persn_params_batch = jnp.array(frozen_persn_params)
    L_peaks_arr = jnp.array(frozen_L_peaks)
    redshifts_arr = jnp.array(redshifts)

    print(f"  Using {len(snids)} SNe for MCMC")
    print()

    # Setup NUTS sampler
    print("Setting up NUTS sampler...")
    nuts_kernel = NUTS(
        numpyro_model,
        target_accept_prob=0.8,  # Higher = more accurate, slower
        max_tree_depth=10,  # Maximum trajectory length
        init_strategy=numpyro.infer.init_to_median  # Start at median of prior
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarmup,
        num_samples=args.nsamples,
        num_chains=args.nchains,
        chain_method='parallel',  # Run chains in parallel on GPU
        progress_bar=True
    )

    print(f"  Warmup: {args.nwarmup} steps")
    print(f"  Sampling: {args.nsamples} samples × {args.nchains} chains = {args.nsamples * args.nchains} total")
    print(f"  Expected time: ~{(args.nwarmup + args.nsamples) / 10:.1f} minutes")
    print()

    # Run MCMC
    print("Running MCMC...")
    start_time = time.time()

    rng_key = jax.random.PRNGKey(0)
    mcmc.run(
        rng_key,
        persn_params_batch,
        L_peaks_arr,
        photometries,
        redshifts_arr
    )

    elapsed = time.time() - start_time
    print(f"  MCMC complete in {elapsed/60:.1f} minutes")
    print()

    # Get samples
    print("Extracting samples...")
    samples = mcmc.get_samples()

    # Compute summary statistics
    k_J_samples = np.array(samples['k_J'])
    eta_prime_samples = np.array(samples['eta_prime'])
    xi_samples = np.array(samples['xi'])

    # Best-fit (median of posterior)
    k_J_best = float(np.median(k_J_samples))
    eta_prime_best = float(np.median(eta_prime_samples))
    xi_best = float(np.median(xi_samples))

    # Uncertainties (standard deviation)
    k_J_std = float(np.std(k_J_samples))
    eta_prime_std = float(np.std(eta_prime_samples))
    xi_std = float(np.std(xi_samples))

    print("=" * 80)
    print("MCMC RESULTS")
    print("=" * 80)
    print(f"Best-fit parameters (median):")
    print(f"  k_J = {k_J_best:.2f} ± {k_J_std:.4f}")
    print(f"  eta' = {eta_prime_best:.4f} ± {eta_prime_std:.5f}")
    print(f"  xi = {xi_best:.2f} ± {xi_std:.4f}")
    print()

    # Print diagnostics
    print("MCMC Diagnostics:")
    mcmc.print_summary()
    print()

    # Check for divergences
    divergences = mcmc.get_extra_fields()['diverging']
    n_divergences = np.sum(divergences)
    if n_divergences > 0:
        print(f"⚠️  WARNING: {n_divergences} divergent transitions detected!")
        print("   Consider increasing target_accept_prob or max_tree_depth")
    else:
        print("✅ No divergences detected")
    print()

    # Save results
    print("Saving results...")

    # Save samples as JSON
    samples_dict = {
        'params': ['k_J', 'eta_prime', 'xi'],
        'samples': np.column_stack([k_J_samples, eta_prime_samples, xi_samples]).tolist(),
        'mean': [float(np.mean(k_J_samples)), float(np.mean(eta_prime_samples)), float(np.mean(xi_samples))],
        'median': [k_J_best, eta_prime_best, xi_best],
        'std': [k_J_std, eta_prime_std, xi_std],
        'n_chains': args.nchains,
        'n_samples_per_chain': args.nsamples,
        'n_warmup': args.nwarmup,
        'n_divergences': int(n_divergences),
        'runtime_minutes': elapsed / 60,
        'n_snids': len(snids)
    }

    with open(outdir / 'samples.json', 'w') as f:
        json.dump(samples_dict, f, indent=2)
    print(f"  Saved samples to: {outdir / 'samples.json'}")

    # Save best-fit parameters
    best_fit = {
        'k_J': k_J_best,
        'eta_prime': eta_prime_best,
        'xi': xi_best,
        'k_J_std': k_J_std,
        'eta_prime_std': eta_prime_std,
        'xi_std': xi_std
    }

    with open(outdir / 'best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)
    print(f"  Saved best-fit to: {outdir / 'best_fit.json'}")

    # Save raw samples as numpy arrays
    np.save(outdir / 'k_J_samples.npy', k_J_samples)
    np.save(outdir / 'eta_prime_samples.npy', eta_prime_samples)
    np.save(outdir / 'xi_samples.npy', xi_samples)
    print(f"  Saved numpy arrays to: {outdir / '*.npy'}")

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print()

    return 0

if __name__ == '__main__':
    exit(main())
