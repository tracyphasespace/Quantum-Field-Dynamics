#!/usr/bin/env python3
"""
Stage 2: Global MCMC - Clean Implementation from Pseudocode

Based on Supernovae_Pseudocode.md lines 102-144 and November 5, 2024 working results.
Uses standardized feature space (NOT QR orthogonalization) with Student-t likelihood.

Usage:
    python stage2_simple.py \
        --stage1-results ../results/v15_clean/stage1_fullscale \
        --lightcurves data/lightcurves_unified_v2_min3.csv \
        --out ../results/v15_clean/stage2_simple \
        --nchains 2 \
        --nsamples 2000 \
        --nwarmup 1000
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time

# JAX and NumPyro
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_data import LightcurveLoader


def load_stage1_alpha_values(stage1_dir, lightcurves_dict, quality_cut=2000):
    """
    Load alpha values from Stage 1 results.

    Returns:
        dict with keys: 'alpha', 'z', 'snids' (all numpy arrays)
    """
    stage1_path = Path(stage1_dir)

    alpha_list = []
    z_list = []
    snid_list = []
    excluded = 0

    for snid_str, lc in lightcurves_dict.items():
        snid = int(snid_str)
        # Stage 1 results are in subdirectories: stage1_dir/snid/metrics.json
        snid_dir = stage1_path / str(snid)
        metrics_file = snid_dir / "metrics.json"
        persn_file = snid_dir / "persn_best.npy"

        if not (metrics_file.exists() and persn_file.exists()):
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            chi2 = abs(metrics['chi2'])

            # Quality cut: chi2
            if chi2 >= quality_cut:
                excluded += 1
                continue

            # Load per-SN parameters
            persn_best = np.load(persn_file)
            # persn_best = [t0, A_plasma, beta, alpha]

            # Quality cut: Check for boundary failures (from stage2_mcmc_numpyro.py)
            ln_A = persn_best[3]
            if ln_A >= 28 or ln_A <= -28:  # Near ±30 boundaries
                excluded += 1
                continue

            A_plasma, beta = persn_best[1], persn_best[2]
            if A_plasma <= 0.001 or A_plasma >= 0.999:  # Near [0, 1] boundaries
                excluded += 1
                continue
            if beta <= 0.001 or beta >= 3.999:  # Near [0, 4] boundaries
                excluded += 1
                continue

            # Quality cut: minimum iterations (ensure convergence)
            iters = metrics.get('iters', 0)
            if iters < 1:
                excluded += 1
                continue

            alpha = ln_A
            alpha_list.append(alpha)
            z_list.append(lc.z)  # SupernovaData object, use attribute access
            snid_list.append(snid)

        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    print(f"  Loaded {len(alpha_list)} SNe with chi2 < {quality_cut}")
    print(f"  Excluded {excluded} SNe with chi2 >= {quality_cut}")

    # Convert alpha array to numpy
    alpha_arr = np.array(alpha_list)

    # Alpha conversion: magnitude space → natural-log space (from stage2_mcmc_numpyro.py)
    K_MAG_PER_LN = 2.5 / np.log(10.0)  # ≈ 1.0857
    median_abs = np.median(np.abs(alpha_arr))

    if median_abs > 5.0:
        # Alpha in magnitude space - convert to natural-log space
        print(f"\n  Alpha conversion: median |α| = {median_abs:.1f} > 5.0")
        print(f"  Converting from magnitude space: α_nat = -α_mag / K")
        print(f"  Before: [{alpha_arr.min():.1f}, {np.median(alpha_arr):.1f}, {alpha_arr.max():.1f}]")
        alpha_arr = -alpha_arr / K_MAG_PER_LN
        print(f"  After:  [{alpha_arr.min():.1f}, {np.median(alpha_arr):.1f}, {alpha_arr.max():.1f}]")
    else:
        print(f"\n  Alpha already in natural-log space (median |α| = {median_abs:.1f})")

    return {
        'alpha': alpha_arr,
        'z': np.array(z_list),
        'snids': np.array(snid_list)
    }


def compute_features(z):
    """
    Compute feature matrix Φ from pseudocode line 252.

    Φ = [ln(1 + z), z, z / (1 + z)]

    Returns:
        Phi: [N, 3] feature matrix
    """
    z = np.asarray(z)
    phi1 = np.log(1 + z)
    phi2 = z
    phi3 = z / (1 + z)

    return np.stack([phi1, phi2, phi3], axis=1)


def standardize_features(Phi):
    """
    Standardize features to zero mean, unit variance.

    Returns:
        Phi_std: standardized features [N, 3]
        means: mean of each feature [3]
        scales: std of each feature [3]
    """
    means = np.mean(Phi, axis=0)
    scales = np.std(Phi, axis=0)

    Phi_std = (Phi - means) / scales

    return Phi_std, means, scales


def numpyro_model(Phi_std, alpha_obs, use_informed_priors=False):
    """
    NumPyro model from pseudocode lines 254-260.

    Model:
      c ~ Normal(0, 1) for each of 3 components (or informed priors)
      ln_A0 ~ Normal(0, 5) - intercept
      α_pred = ln_A0 + Φ_std · c
      σ_alpha ~ HalfNormal(2.0)
      ν ~ Exponential(0.1) + 2.0
      α_obs ~ StudentT(ν, α_pred, σ_alpha)
    """
    # Sample standardized coefficients
    if use_informed_priors:
        # Informed priors DIRECTLY on standardized coefficients c
        # Using golden values from November 5, 2024 run as the mean
        # This makes priors data-independent and stable
        c0_golden, c1_golden, c2_golden = 1.857, -2.227, -0.766

        c0 = numpyro.sample('c0', dist.Normal(c0_golden, 0.5))
        c1 = numpyro.sample('c1', dist.Normal(c1_golden, 0.5))
        c2 = numpyro.sample('c2', dist.Normal(c2_golden, 0.3))
        c = jnp.array([c0, c1, c2])
    else:
        # Uninformative priors
        c = numpyro.sample('c', dist.Normal(0.0, 1.0).expand([3]).to_event(1))

    # Sample intercept
    ln_A0 = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))

    # Predict alpha in standardized space
    # With standardized features and Normal(0,1) priors, c can be +/-
    # The negative signs from physics are absorbed into the c coefficients
    alpha_pred = ln_A0 + jnp.dot(Phi_std, c)

    # Sample heteroscedastic noise
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(2.0))

    # Sample Student-t degrees of freedom
    nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0

    # Student-t likelihood (heavy tails for BBH/lensing outliers)
    with numpyro.plate('data', Phi_std.shape[0]):
        numpyro.sample('alpha_obs',
                       dist.StudentT(df=nu, loc=alpha_pred, scale=sigma_alpha),
                       obs=alpha_obs)


def run_mcmc(Phi_std, alpha_obs, nchains=2, nsamples=2000, nwarmup=1000, use_informed_priors=False):
    """
    Run MCMC using NUTS sampler (pseudocode line 260).
    """
    print("\nRunning MCMC...")
    print(f"  Chains: {nchains}")
    print(f"  Samples per chain: {nsamples}")
    print(f"  Warmup: {nwarmup}")
    print(f"  Total samples: {nchains * nsamples}")
    if use_informed_priors:
        print(f"  Using informed priors on c (golden values from Nov 5, 2024)")

    # Set up NUTS kernel with partial application for extra args
    def model_with_priors(Phi, alpha):
        return numpyro_model(Phi, alpha, use_informed_priors=use_informed_priors)

    kernel = NUTS(model_with_priors)

    # Set up MCMC
    mcmc = MCMC(
        kernel,
        num_chains=nchains,
        num_samples=nsamples,
        num_warmup=nwarmup,
        progress_bar=True
    )

    # Run MCMC
    start_time = time.time()
    rng_key = jax.random.PRNGKey(42)
    mcmc.run(rng_key, Phi_std, alpha_obs)
    elapsed = time.time() - start_time

    print(f"\n  MCMC complete in {elapsed/60:.1f} minutes")

    # Get samples
    samples = mcmc.get_samples()

    # Check for divergences
    num_divergences = np.sum(mcmc.get_extra_fields()['diverging'])
    print(f"  Divergences: {num_divergences}")

    return samples, mcmc


def back_transform_to_physics(c_samples, scales):
    """
    Back-transform from standardized space to physics space.

    From November 5 results:
      c = k_phys * scales (forward)
      k_phys = c / scales (back-transform)
    """
    k_J_samples = c_samples[:, 0] / scales[0]
    eta_prime_samples = c_samples[:, 1] / scales[1]
    xi_samples = c_samples[:, 2] / scales[2]

    return k_J_samples, eta_prime_samples, xi_samples


def compute_diagnostics(samples, param_name='parameter'):
    """Compute MCMC diagnostics for a parameter."""
    median = float(np.median(samples))
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    q05 = float(np.percentile(samples, 5))
    q95 = float(np.percentile(samples, 95))

    return {
        'median': median,
        'mean': mean,
        'std': std,
        'q05': q05,
        'q95': q95,
        'min': float(np.min(samples)),
        'max': float(np.max(samples))
    }


def save_results(samples, k_J_samples, eta_prime_samples, xi_samples,
                 means, scales, out_dir):
    """Save MCMC results to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(out_path / 'k_J_samples.npy', k_J_samples)
    np.save(out_path / 'eta_prime_samples.npy', eta_prime_samples)
    np.save(out_path / 'xi_samples.npy', xi_samples)
    # Handle both informed (c0, c1, c2) and uninformed (c) priors
    if 'c' in samples:
        c_samples_to_save = np.asarray(samples['c'])
    else:
        c_samples_to_save = np.stack([samples['c0'], samples['c1'], samples['c2']], axis=1)
    np.save(out_path / 'c_samples.npy', c_samples_to_save)

    # Compute diagnostics
    k_J_stats = compute_diagnostics(k_J_samples, 'k_J')
    eta_prime_stats = compute_diagnostics(eta_prime_samples, 'eta_prime')
    xi_stats = compute_diagnostics(xi_samples, 'xi')

    # Save best-fit (median values)
    best_fit = {
        'k_J': k_J_stats['median'],
        'eta_prime': eta_prime_stats['median'],
        'xi': xi_stats['median'],
        'k_J_std': k_J_stats['std'],
        'eta_prime_std': eta_prime_stats['std'],
        'xi_std': xi_stats['std']
    }

    with open(out_path / 'best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)

    # Save full statistics
    results = {
        'physical': {
            'k_J': k_J_stats,
            'eta_prime': eta_prime_stats,
            'xi': xi_stats
        },
        'standardized': {
            'c0': compute_diagnostics(c_samples_to_save[:, 0], 'c0'),
            'c1': compute_diagnostics(c_samples_to_save[:, 1], 'c1'),
            'c2': compute_diagnostics(c_samples_to_save[:, 2], 'c2')
        },
        'meta': {
            'standardizer': {
                'means': means.tolist(),
                'scales': scales.tolist()
            },
            'n_samples': len(k_J_samples)
        }
    }

    with open(out_path / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"  best_fit.json - Median parameter values")
    print(f"  summary.json - Full statistics")
    print(f"  *_samples.npy - MCMC samples")


def print_results(k_J_samples, eta_prime_samples, xi_samples):
    """Print results summary."""
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nPhysical Parameters (median ± std):")
    print(f"  k_J       = {np.median(k_J_samples):.3f} ± {np.std(k_J_samples):.3f} km/s/Mpc")
    print(f"  eta'      = {np.median(eta_prime_samples):.3f} ± {np.std(eta_prime_samples):.3f}")
    print(f"  xi        = {np.median(xi_samples):.3f} ± {np.std(xi_samples):.3f}")

    print("\nExpected (November 5, 2024 golden reference):")
    print(f"  k_J       = 10.770 ± 4.567")
    print(f"  eta'      = -7.988 ± 1.439")
    print(f"  xi        = -6.908 ± 3.746")

    # Check if results are in expected range
    k_J_ok = 7.5 < np.median(k_J_samples) < 13.9
    eta_ok = -10.4 < np.median(eta_prime_samples) < -5.6
    xi_ok = -9.0 < np.median(xi_samples) < -4.8

    print("\nValidation (within ±30% of Nov 5 results):")
    print(f"  k_J:  {'✓ PASS' if k_J_ok else '✗ FAIL'}")
    print(f"  eta': {'✓ PASS' if eta_ok else '✗ FAIL'}")
    print(f"  xi:   {'✓ PASS' if xi_ok else '✗ FAIL'}")

    if k_J_ok and eta_ok and xi_ok:
        print("\n✓ ALL PARAMETERS WITHIN EXPECTED RANGE")
    else:
        print("\n✗ SOME PARAMETERS OUT OF RANGE - CHECK RESULTS")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Global MCMC (Clean Implementation)')
    parser.add_argument('--stage1-results', required=True, help='Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True, help='Lightcurves CSV file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--nchains', type=int, default=2, help='Number of MCMC chains')
    parser.add_argument('--nsamples', type=int, default=2000, help='Samples per chain')
    parser.add_argument('--nwarmup', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for Stage 1 quality cut')
    parser.add_argument('--use-informed-priors', action='store_true',
                       help='Use informed priors on standardized coefficients c (from Nov 5, 2024 golden run)')

    args = parser.parse_args()

    print("="*80)
    print("STAGE 2: GLOBAL MCMC (CLEAN IMPLEMENTATION)")
    print("="*80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Lightcurves: {args.lightcurves}")
    print(f"Output: {args.out}")
    print(f"Quality cut: chi2 < {args.quality_cut}")

    # Load lightcurves
    print("\nLoading lightcurves...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")

    # Load Stage 1 alpha values
    print("\nLoading Stage 1 results...")
    data = load_stage1_alpha_values(args.stage1_results, all_lcs, args.quality_cut)

    alpha_obs = data['alpha']
    z = data['z']
    n_sne = len(alpha_obs)

    print(f"\nData summary:")
    print(f"  N_SNe: {n_sne}")
    print(f"  Redshift range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  Alpha range: [{alpha_obs.min():.3f}, {alpha_obs.max():.3f}]")

    # Compute and standardize features
    print("\nComputing features...")
    Phi = compute_features(z)
    print(f"  Feature matrix shape: {Phi.shape}")

    print("Standardizing features...")
    Phi_std, means, scales = standardize_features(Phi)
    print(f"  Means: [{means[0]:.3f}, {means[1]:.3f}, {means[2]:.3f}]")
    print(f"  Scales: [{scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f}]")

    # Convert to JAX arrays
    Phi_std_jax = jnp.array(Phi_std)
    alpha_obs_jax = jnp.array(alpha_obs)

    # Run MCMC
    samples, mcmc = run_mcmc(Phi_std_jax, alpha_obs_jax,
                             args.nchains, args.nsamples, args.nwarmup,
                             use_informed_priors=args.use_informed_priors)

    # Back-transform to physics space
    print("\nBack-transforming to physics space...")
    # Handle both informed (c0, c1, c2) and uninformed (c) priors
    if 'c' in samples:
        c_samples = np.asarray(samples['c'])
    else:
        # Reconstruct c from c0, c1, c2
        c_samples = np.stack([samples['c0'], samples['c1'], samples['c2']], axis=1)
    k_J_samples, eta_prime_samples, xi_samples = back_transform_to_physics(c_samples, scales)

    print(f"  c samples shape: {c_samples.shape}")
    print(f"  c[0] median: {np.median(c_samples[:, 0]):.3f}")
    print(f"  c[1] median: {np.median(c_samples[:, 1]):.3f}")
    print(f"  c[2] median: {np.median(c_samples[:, 2]):.3f}")

    # Print results
    print_results(k_J_samples, eta_prime_samples, xi_samples)

    # Save results
    save_results(samples, k_J_samples, eta_prime_samples, xi_samples,
                 means, scales, args.out)

    print("\n" + "="*80)
    print("STAGE 2 COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
