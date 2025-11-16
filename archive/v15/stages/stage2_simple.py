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


def load_stage1_ln_A_values(stage1_dir, lightcurves_dict, quality_cut=2000, allowed_snids=None):
    """
    Load ln_A values (natural log of amplitude) from Stage 1 results.

    Args:
        allowed_snids: Optional set of SNIDs to restrict to (for staged validation)

    Returns:
        dict with keys: 'ln_A', 'z', 'snids' (all numpy arrays)
    """
    stage1_path = Path(stage1_dir)

    ln_A_list = []
    z_list = []
    snid_list = []
    excluded = 0

    for snid_str, lc in lightcurves_dict.items():
        snid = int(snid_str)

        # Skip if not in allowed list (for staged validation)
        if allowed_snids is not None and snid not in allowed_snids:
            continue

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
            # persn_best = [t0, A_plasma, beta, ln_A]

            # Quality cut: Check for boundary failures
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

            ln_A_list.append(ln_A)
            z_list.append(lc.z)  # SupernovaData object, use attribute access
            snid_list.append(snid)

        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    print(f"  Loaded {len(ln_A_list)} SNe with chi2 < {quality_cut}")
    print(f"  Excluded {excluded} SNe with chi2 >= {quality_cut}")

    # Convert ln_A array to numpy
    ln_A_arr = np.array(ln_A_list)

    # Stage 1 saves ln_A (natural logarithm of amplitude) directly
    # No conversion needed - ln_A is already in natural log space
    median_abs = np.median(np.abs(ln_A_arr))
    print(f"\n  ln_A loaded from Stage 1 (median |ln_A| = {median_abs:.1f})")
    print(f"  Range: [{ln_A_arr.min():.1f}, {np.median(ln_A_arr):.1f}, {ln_A_arr.max():.1f}]")

    return {
        'ln_A': ln_A_arr,
        'z': np.array(z_list),
        'snids': np.array(snid_list)
    }


def compute_features(z):
    """
    Compute feature matrix Φ for ANOMALOUS dimming only.

    k_J is FIXED at 70.0 km/s/Mpc (QVD baseline cosmology).
    Only fit anomalous components: plasma veil (η') and FDR (ξ).

    Φ = [z, z / (1 + z)]

    Returns:
        Phi: [N, 2] feature matrix
    """
    z = np.asarray(z)
    phi2 = z  # Plasma veil basis (corresponds to eta_prime)
    phi3 = z / (1 + z)  # FDR basis (corresponds to xi)

    return np.stack([phi2, phi3], axis=1)


def standardize_features(Phi):
    """
    Standardize features to zero mean, unit variance.

    Returns:
        Phi_std: standardized features [N, 2]
        means: mean of each feature [2]
        scales: std of each feature [2]
    """
    means = np.mean(Phi, axis=0)
    scales = np.std(Phi, axis=0)

    Phi_std = (Phi - means) / scales

    return Phi_std, means, scales


def numpyro_model(Phi_std, ln_A_obs, use_informed_priors=False, fix_nu=None):
    """
    NumPyro model for ANOMALOUS dimming only (k_J FIXED at 70.0).

    Model:
      c ~ Normal(0, 1) for each of 2 components: eta_prime, xi
      ln_A0 ~ Normal(0, 5) - intercept
      ln_A_pred = ln_A0 + Φ_std · c
      σ_ln_A ~ HalfNormal(2.0)
      ν ~ Exponential(0.1) + 2.0 (or fixed if fix_nu is provided)
      ln_A_obs ~ StudentT(ν, ln_A_pred, σ_ln_A)
    """
    # Sample standardized coefficients for ANOMALOUS dimming only
    if use_informed_priors:
        # Informed priors on standardized coefficients c
        # NOTE: These are c1 and c2 from the original 3-parameter model
        # c0 (k_J) is now FIXED, so we only sample c1 (eta_prime) and c2 (xi)
        c0_golden, c1_golden = -2.227, -0.766  # eta_prime, xi from Nov 5, 2024

        c0 = numpyro.sample('c0', dist.Normal(c0_golden, 0.5))
        c1 = numpyro.sample('c1', dist.Normal(c1_golden, 0.3))
        c = jnp.array([c0, c1])
    else:
        # Uninformative priors
        c = numpyro.sample('c', dist.Normal(0.0, 1.0).expand([2]).to_event(1))

    # Sample intercept
    ln_A0 = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))

    # Predict ln_A in standardized space
    # With standardized features and Normal(0,1) priors, c can be +/-
    # The negative signs from physics are absorbed into the c coefficients
    ln_A_pred = ln_A0 + jnp.dot(Phi_std, c)

    # Sample heteroscedastic noise
    sigma_ln_A = numpyro.sample('sigma_ln_A', dist.HalfNormal(2.0))

    # Sample Student-t degrees of freedom (or fix to golden value)
    if fix_nu is not None:
        # Fix nu to avoid Student-t funnel geometry
        nu = fix_nu
        numpyro.deterministic('nu', nu)
    else:
        nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0

    # Student-t likelihood (heavy tails for BBH/lensing outliers)
    with numpyro.plate('data', Phi_std.shape[0]):
        numpyro.sample('ln_A_obs',
                       dist.StudentT(df=nu, loc=ln_A_pred, scale=sigma_ln_A),
                       obs=ln_A_obs)


def run_mcmc(Phi_std, ln_A_obs, nchains=2, nsamples=2000, nwarmup=1000, use_informed_priors=False, fix_nu=None):
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
    if fix_nu is not None:
        print(f"  Fixing nu = {fix_nu:.3f} (avoids Student-t funnel)")

    # Set up NUTS kernel with partial application for extra args
    def model_with_priors(Phi, ln_A):
        return numpyro_model(Phi, ln_A, use_informed_priors=use_informed_priors, fix_nu=fix_nu)

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
    mcmc.run(rng_key, Phi_std, ln_A_obs)
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
    Back-transform from standardized space to physics space for ANOMALOUS dimming.

    k_J is FIXED at 70.0 km/s/Mpc (QVD baseline), so we only transform:
      - eta_prime (plasma veil)
      - xi (FDR)

    From standardization:
      c = k_phys * scales (forward)
      k_phys = c / scales (back-transform)
    """
    eta_prime_samples = c_samples[:, 0] / scales[0]
    xi_samples = c_samples[:, 1] / scales[1]

    return eta_prime_samples, xi_samples


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


def save_results(samples, eta_prime_samples, xi_samples,
                 means, scales, out_dir, use_informed_priors=False):
    """
    Save MCMC results to disk for ANOMALOUS dimming parameters only.

    k_J is FIXED at 70.0 km/s/Mpc, so we only save (eta_prime, xi).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Reconstruct c array if using informed priors
    if use_informed_priors:
        c0_samples = np.asarray(samples['c0'])  # eta_prime in standardized space
        c1_samples = np.asarray(samples['c1'])  # xi in standardized space
        c_samples_arr = np.stack([c0_samples, c1_samples], axis=1)
    else:
        c_samples_arr = np.asarray(samples['c'])

    # Save numpy arrays (2 parameters only)
    np.save(out_path / 'eta_prime_samples.npy', eta_prime_samples)
    np.save(out_path / 'xi_samples.npy', xi_samples)
    np.save(out_path / 'c_samples.npy', c_samples_arr)

    # Compute diagnostics
    eta_prime_stats = compute_diagnostics(eta_prime_samples, 'eta_prime')
    xi_stats = compute_diagnostics(xi_samples, 'xi')

    # Save best-fit (median values)
    # NOTE: k_J = 70.0 km/s/Mpc is FIXED (from QVD baseline cosmology)
    best_fit = {
        'k_J': 70.0,  # FIXED (QVD baseline)
        'eta_prime': eta_prime_stats['median'],
        'xi': xi_stats['median'],
        'k_J_std': 0.0,  # FIXED (no uncertainty)
        'eta_prime_std': eta_prime_stats['std'],
        'xi_std': xi_stats['std']
    }

    # Save as numpy array for compatibility with v15_model.py
    np.save(out_path / 'best_fit.npy', np.array([eta_prime_stats['median'], xi_stats['median']]))

    with open(out_path / 'best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)

    # Save full statistics
    results = {
        'physical': {
            'k_J': {'median': 70.0, 'std': 0.0, 'note': 'FIXED from QVD baseline cosmology'},
            'eta_prime': eta_prime_stats,
            'xi': xi_stats
        },
        'standardized': {
            'c0': compute_diagnostics(c_samples_arr[:, 0], 'c0_eta_prime'),
            'c1': compute_diagnostics(c_samples_arr[:, 1], 'c1_xi')
        },
        'meta': {
            'standardizer': {
                'means': means.tolist(),
                'scales': scales.tolist()
            },
            'n_samples': len(eta_prime_samples),
            'n_parameters': 2,
            'note': 'k_J = 70.0 km/s/Mpc FIXED (QVD baseline)'
        }
    }

    with open(out_path / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"  best_fit.json - Median parameter values (k_J FIXED at 70.0)")
    print(f"  best_fit.npy - [eta_prime, xi] for v15_model.py")
    print(f"  summary.json - Full statistics")
    print(f"  *_samples.npy - MCMC samples")


def print_results(eta_prime_samples, xi_samples):
    """
    Print results summary for ANOMALOUS dimming parameters.

    k_J is FIXED at 70.0 km/s/Mpc (QVD baseline cosmology).
    """
    print("\n" + "="*80)
    print("RESULTS: ANOMALOUS DIMMING PARAMETERS")
    print("="*80)

    print("\nPhysical Parameters (median ± std):")
    print(f"  k_J       = 70.000 (FIXED from QVD baseline cosmology)")
    print(f"  eta'      = {np.median(eta_prime_samples):.3f} ± {np.std(eta_prime_samples):.3f}")
    print(f"  xi        = {np.median(xi_samples):.3f} ± {np.std(xi_samples):.3f}")

    print("\nNOTE: This model fits ONLY the anomalous dimming component (~0.5 mag at z=0.5)")
    print("      from plasma veil (η') and FDR (ξ) effects.")
    print("      The baseline Hubble Law (H₀ ≈ 70 km/s/Mpc) is ALREADY explained")
    print("      by the QVD redshift model (see RedShift directory).")

    print("\nModel Assumptions (V15 Preliminary):")
    print("  - 2-WD progenitor system (barycentric mass)")
    print("  - Small black hole present")
    print("  - Planck/Wien thermal broadening (NOT ΛCDM time dilation)")
    print("  - BBH orbital lensing deferred to V16 (outliers only)")

    # Expected ranges are from the previous 3-parameter model
    # We need to see what the new 2-parameter model produces
    print("\n" + "="*80)
    print("Previous 3-Parameter Model Results (November 5, 2024):")
    print("  k_J       = 10.770 ± 4.567 (NOW FIXED at 70.0)")
    print("  eta'      = -7.988 ± 1.439")
    print("  xi        = -6.908 ± 3.746")
    print("="*80)


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
    parser.add_argument('--fix-nu', type=float, default=None,
                       help='Fix Student-t nu parameter (avoids funnel geometry, use 6.522 from golden run)')
    parser.add_argument('--snid-list', type=str, default=None,
                       help='JSON file with list of SNIDs to include (for staged validation)')

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

    # Load SNID list if provided (for staged validation)
    allowed_snids = None
    if args.snid_list:
        print(f"\nLoading SNID list from {args.snid_list}...")
        with open(args.snid_list) as f:
            snid_data = json.load(f)
            allowed_snids = set(snid_data['snids'])
        print(f"  Restricting to {len(allowed_snids)} SNe")

    # Load Stage 1 ln_A values
    print("\nLoading Stage 1 results...")
    data = load_stage1_ln_A_values(args.stage1_results, all_lcs, args.quality_cut, allowed_snids)

    ln_A_obs = data['ln_A']
    z = data['z']
    n_sne = len(ln_A_obs)

    print(f"\nData summary:")
    print(f"  N_SNe: {n_sne}")
    print(f"  Redshift range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  ln_A range: [{ln_A_obs.min():.3f}, {ln_A_obs.max():.3f}]")

    # Compute and standardize features
    print("\nComputing features...")
    Phi = compute_features(z)
    print(f"  Feature matrix shape: {Phi.shape}")

    print("Standardizing features...")
    Phi_std, means, scales = standardize_features(Phi)
    print(f"  Means: [{means[0]:.3f}, {means[1]:.3f}]")
    print(f"  Scales: [{scales[0]:.3f}, {scales[1]:.3f}]")

    # Convert to JAX arrays
    Phi_std_jax = jnp.array(Phi_std)
    ln_A_obs_jax = jnp.array(ln_A_obs)

    # Run MCMC
    samples, mcmc = run_mcmc(Phi_std_jax, ln_A_obs_jax,
                             args.nchains, args.nsamples, args.nwarmup,
                             use_informed_priors=args.use_informed_priors,
                             fix_nu=args.fix_nu)

    # Back-transform to physics space
    print("\nBack-transforming to physics space...")

    # Handle both informed and uninformed prior cases
    if args.use_informed_priors:
        # Reconstruct c from individual components (2 parameters: eta_prime, xi)
        c0_samples = np.asarray(samples['c0'])  # eta_prime in standardized space
        c1_samples = np.asarray(samples['c1'])  # xi in standardized space
        c_samples = np.stack([c0_samples, c1_samples], axis=1)
    else:
        # Access c directly
        c_samples = np.asarray(samples['c'])

    eta_prime_samples, xi_samples = back_transform_to_physics(c_samples, scales)

    print(f"  c samples shape: {c_samples.shape}")
    print(f"  c[0] (eta_prime) median: {np.median(c_samples[:, 0]):.3f}")
    print(f"  c[1] (xi) median: {np.median(c_samples[:, 1]):.3f}")

    # Print results
    print_results(eta_prime_samples, xi_samples)

    # Save results
    save_results(samples, eta_prime_samples, xi_samples,
                 means, scales, args.out, use_informed_priors=args.use_informed_priors)

    print("\n" + "="*80)
    print("STAGE 2 COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
