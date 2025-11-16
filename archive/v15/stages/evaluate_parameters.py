#!/usr/bin/env python3
"""
Evaluate fit quality of known parameter sets against full dataset.

Forward problem: Given parameters, compute chi-squared and residuals.

Usage:
    python3 evaluate_parameters.py \
        --stage1-results ../results/v15_clean/stage1_fullscale \
        --lightcurves data/lightcurves_unified_v2_min3.csv
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_data import LightcurveLoader


def load_stage1_ln_A_values(stage1_dir, lightcurves_dict, quality_cut=2000):
    """Load ln_A values from Stage 1 results."""
    stage1_path = Path(stage1_dir)

    ln_A_list = []
    z_list = []
    snid_list = []
    excluded = 0

    for snid_str, lc in lightcurves_dict.items():
        snid = int(snid_str)
        snid_dir = stage1_path / str(snid)
        metrics_file = snid_dir / "metrics.json"
        persn_file = snid_dir / "persn_best.npy"

        if not (metrics_file.exists() and persn_file.exists()):
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            chi2 = abs(metrics['chi2'])
            if chi2 >= quality_cut:
                excluded += 1
                continue

            persn_best = np.load(persn_file)
            ln_A = persn_best[3]

            if ln_A >= 28 or ln_A <= -28:
                excluded += 1
                continue

            A_plasma, beta = persn_best[1], persn_best[2]
            if A_plasma <= 0.001 or A_plasma >= 0.999:
                excluded += 1
                continue
            if beta <= 0.001 or beta >= 3.999:
                excluded += 1
                continue

            iters = metrics.get('iters', 0)
            if iters < 1:
                excluded += 1
                continue

            ln_A_list.append(ln_A)
            z_list.append(lc.z)
            snid_list.append(snid)

        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    print(f"  Loaded {len(ln_A_list)} SNe (excluded {excluded} with chi2 >= {quality_cut})")

    return {
        'ln_A': np.array(ln_A_list),
        'z': np.array(z_list),
        'snids': np.array(snid_list)
    }


def compute_features(z):
    """Compute feature matrix Φ = [ln(1 + z), z, z / (1 + z)]"""
    z = np.asarray(z)
    phi1 = np.log(1 + z)
    phi2 = z
    phi3 = z / (1 + z)
    return np.stack([phi1, phi2, phi3], axis=1)


def standardize_features(Phi):
    """Standardize features to zero mean, unit variance."""
    means = np.mean(Phi, axis=0)
    scales = np.std(Phi, axis=0)
    Phi_std = (Phi - means) / scales
    return Phi_std, means, scales


def transform_physics_to_standardized(k_J, eta_prime, xi, scales):
    """Transform physics parameters to standardized coefficients."""
    c0 = k_J * scales[0]
    c1 = eta_prime * scales[1]
    c2 = xi * scales[2]
    return np.array([c0, c1, c2])


def evaluate_fit(c, Phi_std, ln_A_obs, param_name):
    """
    Evaluate fit quality for given standardized coefficients.

    Returns:
        dict with chi2, reduced_chi2, residuals, etc.
    """
    # Predict ln_A (assuming ln_A0 ≈ 0 for simplicity, or fit it)
    # For proper evaluation, we should include ln_A0
    # Let's fit ln_A0 that minimizes residuals

    # ln_A_pred = ln_A0 + Φ_std · c
    # Optimal ln_A0 = mean(ln_A_obs - Φ_std · c)

    Phi_c = np.dot(Phi_std, c)
    ln_A0_fit = np.mean(ln_A_obs - Phi_c)
    ln_A_pred = ln_A0_fit + Phi_c

    # Compute residuals
    residuals = ln_A_obs - ln_A_pred

    # Compute chi-squared (assuming constant sigma)
    # Estimate sigma from residuals
    sigma = np.std(residuals)
    chi2 = np.sum((residuals / sigma)**2)
    dof = len(ln_A_obs) - 4  # 4 parameters: c0, c1, c2, ln_A0
    reduced_chi2 = chi2 / dof

    # Compute RMS error
    rms = np.sqrt(np.mean(residuals**2))

    print(f"\n{'='*80}")
    print(f"{param_name}")
    print(f"{'='*80}")
    print(f"Standardized coefficients:")
    print(f"  c0 = {c[0]:.4f}")
    print(f"  c1 = {c[1]:.4f}")
    print(f"  c2 = {c[2]:.4f}")
    print(f"  ln_A0 = {ln_A0_fit:.4f} (fitted intercept)")
    print(f"\nFit quality:")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  DOF: {dof}")
    print(f"  Reduced chi2: {reduced_chi2:.4f}")
    print(f"  RMS error: {rms:.4f}")
    print(f"  Estimated sigma: {sigma:.4f}")
    print(f"\nResidual statistics:")
    print(f"  Mean: {np.mean(residuals):.4f}")
    print(f"  Std: {np.std(residuals):.4f}")
    print(f"  Min: {np.min(residuals):.4f}")
    print(f"  Max: {np.max(residuals):.4f}")

    # Count outliers (|residual| > 3σ)
    outliers = np.abs(residuals) > 3 * sigma
    n_outliers = np.sum(outliers)
    print(f"  Outliers (>3σ): {n_outliers} / {len(residuals)} ({100*n_outliers/len(residuals):.1f}%)")

    return {
        'c': c,
        'ln_A0': ln_A0_fit,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'rms': rms,
        'sigma': sigma,
        'residuals': residuals,
        'n_outliers': n_outliers
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate parameter fit quality')
    parser.add_argument('--stage1-results', required=True, help='Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True, help='Lightcurves CSV file')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for Stage 1 quality cut')

    args = parser.parse_args()

    print("="*80)
    print("PARAMETER EVALUATION: FORWARD PROBLEM")
    print("="*80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Quality cut: chi2 < {args.quality_cut}")

    # Load data
    print("\nLoading lightcurves...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")

    print("\nLoading Stage 1 results...")
    data = load_stage1_ln_A_values(args.stage1_results, all_lcs, args.quality_cut)

    ln_A_obs = data['ln_A']
    z = data['z']
    n_sne = len(ln_A_obs)

    print(f"\nDataset summary:")
    print(f"  N_SNe: {n_sne}")
    print(f"  Redshift range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  ln_A range: [{ln_A_obs.min():.1f}, {ln_A_obs.max():.1f}]")

    # Compute and standardize features
    print("\nComputing features...")
    Phi = compute_features(z)
    Phi_std, means, scales = standardize_features(Phi)
    print(f"  Standardization scales: [{scales[0]:.4f}, {scales[1]:.4f}, {scales[2]:.4f}]")

    # Define parameter sets to test
    param_sets = [
        {
            'name': 'FIXED_TEST Results (200 samples, no priors)',
            'k_J': 5.381956298963422,
            'eta_prime': -1.900259394073368,
            'xi': -3.8146692019334054
        },
        {
            'name': 'Recovery Golden Values',
            'k_J': 10.770038588319618,
            'eta_prime': -7.987900510670775,
            'xi': -6.907618767280434
        }
    ]

    # Evaluate each parameter set
    results = []
    for params in param_sets:
        # Transform to standardized space
        c = transform_physics_to_standardized(
            params['k_J'],
            params['eta_prime'],
            params['xi'],
            scales
        )

        # Evaluate fit
        result = evaluate_fit(c, Phi_std, ln_A_obs, params['name'])
        result['physics'] = params
        results.append(result)

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Parameter Set':<50} {'Reduced χ²':<15} {'RMS':<10}")
    print("-" * 80)
    for i, params in enumerate(param_sets):
        print(f"{params['name']:<50} {results[i]['reduced_chi2']:<15.4f} {results[i]['rms']:<10.4f}")

    # Determine which fits better
    best_idx = np.argmin([r['reduced_chi2'] for r in results])
    print(f"\nBest fit: {param_sets[best_idx]['name']}")
    print(f"  Reduced χ² = {results[best_idx]['reduced_chi2']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
