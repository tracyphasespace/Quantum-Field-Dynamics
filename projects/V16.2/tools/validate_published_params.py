#!/usr/bin/env python3
"""
Forward Model Validation Using Published Parameters

Instead of running expensive MCMC, use the PUBLISHED parameters from the papers:
- k_J ≈ 10.74 km/s/Mpc
- η' ≈ -7.97
- ξ ≈ -6.95

And evaluate how well they fit the lightcurve data.

This is a GRAPHING problem, not a supercomputer problem!

Usage:
    python3 validate_published_params.py \
        --stage1-results ../path/to/stage1 \
        --lightcurves ../path/to/lightcurves.csv \
        --out validation_output

Output:
    - Hubble diagram with residuals
    - Per-SN chi-squared distribution
    - Outlier identification
    - Goodness-of-fit statistics
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Published parameters from the papers (November 5, 2024 golden run)
PUBLISHED_PARAMS = {
    'k_J': 10.74,      # km/s/Mpc
    'eta_prime': -7.97,
    'xi': -6.95,
    'sigma_alpha': 1.398,
    'nu': 6.522,
}

# Published standardized coefficients
PUBLISHED_C = {
    'c0': 1.857,
    'c1': -2.227,
    'c2': -0.766,
}


def load_stage1_results(stage1_dir: Path, quality_cut: float = 2000) -> Dict:
    """
    Load Stage 1 results (per-SN alpha estimates).

    Returns:
        Dict with keys: snid, alpha, z, chi2, status
    """
    results = {
        'snid': [],
        'alpha': [],
        'z': [],
        'chi2': [],
        'status': [],
    }

    # Load lightcurves for redshifts
    stage1_dir = Path(stage1_dir)

    for sn_dir in stage1_dir.iterdir():
        if not sn_dir.is_dir():
            continue

        snid = sn_dir.name

        # Check status
        status_file = sn_dir / 'status.txt'
        if not status_file.exists():
            continue

        status = status_file.read_text().strip()

        # Load metrics
        metrics_file = sn_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        chi2 = metrics.get('chi2', np.inf)

        # Quality cut
        if chi2 > quality_cut:
            continue

        # Load persn_best (contains alpha)
        persn_file = sn_dir / 'persn_best.npy'
        if not persn_file.exists():
            continue

        persn = np.load(persn_file)

        # Check parameter count
        if len(persn) == 4:
            # V16 format: (t0, A_plasma, beta, alpha)
            alpha = persn[3]
        elif len(persn) == 5:
            # V16 RESTORED format: (t0, A_plasma, beta, ln_A, A_lens)
            # Need to convert ln_A to alpha
            alpha = persn[3]
        else:
            print(f"Warning: {snid} has unexpected parameter count: {len(persn)}")
            continue

        results['snid'].append(snid)
        results['alpha'].append(float(alpha))
        results['chi2'].append(float(chi2))
        results['status'].append(status)
        results['z'].append(None)  # Will fill from lightcurves

    return results


def load_redshifts(lightcurves_csv: Path, snids: List[str]) -> Dict[str, float]:
    """Load redshifts for given SNIDs from lightcurves CSV."""
    df = pd.read_csv(lightcurves_csv)

    # Get unique z for each SNID
    z_dict = {}
    for snid in snids:
        sn_data = df[df['SNID'] == snid]
        if len(sn_data) > 0:
            z_dict[snid] = sn_data['z'].iloc[0]

    return z_dict


def compute_features(z_array: np.ndarray) -> np.ndarray:
    """
    Compute the three QFD basis features.

    From the papers:
    - φ₁(z) = ln(1+z)
    - φ₂(z) = z
    - φ₃(z) = z/(1+z)

    Returns:
        Phi: [N, 3] array of features
    """
    phi1 = np.log(1 + z_array)
    phi2 = z_array
    phi3 = z_array / (1 + z_array)

    return np.stack([phi1, phi2, phi3], axis=1)


def predict_alpha(z_array: np.ndarray, k_J: float, eta_prime: float, xi: float,
                  ln_A0: float = 0.0) -> np.ndarray:
    """
    Predict alpha using published parameters.

    Model:
        α_pred = ln_A0 + k_J·φ₁(z) + η'·φ₂(z) + ξ·φ₃(z)

    Where:
        φ₁(z) = ln(1+z)
        φ₂(z) = z
        φ₃(z) = z/(1+z)
    """
    Phi = compute_features(z_array)

    # Linear prediction
    alpha_pred = ln_A0 + k_J * Phi[:, 0] + eta_prime * Phi[:, 1] + xi * Phi[:, 2]

    return alpha_pred


def compute_distance_modulus(alpha: np.ndarray) -> np.ndarray:
    """
    Convert alpha to distance modulus.

    From the papers:
        μ = -(2.5 / ln(10)) * alpha
    """
    return -(2.5 / np.log(10)) * alpha


def plot_hubble_diagram(z: np.ndarray, mu_obs: np.ndarray, mu_pred: np.ndarray,
                        out_path: Path, title: str = "Hubble Diagram"):
    """Generate Hubble diagram with residuals."""
    residuals = mu_obs - mu_pred

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top: Hubble diagram
    ax1.scatter(z, mu_obs, s=1, alpha=0.5, label='Observed')
    ax1.plot(np.sort(z), mu_pred[np.argsort(z)], 'r-', linewidth=2, label='Model')
    ax1.set_ylabel('Distance Modulus μ (mag)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Residuals
    ax2.scatter(z, residuals, s=1, alpha=0.5, color='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.axhline(residuals.mean(), color='blue', linestyle='--', linewidth=1,
                label=f'Mean: {residuals.mean():.3f}')
    ax2.axhline(residuals.mean() + residuals.std(), color='gray', linestyle=':',
                label=f'±1σ: {residuals.std():.3f}')
    ax2.axhline(residuals.mean() - residuals.std(), color='gray', linestyle=':')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residual (mag)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")


def plot_residual_histogram(residuals: np.ndarray, out_path: Path):
    """Plot histogram of residuals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(residuals.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {residuals.mean():.3f} mag')

    # Add statistics
    stats_text = f'RMS: {np.sqrt(np.mean(residuals**2)):.3f} mag\n'
    stats_text += f'Std: {residuals.std():.3f} mag\n'
    stats_text += f'N: {len(residuals)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Residual (mag)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")


def identify_outliers(residuals: np.ndarray, threshold_sigma: float = 3.0) -> np.ndarray:
    """Identify outliers based on residual threshold."""
    rms = np.sqrt(np.mean(residuals**2))
    outlier_mask = np.abs(residuals) > threshold_sigma * rms
    return outlier_mask


def main():
    parser = argparse.ArgumentParser(
        description='Validate published parameters against lightcurve data'
    )
    parser.add_argument('--stage1-results', required=True,
                       help='Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True,
                       help='Lightcurves CSV file')
    parser.add_argument('--out', default='validation_output',
                       help='Output directory')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for Stage 1 quality cut')
    parser.add_argument('--k-J', type=float, default=PUBLISHED_PARAMS['k_J'],
                       help=f'k_J parameter (default: {PUBLISHED_PARAMS["k_J"]})')
    parser.add_argument('--eta-prime', type=float, default=PUBLISHED_PARAMS['eta_prime'],
                       help=f'eta\' parameter (default: {PUBLISHED_PARAMS["eta_prime"]})')
    parser.add_argument('--xi', type=float, default=PUBLISHED_PARAMS['xi'],
                       help=f'xi parameter (default: {PUBLISHED_PARAMS["xi"]})')

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Forward Model Validation Using Published Parameters")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  k_J     = {args.k_J:.3f} km/s/Mpc")
    print(f"  η'      = {args.eta_prime:.3f}")
    print(f"  ξ       = {args.xi:.3f}")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    results = load_stage1_results(Path(args.stage1_results), args.quality_cut)
    print(f"  Loaded {len(results['snid'])} SNe (chi2 < {args.quality_cut})")

    # Load redshifts
    print("\nLoading redshifts from lightcurves...")
    z_dict = load_redshifts(Path(args.lightcurves), results['snid'])

    # Match redshifts to results
    valid_indices = []
    z_array = []
    for i, snid in enumerate(results['snid']):
        if snid in z_dict:
            valid_indices.append(i)
            z_array.append(z_dict[snid])
        else:
            print(f"  Warning: No redshift found for {snid}")

    # Filter to valid SNe
    alpha_obs = np.array([results['alpha'][i] for i in valid_indices])
    z_array = np.array(z_array)
    snids = [results['snid'][i] for i in valid_indices]

    print(f"  Matched {len(snids)} SNe with redshifts")
    print(f"  Redshift range: [{z_array.min():.3f}, {z_array.max():.3f}]")

    # Compute predictions
    print("\nComputing model predictions...")
    alpha_pred = predict_alpha(z_array, args.k_J, args.eta_prime, args.xi)

    # Convert to distance modulus
    mu_obs = compute_distance_modulus(alpha_obs)
    mu_pred = compute_distance_modulus(alpha_pred)

    # Compute residuals
    residuals = mu_obs - mu_pred
    rms = np.sqrt(np.mean(residuals**2))

    print(f"\nGoodness of Fit:")
    print(f"  RMS residual:  {rms:.3f} mag")
    print(f"  Mean residual: {residuals.mean():.3f} mag")
    print(f"  Std residual:  {residuals.std():.3f} mag")

    # Identify outliers
    outlier_mask = identify_outliers(residuals, threshold_sigma=3.0)
    n_outliers = outlier_mask.sum()
    outlier_fraction = n_outliers / len(residuals)

    print(f"\nOutliers (>3σ):")
    print(f"  Count:    {n_outliers}")
    print(f"  Fraction: {outlier_fraction*100:.1f}%")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_hubble_diagram(z_array, mu_obs, mu_pred,
                       out_dir / 'hubble_diagram.png',
                       title=f'Hubble Diagram (RMS={rms:.3f} mag)')

    plot_residual_histogram(residuals, out_dir / 'residual_histogram.png')

    # Save summary
    summary = {
        'parameters': {
            'k_J': args.k_J,
            'eta_prime': args.eta_prime,
            'xi': args.xi,
        },
        'dataset': {
            'n_sne': len(snids),
            'z_min': float(z_array.min()),
            'z_max': float(z_array.max()),
        },
        'goodness_of_fit': {
            'rms_mag': float(rms),
            'mean_residual_mag': float(residuals.mean()),
            'std_residual_mag': float(residuals.std()),
        },
        'outliers': {
            'count': int(n_outliers),
            'fraction': float(outlier_fraction),
            'threshold_sigma': 3.0,
        },
    }

    with open(out_dir / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {out_dir / 'validation_summary.json'}")

    # Save outlier list
    outlier_snids = [snids[i] for i in range(len(snids)) if outlier_mask[i]]
    with open(out_dir / 'outlier_snids.txt', 'w') as f:
        for snid in outlier_snids:
            f.write(f"{snid}\n")

    print(f"  Saved: {out_dir / 'outlier_snids.txt'}")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
