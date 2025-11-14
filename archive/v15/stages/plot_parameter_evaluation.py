#!/usr/bin/env python3
"""
Visualize parameter evaluation - show what the "forward problem" means.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_data import LightcurveLoader


def load_stage1_ln_A_values(stage1_dir, lightcurves_dict, quality_cut=2000):
    """Load ln_A values from Stage 1 results."""
    stage1_path = Path(stage1_dir)

    ln_A_list = []
    z_list = []
    snid_list = []

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
                continue

            persn_best = np.load(persn_file)
            ln_A = persn_best[3]

            if ln_A >= 28 or ln_A <= -28:
                continue

            A_plasma, beta = persn_best[1], persn_best[2]
            if A_plasma <= 0.001 or A_plasma >= 0.999:
                continue
            if beta <= 0.001 or beta >= 3.999:
                continue

            iters = metrics.get('iters', 0)
            if iters < 1:
                continue

            ln_A_list.append(ln_A)
            z_list.append(lc.z)
            snid_list.append(snid)

        except Exception:
            continue

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


def predict_ln_A(c, Phi_std, ln_A_obs):
    """
    Predict ln_A for given parameters.

    This is the FAST part - just one dot product!
    ln_A_pred = ln_A0 + Φ_std · c
    """
    Phi_c = np.dot(Phi_std, c)
    ln_A0_fit = np.mean(ln_A_obs - Phi_c)
    ln_A_pred = ln_A0_fit + Phi_c
    return ln_A_pred, ln_A0_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-results', required=True)
    parser.add_argument('--lightcurves', required=True)
    parser.add_argument('--quality-cut', type=float, default=2000)
    parser.add_argument('--out', default='/tmp/parameter_evaluation.png')

    args = parser.parse_args()

    print("Loading data...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    data = load_stage1_ln_A_values(args.stage1_results, all_lcs, args.quality_cut)

    ln_A_obs = data['ln_A']
    z = data['z']
    n_sne = len(ln_A_obs)

    print(f"Loaded {n_sne} SNe")

    # Compute features (same for both parameter sets)
    Phi = compute_features(z)
    Phi_std, means, scales = standardize_features(Phi)

    # Define parameter sets
    params1 = {
        'name': 'FIXED_TEST\n(k_J=5.38)',
        'k_J': 5.381956298963422,
        'eta_prime': -1.900259394073368,
        'xi': -3.8146692019334054,
        'color': 'blue'
    }

    params2 = {
        'name': 'Recovery\n(k_J=10.77)',
        'k_J': 10.770038588319618,
        'eta_prime': -7.987900510670775,
        'xi': -6.907618767280434,
        'color': 'red'
    }

    # Predict ln_A for both parameter sets (THIS IS THE FAST PART!)
    print("\nComputing predictions (instant!)...")

    c1 = transform_physics_to_standardized(params1['k_J'], params1['eta_prime'],
                                           params1['xi'], scales)
    ln_A_pred1, ln_A0_1 = predict_ln_A(c1, Phi_std, ln_A_obs)
    residuals1 = ln_A_obs - ln_A_pred1
    rms1 = np.sqrt(np.mean(residuals1**2))

    c2 = transform_physics_to_standardized(params2['k_J'], params2['eta_prime'],
                                           params2['xi'], scales)
    ln_A_pred2, ln_A0_2 = predict_ln_A(c2, Phi_std, ln_A_obs)
    residuals2 = ln_A_obs - ln_A_pred2
    rms2 = np.sqrt(np.mean(residuals2**2))

    print(f"Done! That's why it's fast - just matrix multiplication.")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Parameter Evaluation: Forward Problem (Why It\'s Fast)',
                 fontsize=16, fontweight='bold')

    # Row 1: FIXED_TEST parameters
    # Observed vs Predicted
    ax = axes[0, 0]
    ax.scatter(ln_A_obs, ln_A_pred1, alpha=0.3, s=10, color=params1['color'])
    lim = [ln_A_obs.min()-1, ln_A_obs.max()+1]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='Perfect fit')
    ax.set_xlabel('Observed ln(A)')
    ax.set_ylabel('Predicted ln(A)')
    ax.set_title(f'{params1["name"]}\nObserved vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'RMS = {rms1:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Residuals vs redshift
    ax = axes[0, 1]
    ax.scatter(z, residuals1, alpha=0.3, s=10, color=params1['color'])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(3*np.std(residuals1), color='gray', linestyle=':', alpha=0.5, label='±3σ')
    ax.axhline(-3*np.std(residuals1), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residual (Observed - Predicted)')
    ax.set_title('Residuals vs Redshift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[0, 2]
    ax.hist(residuals1, bins=50, alpha=0.7, color=params1['color'], edgecolor='black')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.text(0.05, 0.95, f'Mean = {np.mean(residuals1):.3f}\nStd = {np.std(residuals1):.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)

    # Row 2: Recovery parameters
    # Observed vs Predicted
    ax = axes[1, 0]
    ax.scatter(ln_A_obs, ln_A_pred2, alpha=0.3, s=10, color=params2['color'])
    ax.plot(lim, lim, 'k--', alpha=0.5, label='Perfect fit')
    ax.set_xlabel('Observed ln(A)')
    ax.set_ylabel('Predicted ln(A)')
    ax.set_title(f'{params2["name"]}\nObserved vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'RMS = {rms2:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Residuals vs redshift
    ax = axes[1, 1]
    ax.scatter(z, residuals2, alpha=0.3, s=10, color=params2['color'])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(3*np.std(residuals2), color='gray', linestyle=':', alpha=0.5, label='±3σ')
    ax.axhline(-3*np.std(residuals2), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residual (Observed - Predicted)')
    ax.set_title('Residuals vs Redshift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[1, 2]
    ax.hist(residuals2, bins=50, alpha=0.7, color=params2['color'], edgecolor='black')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.text(0.05, 0.95, f'Mean = {np.mean(residuals2):.3f}\nStd = {np.std(residuals2):.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.out}")

    # Add explanation
    print("\n" + "="*80)
    print("WHY IT'S FAST:")
    print("="*80)
    print("MCMC (days):   Try many parameter values → pick best ones")
    print("Forward (1s):  Given parameters → compute fit quality")
    print("\nThe math:")
    print("  ln_A_predicted = ln_A0 + Φ · c")
    print("  where Φ is 4727×3 matrix (one row per supernova)")
    print("  and c is 3×1 vector (your parameters)")
    print("\n  This is just ONE matrix multiplication!")
    print("  4727 predictions in ~0.001 seconds")
    print("="*80)


if __name__ == '__main__':
    sys.exit(main())
