#!/usr/bin/env python3
"""
Holdout Validation Tool

Implements pseudocode lines 163-168, 266-267, 341:
- Predict alpha on holdout SNe (chi2 >= quality_cut)
- Compute RMS residuals
- Compare to clean set RMS
- Identify BBH/lensing candidates vs. bad data

Usage:
    python tools/validate_holdout.py \
        --stage1-results ../results/v15_clean/stage1_fullscale \
        --stage2-results ../results/v15_clean/stage2_production_unconstrained \
        --lightcurves data/lightcurves_unified_v2_min3.csv \
        --quality-cut 2000 \
        --out ../results/v15_clean/holdout_validation
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_model import ln_A_pred_batch
from v15_data import load_lightcurves

def load_holdout_sne(stage1_dir, lightcurves_dict, quality_cut=2000):
    """
    Load SNe EXCLUDED from Stage 2 (chi2 >= quality_cut).

    These are the "difficult" SNe that should be validated
    on the model trained on clean data.
    """
    holdout = {}
    stage1_path = Path(stage1_dir)

    for snid_str, lc in lightcurves_dict.items():
        snid = int(snid_str)
        metrics_file = stage1_path / f"{snid}_metrics.json"

        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            chi2 = metrics['chi2']

            # HOLDOUT: chi2 >= quality_cut (opposite of clean set)
            if abs(chi2) >= quality_cut:
                persn_file = stage1_path / f"{snid}_best.npy"
                if persn_file.exists():
                    persn_best = np.load(persn_file)
                    holdout[snid] = {
                        'snid': snid,
                        'z': lc['z'],
                        'persn_best': persn_best,  # [t0, A_plasma, beta, alpha]
                        'chi2': chi2,
                        'metrics': metrics
                    }
        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    return holdout

def validate_holdout(holdout_results, stage2_samples, verbose=True):
    """
    Validate model on holdout SNe.

    From pseudocode line 341:
      residuals = α_obs_holdout - α_pred_mean
      RMS = √(mean(residuals²))
    """
    # Extract global parameters (mean over MCMC samples)
    k_J_mean = np.mean(stage2_samples['k_J'])
    eta_prime_mean = np.mean(stage2_samples['eta_prime'])
    xi_mean = np.mean(stage2_samples['xi'])

    if verbose:
        print(f"\nGlobal parameters (mean):")
        print(f"  k_J = {k_J_mean:.4f}")
        print(f"  η' = {eta_prime_mean:.4f}")
        print(f"  ξ = {xi_mean:.4f}")

    # Prepare holdout data
    z_holdout = np.array([r['z'] for r in holdout_results.values()])
    alpha_obs_holdout = np.array([r['persn_best'][3] for r in holdout_results.values()])
    snids = np.array([r['snid'] for r in holdout_results.values()])
    chi2_holdout = np.array([r['chi2'] for r in holdout_results.values()])

    # Predict alpha using global model
    alpha_pred_holdout = ln_A_pred_batch(z_holdout, k_J_mean, eta_prime_mean, xi_mean)

    # Compute residuals
    residuals = alpha_obs_holdout - alpha_pred_holdout
    rms = np.sqrt(np.mean(residuals**2))

    if verbose:
        print(f"\nHoldout Validation:")
        print(f"  N_holdout = {len(holdout_results)}")
        print(f"  RMS = {rms:.4f}")
        print(f"  Median residual = {np.median(residuals):.4f}")
        print(f"  Std residual = {np.std(residuals):.4f}")
        print(f"  Max |residual| = {np.max(np.abs(residuals)):.4f}")

    # Identify extreme outliers (potential BBH/lensing)
    # Use 3-sigma threshold
    threshold = 3 * np.std(residuals)
    extreme_mask = np.abs(residuals) > threshold

    if verbose:
        print(f"\nExtreme outliers (|residual| > {threshold:.2f}):")
        print(f"  N_extreme = {np.sum(extreme_mask)} ({100*np.mean(extreme_mask):.1f}%)")

    return {
        'rms': rms,
        'residuals': residuals,
        'alpha_obs': alpha_obs_holdout,
        'alpha_pred': alpha_pred_holdout,
        'z': z_holdout,
        'snids': snids,
        'chi2': chi2_holdout,
        'extreme_mask': extreme_mask,
        'k_J_mean': k_J_mean,
        'eta_prime_mean': eta_prime_mean,
        'xi_mean': xi_mean
    }

def plot_holdout_diagnostics(validation_results, clean_rms, out_dir):
    """Generate diagnostic plots for holdout validation"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residuals = validation_results['residuals']
    z = validation_results['z']
    chi2 = validation_results['chi2']
    extreme_mask = validation_results['extreme_mask']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Residuals vs redshift
    ax = axes[0, 0]
    ax.scatter(z[~extreme_mask], residuals[~extreme_mask], alpha=0.5, s=10, label='Normal')
    ax.scatter(z[extreme_mask], residuals[extreme_mask], alpha=0.8, s=30,
               color='red', label='Extreme outliers')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(3*np.std(residuals), color='r', linestyle=':', alpha=0.3)
    ax.axhline(-3*np.std(residuals), color='r', linestyle=':', alpha=0.3)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residual (α_obs - α_pred)')
    ax.set_title(f'Holdout Residuals vs Redshift\nRMS={validation_results["rms"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residual histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(np.median(residuals), color='r', linestyle='-',
               label=f'Median={np.median(residuals):.3f}')
    ax.set_xlabel('Residual (α_obs - α_pred)')
    ax.set_ylabel('Count')
    ax.set_title(f'Holdout Residual Distribution\nStd={np.std(residuals):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals vs Stage 1 chi2
    ax = axes[1, 0]
    ax.scatter(chi2[~extreme_mask], residuals[~extreme_mask], alpha=0.5, s=10)
    ax.scatter(chi2[extreme_mask], residuals[extreme_mask], alpha=0.8, s=30, color='red')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Stage 1 chi² (per-SN fit quality)')
    ax.set_ylabel('Residual (α_obs - α_pred)')
    ax.set_title('Residuals vs Fit Quality')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Clean vs Holdout RMS comparison
    ax = axes[1, 1]
    categories = ['Clean Set\n(chi²<2000)', 'Holdout Set\n(chi²≥2000)']
    rms_values = [clean_rms, validation_results['rms']]
    colors = ['green', 'orange']
    bars = ax.bar(categories, rms_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMS Residual')
    ax.set_title('Model Performance: Clean vs Holdout')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, rms_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'holdout_diagnostics.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(out_dir / 'holdout_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plots to {out_dir}/holdout_diagnostics.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Validate model on holdout SNe')
    parser.add_argument('--stage1-results', required=True, help='Stage 1 results directory')
    parser.add_argument('--stage2-results', required=True, help='Stage 2 results directory')
    parser.add_argument('--lightcurves', required=True, help='Lightcurves CSV file')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold (holdout = chi2 >= this)')
    parser.add_argument('--out', required=True, help='Output directory')

    args = parser.parse_args()

    print("="*80)
    print("HOLDOUT VALIDATION")
    print("="*80)
    print(f"Stage 1 results: {args.stage1_results}")
    print(f"Stage 2 results: {args.stage2_results}")
    print(f"Quality cut: chi² >= {args.quality_cut}")

    # Load lightcurves
    print("\nLoading lightcurves...")
    all_lcs = load_lightcurves(args.lightcurves)
    print(f"  Loaded {len(all_lcs)} lightcurves")

    # Load holdout SNe
    print("\nLoading holdout SNe (excluded from Stage 2)...")
    holdout_results = load_holdout_sne(args.stage1_results, all_lcs, args.quality_cut)
    print(f"  Loaded {len(holdout_results)} holdout SNe")

    if len(holdout_results) == 0:
        print("ERROR: No holdout SNe found!")
        return 1

    # Load Stage 2 MCMC samples
    print("\nLoading Stage 2 MCMC samples...")
    stage2_path = Path(args.stage2_results)

    # Try loading from samples.npz first
    samples_file = stage2_path / 'samples.npz'
    if samples_file.exists():
        samples_data = np.load(samples_file)
        stage2_samples = {
            'k_J': samples_data['k_J'],
            'eta_prime': samples_data['eta_prime'],
            'xi': samples_data['xi']
        }
    else:
        # Fall back to JSON
        samples_file = stage2_path / 'samples.json'
        with open(samples_file) as f:
            samples_json = json.load(f)
        stage2_samples = {
            'k_J': np.array(samples_json['k_J']),
            'eta_prime': np.array(samples_json['eta_prime']),
            'xi': np.array(samples_json['xi'])
        }

    print(f"  Loaded {len(stage2_samples['k_J'])} MCMC samples")

    # Load clean set RMS for comparison
    best_fit_file = stage2_path / 'best_fit.json'
    with open(best_fit_file) as f:
        best_fit = json.load(f)

    # Estimate clean RMS from sigma_ln_A if available
    clean_rms = best_fit.get('sigma_ln_A', 0.2)  # Default fallback

    # Run validation
    print("\nValidating model on holdout set...")
    validation_results = validate_holdout(holdout_results, stage2_samples)

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_holdout_diagnostics(validation_results, clean_rms, args.out)

    # Save results
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    output = {
        'n_holdout': len(holdout_results),
        'quality_cut': args.quality_cut,
        'rms_holdout': float(validation_results['rms']),
        'rms_clean': float(clean_rms),
        'ratio': float(validation_results['rms'] / clean_rms) if clean_rms > 0 else None,
        'median_residual': float(np.median(validation_results['residuals'])),
        'std_residual': float(np.std(validation_results['residuals'])),
        'n_extreme': int(np.sum(validation_results['extreme_mask'])),
        'pct_extreme': float(100 * np.mean(validation_results['extreme_mask'])),
        'global_params': {
            'k_J': float(validation_results['k_J_mean']),
            'eta_prime': float(validation_results['eta_prime_mean']),
            'xi': float(validation_results['xi_mean'])
        }
    }

    with open(out_path / 'holdout_summary.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save detailed results
    np.savez(out_path / 'holdout_results.npz',
             snids=validation_results['snids'],
             z=validation_results['z'],
             alpha_obs=validation_results['alpha_obs'],
             alpha_pred=validation_results['alpha_pred'],
             residuals=validation_results['residuals'],
             chi2=validation_results['chi2'],
             extreme_mask=validation_results['extreme_mask'])

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Clean set RMS:   {clean_rms:.4f}")
    print(f"Holdout RMS:     {validation_results['rms']:.4f}")
    print(f"Ratio (H/C):     {validation_results['rms']/clean_rms:.2f}x")
    print(f"\nExtreme outliers: {np.sum(validation_results['extreme_mask'])} "
          f"({100*np.mean(validation_results['extreme_mask']):.1f}%)")
    print(f"\nResults saved to: {args.out}")
    print("="*80)

    # Interpretation
    ratio = validation_results['rms'] / clean_rms
    print("\nInterpretation:")
    if ratio < 1.5:
        print("  ✓ Model generalizes well to holdout set")
        print("    Excluded SNe are likely bad data, not physical outliers")
    elif ratio < 3.0:
        print("  ⚠ Moderate degradation on holdout set")
        print("    Mix of bad data and potential BBH/lensing candidates")
    else:
        print("  ✗ Severe degradation on holdout set")
        print("    Many holdout SNe may be physical outliers (BBH/lensing)")
        print("    Consider using Student-t likelihood or GMM gating")

    return 0

if __name__ == '__main__':
    sys.exit(main())
