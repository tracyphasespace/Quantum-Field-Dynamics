#!/usr/bin/env python3
"""
Holdout Validation: Evaluate Excluded SNe

Loads the 637 excluded SNe (chi2 > 2000) and evaluates how well they fit
using the best-fit parameters from Model A (training set).

Generates comparison figure showing:
- Training set performance (4831 SNe)
- Holdout set performance (637 SNe)
- Residual distributions and trends

This addresses the concern: "we don't just throw out 6% of the data"
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import namedtuple

# Lightcurve structure
LightCurve = namedtuple('LightCurve', ['z', 'mjd', 'flux', 'flux_err', 'band'])


def load_lightcurves(lightcurves_file):
    """Load lightcurves from unified CSV."""
    df = pd.read_csv(lightcurves_file)

    lcs = {}
    for snid, group in df.groupby('snid'):
        lc = LightCurve(
            z=group['z'].iloc[0],
            mjd=group['mjd'].values,
            flux=group['flux_nu_jy'].values,
            flux_err=group['flux_nu_jy_err'].values,
            band=group['band'].values
        )
        lcs[str(snid)] = lc

    return lcs


def load_stage1_results_with_lightcurves(stage1_dir, lightcurves_dict):
    """Load all Stage 1 results with z and alpha_obs from lightcurves."""

    all_results = []

    for sn_dir in stage1_dir.iterdir():
        if not sn_dir.is_dir():
            continue

        snid = sn_dir.name
        metrics_file = sn_dir / 'metrics.json'
        persn_file = sn_dir / 'persn_best.npy'

        if not metrics_file.exists() or not persn_file.exists():
            continue

        if snid not in lightcurves_dict:
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Load persn_best
            persn_best = np.load(persn_file)

            # Get z from lightcurve
            lc = lightcurves_dict[snid]
            z = lc.z

            # Get alpha_obs from persn_best (order: t0, A_plasma, beta, alpha)
            alpha_obs = persn_best[3] if len(persn_best) == 4 else persn_best[-1]

            # Get chi2
            chi2 = metrics['chi2']
            n_obs = len(lc.mjd)
            chi2_per_obs = chi2 / n_obs if n_obs > 0 else chi2

            result = {
                'snid': snid,
                'z': z,
                'alpha_obs': alpha_obs,
                'chi2': chi2,
                'chi2_per_obs': chi2_per_obs,
                'iters': metrics['iters'],
                'n_obs': n_obs
            }

            all_results.append(result)

        except Exception as e:
            print(f"Warning: Could not load {snid}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Apply quality filter
    df['is_training'] = (df['chi2_per_obs'] < 2000) & (df['iters'] >= 5)

    training = df[df['is_training']].copy()
    holdout = df[~df['is_training']].copy()

    print(f"Total SNe: {len(df)}")
    print(f"Training set: {len(training)} ({100*len(training)/len(df):.1f}%)")
    print(f"Holdout set: {len(holdout)} ({100*len(holdout)/len(df):.1f}%)")
    print()

    return training, holdout


def compute_alpha_pred(z_array, k_J, eta_prime, xi, standardizer):
    """
    Compute α_pred(z) using the standardized basis.

    α(z) = α₀ - (k_J·φ₁ + η'·φ₂ + ξ·φ₃)

    where φᵢ are standardized basis functions.
    """

    # Raw basis functions
    phi1 = np.log(1 + z_array)
    phi2 = z_array
    phi3 = z_array / (1 + z_array)

    # Stack and standardize
    Phi_raw = np.column_stack([phi1, phi2, phi3])
    Phi = (Phi_raw - standardizer['means']) / standardizer['scales']

    # Convert physics params to standardized coefficients
    # Using the Jacobian transformation
    c0 = -k_J
    c1 = -eta_prime
    c2 = -xi

    # Compute α_pred(z) in standardized space
    # α(z) = α₀ + c·Φ (note: c absorbs the minus sign from physics convention)
    alpha_pred = np.dot(Phi, np.array([c0, c1, c2]))

    return alpha_pred


def evaluate_holdout(training_df, holdout_df, best_fit, standardizer):
    """Compute predictions and residuals for both sets."""

    # Extract best-fit parameters (use median for robust estimate)
    k_J = best_fit['k_J']
    eta_prime = best_fit['eta_prime']
    xi = best_fit['xi']
    alpha0 = best_fit['alpha0']

    print("Best-fit parameters (Model A):")
    print(f"  α₀        = {alpha0:.4f}")
    print(f"  k_J       = {k_J:.4f}")
    print(f"  η'        = {eta_prime:.4f}")
    print(f"  ξ         = {xi:.4f}")
    print()

    # Compute predictions for training set
    training_alpha_pred = compute_alpha_pred(
        training_df['z'].values,
        k_J, eta_prime, xi,
        standardizer
    )
    training_df['alpha_pred'] = alpha0 + training_alpha_pred
    training_df['residual'] = training_df['alpha_obs'] - training_df['alpha_pred']

    # Compute predictions for holdout set
    holdout_alpha_pred = compute_alpha_pred(
        holdout_df['z'].values,
        k_J, eta_prime, xi,
        standardizer
    )
    holdout_df['alpha_pred'] = alpha0 + holdout_alpha_pred
    holdout_df['residual'] = holdout_df['alpha_obs'] - holdout_df['alpha_pred']

    # Compute statistics
    training_stats = {
        'rms': np.std(training_df['residual']),
        'mean': np.mean(training_df['residual']),
        'median': np.median(training_df['residual']),
        'mad': np.median(np.abs(training_df['residual'] - np.median(training_df['residual']))),
        'n': len(training_df)
    }

    holdout_stats = {
        'rms': np.std(holdout_df['residual']),
        'mean': np.mean(holdout_df['residual']),
        'median': np.median(holdout_df['residual']),
        'mad': np.median(np.abs(holdout_df['residual'] - np.median(holdout_df['residual']))),
        'n': len(holdout_df)
    }

    print("Training Set Statistics:")
    print(f"  N         = {training_stats['n']}")
    print(f"  RMS       = {training_stats['rms']:.4f} mag")
    print(f"  Mean      = {training_stats['mean']:.4f} mag")
    print(f"  Median    = {training_stats['median']:.4f} mag")
    print(f"  MAD       = {training_stats['mad']:.4f} mag")
    print()

    print("Holdout Set Statistics:")
    print(f"  N         = {holdout_stats['n']}")
    print(f"  RMS       = {holdout_stats['rms']:.4f} mag")
    print(f"  Mean      = {holdout_stats['mean']:.4f} mag")
    print(f"  Median    = {holdout_stats['median']:.4f} mag")
    print(f"  MAD       = {holdout_stats['mad']:.4f} mag")
    print()

    delta_rms = holdout_stats['rms'] - training_stats['rms']
    print(f"ΔRMS (holdout - training): {delta_rms:+.4f} mag")

    if abs(delta_rms) <= 0.05:
        print("✅ SUCCESS: Holdout performance within 0.05 mag of training")
    else:
        print("⚠️  WARNING: Holdout performance degraded by > 0.05 mag")
    print()

    return training_stats, holdout_stats


def plot_holdout_validation(training_df, holdout_df, training_stats, holdout_stats, outfile):
    """Generate comprehensive holdout validation figure."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Color scheme
    color_train = '#1f77b4'  # Blue
    color_hold = '#ff7f0e'   # Orange

    # ==== Panel 1: Residuals vs Redshift ====
    ax = axes[0, 0]

    # Plot training set
    ax.scatter(training_df['z'], training_df['residual'],
               alpha=0.3, s=10, color=color_train, label='Training (N=4831)')

    # Plot holdout set
    ax.scatter(holdout_df['z'], holdout_df['residual'],
               alpha=0.5, s=20, color=color_hold, marker='^',
               edgecolors='black', linewidths=0.5,
               label='Holdout (N=637)')

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Residual α_obs - α_pred [mag]', fontsize=12)
    ax.set_title('Residuals vs Redshift', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    # ==== Panel 2: Residual Histograms ====
    ax = axes[0, 1]

    bins = np.linspace(-6, 6, 60)

    ax.hist(training_df['residual'], bins=bins, alpha=0.6, color=color_train,
            label=f'Training (RMS={training_stats["rms"]:.3f})', density=True)
    ax.hist(holdout_df['residual'], bins=bins, alpha=0.6, color=color_hold,
            label=f'Holdout (RMS={holdout_stats["rms"]:.3f})', density=True)

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual [mag]', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    # ==== Panel 3: Q-Q Plot ====
    ax = axes[0, 2]

    # Standardize residuals
    train_std = (training_df['residual'] - training_stats['median']) / training_stats['mad']
    hold_std = (holdout_df['residual'] - holdout_stats['median']) / holdout_stats['mad']

    # Compute quantiles
    quantiles = np.linspace(0.01, 0.99, 100)
    train_q = np.quantile(train_std, quantiles)
    hold_q = np.quantile(hold_std, quantiles)

    ax.plot(train_q, hold_q, 'o', color=color_hold, alpha=0.6, markersize=4)

    # 1:1 line
    lim = max(abs(train_q).max(), abs(hold_q).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='1:1')

    ax.set_xlabel('Training Quantiles (standardized)', fontsize=12)
    ax.set_ylabel('Holdout Quantiles (standardized)', fontsize=12)
    ax.set_title('Q-Q Plot (Holdout vs Training)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.axis('equal')

    # ==== Panel 4: Chi2 Distribution (Holdout) ====
    ax = axes[1, 0]

    ax.hist(holdout_df['chi2_per_obs'], bins=50, color=color_hold, alpha=0.7, edgecolor='black')
    ax.axvline(2000, color='red', linestyle='--', linewidth=2,
               label='Training cutoff (χ²=2000)')
    ax.set_xlabel('Chi-squared per observation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Holdout Set: Chi² Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10000)

    # ==== Panel 5: Residual vs Chi2 (Holdout) ====
    ax = axes[1, 1]

    ax.scatter(holdout_df['chi2_per_obs'], np.abs(holdout_df['residual']),
               alpha=0.6, s=20, color=color_hold)
    ax.axhline(holdout_stats['rms'], color='red', linestyle='--',
               label=f'RMS={holdout_stats["rms"]:.3f}')
    ax.set_xlabel('Chi-squared per observation', fontsize=12)
    ax.set_ylabel('|Residual| [mag]', fontsize=12)
    ax.set_title('Holdout: |Residual| vs Chi²', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 8)

    # ==== Panel 6: Summary Statistics Table ====
    ax = axes[1, 2]
    ax.axis('off')

    delta_rms = holdout_stats['rms'] - training_stats['rms']
    success = "✅ PASS" if abs(delta_rms) <= 0.05 else "⚠️ FAIL"

    table_text = f"""
HOLDOUT VALIDATION SUMMARY

Training Set (N={training_stats['n']}):
  RMS       {training_stats['rms']:.4f} mag
  Mean      {training_stats['mean']:.4f} mag
  Median    {training_stats['median']:.4f} mag
  MAD       {training_stats['mad']:.4f} mag

Holdout Set (N={holdout_stats['n']}):
  RMS       {holdout_stats['rms']:.4f} mag
  Mean      {holdout_stats['mean']:.4f} mag
  Median    {holdout_stats['median']:.4f} mag
  MAD       {holdout_stats['mad']:.4f} mag

Performance:
  ΔRMS      {delta_rms:+.4f} mag
  Status    {success}

Success Criterion: |ΔRMS| ≤ 0.05 mag

Interpretation:
The holdout set (excluded due to poor
Stage 1 fits) still has {holdout_stats['n']} SNe.
We evaluate them using the best-fit
parameters from the training set.

If ΔRMS is small, the model generalizes
well even to challenging data.
"""

    ax.text(0.1, 0.5, table_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    # Overall title
    fig.suptitle('Holdout Validation: External Validity Check (Model A)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {outfile}")

    return fig


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    stage1_dir = base_dir / "results" / "v15_production" / "stage1"
    stage2_dir = base_dir / "results" / "v15_production" / "stage2"
    figures_dir = base_dir / "results" / "v15_production" / "figures"
    lightcurves_file = base_dir / "data" / "lightcurves_unified_v2_min3.csv"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load best-fit parameters from Model A (production)
    with open(stage2_dir / "best_fit.json", 'r') as f:
        best_fit = json.load(f)

    with open(stage2_dir / "summary.json", 'r') as f:
        summary = json.load(f)

    standardizer = summary['meta']['standardizer']

    print("=" * 80)
    print("HOLDOUT VALIDATION: EXTERNAL VALIDITY CHECK")
    print("=" * 80)
    print()

    # Load lightcurves
    print("Loading lightcurves...")
    lightcurves_dict = load_lightcurves(lightcurves_file)
    print(f"  Loaded {len(lightcurves_dict)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    training_df, holdout_df = load_stage1_results_with_lightcurves(stage1_dir, lightcurves_dict)

    # Evaluate holdout set
    print("Evaluating holdout set with Model A parameters...")
    training_stats, holdout_stats = evaluate_holdout(
        training_df, holdout_df, best_fit, standardizer
    )

    # Generate figure
    print("Generating validation figure...")
    outfile = figures_dir / "holdout_validation.png"
    plot_holdout_validation(training_df, holdout_df, training_stats, holdout_stats, outfile)

    # Save detailed results
    results_file = figures_dir / "holdout_validation_results.json"

    # Convert numpy types to native python for JSON serialization
    def convert_to_native(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, (np.int64, np.int32)):
                result[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                result[k] = float(v)
            else:
                result[k] = v
        return result

    results = {
        'training_stats': convert_to_native(training_stats),
        'holdout_stats': convert_to_native(holdout_stats),
        'delta_rms': float(holdout_stats['rms'] - training_stats['rms']),
        'success': bool(abs(holdout_stats['rms'] - training_stats['rms']) <= 0.05),
        'best_fit': best_fit
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {results_file}")
    print()
    print("=" * 80)
    print("HOLDOUT VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
