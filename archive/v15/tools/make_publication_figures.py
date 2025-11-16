#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for V15 QFD Pipeline

Generates standardized figures for publication:
- Fig 4: Hubble diagram (μ-space visualization)
- Fig 6: Per-survey residuals
- Fig 8: Out-of-sample performance

Style: Consistent fonts, colors, sizes for publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import argparse

# Publication style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

K = 2.5 / np.log(10.0)  # α → μ conversion

# Survey color palette (consistent across figures)
SURVEY_COLORS = {
    'Pantheon+': '#1f77b4',
    'HST': '#ff7f0e',
    'DES': '#2ca02c',
    'SDSS': '#d62728',
    'PS1': '#9467bd',
    'SNLS': '#8c564b',
    'CSP': '#e377c2',
    'CfA': '#7f7f7f',
}


def figure4_hubble_diagram(df, out_path):
    """
    Figure 4: Hubble Diagram with residuals

    Top: μ_obs vs z with QFD and ΛCDM curves
    Bottom: Residuals vs z with binned statistics
    """
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Top panel: Hubble diagram
    for survey, group in df.groupby('survey'):
        color = SURVEY_COLORS.get(survey, 'gray')
        ax1.scatter(group['z'], group['mu_obs'],
                   alpha=0.5, s=20, c=color, label=survey, edgecolors='none')

    # QFD curve (if available)
    if 'mu_qfd' in df.columns:
        z_curve = np.linspace(df['z'].min(), df['z'].max(), 200)
        # Interpolate or plot actual model predictions
        ax1.plot(df['z'].values, df['mu_qfd'].values,
                'b-', linewidth=2, label='QFD', alpha=0.8, zorder=10)

    # ΛCDM curve (if available)
    if 'mu_lcdm' in df.columns:
        ax1.plot(df['z'].values, df['mu_lcdm'].values,
                'r--', linewidth=2, label='ΛCDM', alpha=0.8, zorder=10)

    ax1.set_ylabel('μ (mag)', fontsize=12)
    ax1.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax1.set_title('Hubble Diagram: QFD vs ΛCDM', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Bottom panel: Residuals
    residual_col = 'residual_mu' if 'residual_mu' in df.columns else 'residual_alpha'
    ylabel = r'$r_\mu$ (mag)' if residual_col == 'residual_mu' else r'$r_\alpha$'

    for survey, group in df.groupby('survey'):
        color = SURVEY_COLORS.get(survey, 'gray')
        ax2.scatter(group['z'], group[residual_col],
                   alpha=0.5, s=20, c=color, edgecolors='none')

    # Binned statistics
    z_bins = np.arange(0.0, df['z'].max() + 0.1, 0.05)
    z_centers = []
    medians = []
    lo68 = []
    hi68 = []

    for i in range(len(z_bins) - 1):
        mask = (df['z'] >= z_bins[i]) & (df['z'] < z_bins[i+1])
        if mask.sum() > 5:
            z_centers.append(0.5 * (z_bins[i] + z_bins[i+1]))
            resid_bin = df[mask][residual_col].values
            medians.append(np.median(resid_bin))
            lo68.append(np.percentile(resid_bin, 16))
            hi68.append(np.percentile(resid_bin, 84))

    if z_centers:
        ax2.plot(z_centers, medians, 'k-', linewidth=2, label='Binned median', zorder=10)
        ax2.fill_between(z_centers, lo68, hi68, alpha=0.3, color='gray', label='68% CI')

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved: {out_path}")
    plt.close()


def figure6_per_survey_residuals(df, report_dir, out_path):
    """
    Figure 6: Per-survey residual diagnostics

    (a) Box plots by survey
    (b) Z-binned means by survey
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    residual_col = 'residual_mu' if 'residual_mu' in df.columns else 'residual_alpha'
    ylabel = r'$r_\mu$ (mag)' if residual_col == 'residual_mu' else r'$r_\alpha$'

    # Panel (a): Box plots
    surveys = sorted(df['survey'].unique())
    box_data = [df[df['survey'] == s][residual_col].values for s in surveys]
    colors = [SURVEY_COLORS.get(s, 'gray') for s in surveys]

    bp = ax1.boxplot(box_data, labels=surveys, patch_artist=True,
                     showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('(a) Per-Survey Distribution', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Panel (b): Z-binned means
    zbin_file = Path(report_dir) / "zbin_alpha_by_survey.csv"
    if zbin_file.exists():
        zbin_df = pd.read_csv(zbin_file)

        for survey in surveys:
            s_data = zbin_df[zbin_df['survey'] == survey]
            if len(s_data) > 0:
                z_mid = 0.5 * (s_data['z_lo'] + s_data['z_hi'])
                color = SURVEY_COLORS.get(survey, 'gray')

                # Plot mean with error bars
                ax2.errorbar(z_mid, s_data['mean'], yerr=s_data['std'],
                           fmt='o-', color=color, label=survey,
                           alpha=0.7, capsize=3, markersize=5)
    else:
        print(f"Warning: {zbin_file} not found, skipping panel (b)")

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title('(b) Z-Binned Means ± 1σ', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved: {out_path}")
    plt.close()


def figure8_holdout_performance(train_results, test_results, out_path):
    """
    Figure 8: Out-of-sample (hold-out) performance

    Bar plot: RMS by survey for train vs test
    Inset: Parity plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    surveys = sorted(train_results['survey'].unique())
    x = np.arange(len(surveys))
    width = 0.35

    train_rms = [train_results[train_results['survey'] == s]['rms_alpha'].values[0]
                 for s in surveys]
    test_rms = [test_results[test_results['survey'] == s]['rms_alpha'].values[0]
                for s in surveys]

    colors_train = [SURVEY_COLORS.get(s, 'gray') for s in surveys]
    colors_test = [c for c in colors_train]  # Same colors, different alpha

    ax.bar(x - width/2, train_rms, width, label='Train', alpha=0.8, color=colors_train)
    ax.bar(x + width/2, test_rms, width, label='Test', alpha=0.5, color=colors_test)

    ax.set_xlabel('Survey', fontsize=12)
    ax.set_ylabel(r'RMS($r_\alpha$)', fontsize=12)
    ax.set_title('Out-of-Sample Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(surveys, rotation=45)
    ax.legend()

    # Inset: Parity plot
    ax_inset = fig.add_axes([0.65, 0.65, 0.25, 0.25])
    ax_inset.scatter(train_rms, test_rms, s=50, alpha=0.7,
                    c=[SURVEY_COLORS.get(s, 'gray') for s in surveys])

    # 1:1 line
    lims = [min(train_rms + test_rms) * 0.95, max(train_rms + test_rms) * 1.05]
    ax_inset.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

    ax_inset.set_xlabel('Train RMS', fontsize=9)
    ax_inset.set_ylabel('Test RMS', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_title('Parity', fontsize=9)

    plt.savefig(out_path, dpi=300)
    print(f"✓ Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for V15 pipeline"
    )
    parser.add_argument("--stage3-csv", required=True,
                       help="Path to Stage 3 results CSV")
    parser.add_argument("--report-dir", required=True,
                       help="Directory containing per-survey reports")
    parser.add_argument("--out-dir", default="results/v15_production/figures",
                       help="Output directory for figures")
    parser.add_argument("--train-csv", help="Training set results (for Fig 8)")
    parser.add_argument("--test-csv", help="Test set results (for Fig 8)")

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading Stage 3 data from {args.stage3_csv}")
    df = pd.read_csv(args.stage3_csv)

    # Add μ-space residuals if needed
    if 'residual_mu' not in df.columns and 'residual_alpha' in df.columns:
        df['residual_mu'] = -K * df['residual_alpha']

    # Generate figures
    print("\nGenerating publication figures...")

    # Figure 4: Hubble diagram
    figure4_hubble_diagram(df, out / "fig4_hubble_diagram.png")

    # Figure 6: Per-survey residuals
    figure6_per_survey_residuals(df, args.report_dir, out / "fig6_per_survey_residuals.png")

    # Figure 8: Hold-out performance (if train/test provided)
    if args.train_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        figure8_holdout_performance(train_df, test_df, out / "fig8_holdout_performance.png")
    else:
        print("Skipping Figure 8 (no train/test data provided)")

    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {out.absolute()}")
    print("\nGenerated files:")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
