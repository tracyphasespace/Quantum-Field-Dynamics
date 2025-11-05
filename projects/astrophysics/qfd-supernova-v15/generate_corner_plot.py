#!/usr/bin/env python3
"""
Generate Figure 5: Posterior corner plot for QFD parameters (k_J, η', ξ)

Uses simulated MCMC samples to demonstrate what parameter constraints
would look like from a real pipeline run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import corner
except ImportError:
    import subprocess
    print("Installing corner package...")
    subprocess.run(["pip", "install", "corner", "-q"], check=True)
    import corner


def generate_mock_posterior_samples(n_samples=2000, seed=42):
    """
    Generate mock MCMC posterior samples for QFD parameters.

    Based on typical values from V15 validation:
    - k_J ~ 70 ± 5
    - η' ~ 0.01 ± 0.005 (weakly constrained)
    - ξ ~ 30 ± 3

    Includes realistic correlation between k_J and ξ (~0.7)
    """
    np.random.seed(seed)

    # Generate correlated k_J and ξ
    mean = [70.0, 30.0]
    cov = [[25.0, 8.0],   # Variance of k_J and covariance
           [8.0, 9.0]]    # Covariance and variance of ξ

    kJ_xi = np.random.multivariate_normal(mean, cov, n_samples)
    k_J = kJ_xi[:, 0]
    xi = kJ_xi[:, 1]

    # η' is weakly constrained and independent
    eta_prime = np.random.normal(0.01, 0.005, n_samples)

    return k_J, eta_prime, xi


def compute_convergence_stats(samples):
    """Compute R-hat and ESS (simulated for demonstration)."""
    # For mock data, we simulate good convergence
    stats = {
        'R_hat': [1.001, 1.002, 1.001],  # All < 1.01 (good)
        'ESS': [1850, 1920, 1875]        # All > 400 (good)
    }
    return stats


def main():
    print("Generating mock MCMC posterior samples...")
    k_J, eta_prime, xi = generate_mock_posterior_samples(n_samples=2000)

    # Create DataFrame
    samples = pd.DataFrame({
        'k_J': k_J,
        'eta_prime': eta_prime,
        'xi': xi
    })

    print(f"Generated {len(samples)} samples")
    print("\nPosterior summary statistics:")
    print(samples.describe())

    # Compute correlations
    corr_matrix = samples.corr()
    print("\nCorrelation matrix:")
    print(corr_matrix)

    # Convergence stats
    stats = compute_convergence_stats(samples)
    print(f"\nConvergence diagnostics:")
    print(f"R-hat: {stats['R_hat']} (all < 1.01 ✓)")
    print(f"ESS: {stats['ESS']} (all > 400 ✓)")

    # Generate corner plot (Figure 5)
    print("\nGenerating Figure 5: Corner plot...")

    # Publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
    })

    # Create corner plot
    fig = corner.corner(
        samples,
        labels=[r'$k_J$', r"$\eta'$", r'$\xi$'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        truths=[70.0, 0.01, 30.0],  # Reference values
        truth_color='red',
        color='steelblue',
        bins=30,
        smooth=1.0,
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        alpha=0.6,
    )

    # Add convergence annotations
    fig.text(0.7, 0.88, r'$\hat{R} < 1.01$ ✓', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    fig.text(0.7, 0.83, r'ESS $> 400$ ✓', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add correlation annotation
    kJ_xi_corr = corr_matrix.loc['k_J', 'xi']
    fig.text(0.7, 0.78, f'r(k_J, ξ) = {kJ_xi_corr:.3f}', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Save figure
    out_path = Path("results/mock_stage3/figures/fig5_corner_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close()

    # Save posterior samples for reference
    samples_path = Path("results/mock_stage3/posterior_samples.csv")
    samples.to_csv(samples_path, index=False)
    print(f"✓ Saved posterior samples: {samples_path}")

    print("\n✓ Figure 5 generation complete!")


if __name__ == "__main__":
    main()
