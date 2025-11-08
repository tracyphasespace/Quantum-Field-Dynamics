#!/usr/bin/env python3
"""
Generate mock/demo results for testing MNRAS figure generation.

This creates realistic-looking results files without running the full pipeline.
Useful for testing figure scripts and layout before real analysis completes.

Usage:
    python generate_mock_results.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def create_directories():
    """Create results directory structure."""
    dirs = [
        'results/v15_production/stage1',
        'results/v15_production/stage2',
        'results/v15_production/stage3',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Created results directories")

def generate_mock_stage3_results():
    """Generate mock Stage 3 results (Hubble diagram data)."""
    print("Generating mock Stage 3 results...")

    # Generate realistic redshift distribution
    np.random.seed(42)
    n_sne = 500  # Smaller for demo

    # Redshift from 0.05 to 1.0 (typical DES range)
    z = np.random.beta(2, 5, n_sne) * 0.95 + 0.05

    # Mock distance modulus (ΛCDM-ish with scatter)
    def mu_lcdm(z, H0=70, Om=0.3):
        """Simplified ΛCDM distance modulus."""
        # Approximation for quick calculation
        D_L = 3000 * z * (1 + z/2)  # Simplified
        return 5 * np.log10(D_L) + 25

    mu_lcdm_true = mu_lcdm(z)

    # Add intrinsic scatter + QFD deviation
    sigma_int = 0.15
    scatter = np.random.normal(0, sigma_int, n_sne)

    # Mock QFD effect (small systematic deviation)
    qfd_effect = -0.05 * z  # Small dimming with z

    mu_obs = mu_lcdm_true + scatter + qfd_effect

    # Mock alpha_obs (from Stage 1)
    K = 2.5 / np.log(10)
    alpha_obs = -(scatter + qfd_effect) / K

    # Mock residuals
    # QFD model: predict alpha from z
    k_J = 10.5
    eta_prime = 0.008
    xi = 6.5

    def alpha_pred(z, k_J, eta_prime, xi):
        phi1 = np.log1p(z)
        phi2 = z
        phi3 = z / (1.0 + z)
        return -(k_J * phi1 + eta_prime * phi2 + xi * phi3)

    alpha_th = alpha_pred(z, k_J, eta_prime, xi)
    residual_alpha = alpha_obs - alpha_th
    residual_qfd = -K * residual_alpha

    # Create DataFrame
    df = pd.DataFrame({
        'snid': [f'SN{i:06d}' for i in range(n_sne)],
        'z': z,
        'mu_obs': mu_obs,
        'alpha_obs': alpha_obs,
        'alpha_th': alpha_th,
        'residual_alpha': residual_alpha,
        'residual_qfd': residual_qfd,
        'residual_lcdm': mu_obs - mu_lcdm_true,
    })

    # Save
    output_file = 'results/v15_production/stage3/stage3_results.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Created {output_file} ({len(df)} SNe)")

    # Also create hubble_data.csv (alternative name)
    df.to_csv('results/v15_production/stage3/hubble_data.csv', index=False)
    print(f"✓ Created hubble_data.csv")

    return df, k_J, eta_prime, xi

def generate_mock_stage2_results(k_J, eta_prime, xi):
    """Generate mock Stage 2 results (MCMC samples)."""
    print("Generating mock Stage 2 results...")

    # Mock MCMC samples
    np.random.seed(42)
    n_samples = 4000

    # Generate correlated samples (realistic posteriors)
    # Use multivariate normal with some correlation
    mean = [k_J, eta_prime, xi]
    cov = [
        [2.0, -0.02, -0.3],    # k_J variance and correlations
        [-0.02, 0.0001, -0.001],  # eta_prime
        [-0.3, -0.001, 1.5],   # xi
    ]

    samples = np.random.multivariate_normal(mean, cov, n_samples)

    # Best fit (mean of samples)
    best_fit = {
        'k_J': float(k_J),
        'eta_prime': float(eta_prime),
        'xi': float(xi),
    }

    # Save best fit
    with open('results/v15_production/stage2/best_fit.json', 'w') as f:
        json.dump(best_fit, f, indent=2)
    print("✓ Created best_fit.json")

    # Save samples (simplified format)
    samples_dict = {
        'k_J': samples[:, 0].tolist(),
        'eta_prime': samples[:, 1].tolist(),
        'xi': samples[:, 2].tolist(),
    }

    with open('results/v15_production/stage2/samples.json', 'w') as f:
        json.dump(samples_dict, f)
    print("✓ Created samples.json")

    # Save summary
    summary = {
        'n_samples': n_samples,
        'parameters': ['k_J', 'eta_prime', 'xi'],
        'mean': {
            'k_J': float(np.mean(samples[:, 0])),
            'eta_prime': float(np.mean(samples[:, 1])),
            'xi': float(np.mean(samples[:, 2])),
        },
        'std': {
            'k_J': float(np.std(samples[:, 0])),
            'eta_prime': float(np.std(samples[:, 1])),
            'xi': float(np.std(samples[:, 2])),
        }
    }

    with open('results/v15_production/stage2/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Created summary.json")

def main():
    print("=" * 60)
    print("Generating Mock Results for Figure Testing")
    print("=" * 60)
    print()

    # Create directories
    create_directories()
    print()

    # Generate Stage 3 results
    df, k_J, eta_prime, xi = generate_mock_stage3_results()
    print()

    # Generate Stage 2 results
    generate_mock_stage2_results(k_J, eta_prime, xi)
    print()

    # Summary
    print("=" * 60)
    print("Mock Results Summary")
    print("=" * 60)
    print(f"SNe count: {len(df)}")
    print(f"Redshift range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
    print(f"Best-fit parameters:")
    print(f"  k_J = {k_J:.3f}")
    print(f"  η' = {eta_prime:.6f}")
    print(f"  ξ = {xi:.3f}")
    print()
    print("Mock RMS residuals:")
    print(f"  QFD: {df['residual_qfd'].std():.3f} mag")
    print(f"  ΛCDM: {df['residual_lcdm'].std():.3f} mag")
    print()
    print("=" * 60)
    print("✓ Mock results generated successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  cd figures/")
    print("  make all")
    print()
    print("Note: These are MOCK results for testing only.")
    print("For publication, run the full pipeline on real data.")

if __name__ == '__main__':
    main()
