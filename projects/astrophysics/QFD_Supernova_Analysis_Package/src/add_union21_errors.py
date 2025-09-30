#!/usr/bin/env python3
"""
add_union21_errors.py

Add realistic uncertainties to Union2.1 dataset based on:
1. Systematic floor: ~0.13 mag from calibration/K-corrections
2. Distance-dependent scatter: increases with redshift
3. Low-z/high-z different error characteristics
"""

import numpy as np
import argparse

def add_realistic_errors(z: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Add realistic distance modulus uncertainties."""

    # Basic error model based on Union2.1 compilation
    # See Suzuki et al. 2012, ApJ 746, 85

    sigma_mu = np.zeros_like(z)

    # Low-z (z < 0.1): dominated by peculiar velocities and calibration
    low_z = z < 0.1
    sigma_mu[low_z] = np.sqrt(0.15**2 + (150 * z[low_z] / 3e5)**2)  # 150 km/s peculiar + calib

    # Intermediate-z (0.1 < z < 0.7): SDSS/SNLS quality
    med_z = (z >= 0.1) & (z < 0.7)
    sigma_mu[med_z] = np.sqrt(0.12**2 + (0.05 * z[med_z])**2)

    # High-z (z > 0.7): HST and ground-based, larger errors
    high_z = z >= 0.7
    sigma_mu[high_z] = np.sqrt(0.15**2 + (0.1 * (z[high_z] - 0.7))**2)

    # Add some realistic scatter
    np.random.seed(42)  # Reproducible
    scatter_factor = np.random.normal(1.0, 0.2, len(z))
    scatter_factor = np.clip(scatter_factor, 0.5, 2.0)

    sigma_mu *= scatter_factor

    # Minimum error floor
    sigma_mu = np.maximum(sigma_mu, 0.08)

    return sigma_mu

def main():
    parser = argparse.ArgumentParser(description="Add realistic errors to Union2.1 data")
    parser.add_argument("--input", default="union2.1_data.txt", help="Input file (z, mu)")
    parser.add_argument("--output", default="union2.1_data_with_errors.txt", help="Output file (z, mu, sigma_mu)")

    args = parser.parse_args()

    # Load data
    data = np.loadtxt(args.input)
    z = data[:, 0]
    mu = data[:, 1]

    print(f"Loaded {len(z)} supernovae from {args.input}")
    print(f"Redshift range: {z.min():.3f} - {z.max():.3f}")

    # Add errors
    sigma_mu = add_realistic_errors(z, mu)

    print(f"Error statistics:")
    print(f"  Mean: {np.mean(sigma_mu):.3f} mag")
    print(f"  Median: {np.median(sigma_mu):.3f} mag")
    print(f"  Range: {np.min(sigma_mu):.3f} - {np.max(sigma_mu):.3f} mag")

    # Save with errors
    output_data = np.column_stack([z, mu, sigma_mu])
    np.savetxt(args.output, output_data, fmt='%.6f %.8f %.6f',
               header='redshift distance_modulus sigma_mu')

    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()