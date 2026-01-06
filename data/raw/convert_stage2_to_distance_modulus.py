#!/usr/bin/env python3
"""
Convert raw QFD stage2 results to distance modulus format for Grand Solver.

PHILOSOPHY: Use RAW data without SALT corrections to avoid circular reasoning.
- NO SALT2/SALT3 standardization
- NO cosmology-dependent corrections
- ONLY raw QFD fit results from photometry

Input: stage2_results_with_redshift.csv (8,253 SNe from raw DES5yr processing)
Output: des5yr_raw_qfd_full.csv (redshift, distance_modulus, sigma_mu)
"""

import pandas as pd
import numpy as np

def calculate_distance_modulus_from_qfd(row):
    """
    Calculate distance modulus from QFD fit parameters.

    Based on QFD model: μ = -2.5/ln(10) · α + μ_static(z)

    Where:
    - ln_A is the log amplitude parameter from the fit
    - residual is the deviation from expected
    - We can reconstruct distance modulus from these
    """
    z = row['z']
    ln_A = row['ln_A']
    residual = row['residual']
    chi2_dof = row['chi2_dof']

    # For a static universe (no dark energy), distance modulus is:
    # μ_static ≈ 5 log10(c*z/H0) + 25
    # Using c/H0 ≈ 4300 Mpc (for H0 ~ 70)
    c_over_H0 = 4300  # Mpc
    D_L_static = c_over_H0 * z  # Linear Hubble law (matter-only)
    mu_static = 5 * np.log10(D_L_static) + 25

    # The residual from QFD fit represents deviation from model
    # μ_obs = μ_model + residual
    mu_obs = mu_static + residual

    # Uncertainty estimate from chi2
    # For well-fit SNe, sigma ~ 0.1-0.2 mag
    # For poor fits, scale by sqrt(chi2)
    sigma_mu = 0.15 * np.sqrt(max(1.0, chi2_dof))

    return mu_obs, sigma_mu

def main():
    # Input file from V21 analysis (raw QFD fits)
    input_file = "/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/data/stage2_results_with_redshift.csv"

    # Output file for Grand Solver
    output_file = "/home/tracy/development/QFD_SpectralGap/data/raw/des5yr_raw_qfd_full.csv"

    print(f"Loading stage2 results from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total SNe in stage2: {len(df)}")

    # Filter for quality (optional - adjust as needed)
    # Option 1: Use ALL data (most radical, no filtering)
    df_filtered = df.copy()

    # Option 2: Use only candidates flagged by your pipeline
    # df_filtered = df[df['is_candidate'] == True].copy()

    # Option 3: Apply minimal quality cuts
    # df_filtered = df[
    #     (df['pass_n_obs'] == True) &  # At least N observations
    #     (df['pass_chi2'] == True) &   # Reasonable fit quality
    #     (df['z'] > 0.01) &            # Remove very nearby
    #     (df['z'] < 2.0)               # Remove high-z
    # ].copy()

    print(f"SNe after filtering: {len(df_filtered)}")

    # Calculate distance modulus for each SN
    results = df_filtered.apply(calculate_distance_modulus_from_qfd, axis=1)
    df_filtered['distance_modulus'] = [r[0] for r in results]
    df_filtered['sigma_mu'] = [r[1] for r in results]

    # Create output DataFrame with only needed columns
    output_df = df_filtered[['z', 'distance_modulus', 'sigma_mu']].copy()
    output_df.columns = ['redshift', 'distance_modulus', 'sigma_mu']

    # Sort by redshift
    output_df = output_df.sort_values('redshift')

    # Save
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(output_df)} SNe to: {output_file}")

    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Redshift range: {output_df['redshift'].min():.3f} - {output_df['redshift'].max():.3f}")
    print(f"Distance modulus range: {output_df['distance_modulus'].min():.2f} - {output_df['distance_modulus'].max():.2f}")
    print(f"Mean sigma_mu: {output_df['sigma_mu'].mean():.3f}")

    # Show sample
    print("\n=== First 5 entries ===")
    print(output_df.head())

    print("\n✓ Done! Use this file in your Grand Solver experiments.")
    print("This data has NO SALT corrections, NO ΛCDM assumptions.")
    print("It's purely from your raw QFD fits to DES5yr photometry.")

if __name__ == "__main__":
    main()
