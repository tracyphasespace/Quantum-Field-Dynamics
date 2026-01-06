#!/usr/bin/env python3
"""
Extract distance modulus from V21 stage2 results using EXACT V21 method.

This replicates plot_canonical_comparison.py processing:
1. mu_obs_uncal = -1.0857 * ln_A
2. Calculate M_corr from filtered low-z data
3. mu_obs = mu_obs_uncal + M_corr + 5.0 (manual offset)
4. Save ALL SNe with distance modulus

NO SALT corrections, NO cosmology assumptions - pure raw QFD processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Physical constants
C_KM_S = 299792.458  # km/s
H0 = 70.0  # km/s/Mpc (for calibration only)

def calculate_M_corr(df):
    """
    Calculate M_corr by forcing median residual at z<0.1 to be zero.
    Exact copy from plot_canonical_comparison.py lines 153-181.
    """
    print("\nCalculating M_corr for calibration...")

    # Select low-z SNe for calibration
    low_z = df[df['z'] < 0.1].copy()

    if len(low_z) == 0:
        print("  WARNING: No SNe with z<0.1 for M_corr calculation!")
        return 0.0
    else:
        # ROBUST ANCHORING
        mu_linear = 5.0 * np.log10((C_KM_S / H0) * low_z['z']) + 25.0
        residuals = low_z['mu_obs_uncal'] - mu_linear

        # Clip 3-sigma outliers to get a clean zero-point
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        clean_mask = np.abs(residuals - med) < 3 * mad

        M_corr = -np.median(residuals[clean_mask])

        print(f"  N(z<0.1) = {len(low_z)} for M_corr calculation")
        print(f"  Median residual: {med:.3f}")
        print(f"  Calculated M_corr = {M_corr:.3f}")
        return M_corr

def apply_calibration_offset(df, M_corr):
    """
    Apply the calculated M_corr and the manual 5.0 magnitude offset.
    Exact copy from plot_canonical_comparison.py lines 184-193.
    """
    df['mu_obs'] = df['mu_obs_uncal'] + M_corr

    # MANUAL CALIBRATION OFFSET (as per user's diagnosis)
    # This shifts the data points up by 5 magnitudes to align with the model predictions
    df['mu_obs'] = df['mu_obs'] + 5.0

    return df

def main():
    print("="*80)
    print("V21 DISTANCE MODULUS EXTRACTION (EXACT METHOD)")
    print("="*80)
    print()

    # Input: stage2 results from V21
    input_file = Path("/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/data/stage2_results_with_redshift.csv")

    # Output: distance modulus CSV for V22
    output_file = Path("/home/tracy/development/QFD_SpectralGap/data/raw/des5yr_v21_all_sne.csv")

    print(f"Loading stage2 results from:\n  {input_file}")
    raw_df = pd.read_csv(input_file)
    print(f"\nTotal SNe: {len(raw_df)}")
    print(f"SNe with redshift: {(~raw_df['z'].isna()).sum()}")

    # Step 1: Calculate uncalibrated distance modulus
    print("\nStep 1: Calculating mu_obs_uncal = -1.0857 * ln_A...")
    raw_df['mu_obs_uncal'] = -1.0857 * raw_df['ln_A']

    # Step 2: Create filtered sample for M_corr calculation
    print("\nStep 2: Filtering for M_corr calculation...")
    filtered_df = raw_df[
        (raw_df['stretch'] > 0.5) &
        (raw_df['stretch'] < 2.8) &
        (raw_df['z'] > 0.01) &
        (~raw_df['z'].isna())
    ].copy()
    print(f"Filtered SNe for calibration: {len(filtered_df)}")

    # Step 3: Calculate M_corr from filtered data
    M_corr = calculate_M_corr(filtered_df)

    # Step 4: Apply calibration to ALL data (not just filtered)
    print(f"\nStep 3: Applying calibration to ALL {len(raw_df)} SNe...")
    print(f"  Formula: mu_obs = mu_obs_uncal + {M_corr:.3f} + 5.0")
    raw_df_calibrated = apply_calibration_offset(raw_df.copy(), M_corr)

    # Step 5: Calculate uncertainties from chi2
    print("\nStep 4: Calculating uncertainties from chi2_dof...")
    raw_df_calibrated['sigma_mu'] = 0.15 * np.sqrt(np.maximum(1.0, raw_df_calibrated['chi2_dof']))

    # Step 6: Save output with all SNe that have valid redshift
    output_df = raw_df_calibrated[~raw_df_calibrated['z'].isna()].copy()
    output_df = output_df[['z', 'mu_obs', 'sigma_mu']].copy()
    output_df.columns = ['redshift', 'distance_modulus', 'sigma_mu']

    # Sort by redshift
    output_df = output_df.sort_values('redshift').reset_index(drop=True)

    # Save
    output_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Saved {len(output_df)} SNe to:\n  {output_file}")
    print(f"{'='*80}")

    # Statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"Redshift range: {output_df['redshift'].min():.4f} - {output_df['redshift'].max():.4f}")
    print(f"Distance modulus range: {output_df['distance_modulus'].min():.2f} - {output_df['distance_modulus'].max():.2f}")
    print(f"Mean sigma_mu: {output_df['sigma_mu'].mean():.3f}")
    print(f"Median sigma_mu: {output_df['sigma_mu'].median():.3f}")

    # Sanity check
    c_km_s = 299792.458
    H0_check = 70.0
    D_check = (c_km_s / H0_check) * output_df['redshift']
    mu_check = 5.0 * np.log10(D_check) + 25.0
    residuals_check = output_df['distance_modulus'] - mu_check

    print("\n=== SANITY CHECK (vs linear Hubble law, H0=70) ===")
    print(f"Mean residual: {residuals_check.mean():.3f} mag (should be ~0)")
    print(f"Std residual: {residuals_check.std():.3f} mag")
    print(f"Median residual: {residuals_check.median():.3f} mag")
    print(f"|Residual| > 2 mag: {(abs(residuals_check) > 2).sum()} SNe ({100*(abs(residuals_check) > 2).sum()/len(output_df):.1f}%)")

    # Sample
    print("\n=== SAMPLE DATA ===")
    print(output_df.head(20))

    print("\n" + "="*80)
    print("STATUS: ✓ EXTRACTION COMPLETE")
    print("="*80)
    print("\nThis data:")
    print("  ✓ Uses EXACT V21 processing method")
    print("  ✓ NO SALT corrections")
    print("  ✓ NO ΛCDM assumptions")
    print("  ✓ Pure raw QFD fits from DES5yr photometry")
    print(f"\nReady for V22 analysis!")

if __name__ == "__main__":
    main()
