#!/usr/bin/env python3
"""
Extract distance moduli from V21 Stage1 results using V21's EXACT method.

This replicates plot_canonical_comparison.py processing line-by-line:
1. mu_obs_uncal = -1.0857 * ln_A (NEGATIVE sign as in V21)
2. Calculate M_corr from low-z calibration
3. Apply M_corr + 5.0 manual offset
4. Filter: 0.5 < stretch < 2.8, z > 0.01

NO modifications - exact V21 code.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Constants (from V21)
C_KM_S = 299792.458
H0 = 70.0

def main():
    print("="*80)
    print("V21 EXACT DISTANCE MODULUS EXTRACTION")
    print("="*80)
    print()

    # Input: V21 Stage1 results
    input_file = Path("/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/data/stage2_results_with_redshift.csv")

    # Output
    output_file = Path("/home/tracy/development/QFD_SpectralGap/data/raw/des5yr_v21_exact.csv")

    print(f"Loading V21 Stage1 results from:\n  {input_file}")
    df = pd.read_csv(input_file)
    print(f"\nTotal SNe: {len(df)}")
    print(f"SNe with redshift: {(~df['z'].isna()).sum()}")

    # Apply V21's exact filters
    print("\nApplying V21 filters...")
    df_filtered = df[
        (df['stretch'] > 0.5) &
        (df['stretch'] < 2.8) &
        (df['z'] > 0.01) &
        (~df['z'].isna())
    ].copy()

    print(f"After filtering: {len(df_filtered)} SNe")
    print("  Criteria: 0.5 < stretch < 2.8 and z > 0.01")

    # Step 1: Calculate uncalibrated distance modulus (V21 line 148)
    print("\nStep 1: mu_obs_uncal = -1.0857 * ln_A")
    df_filtered['mu_obs_uncal'] = -1.0857 * df_filtered['ln_A']

    # Step 2: Calculate M_corr (V21 lines 153-181)
    print("\nStep 2: Calculating M_corr from low-z calibration...")
    low_z = df_filtered[df_filtered['z'] < 0.1].copy()

    if len(low_z) < 10:
        print("  WARNING: Too few low-z SNe!")
        M_corr = 0.0
    else:
        # Linear Hubble law for calibration
        mu_linear = 5.0 * np.log10((C_KM_S / H0) * low_z['z']) + 25.0
        residuals = low_z['mu_obs_uncal'] - mu_linear

        # Robust estimator (3-sigma clipping)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        clean_mask = np.abs(residuals - med) < 3 * mad

        M_corr = -np.median(residuals[clean_mask])

        print(f"  N(z<0.1) = {len(low_z)}")
        print(f"  Median residual: {med:.3f}")
        print(f"  M_corr = {M_corr:.3f}")

    # Step 3: Apply calibration (V21 lines 184-194)
    print(f"\nStep 3: Applying calibration...")
    print(f"  mu_obs = mu_obs_uncal + M_corr + 5.0")
    print(f"  mu_obs = mu_obs_uncal + {M_corr:.3f} + 5.0")

    df_filtered['mu_obs'] = df_filtered['mu_obs_uncal'] + M_corr + 5.0

    # Step 4: Estimate uncertainties
    print("\nStep 4: Estimating uncertainties from chi2_dof...")
    df_filtered['sigma_mu'] = 0.15 * np.sqrt(np.maximum(1.0, df_filtered['chi2_dof']))

    # Create output
    output_df = df_filtered[['z', 'mu_obs', 'sigma_mu']].copy()
    output_df.columns = ['redshift', 'distance_modulus', 'sigma_mu']
    output_df = output_df.sort_values('redshift').reset_index(drop=True)

    # Save
    output_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"✓ Saved {len(output_df)} SNe to:\n  {output_file}")
    print(f"{'='*80}")

    # Statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"SNe count: {len(output_df)}")
    print(f"Redshift range: {output_df['redshift'].min():.4f} - {output_df['redshift'].max():.4f}")
    print(f"Distance modulus range: {output_df['distance_modulus'].min():.2f} - {output_df['distance_modulus'].max():.2f}")
    print(f"Mean sigma_mu: {output_df['sigma_mu'].mean():.3f} mag")
    print(f"Median sigma_mu: {output_df['sigma_mu'].median():.3f} mag")

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
    print("\nThis data uses V21's EXACT processing:")
    print("  ✓ Negative sign: mu = -1.0857 * ln_A")
    print("  ✓ Low-z calibration (M_corr)")
    print("  ✓ Manual +5.0 mag offset")
    print("  ✓ Quality filters applied")
    print(f"\nReady for V22 QFD parameter fitting!")

if __name__ == "__main__":
    main()
