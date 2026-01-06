#!/usr/bin/env python3
"""
Convert raw QFD stage2 results to distance modulus format - CORRECTED VERSION

CRITICAL FIX: Use ln_A (log amplitude) to calculate distance modulus, NOT residual!

Based on V21 plot_canonical_comparison.py line 148:
    mu_obs_uncal = -1.0857 * ln_A

Where -1.0857 = -2.5/ln(10) converts natural log to magnitude scale.

Input: stage2_results_with_redshift.csv (8,253 SNe from raw DES5yr processing)
Output: des5yr_raw_qfd_CORRECTED.csv (redshift, distance_modulus, sigma_mu)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_distance_modulus_from_ln_A(row):
    """
    Calculate distance modulus from QFD fit amplitude parameter.

    Physics:
    - ln_A is the natural log of the fitted light curve amplitude
    - Distance modulus μ = m - M = -2.5 log10(amplitude) + constant
    - Converting: μ = -2.5/ln(10) * ln(amplitude) + constant
    -           = -1.0857 * ln_A + M_calibration

    We use uncalibrated values and let the fit determine absolute calibration.
    """
    z = row['z']
    ln_A = row['ln_A']
    chi2_dof = row['chi2_dof']

    # Distance modulus (uncalibrated)
    # This is the CORRECT formula from V21
    mu_obs_uncal = -1.0857 * ln_A

    # Uncertainty estimate
    # For well-fit SNe: sigma ~ 0.15 mag
    # For poor fits: scale by sqrt(chi2)
    sigma_mu = 0.15 * np.sqrt(max(1.0, chi2_dof))

    return mu_obs_uncal, sigma_mu

def apply_calibration_offset(df):
    """
    Apply calibration offset to match low-z Hubble flow.

    Following V21 method:
    - Use SNe at z < 0.1 (Hubble flow)
    - Force median residual vs linear Hubble law to be zero
    - This calibrates absolute magnitude M
    """
    # Linear Hubble law for calibration
    c_km_s = 299792.458
    H0_calib = 70.0  # Nominal H0 for calibration

    # Low-z sample (column is still named 'z' at this point)
    low_z = df[df['z'] < 0.1].copy()

    if len(low_z) < 10:
        print("WARNING: Too few low-z SNe for calibration. Skipping offset correction.")
        return 0.0

    # Expected mu for linear Hubble law
    D_static = (c_km_s / H0_calib) * low_z['z']
    mu_linear = 5.0 * np.log10(D_static) + 25.0

    # Residuals
    residuals = low_z['mu_uncal'] - mu_linear

    # Robust estimator (MAD)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    clean_mask = np.abs(residuals - med) < 3 * mad

    # Calibration offset
    M_corr = -np.median(residuals[clean_mask])

    print(f"\nCalibration Statistics (z < 0.1):")
    print(f"  Low-z SNe: {len(low_z)}")
    print(f"  Median residual before calibration: {med:.3f} mag")
    print(f"  MAD: {mad:.3f} mag")
    print(f"  Clean fraction: {clean_mask.sum()}/{len(low_z)} ({100*clean_mask.sum()/len(low_z):.1f}%)")
    print(f"  Calibration offset M_corr: {M_corr:.3f} mag")

    return M_corr

def main():
    print("=" * 80)
    print("CORRECTED STAGE2 → DISTANCE MODULUS CONVERSION")
    print("=" * 80)
    print()

    # Input file from V21 analysis (raw QFD fits)
    input_file = Path("/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/data/stage2_results_with_redshift.csv")

    # Output file for V22 analysis
    output_file = Path("/home/tracy/development/QFD_SpectralGap/data/raw/des5yr_raw_qfd_CORRECTED.csv")

    print(f"Loading stage2 results from:\n  {input_file}")
    df = pd.read_csv(input_file)
    print(f"\nTotal SNe in stage2: {len(df)}")

    # Apply quality cuts
    print("\nApplying Quality Cuts:")
    print(f"  Initial: {len(df)} SNe")

    # Quality cuts (similar to V21)
    quality_mask = (
        (df['pass_n_obs'] == True) &      # Sufficient observations
        (df['pass_chi2'] == True) &       # Reasonable fit quality
        (df['z'] > 0.01) &                # Remove very nearby (peculiar velocities)
        (df['z'] < 1.5) &                 # Remove extreme high-z (uncertain)
        (~df['is_flashlight']) &          # Remove flashlight candidates
        (np.isfinite(df['ln_A'])) &       # Valid amplitude
        (np.isfinite(df['z']))            # Valid redshift
    )

    df_quality = df[quality_mask].copy()
    print(f"  After quality cuts: {len(df_quality)} SNe")
    print(f"  Removed: {len(df) - len(df_quality)} SNe")

    # Calculate distance modulus from ln_A (CORRECT METHOD)
    print("\nCalculating distance modulus from ln_A...")
    results = df_quality.apply(calculate_distance_modulus_from_ln_A, axis=1)
    df_quality['mu_uncal'] = [r[0] for r in results]
    df_quality['sigma_mu'] = [r[1] for r in results]

    # Apply calibration offset
    M_corr = apply_calibration_offset(df_quality)

    # Apply M_corr + MANUAL +5.0 mag offset (from V21 plot_canonical_comparison.py line 192)
    df_quality['distance_modulus'] = df_quality['mu_uncal'] + M_corr + 5.0

    print(f"Applied total offset: M_corr ({M_corr:.3f}) + manual (5.0) = {M_corr + 5.0:.3f} mag")

    # Create output DataFrame
    output_df = df_quality[['z', 'distance_modulus', 'sigma_mu']].copy()
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

    # Sanity check: residuals vs simple Hubble law
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

    # Show sample
    print("\n=== SAMPLE DATA ===")
    print(output_df.head(10))

    print("\n" + "="*80)
    print("STATUS: ✓ CONVERSION COMPLETE")
    print("="*80)
    print("\nNOTES:")
    print("  ✓ Used ln_A (log amplitude) NOT residual")
    print("  ✓ Applied calibration offset from low-z Hubble flow")
    print("  ✓ Quality cuts applied (n_obs, chi2, redshift range)")
    print("  ✓ NO SALT corrections, NO ΛCDM assumptions")
    print("\nThis data is ready for V22 QFD vs ΛCDM comparison!")

if __name__ == "__main__":
    main()
