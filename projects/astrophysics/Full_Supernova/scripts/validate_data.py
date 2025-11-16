#!/usr/bin/env python3
"""
Validate extracted DES-SN5YR photometry data.

Performs quality checks on the extracted CSV file to ensure:
- Data completeness
- Valid ranges for all fields
- No missing critical values
- Consistency with QFD pipeline requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def validate_lightcurves(csv_path: Path, verbose: bool = True) -> bool:
    """
    Validate extracted light curve CSV.

    Parameters
    ----------
    csv_path : Path
        Path to lightcurves CSV
    verbose : bool
        Print detailed validation results

    Returns
    -------
    bool
        True if all validation checks pass
    """
    if verbose:
        print("=" * 70)
        print("DES-SN5YR Data Validation")
        print("=" * 70)
        print(f"File: {csv_path}")
        print(f"Size: {csv_path.stat().st_size / 1e6:.1f} MB")
        print()

    # Load data
    df = pd.read_csv(csv_path)

    if verbose:
        print(f"Total rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print()

    all_checks_passed = True

    # Check 1: Required columns
    required_cols = [
        'snid', 'mjd', 'band', 'flux_nu_jy', 'flux_nu_jy_err',
        'wavelength_eff_nm', 'z', 'ra', 'dec', 'survey'
    ]

    if verbose:
        print("CHECK 1: Required columns")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ❌ FAIL: Missing columns: {missing_cols}")
        all_checks_passed = False
    else:
        if verbose:
            print(f"  ✓ PASS: All {len(required_cols)} required columns present")

    # Check 2: No nulls in critical columns
    if verbose:
        print("\nCHECK 2: Missing values in critical columns")
    critical_cols = ['snid', 'mjd', 'band', 'flux_nu_jy', 'flux_nu_jy_err', 'z']
    nulls_found = False
    for col in critical_cols:
        if col in df.columns:
            n_nulls = df[col].isnull().sum()
            if n_nulls > 0:
                print(f"  ❌ FAIL: {col} has {n_nulls} null values")
                nulls_found = True
                all_checks_passed = False
    if not nulls_found and verbose:
        print("  ✓ PASS: No nulls in critical columns")

    # Check 3: Valid ranges
    if verbose:
        print("\nCHECK 3: Valid value ranges")

    checks = []

    # MJD should be reasonable (2010-2030)
    mjd_min, mjd_max = 55562, 62502  # Jan 1, 2011 to Dec 31, 2032
    mjd_valid = df['mjd'].between(mjd_min, mjd_max).all()
    checks.append(("MJD in [2011, 2032]", mjd_valid))

    # Redshift should be positive and < 2.0
    z_valid = df['z'].between(0.0, 2.0).all()
    checks.append(("Redshift in [0, 2]", z_valid))

    # RA/Dec should be valid coordinates
    ra_valid = df['ra'].between(-360, 360).all()
    dec_valid = df['dec'].between(-90, 90).all()
    checks.append(("RA in [-360, 360]", ra_valid))
    checks.append(("Dec in [-90, 90]", dec_valid))

    # Flux errors should be positive
    fluxerr_positive = (df['flux_nu_jy_err'] > 0).all()
    checks.append(("Flux errors > 0", fluxerr_positive))

    # Wavelengths should be valid for DES bands
    wavelength_valid = df['wavelength_eff_nm'].between(300, 1100).all()
    checks.append(("Wavelengths in [300, 1100] nm", wavelength_valid))

    for check_name, passed in checks:
        if passed:
            if verbose:
                print(f"  ✓ PASS: {check_name}")
        else:
            print(f"  ❌ FAIL: {check_name}")
            all_checks_passed = False

    # Check 4: Data completeness
    if verbose:
        print("\nCHECK 4: Data completeness")

    n_sne = df['snid'].nunique()
    n_measurements = len(df)
    avg_per_sn = n_measurements / n_sne
    n_bands = df['band'].nunique()

    if verbose:
        print(f"  Unique SNe: {n_sne:,}")
        print(f"  Total measurements: {n_measurements:,}")
        print(f"  Average per SN: {avg_per_sn:.1f}")
        print(f"  Unique bands: {n_bands}")

    # Each SN should have at least 5 measurements
    sn_counts = df.groupby('snid').size()
    min_count = sn_counts.min()
    if min_count >= 5:
        if verbose:
            print(f"  ✓ PASS: All SNe have ≥5 measurements (min={min_count})")
    else:
        print(f"  ❌ FAIL: Some SNe have <5 measurements (min={min_count})")
        all_checks_passed = False

    # Check 5: Redshift distribution
    if verbose:
        print("\nCHECK 5: Redshift distribution")

    z_bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
    z_counts = pd.cut(df.groupby('snid')['z'].first(), bins=z_bins).value_counts().sort_index()

    if verbose:
        for interval, count in z_counts.items():
            print(f"  {interval}: {count:,} SNe ({count/n_sne*100:.1f}%)")

    # Should have good coverage in 0.05 < z < 1.0
    z_050_100 = df.groupby('snid')['z'].first().between(0.05, 1.0).sum()
    if z_050_100 / n_sne >= 0.8:
        if verbose:
            print(f"  ✓ PASS: {z_050_100/n_sne*100:.1f}% of SNe in primary range [0.05, 1.0]")
    else:
        print(f"  ⚠ WARNING: Only {z_050_100/n_sne*100:.1f}% in primary range")

    # Check 6: Band coverage
    if verbose:
        print("\nCHECK 6: Band coverage")

    band_counts = df['band'].str.strip().value_counts()
    if verbose:
        for band, count in band_counts.items():
            print(f"  {band}: {count:,} measurements ({count/n_measurements*100:.1f}%)")

    # Should have all 4 main DES bands
    expected_bands = {'g', 'r', 'i', 'z'}
    found_bands = set(df['band'].str.strip().unique())
    if expected_bands.issubset(found_bands):
        if verbose:
            print(f"  ✓ PASS: All DES bands (g, r, i, z) present")
    else:
        missing = expected_bands - found_bands
        print(f"  ❌ FAIL: Missing DES bands: {missing}")
        all_checks_passed = False

    # Check 7: SNR distribution
    if verbose:
        print("\nCHECK 7: Signal-to-noise ratio")

    if 'snr' in df.columns:
        snr_median = df['snr'].median()
        snr_high = (df['snr'] > 5).sum() / len(df) * 100
        snr_low = (df['snr'] < 3).sum() / len(df) * 100

        if verbose:
            print(f"  Median SNR: {snr_median:.2f}")
            print(f"  High SNR (>5): {snr_high:.1f}% of measurements")
            print(f"  Low SNR (<3): {snr_low:.1f}% of measurements")

        if snr_median > 3:
            if verbose:
                print(f"  ✓ PASS: Median SNR is adequate ({snr_median:.2f})")
        else:
            print(f"  ⚠ WARNING: Low median SNR ({snr_median:.2f})")

    # Final summary
    if verbose:
        print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        if verbose:
            print("\nData is ready for QFD pipeline:")
            print("  python ../qfd-supernova-v15/src/stage1_optimize.py \\")
            print(f"      --lightcurves {csv_path} \\")
            print("      --out results/stage1_full \\")
            print("      --workers 8")
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("Please review the errors above before proceeding.")

    if verbose:
        print("=" * 70)

    return all_checks_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DES-SN5YR extracted data")
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("data/processed/lightcurves_unified_full.csv"),
        help="Path to lightcurves CSV (default: data/processed/lightcurves_unified_full.csv)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show pass/fail, not detailed output"
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: File not found: {args.csv_path}")
        exit(1)

    passed = validate_lightcurves(args.csv_path, verbose=not args.quiet)
    exit(0 if passed else 1)
