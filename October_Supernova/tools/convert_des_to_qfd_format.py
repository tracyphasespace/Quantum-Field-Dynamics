#!/usr/bin/env python3
"""
DES-SN 5YR Format Conversion to QFD CSV
Instance 2 (I2) - Data Infrastructure

Converts DES-SN 5YR data from FITS/SNANA format to QFD-native CSV format.

Input:  DES-SN 5YR raw data (FITS tables or SNANA format)
Output: QFD CSV (survey, snid, ra, dec, z, band, mjd, mag, flux_Jy, ...)

Usage:
    python tools/convert_des_to_qfd_format.py \
        --input data/des_sn5yr/raw/ \
        --output data/des_sn5yr/lightcurves_des_sn5yr.csv \
        --schema data/quality_gates_schema_v1.json

Author: Instance 2 (Data Infrastructure)
Date: 2025-11-01
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try importing astropy for FITS, fall back to manual parsing if unavailable
try:
    from astropy.io import fits
    from astropy.table import Table
    HAVE_ASTROPY = True
except ImportError:
    print("Warning: astropy not available. Will only support CSV/text formats.")
    HAVE_ASTROPY = False


# DES band effective wavelengths (nm)
# Source: DES filter curves (DES Data Release documentation)
BAND_TO_WAVELENGTH_DES = {
    'g': 475.0,  # DES g-band (4750 Å)
    'r': 635.0,  # DES r-band (6350 Å)
    'i': 775.0,  # DES i-band (7750 Å)
    'z': 925.0,  # DES z-band (9250 Å)
    'Y': 1000.0, # DES Y-band (10000 Å) - if present
}

# AB magnitude zeropoint (in Jy)
# AB system: m_AB = -2.5 log10(f_nu / 3631 Jy)
# → f_nu [Jy] = 3631 × 10^(-0.4 × m_AB)
AB_ZP_JY = 3631.0


def mag_to_flux_jy(mag: np.ndarray, mag_err: np.ndarray,
                    zp: float = 27.5, zpsys: str = 'AB') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert AB magnitudes to flux in Janskys.

    Parameters
    ----------
    mag : array
        AB magnitudes
    mag_err : array
        Magnitude uncertainties
    zp : float
        Magnitude zeropoint (default: 27.5 for natural AB)
    zpsys : str
        Zeropoint system ('AB' expected)

    Returns
    -------
    flux_jy : array
        Flux in Janskys
    flux_err_jy : array
        Flux uncertainty in Janskys

    Notes
    -----
    AB magnitude system:
        m_AB = -2.5 log10(f_nu / 3631 Jy)

    For instrumental magnitudes with zeropoint:
        m_inst + zp = m_AB
        → f_nu = 3631 × 10^(-0.4 × (m_inst + zp))

    For natural AB magnitudes (zp already applied):
        zp = 27.5 (standard convention)
        → f_nu = 3631 × 10^(-0.4 × m_AB)

    Uncertainty propagation:
        df/dm = -0.4 ln(10) × f
        → σ_f = 0.4 ln(10) × f × σ_m
    """
    if zpsys.upper() != 'AB':
        raise ValueError(f"Only AB magnitude system supported, got {zpsys}")

    # Convert magnitude to flux
    # If zp is standard AB (27.5), mag is already in AB system
    # Otherwise, mag is instrumental and needs zeropoint correction
    mag_ab = mag if zp == 27.5 else mag + zp - 27.5

    flux_jy = AB_ZP_JY * 10**(-0.4 * mag_ab)

    # Propagate uncertainty: σ_f = 0.4 ln(10) × f × σ_m
    flux_err_jy = 0.4 * np.log(10) * flux_jy * mag_err

    return flux_jy, flux_err_jy


def read_snana_format(filepath: Path) -> pd.DataFrame:
    """
    Read SNANA-format light curve file.

    SNANA format (simplified):
        SURVEY: DES
        SNID: 12345
        RA: 53.12345
        DEC: -27.98765
        REDSHIFT: 0.45 +- 0.01

        VARLIST: MJD FLT FIELD FLUXCAL FLUXCALERR MAG MAGERR
        OBS: 56500.123 g X3 1234.5 45.6 23.456 0.040
        OBS: 56501.234 r X3 2345.6 56.7 22.345 0.025
        ...

    Returns
    -------
    df : DataFrame
        Columns: survey, snid, ra, dec, z, band, mjd, mag, mag_err, flux_cal, flux_cal_err
    """
    metadata = {}
    observations = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse metadata
            if line.startswith('SURVEY:'):
                metadata['survey'] = line.split(':')[1].strip()
            elif line.startswith('SNID:'):
                metadata['snid'] = line.split(':')[1].strip()
            elif line.startswith('RA:'):
                metadata['ra'] = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('DEC:'):
                metadata['dec'] = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('REDSHIFT:'):
                parts = line.split(':')[1].strip().split()
                metadata['z'] = float(parts[0])

            # Parse observations
            elif line.startswith('OBS:'):
                parts = line.split()
                obs = {
                    'mjd': float(parts[1]),
                    'band': parts[2],
                    'field': parts[3],
                    'flux_cal': float(parts[4]) if parts[4] != 'NULL' else np.nan,
                    'flux_cal_err': float(parts[5]) if parts[5] != 'NULL' else np.nan,
                    'mag': float(parts[6]) if parts[6] != 'NULL' else np.nan,
                    'mag_err': float(parts[7]) if parts[7] != 'NULL' else np.nan,
                }
                observations.append(obs)

    if not observations:
        return pd.DataFrame()

    # Combine metadata with observations
    df = pd.DataFrame(observations)
    for key, value in metadata.items():
        df[key] = value

    return df


def read_fits_table(filepath: Path) -> pd.DataFrame:
    """
    Read DES-SN 5YR FITS table.

    Expected columns (may vary):
        SNID, RA, DEC, Z, MJD, BAND, MAG, MAGERR, FLUX, FLUXERR

    Returns
    -------
    df : DataFrame
        Standardized columns matching SNANA format
    """
    if not HAVE_ASTROPY:
        raise ImportError("astropy required to read FITS files. Install: pip install astropy")

    with fits.open(filepath) as hdul:
        # Try to find the light curve table
        # Typically in HDU 1, but check all
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                table = Table.read(hdu)
                df = table.to_pandas()

                # Standardize column names (case-insensitive mapping)
                col_map = {col.upper(): col for col in df.columns}

                # Rename columns to standard names
                rename_dict = {}
                if 'SNID' in col_map:
                    rename_dict[col_map['SNID']] = 'snid'
                if 'CID' in col_map:  # Alternative SN ID column
                    rename_dict[col_map['CID']] = 'snid'
                if 'RA' in col_map:
                    rename_dict[col_map['RA']] = 'ra'
                if 'DEC' in col_map or 'DECL' in col_map:
                    rename_dict[col_map.get('DEC', col_map.get('DECL'))] = 'dec'
                if 'Z' in col_map or 'REDSHIFT' in col_map:
                    rename_dict[col_map.get('Z', col_map.get('REDSHIFT'))] = 'z'
                if 'MJD' in col_map:
                    rename_dict[col_map['MJD']] = 'mjd'
                if 'BAND' in col_map or 'FLT' in col_map or 'FILTER' in col_map:
                    rename_dict[col_map.get('BAND', col_map.get('FLT', col_map.get('FILTER')))] = 'band'
                if 'MAG' in col_map:
                    rename_dict[col_map['MAG']] = 'mag'
                if 'MAGERR' in col_map or 'MAG_ERR' in col_map:
                    rename_dict[col_map.get('MAGERR', col_map.get('MAG_ERR'))] = 'mag_err'

                df.rename(columns=rename_dict, inplace=True)

                # Add survey column if missing
                if 'survey' not in df.columns:
                    df['survey'] = 'DES'

                return df

    raise ValueError(f"No valid table found in FITS file: {filepath}")


def convert_to_qfd_format(df: pd.DataFrame,
                          band_to_wavelength: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Convert DES data to QFD CSV format.

    Parameters
    ----------
    df : DataFrame
        Input data with columns: survey, snid, ra, dec, z, band, mjd, mag, mag_err
    band_to_wavelength : dict, optional
        Band -> effective wavelength mapping (nm)
        Default: BAND_TO_WAVELENGTH_DES

    Returns
    -------
    df_qfd : DataFrame
        QFD format with columns matching Pantheon+ schema
    """
    if band_to_wavelength is None:
        band_to_wavelength = BAND_TO_WAVELENGTH_DES

    # Make a copy
    df_qfd = df.copy()

    # Ensure required columns exist
    required = ['survey', 'snid', 'ra', 'dec', 'z', 'band', 'mjd', 'mag', 'mag_err']
    missing = [col for col in required if col not in df_qfd.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert magnitudes to flux (Jy)
    # Assume AB system with natural zeropoint (27.5)
    flux_jy, flux_err_jy = mag_to_flux_jy(
        df_qfd['mag'].values,
        df_qfd['mag_err'].values,
        zp=27.5,
        zpsys='AB'
    )

    df_qfd['flux_nu_jy'] = flux_jy
    df_qfd['flux_nu_jy_err'] = flux_err_jy

    # Add zeropoint columns
    df_qfd['zp'] = 27.5  # AB natural zeropoint
    df_qfd['zpsys'] = 'AB'

    # Map bands to effective wavelengths
    df_qfd['wavelength_eff_nm'] = df_qfd['band'].map(band_to_wavelength)

    # Check for unmapped bands
    unmapped = df_qfd[df_qfd['wavelength_eff_nm'].isna()]['band'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped bands (will be dropped): {unmapped}")
        df_qfd = df_qfd[df_qfd['wavelength_eff_nm'].notna()]

    # Add selection flag (all True initially, filtering happens later)
    df_qfd['__selected_winner__'] = True

    # Reorder columns to match Pantheon+ schema
    column_order = [
        'survey', 'snid', 'ra', 'dec', 'z', 'band', 'mjd',
        'mag', 'mag_err', 'flux_nu_jy', 'flux_nu_jy_err',
        'zp', 'zpsys', 'wavelength_eff_nm', '__selected_winner__'
    ]

    # Only keep columns that exist
    column_order = [col for col in column_order if col in df_qfd.columns]

    return df_qfd[column_order]


def main():
    parser = argparse.ArgumentParser(
        description="Convert DES-SN 5YR data to QFD CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single SNANA file
    python tools/convert_des_to_qfd_format.py \\
        --input data/des_sn5yr/raw/DES_SN12345.DAT \\
        --output data/des_sn5yr/lightcurves_des_sn5yr.csv \\
        --format snana

    # Convert all files in directory
    python tools/convert_des_to_qfd_format.py \\
        --input data/des_sn5yr/raw/ \\
        --output data/des_sn5yr/lightcurves_des_sn5yr.csv \\
        --format auto

    # Convert FITS table
    python tools/convert_des_to_qfd_format.py \\
        --input data/des_sn5yr/raw/des_sn5yr_photometry.fits \\
        --output data/des_sn5yr/lightcurves_des_sn5yr.csv \\
        --format fits
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input file or directory (FITS, SNANA, or CSV)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--format',
        choices=['auto', 'fits', 'snana', 'csv'],
        default='auto',
        help='Input format (default: auto-detect)'
    )
    parser.add_argument(
        '--schema',
        help='Quality gates schema JSON (for validation)'
    )
    parser.add_argument(
        '--band-mapping',
        help='JSON file with custom band->wavelength mapping'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load custom band mapping if provided
    band_to_wavelength = BAND_TO_WAVELENGTH_DES
    if args.band_mapping:
        with open(args.band_mapping, 'r') as f:
            band_to_wavelength = json.load(f)

    # Determine format
    fmt = args.format
    if fmt == 'auto':
        if input_path.suffix.lower() in ['.fits', '.fit']:
            fmt = 'fits'
        elif input_path.suffix.lower() in ['.dat', '.txt']:
            fmt = 'snana'
        elif input_path.suffix.lower() == '.csv':
            fmt = 'csv'
        else:
            print(f"Error: Cannot auto-detect format from {input_path.suffix}")
            print("Please specify --format explicitly")
            sys.exit(1)

    # Read data
    print(f"Reading {fmt.upper()} data from: {input_path}")

    all_data = []

    if input_path.is_dir():
        # Process all files in directory
        patterns = {
            'fits': ['*.fits', '*.fit'],
            'snana': ['*.DAT', '*.dat', '*.txt'],
            'csv': ['*.csv']
        }

        files = []
        for pattern in patterns.get(fmt, ['*']):
            files.extend(input_path.glob(pattern))

        if not files:
            print(f"Error: No {fmt} files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(files)} {fmt} files")

        for i, filepath in enumerate(files, 1):
            try:
                if fmt == 'fits':
                    df = read_fits_table(filepath)
                elif fmt == 'snana':
                    df = read_snana_format(filepath)
                elif fmt == 'csv':
                    df = pd.read_csv(filepath)
                else:
                    raise ValueError(f"Unknown format: {fmt}")

                all_data.append(df)

                if i % 100 == 0:
                    print(f"  Processed {i}/{len(files)} files...")

            except Exception as e:
                print(f"  Warning: Failed to read {filepath.name}: {e}")
                continue

        if not all_data:
            print("Error: No data successfully loaded")
            sys.exit(1)

        df_raw = pd.concat(all_data, ignore_index=True)

    else:
        # Single file
        if fmt == 'fits':
            df_raw = read_fits_table(input_path)
        elif fmt == 'snana':
            df_raw = read_snana_format(input_path)
        elif fmt == 'csv':
            df_raw = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    print(f"Loaded {len(df_raw)} observations from {df_raw['snid'].nunique()} SNe")

    # Convert to QFD format
    print("Converting to QFD format...")
    df_qfd = convert_to_qfd_format(df_raw, band_to_wavelength)

    # Basic validation
    print("\nData summary:")
    print(f"  SNe: {df_qfd['snid'].nunique()}")
    print(f"  Observations: {len(df_qfd)}")
    print(f"  Redshift range: z = {df_qfd['z'].min():.3f} - {df_qfd['z'].max():.3f}")
    print(f"  Bands: {sorted(df_qfd['band'].unique())}")
    print(f"  MJD range: {df_qfd['mjd'].min():.1f} - {df_qfd['mjd'].max():.1f}")

    # Check for missing data
    n_missing_flux = df_qfd['flux_nu_jy'].isna().sum()
    if n_missing_flux > 0:
        print(f"\n  Warning: {n_missing_flux} observations with missing flux (will be dropped)")
        df_qfd = df_qfd[df_qfd['flux_nu_jy'].notna()]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_qfd.to_csv(output_path, index=False, float_format='%.10g')

    print(f"\nSaved QFD CSV to: {output_path}")
    print(f"Final: {len(df_qfd)} observations from {df_qfd['snid'].nunique()} SNe")

    # Validate against schema if provided
    if args.schema:
        with open(args.schema, 'r') as f:
            schema = json.load(f)

        print(f"\nValidating against schema: {args.schema}")

        # Check required columns
        required_cols = schema['implementation_notes']['data_format_requirements']['required_for_qfd']
        missing_cols = [col for col in required_cols if col not in df_qfd.columns]

        if missing_cols:
            print(f"  ❌ Missing required columns: {missing_cols}")
        else:
            print(f"  ✅ All required columns present")

        # Quick quality check
        gates = schema['filter_categories']['standard_community_filters']['gates']
        z_min = gates['z_min']['value']
        z_max = gates['z_max']['value']

        n_below_z = (df_qfd['z'] < z_min).sum()
        n_above_z = (df_qfd['z'] > z_max).sum()

        if n_below_z > 0:
            print(f"  ⚠️  {n_below_z} obs below z_min ({z_min}) - will be filtered")
        if n_above_z > 0:
            print(f"  ⚠️  {n_above_z} obs above z_max ({z_max}) - will be filtered")

    print("\n✅ Conversion complete!")
    print("\nNext steps:")
    print(f"  1. Review: less {output_path}")
    print(f"  2. Apply quality gates: python tools/apply_quality_gates.py --input {output_path}")
    print(f"  3. Signal ready: touch {output_path.parent}/.READY_FOR_FIT")


if __name__ == '__main__':
    main()
