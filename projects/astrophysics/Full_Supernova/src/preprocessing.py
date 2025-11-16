#!/usr/bin/env python3
"""
DES-SN5YR Raw Photometry Preprocessing

Extracts raw light curve photometry from SNANA FITS format and converts
to unified CSV format compatible with QFD pipeline.

Author: QFD Research Team
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


# Band effective wavelengths (nm) for DES filters
BAND_WAVELENGTHS = {
    'g': 475.0,
    'r': 635.0,
    'i': 780.0,
    'z': 915.0,
    'Y': 1000.0,
    'u': 365.0,  # not commonly used in DES SNe
}


def read_fits_header(fits_path: Path) -> Table:
    """
    Read SNANA HEAD FITS file containing SN metadata.

    Parameters
    ----------
    fits_path : Path
        Path to *_HEAD.FITS.gz file

    Returns
    -------
    Table
        Astropy Table with SN metadata (one row per SN)
    """
    if str(fits_path).endswith('.gz'):
        with gzip.open(fits_path, 'rb') as f:
            hdul = fits.open(f)
            header_table = Table(hdul[1].data)
            hdul.close()
    else:
        with fits.open(fits_path) as hdul:
            header_table = Table(hdul[1].data)

    return header_table


def read_fits_photometry(fits_path: Path) -> Table:
    """
    Read SNANA PHOT FITS file containing light curve data.

    Parameters
    ----------
    fits_path : Path
        Path to *_PHOT.FITS.gz file

    Returns
    -------
    Table
        Astropy Table with photometry (all SNe concatenated)
    """
    if str(fits_path).endswith('.gz'):
        with gzip.open(fits_path, 'rb') as f:
            hdul = fits.open(f)
            phot_table = Table(hdul[1].data)
            hdul.close()
    else:
        with fits.open(fits_path) as hdul:
            phot_table = Table(hdul[1].data)

    return phot_table


def fluxcal_to_jy(fluxcal: np.ndarray, zeropt: np.ndarray = None) -> np.ndarray:
    """
    Convert SNANA FLUXCAL to flux in Janskys.

    SNANA convention: mag = 27.5 - 2.5 * log10(FLUXCAL)
    AB magnitude: mag_AB = -2.5 * log10(f_nu / 3631 Jy)

    Therefore: f_nu [Jy] = 3631 * 10^(-0.4 * mag_AB)
                          = 3631 * 10^(-0.4 * (27.5 - 2.5 * log10(FLUXCAL)))
                          = 3631 * 10^(-11) * FLUXCAL

    Parameters
    ----------
    fluxcal : np.ndarray
        SNANA FLUXCAL values
    zeropt : np.ndarray, optional
        Custom zeropoint (default 27.5)

    Returns
    -------
    np.ndarray
        Flux in Janskys
    """
    if zeropt is None:
        zeropt = 27.5

    # Convert to AB magnitude
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mag_ab = zeropt - 2.5 * np.log10(np.abs(fluxcal))

    # Convert AB mag to Jy
    flux_jy = 3631.0 * 10**(-0.4 * mag_ab)

    # Handle negative FLUXCAL (preserve sign)
    flux_jy = np.where(fluxcal < 0, -flux_jy, flux_jy)

    return flux_jy


def extract_sn_photometry(
    snid: int,
    ptrobs_min: int,
    ptrobs_max: int,
    phot_table: Table,
    ra: float,
    dec: float,
    redshift: float,
    survey: str = "DES"
) -> pd.DataFrame:
    """
    Extract photometry for a single SN using pointers.

    Parameters
    ----------
    snid : int
        Supernova ID
    ptrobs_min : int
        Minimum pointer to PHOT table (1-indexed)
    ptrobs_max : int
        Maximum pointer to PHOT table (1-indexed)
    phot_table : Table
        Full photometry table
    ra : float
        Right ascension (deg)
    dec : float
        Declination (deg)
    redshift : float
        Redshift
    survey : str
        Survey name

    Returns
    -------
    pd.DataFrame
        Light curve data for this SN
    """
    # Convert 1-indexed pointers to 0-indexed Python
    idx_min = ptrobs_min - 1
    idx_max = ptrobs_max - 1

    # Extract rows for this SN
    sn_phot = phot_table[idx_min:idx_max+1]

    # Filter out end-of-lightcurve markers (MJD = -777)
    valid_mask = sn_phot['MJD'] > 0
    sn_phot = sn_phot[valid_mask]

    if len(sn_phot) == 0:
        return pd.DataFrame()

    # Convert FLUXCAL to Jy
    flux_jy = fluxcal_to_jy(sn_phot['FLUXCAL'].data)
    fluxerr_jy = fluxcal_to_jy(sn_phot['FLUXCALERR'].data)

    # Get band effective wavelengths
    bands = sn_phot['BAND'].data.astype(str)
    wavelengths = np.array([BAND_WAVELENGTHS.get(b.strip(), -999.0) for b in bands])

    # Calculate SNR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = np.abs(flux_jy) / fluxerr_jy
        snr = np.where(np.isfinite(snr), snr, 0.0)

    # Create DataFrame
    df = pd.DataFrame({
        'snid': snid,
        'mjd': sn_phot['MJD'].data,
        'band': bands,
        'flux_nu_jy': flux_jy,
        'flux_nu_jy_err': fluxerr_jy,
        'wavelength_eff_nm': wavelengths,
        'zeropoint': sn_phot['ZEROPT'].data,
        'psf_fwhm': sn_phot['PSF_SIG1'].data * 2.355,  # sigma to FWHM
        'sky_sig': sn_phot['SKY_SIG'].data,
        'ra': ra,
        'dec': dec,
        'z': redshift,
        'survey': survey,
        'snr': snr,
        'field': sn_phot['FIELD'].data.astype(str),
        'photflag': sn_phot['PHOTFLAG'].data,
    })

    return df


def filter_type_ia(
    header_table: Table,
    require_spec_confirmed: bool = False,
    include_photometric: bool = True
) -> np.ndarray:
    """
    Filter for Type Ia supernovae.

    Parameters
    ----------
    header_table : Table
        Header table with SNTYPE column
    require_spec_confirmed : bool
        If True, only include SNTYPE=1 (spec-confirmed Ia)
    include_photometric : bool
        If True, include SNTYPE=0 (likely photometric Ia)

    Returns
    -------
    np.ndarray
        Boolean mask for Type Ia SNe
    """
    sntype = header_table['SNTYPE'].data

    if require_spec_confirmed:
        # Only spectroscopically confirmed Type Ia
        mask = (sntype == 1)
    elif include_photometric:
        # Photometric + spec-confirmed Type Ia
        # SNTYPE=0 are unclassified but mostly Ia based on selection
        # SNTYPE=1 are spec-confirmed Ia
        # SNTYPE=4 are Ia-pec (peculiar Ia, still useful)
        mask = np.isin(sntype, [0, 1, 4])
    else:
        # Only spec-confirmed normal Ia
        mask = (sntype == 1)

    return mask


def filter_quality(
    header_table: Table,
    min_observations: int = 5,
    min_redshift: float = 0.01,
    max_redshift: float = 1.5,
    require_host_match: bool = False
) -> np.ndarray:
    """
    Apply quality cuts to SN sample.

    Parameters
    ----------
    header_table : Table
        Header table
    min_observations : int
        Minimum number of photometric observations
    min_redshift : float
        Minimum redshift (avoid peculiar velocity regime)
    max_redshift : float
        Maximum redshift (avoid high-z uncertainty)
    require_host_match : bool
        Require host galaxy match (HOSTGAL_NMATCH > 0)

    Returns
    -------
    np.ndarray
        Boolean mask for quality cuts
    """
    nobs = header_table['NOBS'].data
    z = header_table['REDSHIFT_FINAL'].data

    # Basic quality cuts
    mask = (nobs >= min_observations) & (z > min_redshift) & (z < max_redshift)

    # Optional host galaxy requirement
    if require_host_match and 'HOSTGAL_NMATCH' in header_table.colnames:
        nmatch = header_table['HOSTGAL_NMATCH'].data
        mask &= (nmatch > 0)

    return mask


def process_des_sn5yr_dataset(
    data_dir: Path,
    output_path: Path,
    dataset_name: str = "DES-SN5YR_DES",
    require_spec_confirmed: bool = False,
    include_photometric: bool = True,
    min_observations: int = 5,
    min_redshift: float = 0.05,
    max_redshift: float = 1.3,
    max_sne: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process DES-SN5YR dataset and create unified CSV.

    Parameters
    ----------
    data_dir : Path
        Path to 0_DATA/DES-SN5YR_XXX directory
    output_path : Path
        Output CSV path
    dataset_name : str
        Dataset subdirectory name
    require_spec_confirmed : bool
        Only include spec-confirmed Type Ia
    include_photometric : bool
        Include photometric Type Ia candidates
    min_observations : int
        Minimum observations per SN
    min_redshift : float
        Minimum redshift
    max_redshift : float
        Maximum redshift
    max_sne : int, optional
        Maximum number of SNe to process (for testing)
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Unified light curve dataframe
    """
    # Locate FITS files
    dataset_path = data_dir / dataset_name
    head_file = dataset_path / f"{dataset_name}_HEAD.FITS.gz"
    phot_file = dataset_path / f"{dataset_name}_PHOT.FITS.gz"

    if not head_file.exists():
        raise FileNotFoundError(f"HEAD file not found: {head_file}")
    if not phot_file.exists():
        raise FileNotFoundError(f"PHOT file not found: {phot_file}")

    if verbose:
        print(f"Reading {dataset_name}...")
        print(f"  HEAD: {head_file}")
        print(f"  PHOT: {phot_file}")

    # Read FITS files
    header_table = read_fits_header(head_file)
    phot_table = read_fits_photometry(phot_file)

    if verbose:
        print(f"  Total SNe in dataset: {len(header_table)}")
        print(f"  Total photometry rows: {len(phot_table)}")

    # Apply filters
    type_ia_mask = filter_type_ia(
        header_table,
        require_spec_confirmed=require_spec_confirmed,
        include_photometric=include_photometric
    )

    quality_mask = filter_quality(
        header_table,
        min_observations=min_observations,
        min_redshift=min_redshift,
        max_redshift=max_redshift
    )

    combined_mask = type_ia_mask & quality_mask

    if verbose:
        print(f"  Type Ia SNe: {type_ia_mask.sum()}")
        print(f"  Quality cuts pass: {quality_mask.sum()}")
        print(f"  Final selection: {combined_mask.sum()}")

    # Get selected SNe
    selected_header = header_table[combined_mask]

    if max_sne is not None:
        selected_header = selected_header[:max_sne]
        if verbose:
            print(f"  Limiting to first {max_sne} SNe for testing")

    # Extract photometry for each SN
    all_lightcurves = []

    for i, row in enumerate(selected_header):
        if verbose and (i % 500 == 0):
            print(f"  Processing SN {i+1}/{len(selected_header)}...")

        snid = int(row['SNID'])
        ptrobs_min = int(row['PTROBS_MIN'])
        ptrobs_max = int(row['PTROBS_MAX'])
        ra = float(row['RA'])
        dec = float(row['DEC'])
        redshift = float(row['REDSHIFT_FINAL'])

        lc = extract_sn_photometry(
            snid=snid,
            ptrobs_min=ptrobs_min,
            ptrobs_max=ptrobs_max,
            phot_table=phot_table,
            ra=ra,
            dec=dec,
            redshift=redshift,
            survey=dataset_name.split('_')[-1]  # Extract survey name
        )

        if len(lc) > 0:
            # Add additional metadata
            if 'HOSTGAL_LOGMASS' in row.colnames:
                lc['host_logmass'] = row['HOSTGAL_LOGMASS']
            if 'HOSTGAL_LOGSFR' in row.colnames:
                lc['host_logsfr'] = row['HOSTGAL_LOGSFR']
            if 'MWEBV' in row.colnames:
                lc['mwebv'] = row['MWEBV']

            all_lightcurves.append(lc)

    # Combine all light curves
    if len(all_lightcurves) == 0:
        raise ValueError("No light curves extracted!")

    df_combined = pd.concat(all_lightcurves, ignore_index=True)

    if verbose:
        print(f"\nFinal dataset:")
        print(f"  Unique SNe: {df_combined['snid'].nunique()}")
        print(f"  Total measurements: {len(df_combined)}")
        print(f"  Redshift range: {df_combined['z'].min():.3f} - {df_combined['z'].max():.3f}")
        print(f"  Bands: {sorted(df_combined['band'].unique())}")

    # Save to CSV
    df_combined.to_csv(output_path, index=False)

    if verbose:
        print(f"\nSaved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

    return df_combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract raw DES-SN5YR photometry")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to 0_DATA directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path")
    parser.add_argument("--dataset", default="DES-SN5YR_DES",
                        choices=["DES-SN5YR_DES", "DES-SN5YR_LOWZ", "DES-SN5YR_Foundation"],
                        help="Dataset to process")
    parser.add_argument("--spec-only", action="store_true",
                        help="Only include spectroscopically confirmed Type Ia")
    parser.add_argument("--min-obs", type=int, default=5,
                        help="Minimum observations")
    parser.add_argument("--min-z", type=float, default=0.05,
                        help="Minimum redshift")
    parser.add_argument("--max-z", type=float, default=1.3,
                        help="Maximum redshift")
    parser.add_argument("--max-sne", type=int, default=None,
                        help="Maximum SNe to process (for testing)")

    args = parser.parse_args()

    process_des_sn5yr_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        dataset_name=args.dataset,
        require_spec_confirmed=args.spec_only,
        include_photometric=not args.spec_only,
        min_observations=args.min_obs,
        min_redshift=args.min_z,
        max_redshift=args.max_z,
        max_sne=args.max_sne,
        verbose=True
    )
