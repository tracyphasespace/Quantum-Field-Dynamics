"""
V13 Data Loading

Based on V1's data handling, modernized with type hints.
Loads lightcurves and prepares them for emcee sampler.
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

# This is the JAX-compatible data structure for the optimizer
class Photometry(NamedTuple):
    mjd: jnp.ndarray
    flux: jnp.ndarray
    flux_err: jnp.ndarray
    wavelength: jnp.ndarray
    z: float

@dataclass
class SupernovaData:
    """Single supernova lightcurve data."""

    snid: str
    z: float

    # Observations
    mjd: np.ndarray  # Modified Julian Date
    flux_jy: np.ndarray  # Flux in Janskys
    flux_err_jy: np.ndarray  # Flux uncertainty
    wavelength_nm: np.ndarray  # Observed wavelength

    # Survey metadata
    survey: str = "UNKNOWN"

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.mjd)

    def to_photometry(self) -> Photometry:
        """Convert to JAX-ready Photometry object."""
        # Drop any rows with non-finite flux or error to avoid NaNs propagating
        mask = (
            np.isfinite(self.mjd)
            & np.isfinite(self.flux_jy)
            & np.isfinite(self.flux_err_jy)
            & np.isfinite(self.wavelength_nm)
        )
        mjd = self.mjd[mask]
        flux = self.flux_jy[mask]
        flux_err = self.flux_err_jy[mask]
        wavelength = self.wavelength_nm[mask]

        return Photometry(
            mjd=jnp.array(mjd),
            flux=jnp.array(flux),
            flux_err=jnp.array(flux_err),
            wavelength=jnp.array(wavelength),
            z=float(self.z),
        )


class LightcurveLoader:
    """Load supernova lightcurves from CSV file."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = None
        self._loaded_data = None
        self._snid_list_cache = None  # Cache for list of all SNIDs

    def get_snid_list(
        self,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> List[str]:
        """
        Get list of all SNIDs without loading full data.
        Memory-efficient: only reads minimal columns.
        """
        if self._snid_list_cache is not None:
            return self._snid_list_cache

        print(f"Reading SNID list from: {self.csv_path}")

        # Only load SNID and z columns to save memory
        df_minimal = pd.read_csv(self.csv_path, usecols=['snid', 'z'])

        # Clean SNID column - strip whitespace
        df_minimal['snid'] = df_minimal['snid'].astype(str).str.strip()

        # Get unique SNIDs with their redshift
        sn_redshifts = df_minimal.groupby('snid')['z'].first()

        # Apply z filters
        if z_min is not None:
            sn_redshifts = sn_redshifts[sn_redshifts >= z_min]
        if z_max is not None:
            sn_redshifts = sn_redshifts[sn_redshifts <= z_max]

        snid_list = [str(snid) for snid in sn_redshifts.index.values]
        print(f"  Found {len(snid_list)} SNe")

        self._snid_list_cache = snid_list
        return snid_list

    def load_batch(
        self,
        snid_list: List[str],
        batch_size: int = 500,
        batch_index: int = 0,
    ) -> Dict[str, SupernovaData]:
        """
        Load a batch of lightcurves from CSV.
        Memory-efficient: only loads specified SNIDs.

        Args:
            snid_list: List of all SNIDs to consider
            batch_size: Number of SNe per batch
            batch_index: Which batch to load (0-indexed)

        Returns:
            Dictionary mapping SNID to SupernovaData objects for this batch.
        """
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, len(snid_list))

        if start_idx >= len(snid_list):
            return {}

        batch_snids = snid_list[start_idx:end_idx]
        print(f"Loading batch {batch_index}: SNe {start_idx}-{end_idx-1} ({len(batch_snids)} SNe)")

        # Read full CSV but only keep rows for batch SNIDs
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.lower()

        # Clean SNID column - strip whitespace
        df['snid'] = df['snid'].astype(str).str.strip()

        # Map to standardized names
        col_mapping = {
            'fluxcal': 'flux_jy',
            'fluxcalerr': 'flux_err_jy',
            'flux_nu_jy': 'flux_jy',
            'flux_nu_jy_err': 'flux_err_jy',
            'flt': 'band',
            'redshift_final': 'z',
            'z_helio': 'z',
        }
        df.rename(columns=col_mapping, inplace=True)

        # Filter to only batch SNIDs - this drastically reduces memory
        df = df[df['snid'].astype(str).isin(batch_snids)]
        print(f"  Filtered to {len(df)} rows for this batch")

        # Load each SN in batch
        lightcurves = {}
        for snid in batch_snids:
            sn_df = df[df['snid'].astype(str) == snid].copy()

            if len(sn_df) < 3:
                continue

            try:
                # Compute errors from SNR if available
                if 'snr' in sn_df.columns:
                    flux_err = sn_df['flux_jy'].values / np.maximum(sn_df['snr'].values, 0.1)
                else:
                    flux_err = sn_df['flux_err_jy'].values

                flux_err_safe = np.clip(flux_err, 1e-6, None)

                # Convert μJy → Jy
                flux_jy_converted = sn_df['flux_jy'].values * 1.0e-6
                flux_err_jy_converted = flux_err_safe * 1.0e-6

                lc = SupernovaData(
                    snid=str(snid),
                    z=float(sn_df['z'].iloc[0]),
                    mjd=sn_df['mjd'].values,
                    flux_jy=flux_jy_converted,
                    flux_err_jy=flux_err_jy_converted,
                    wavelength_nm=self._get_wavelength(sn_df),
                    survey=sn_df.get('survey', pd.Series(['UNKNOWN'])).iloc[0],
                )

                lightcurves[str(snid)] = lc

            except Exception as e:
                print(f"  Error loading {snid}: {e}")
                continue

        print(f"  Successfully loaded: {len(lightcurves)} SNe in this batch")
        return lightcurves

    def load(
        self,
        n_sne: Optional[int] = None,
        start_sne: int = 0,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        require_peak: bool = True,
    ) -> Dict[str, SupernovaData]:
        """
        Load lightcurves from CSV.

        Returns dictionary mapping SNID to SupernovaData objects.
        Format matches V1's data structure for compatibility.

        WARNING: Loads ALL data into memory. Use load_batch() for large datasets.
        """
        if self._loaded_data:
            return self._loaded_data

        print(f"Loading lightcurves from: {self.csv_path}")

        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"  Total rows in CSV: {len(df)}")
        print(f"  WARNING: Loading all data into memory. Consider using load_batch() for large datasets.")

        # Normalize column names to handle V1 and V2 formats
        # V1: SNID, MJD, FLUXCAL, FLUXCALERR, FLT
        # V2 clean: snid, mjd, flux_nu_jy, flux_nu_jy_err, band, wavelength_eff_nm, z

        # Normalize to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Map to standardized names
        col_mapping = {
            'fluxcal': 'flux_jy',
            'fluxcalerr': 'flux_err_jy',
            'flux_nu_jy': 'flux_jy',
            'flux_nu_jy_err': 'flux_err_jy',
            'flt': 'band',
            'redshift_final': 'z',
            'z_helio': 'z',
        }
        df.rename(columns=col_mapping, inplace=True)

        # Basic validation
        required_cols = ['snid', 'mjd', 'flux_jy', 'flux_err_jy']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Validate redshift column
        if 'z' not in df.columns:
            raise ValueError("No redshift column found (need 'z')")

        # Get unique SNe
        unique_snids = df['snid'].unique()
        print(f"  Unique SNe: {len(unique_snids)}")

        # Apply filters
        if z_min is not None or z_max is not None:
            # Filter by redshift (per-SN, not per-obs)
            sn_redshifts = df.groupby('snid')['z'].first()

            if z_min is not None:
                sn_redshifts = sn_redshifts[sn_redshifts >= z_min]
            if z_max is not None:
                sn_redshifts = sn_redshifts[sn_redshifts <= z_max]

            unique_snids = sn_redshifts.index.values
            print(f"  After z filter [{z_min}, {z_max}]: {len(unique_snids)} SNe")

        # Select subset
        if start_sne > 0:
            unique_snids = unique_snids[start_sne:]
            print(f"  After start_sne={start_sne}: {len(unique_snids)} SNe")

        if n_sne is not None:
            unique_snids = unique_snids[:n_sne]
            print(f"  Limited to n_sne={n_sne}: {len(unique_snids)} SNe")

        # Load each SN
        lightcurves = {}
        for snid in unique_snids:
            sn_df = df[df['snid'] == snid].copy()

            # Basic quality checks
            if len(sn_df) < 3:
                print(f"  Skipping {snid}: Only {len(sn_df)} observations")
                continue

            # Extract data
            try:
                # BUGFIX: Compute real errors from SNR (CSV has placeholder 0.02)
                # flux_err = flux / SNR, with floor to prevent division by zero
                if 'snr' in sn_df.columns:
                    # Compute from SNR (preferred)
                    flux_err = sn_df['flux_jy'].values / np.maximum(sn_df['snr'].values, 0.1)
                else:
                    # Fall back to CSV error column if no SNR
                    flux_err = sn_df['flux_err_jy'].values

                # Floor errors to prevent division by zero/negative
                flux_err_safe = np.clip(flux_err, 1e-6, None)

                # FIX 2025-01-18: Convert DES data from Micro-Janskys (μJy) to Janskys (Jy)
                # The model works in physical units that output Jy (~10^-6 to 10^-3 Jy for distant SNe).
                # The raw DES data values (~10-1000) with column name 'flux_jy' are actually in μJy.
                # Evidence: z~0.4 SN at mag ~18-20 should be ~10^-4 Jy, not ~100 Jy (brighter than Vega!)
                flux_jy_converted = sn_df['flux_jy'].values * 1.0e-6  # μJy → Jy
                flux_err_jy_converted = flux_err_safe * 1.0e-6        # μJy → Jy

                lc = SupernovaData(
                    snid=str(snid),
                    z=float(sn_df['z'].iloc[0]),
                    mjd=sn_df['mjd'].values,
                    flux_jy=flux_jy_converted,
                    flux_err_jy=flux_err_jy_converted,
                    wavelength_nm=self._get_wavelength(sn_df),
                    survey=sn_df.get('survey', pd.Series(['UNKNOWN'])).iloc[0],
                )

                # BUGFIX: Key with string SNID to match SupernovaData.snid field type
                # This ensures Stage 1 directory names match lightcurve dict keys
                lightcurves[str(snid)] = lc

            except Exception as e:
                print(f"  Error loading {snid}: {e}")
                continue

        print(f"  Successfully loaded: {len(lightcurves)} SNe")

        if len(lightcurves) == 0:
            raise ValueError("No lightcurves loaded! Check filters and data quality.")
        
        self._loaded_data = lightcurves
        return lightcurves

    def get_all_photometry(self, **kwargs) -> Dict[str, Photometry]:
        """
        Loads all SN data and returns it in the JAX-compatible Photometry format.
        """
        all_sn_data = self.load(**kwargs)
        return {
            snid: data.to_photometry()
            for snid, data in all_sn_data.items()
        }

    def _get_wavelength(self, sn_df: pd.DataFrame) -> np.ndarray:
        """
        Get observed wavelength from data.

        V2 clean data has wavelength_eff_nm directly.
        V1 data requires filter -> wavelength mapping.
        """
        # If wavelength_eff_nm is available, use it (V2 clean data)
        if 'wavelength_eff_nm' in sn_df.columns:
            return sn_df['wavelength_eff_nm'].values

        # Otherwise, map from filter band (V1 data)
        if 'band' not in sn_df.columns:
            raise ValueError("No wavelength information found (need 'wavelength_eff_nm' or 'band')")

        # Simplified filter -> wavelength mapping
        filter_wavelengths = {
            'g': 475.0,  # g-band
            'r': 625.0,  # r-band
            'i': 775.0,  # i-band
            'z': 925.0,  # z-band
            'y': 1000.0, # Y-band
        }

        wavelengths = []
        for flt in sn_df['band']:
            # Extract first character (e.g., 'g' from 'g-SDSS')
            band = str(flt)[0].lower()
            wavelengths.append(filter_wavelengths.get(band, 625.0))  # Default to r-band

        return np.array(wavelengths)


def prepare_for_sampler(
    lightcurves: Dict[str, SupernovaData]
) -> Tuple[Dict[str, Dict], int]:
    """
    Prepare lightcurves for emcee sampler.

    Returns:
        - Dictionary of JAX-ready data (for GPU physics)
        - Number of parameters (for emcee initialization)
    """
    # Convert to JAX arrays
    jax_lightcurves = {
        snid: lc.to_jax()
        for snid, lc in lightcurves.items()
    }

    # V1 fitted 3 global parameters: k_J, eta_prime, xi
    n_params = 3

    return jax_lightcurves, n_params
