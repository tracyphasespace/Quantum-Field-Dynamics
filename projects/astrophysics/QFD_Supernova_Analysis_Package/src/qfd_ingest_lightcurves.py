#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qfd_ingest_lightcurves.py
Download & normalize raw supernova light curves for QFD time–flux fitting.

Sources:
  - osc  : Open Supernova Catalog via astroquery.oac.OAC
  - sdss : SDSS-II Sako+ (2018) via sndata.sdss.Sako18
  - des  : DES-SN3YR via sndata.des.SN3YR
  - csp  : Carnegie Supernova Project DR3 via sndata.csp.DR3

Outputs:
  <outdir>/lightcurves_<source>.parquet  and  .csv

Schema (one row per epoch):
  source,snid,ra,dec,z_helio,band,mjd,mag,mag_err,flux_nu_jy,flux_nu_jy_err,zp,zpsys,wavelength_eff_nm
"""
from __future__ import annotations

import os, sys, math, argparse, logging
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants

AB_ZP_JY = 3631.0
SNANA_ZP = 27.5  # Standard SNANA zero point for AB magnitudes

WL_EFF_NM = {
    # SDSS
    "u": 355.1, "g": 468.6, "r": 616.5, "i": 748.1, "z": 893.1,
    # SDSS Sako18 specific bands
    "sdss_sako18_u3": 355.1, "sdss_sako18_u4": 355.1, "sdss_sako18_u6": 355.1,
    "sdss_sako18_g3": 468.6, "sdss_sako18_g4": 468.6, "sdss_sako18_g6": 468.6,
    "sdss_sako18_r3": 616.5, "sdss_sako18_r4": 616.5, "sdss_sako18_r6": 616.5,
    "sdss_sako18_i3": 748.1, "sdss_sako18_i4": 748.1, "sdss_sako18_i6": 748.1,
    "sdss_sako18_z3": 893.1, "sdss_sako18_z4": 893.1, "sdss_sako18_z6": 893.1,
    # DES
    "g_des": 477.0, "r_des": 641.0, "i_des": 783.0, "z_des": 917.0,
    # DES SN3YR specific bands
    "des_sn3yr_g": 477.0, "des_sn3yr_r": 641.0, "des_sn3yr_i": 783.0, "des_sn3yr_z": 917.0,
    # CSP (optical)
    "b": 438.0, "v": 545.0, "r": 623.0, "i": 763.0,
    # PS1 (future)
    "g_ps1": 481.0, "r_ps1": 617.0, "i_ps1": 752.0, "z_ps1": 866.0, "y_ps1": 962.0,
    # Common fallbacks
    "clear": 550.0, "white": 550.0, "unfiltered": 550.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers

def to_float(x, default: float = np.nan) -> float:
    """Robust float coercion (str, None, arrays) -> float or np.nan."""
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        # common string 'nan', 'NaN'
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() == "nan":
                return default
        result = float(x)
        # Check for problematic values (rate-limited for large datasets)
        if not np.isfinite(result):
            logging.debug(f"Non-finite value encountered: {x} -> {result}")
        return result
    except (ValueError, TypeError) as e:
        logging.debug(f"Failed to convert {x} to float: {e}")
        return default
    except Exception as e:
        logging.debug(f"Unexpected error converting {x} to float: {e}")
        return default

def ab_mag_to_flux_jy(mag: float, mag_err: Optional[float]) -> Tuple[float, float]:
    if not np.isfinite(mag):
        return (np.nan, np.nan)
    f = AB_ZP_JY * 10 ** (-0.4 * float(mag))
    if mag_err is None or (not np.isfinite(mag_err)):
        return (f, np.nan)
    df = 0.4 * math.log(10) * f * float(mag_err)
    return (f, df)

def nmgy_to_mag(nmgy: float, nmgy_err: Optional[float] = None) -> Tuple[float, float]:
    if nmgy is None or (not np.isfinite(nmgy)) or nmgy <= 0:
        return (np.nan, np.nan)
    mag = 22.5 - 2.5 * math.log10(float(nmgy))
    if nmgy_err is None or (not np.isfinite(nmgy_err)) or nmgy_err <= 0:
        return (mag, np.nan)
    dmag = (2.5 / math.log(10)) * (float(nmgy_err) / float(nmgy))
    return (mag, dmag)

def wl_eff_nm_for(band: str, source: str) -> float:
    """Get effective wavelength for a band, with fallback handling."""
    b = (band or "").strip().lower()

    # Try source-specific mapping first
    if source == "des":
        wl = WL_EFF_NM.get(f"{b}_des", WL_EFF_NM.get(b, None))
    elif source == "ps1":
        wl = WL_EFF_NM.get(f"{b}_ps1", WL_EFF_NM.get(b, None))
    else:
        wl = WL_EFF_NM.get(b, None)

    if wl is None:
        logging.warning(f"Unknown band '{band}' for source '{source}', using 550nm default")
        return 550.0  # Reasonable default for optical surveys

    return wl

def _band_simple(band: str) -> str:
    if not band:
        return ""
    b = str(band)
    if "::" in b:
        b = b.split("::")[-1]
    return b.strip().lower()

def meta_from_row(r) -> Tuple[float, float, float]:
    """Extract RA, Dec, and heliocentric redshift from a table row."""
    ra = to_float(r.get("RA", r.get("ra", np.nan)))
    dec = to_float(r.get("DEC", r.get("dec", np.nan)))
    z = to_float(r.get("REDSHIFT", r.get("zhel", r.get("Z_HELIO", r.get("Z_HEL", np.nan)))))
    return ra, dec, z

def _write_outputs(df: pd.DataFrame, outdir: str, source: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pq = os.path.join(outdir, f"lightcurves_{source}.parquet")
    csv = os.path.join(outdir, f"lightcurves_{source}.csv")
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    print(f"Wrote {len(df)} rows to:\n  {pq}\n  {csv}")

def _combine_dataframes_efficiently(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Memory-efficient combination of DataFrames for large datasets."""
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    # For very large datasets, process in chunks to avoid memory issues
    chunk_size = 50  # Process 50 DataFrames at a time
    if len(dfs) <= chunk_size:
        return pd.concat(dfs, ignore_index=True)

    # Process in chunks
    result_chunks = []
    for i in range(0, len(dfs), chunk_size):
        chunk = dfs[i:i + chunk_size]
        combined_chunk = pd.concat(chunk, ignore_index=True)
        result_chunks.append(combined_chunk)

        # Log progress for large datasets
        if len(dfs) > 20:  # Lower threshold for progress reporting
            logging.info(f"Processed chunk {i//chunk_size + 1}/{(len(dfs) + chunk_size - 1)//chunk_size}")

    return pd.concat(result_chunks, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helper for SNANA-style data

def _process_snana_data(tab, snid: str, source: str, zero_point: float) -> List[dict]:
    """Process SNANA-style photometry table into standardized rows."""
    if tab is None or len(tab) == 0:
        return []

    df = tab.to_pandas()
    rows = []
    for _, r in df.iterrows():
        mjd = to_float(r.get("MJD", np.nan))
        band = _band_simple(r.get("BAND", r.get("band", "")))

        # Try magnitude first
        if np.isfinite(to_float(r.get("MAG", np.nan))):
            mag = to_float(r["MAG"])
            mag_err = to_float(r.get("MAGERR", np.nan))
        else:
            # Convert flux to magnitude
            fc = to_float(r.get("FLUXCAL", np.nan))
            fce = to_float(r.get("FLUXCALERR", np.nan))
            if np.isfinite(fc) and fc > 0:
                mag = zero_point - 2.5 * math.log10(fc)
                mag_err = (2.5 / math.log(10)) * (fce / fc) if (np.isfinite(fce) and fce > 0) else np.nan
            else:
                mag = mag_err = np.nan

        # Convert to flux density
        if np.isfinite(mag):
            f_jy, f_jy_err = ab_mag_to_flux_jy(mag, mag_err if np.isfinite(mag_err) else np.nan)
        else:
            f_jy = f_jy_err = np.nan

        # Skip rows without epoch or band
        if not np.isfinite(mjd) or band == "":
            continue

        # Extract metadata
        ra, dec, z_helio = meta_from_row(r)

        rows.append(dict(
            source=source, snid=str(snid), ra=ra, dec=dec, z_helio=z_helio,
            band=band, mjd=mjd, mag=mag if np.isfinite(mag) else np.nan,
            mag_err=mag_err if np.isfinite(mag_err) else np.nan,
            flux_nu_jy=f_jy, flux_nu_jy_err=f_jy_err,
            zp=zero_point, zpsys="AB",
            wavelength_eff_nm=wl_eff_nm_for(band, source),
        ))
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Fetchers

def fetch_osc(names: List[str]) -> List[pd.DataFrame]:
    """Open Supernova Catalog via astroquery.oac.OAC."""
    try:
        from astroquery.oac import OAC
        from astropy.table import Table
    except Exception as e:
        raise RuntimeError("astroquery.oac not available. Try: pip install astroquery astropy") from e

    dfs: List[pd.DataFrame] = []
    for name in names:
        # IMPORTANT: pass as positional or event= (NOT name=)
        try:
            t = OAC.get_photometry(name)
        except TypeError:
            t = OAC.get_photometry(event=name)

        if t is None:
            continue

        if isinstance(t, list):
            tbls = [tt for tt in t if isinstance(tt, Table) and len(tt) > 0]
            if not tbls:
                continue
            t = Table.vstack(tbls, metadata_conflicts="silent")

        if not isinstance(t, Table) or len(t) == 0:
            continue

        df = t.to_pandas()
        rows = []
        for _, r in df.iterrows():
            mjd   = to_float(r.get("time", r.get("MJD", np.nan)))
            band_raw = str(r.get("band", r.get("BAND", "")) or "")
            band  = _band_simple(band_raw)
            zpsys = str(r.get("zeropoint_system", r.get("ZPSYS", "AB")) or "AB").upper()
            zp    = to_float(r.get("zeropoint", r.get("ZP", np.nan)))

            # Prefer magnitude if present
            mag     = to_float(r.get("magnitude", r.get("MAG", np.nan)))
            mag_err = to_float(r.get("e_magnitude", r.get("MAGERR", np.nan)))

            # If only flux present and ZP known, convert flux→mag (in that system)
            flux  = to_float(r.get("flux", np.nan))
            eflux = to_float(r.get("e_flux", np.nan))
            if (not np.isfinite(mag)) and np.isfinite(flux) and np.isfinite(zp) and (flux > 0):
                mag = zp - 2.5 * math.log10(flux)
                mag_err = (2.5 / math.log(10)) * (eflux / flux) if (np.isfinite(eflux) and eflux > 0) else np.nan

            # Only compute AB flux if the system is AB-like; otherwise leave flux NaN
            if np.isfinite(mag) and zpsys in ("AB", "SDSS", "DES"):
                f_jy, f_jy_err = ab_mag_to_flux_jy(mag, mag_err if np.isfinite(mag_err) else np.nan)
            else:
                f_jy = f_jy_err = np.nan

            # Skip rows without epoch or band
            if not np.isfinite(mjd) or band == "":
                continue

            # Extract metadata
            ra, dec, z_helio = meta_from_row(r)

            rows.append(dict(
                source="osc", snid=name, ra=ra, dec=dec, z_helio=z_helio,
                band=band, mjd=mjd,
                mag=mag if np.isfinite(mag) else np.nan,
                mag_err=mag_err if np.isfinite(mag_err) else np.nan,
                flux_nu_jy=f_jy, flux_nu_jy_err=f_jy_err,
                zp=zp if np.isfinite(zp) else np.nan,  # prefer NaN over 0.0
                zpsys=zpsys,
                wavelength_eff_nm=wl_eff_nm_for(band, "osc"),
            ))
        if rows:
            dfs.append(pd.DataFrame(rows))
    return dfs


def fetch_sdss(limit: Optional[int] = None, ids: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """SDSS-II (Sako18) via sndata (formatted photometry tables)."""
    try:
        from sndata.sdss import Sako18
    except Exception as e:
        raise RuntimeError("sndata not available. Try: pip install sndata") from e

    s = Sako18()
    s.download_module_data()

    if ids is None:
        ids = s.get_available_ids()[: (limit or 50)]

    dfs: List[pd.DataFrame] = []
    for snid in ids:
        try:
            # Get formatted photometry table with standardized columns
            tab = s.get_data_for_id(str(snid), format_table=True)
            if tab is None or len(tab) == 0:
                logging.warning(f"No photometry data found for SDSS object {snid}")
                continue

            df = tab.to_pandas()
            rows = []
            for _, r in df.iterrows():
                mjd = to_float(r.get("time", np.nan))
                band = _band_simple(r.get("band", ""))
                zp = to_float(r.get("zp", SNANA_ZP))
                flux = to_float(r.get("flux", np.nan))
                flux_err = to_float(r.get("fluxerr", np.nan))

                # Convert flux to magnitude if flux is positive
                if np.isfinite(flux) and flux > 0 and np.isfinite(zp):
                    mag = zp - 2.5 * math.log10(flux)
                    mag_err = (2.5 / math.log(10)) * (flux_err / flux) if (np.isfinite(flux_err) and flux_err > 0) else np.nan
                else:
                    mag = mag_err = np.nan

                # Convert to Jy flux density
                if np.isfinite(mag):
                    f_jy, f_jy_err = ab_mag_to_flux_jy(mag, mag_err if np.isfinite(mag_err) else np.nan)
                else:
                    f_jy = f_jy_err = np.nan

                # Skip rows without epoch or band
                if not np.isfinite(mjd) or band == "":
                    continue

                # Extract metadata (need to get from SNANA metadata)
                ra, dec, z_helio = meta_from_row(r)

                rows.append(dict(
                    source="sdss", snid=str(snid), ra=ra, dec=dec, z_helio=z_helio,
                    band=band, mjd=mjd, mag=mag if np.isfinite(mag) else np.nan,
                    mag_err=mag_err if np.isfinite(mag_err) else np.nan,
                    flux_nu_jy=f_jy, flux_nu_jy_err=f_jy_err,
                    zp=zp, zpsys="AB",
                    wavelength_eff_nm=wl_eff_nm_for(band, "sdss"),
                ))

            if rows:
                dfs.append(pd.DataFrame(rows))

        except Exception as e:
            logging.warning(f"Failed to process SDSS object {snid}: {e}")
            continue
    return dfs


def fetch_des(limit: Optional[int] = None, ids: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """DES-SN3YR via sndata (formatted photometry tables)."""
    try:
        from sndata.des import SN3YR
    except Exception as e:
        raise RuntimeError("sndata not available. Try: pip install sndata") from e

    d = SN3YR()
    d.download_module_data()

    if ids is None:
        ids = d.get_available_ids()[: (limit or 50)]

    dfs: List[pd.DataFrame] = []
    for snid in ids:
        try:
            # Get formatted photometry table with standardized columns
            tab = d.get_data_for_id(str(snid), format_table=True)
            if tab is None or len(tab) == 0:
                logging.warning(f"No photometry data found for DES object {snid}")
                continue

            df = tab.to_pandas()
            rows = []
            for _, r in df.iterrows():
                mjd = to_float(r.get("time", np.nan))
                band = _band_simple(r.get("band", ""))
                zp = to_float(r.get("zp", SNANA_ZP))
                flux = to_float(r.get("flux", np.nan))
                flux_err = to_float(r.get("fluxerr", np.nan))

                # Convert flux to magnitude if flux is positive
                if np.isfinite(flux) and flux > 0 and np.isfinite(zp):
                    mag = zp - 2.5 * math.log10(flux)
                    mag_err = (2.5 / math.log(10)) * (flux_err / flux) if (np.isfinite(flux_err) and flux_err > 0) else np.nan
                else:
                    mag = mag_err = np.nan

                # Convert to Jy flux density
                if np.isfinite(mag):
                    f_jy, f_jy_err = ab_mag_to_flux_jy(mag, mag_err if np.isfinite(mag_err) else np.nan)
                else:
                    f_jy = f_jy_err = np.nan

                # Skip rows without epoch or band
                if not np.isfinite(mjd) or band == "":
                    continue

                # Extract metadata (need to get from SNANA metadata)
                ra, dec, z_helio = meta_from_row(r)

                rows.append(dict(
                    source="des", snid=str(snid), ra=ra, dec=dec, z_helio=z_helio,
                    band=band, mjd=mjd, mag=mag if np.isfinite(mag) else np.nan,
                    mag_err=mag_err if np.isfinite(mag_err) else np.nan,
                    flux_nu_jy=f_jy, flux_nu_jy_err=f_jy_err,
                    zp=zp, zpsys="AB",
                    wavelength_eff_nm=wl_eff_nm_for(band, "des"),
                ))

            if rows:
                dfs.append(pd.DataFrame(rows))

        except Exception as e:
            logging.warning(f"Failed to process DES object {snid}: {e}")
            continue
    return dfs


def fetch_csp(limit: Optional[int] = None, ids: Optional[List[str]] = None) -> List[pd.DataFrame]:
    """CSP DR3 via sndata. If the CDS tar endpoint returns HTML (not gzip), we skip gracefully."""
    try:
        from sndata.csp import DR3
    except Exception as e:
        raise RuntimeError("sndata not available or CSP DR3 module missing. Try: pip install sndata") from e

    c = DR3()
    try:
        c.download_module_data()
    except Exception as e:
        print(f"[warn] CSP DR3 download failed ({e.__class__.__name__}: {e}). Skipping CSP for now.")
        return []

    if ids is None:
        ids = c.get_available_ids()[: (limit or 50)]

    dfs: List[pd.DataFrame] = []
    for snid in ids:
        tab = c.get_data_for_id(str(snid), format_table=True)
        if tab is None or len(tab) == 0:
            continue
        df = tab.to_pandas()
        rows = []
        for _, r in df.iterrows():
            mjd  = to_float(r.get("MJD", np.nan))
            band = _band_simple(r.get("BAND", r.get("band", "")))
            zpsys_val = str(r.get("ZPSYS", r.get("zeropoint_system", "")) or "").upper()
            mag = to_float(r.get("MAG", np.nan))
            mag_err = to_float(r.get("MAGERR", np.nan))

            # Only compute AB flux if we know it's AB-calibrated
            if zpsys_val in ("AB", "SDSS", "DES"):  # Systems we're confident are AB-calibrated
                f_jy, f_jy_err = ab_mag_to_flux_jy(mag, mag_err if np.isfinite(mag_err) else np.nan) if np.isfinite(mag) else (np.nan, np.nan)
            else:
                f_jy = f_jy_err = np.nan

            if not np.isfinite(mjd) or band == "":
                continue

            # Extract metadata
            ra, dec, z_helio = meta_from_row(r)

            rows.append(dict(
                source="csp", snid=str(snid), ra=ra, dec=dec, z_helio=z_helio,
                band=band, mjd=mjd, mag=mag if np.isfinite(mag) else np.nan,
                mag_err=mag_err if np.isfinite(mag_err) else np.nan,
                flux_nu_jy=f_jy, flux_nu_jy_err=f_jy_err,
                zp=np.nan,  # Unknown/varies; don't force a constant value
                zpsys=(zpsys_val or "UNKNOWN"),
                wavelength_eff_nm=wl_eff_nm_for(band, "csp"),
            ))
        if rows:
            dfs.append(pd.DataFrame(rows))
    return dfs

# ──────────────────────────────────────────────────────────────────────────────
# CLI

def main():
    ap = argparse.ArgumentParser(description="Download & normalize SN light curves (raw photometry).")
    ap.add_argument("--source", required=True, choices=["osc", "sdss", "des", "csp"], help="Data source to ingest")
    ap.add_argument("--ids", type=str, default=None, help="Comma-separated object names/IDs (if omitted, use --limit)")
    ap.add_argument("--ids-file", type=str, help="Path to file with one ID/name per line")
    ap.add_argument("--limit", type=int, default=25, help="If --ids not provided, take first N objects")
    ap.add_argument("--outdir", type=str, default="data/lightcurves", help="Output directory")
    ap.add_argument("--chunk-size", type=int, default=0, help="Write multiple parquet parts every K rows (0 = single file)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    # Configure logging - show INFO by default for progress, DEBUG only with verbose
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Process IDs from command line and/or file
    ids_from_file = None
    if args.ids_file:
        logging.info(f"Reading IDs from file: {args.ids_file}")
        with open(args.ids_file) as fh:
            ids_from_file = [ln.strip() for ln in fh if ln.strip()]
        logging.info(f"Loaded {len(ids_from_file)} IDs from file")

    if args.ids and ids_from_file:
        # Combine and deduplicate
        ids_from_args = [s.strip() for s in args.ids.split(",") if s.strip()]
        ids = list({*ids_from_file, *ids_from_args})  # Set comprehension for deduplication
        logging.info(f"Combined {len(ids_from_args)} command-line IDs with {len(ids_from_file)} file IDs = {len(ids)} unique IDs")
    else:
        ids = ids_from_file or ([s.strip() for s in args.ids.split(",")] if args.ids else None)

    if args.source == "osc":
        if not ids:
            sys.exit("For --source osc you must pass --ids='SN1,SN2,...' or --ids-file")
        logging.info(f"Fetching OSC data for {len(ids)} objects")
        dfs = fetch_osc(ids)
    elif args.source == "sdss":
        if ids:
            logging.info(f"Fetching SDSS data for {len(ids)} specific objects")
        else:
            logging.info(f"Fetching SDSS data for first {args.limit} objects")
        dfs = fetch_sdss(limit=args.limit, ids=ids)
    elif args.source == "des":
        if ids:
            logging.info(f"Fetching DES data for {len(ids)} specific objects")
        else:
            logging.info(f"Fetching DES data for first {args.limit} objects")
        dfs = fetch_des(limit=args.limit, ids=ids)
    elif args.source == "csp":
        if ids:
            logging.info(f"Fetching CSP data for {len(ids)} specific objects")
        else:
            logging.info(f"Fetching CSP data for first {args.limit} objects")
        dfs = fetch_csp(limit=args.limit, ids=ids)
    else:
        sys.exit("Unknown source.")

    if not dfs:
        sys.exit("No light curves fetched (check IDs/limit/network).")

    logging.info(f"Processing {len(dfs)} DataFrames with total {sum(len(df) for df in dfs)} rows")

    # Handle chunked vs single file output
    if args.chunk_size and args.chunk_size > 0:
        logging.info(f"Writing chunked output with {args.chunk_size} rows per file")
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        buf, n, part = [], 0, 0

        for dfi in dfs:
            buf.append(dfi)
            n += len(dfi)
            if n >= args.chunk_size:
                part += 1
                part_df = pd.concat(buf, ignore_index=True).sort_values(["snid", "mjd", "band"])
                part_path = os.path.join(outdir, f"lightcurves_{args.source}_part{part:03d}.parquet")
                part_df.to_parquet(part_path, index=False)
                logging.info(f"Wrote part {part}: {len(part_df)} rows to {part_path}")
                buf, n = [], 0

        # Handle remaining data
        if buf:
            part += 1
            part_df = pd.concat(buf, ignore_index=True).sort_values(["snid", "mjd", "band"])
            part_path = os.path.join(outdir, f"lightcurves_{args.source}_part{part:03d}.parquet")
            part_df.to_parquet(part_path, index=False)
            logging.info(f"Wrote final part {part}: {len(part_df)} rows to {part_path}")

        print(f"Wrote {part} part files to {outdir}")
    else:
        out = _combine_dataframes_efficiently(dfs).sort_values(["snid", "mjd", "band"])
        _write_outputs(out, args.outdir, args.source)

if __name__ == "__main__":
    main()
