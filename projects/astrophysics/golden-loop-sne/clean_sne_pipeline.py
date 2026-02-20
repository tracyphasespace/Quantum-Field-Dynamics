#!/usr/bin/env python3
"""
clean_sne_pipeline.py — QFD SNe Pipeline v3 (Hsiao Template Fit)
=================================================================

From raw DES-SN5YR photometry to Hubble diagram.
No inherited code from v15/v16/v18/v22.

Physics: v2 Kelvin wave model (0 free physics parameters)
    α = 1/137.036          (measured, sole input)
    β = 3.043233053        (Golden Loop)
    K_J = 85.58            (geometric vacuum drag)
    D_L(z) = (c/K_J) ln(1+z) (1+z)^{2/3}
    τ(z) = η [1 - 1/√(1+z)], η = π²/β²

Light curve fitting: Hsiao+ (2007) SN Ia spectral template via sncosmo
    Multi-band simultaneous fit (DES griz)
    Parameters: t0 (peak time), amplitude (distance-encoded)
    K-corrections handled internally by the template SED

Data: DES-SN5YR raw photometry (5,468 SNe Ia, griz bands)
      Cross-validated against SALT2-reduced Hubble diagram (1,591 overlap)

Free parameters: 1 (absolute magnitude M_0 — calibration, not physics)

Pipeline stages:
    1. Load raw photometry + compute true errors from SNR
    2. Hsiao template fit per SN (sncosmo.fit_lc)
    3. Quality cuts
    4. Hubble diagram vs v2 Kelvin
    5. Cross-validation against SALT2
    6. SALT2 mB test (control)

Author: Tracy + Claude
Date: 2026-02-20
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
import sncosmo
from astropy.table import Table
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONSTANTS (all from Golden Loop — zero free parameters)
# ============================================================

PI = np.pi
ALPHA = 1.0 / 137.035999084
C_KMS = 299792.458  # km/s


def solve_golden_loop(alpha):
    """1/α = 2π²(e^β/β) + 1  →  β"""
    target = (1.0 / alpha) - 1.0
    C = 2.0 * PI**2
    b = 3.0
    for _ in range(100):
        eb = np.exp(b)
        val = C * (eb / b) - target
        deriv = C * eb * (b - 1.0) / b**2
        if abs(deriv) < 1e-30:
            break
        b -= val / deriv
        if abs(val / deriv) < 1e-15:
            break
    return b


BETA = solve_golden_loop(ALPHA)
K_VORTEX = 7.0 * PI / 5.0                    # Hill vortex eigenvalue
XI_QFD = K_VORTEX**2 * (5.0 / 6.0)           # = 49π²/30
K_J = XI_QFD * BETA**1.5                      # ≈ 85.6 km/s/Mpc
ETA = PI**2 / BETA**2                         # ≈ 1.066 (scattering opacity)
K_MAG = 5.0 / np.log(10.0)                    # ≈ 2.172 (mag conversion)

# Band map: DES band letter → sncosmo name
BAND_MAP = {'g': 'desg', 'r': 'desr', 'i': 'desi', 'z': 'desz'}

# ============================================================
# v2 KELVIN DISTANCE MODEL
# ============================================================


def mu_v2k(z, M=0.0):
    """v2 Kelvin distance modulus (0 free physics params).

    μ = 5 log₁₀(D_L/Mpc) + 25 + M + K_MAG × η × [1 - 1/√(1+z)]

    D_L = (c/K_J) ln(1+z) (1+z)^{2/3}
    """
    z = np.asarray(z, dtype=float)
    D_prop = (C_KMS / K_J) * np.log1p(z)      # proper distance [Mpc]
    D_L = D_prop * (1.0 + z)**(2.0 / 3.0)     # luminosity distance
    tau = 1.0 - 1.0 / np.sqrt(1.0 + z)        # scattering opacity shape
    mu = np.full_like(z, np.nan, dtype=float)
    mask = D_L > 0
    mu[mask] = 5.0 * np.log10(D_L[mask]) + 25.0 + M + K_MAG * ETA * tau[mask]
    return mu


# ============================================================
# STAGE 1: LOAD RAW PHOTOMETRY
# ============================================================

DATA_DIR = '/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data'
PHOTOMETRY_FILE = os.path.join(DATA_DIR, 'lightcurves_unified_v2_min3.csv')
HD_FILE = os.path.join(DATA_DIR, 'DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv')


def load_photometry():
    """Load raw DES-SN5YR photometry and compute true errors."""
    df = pd.read_csv(PHOTOMETRY_FILE)

    # The flux_nu_jy_err column is a placeholder (all 0.02).
    # True errors come from the SNR column: err = flux / SNR
    df['flux_err_true'] = df['flux_nu_jy'] / df['snr']

    # Error floor: at least 1% of flux (avoid zero/tiny errors)
    min_err = df.groupby('snid')['flux_err_true'].transform('median') * 0.1
    df['flux_err_true'] = np.maximum(df['flux_err_true'], min_err)
    df['flux_err_true'] = np.maximum(df['flux_err_true'], df['flux_nu_jy'] * 0.01)

    print(f"Loaded {len(df):,} observations for {df['snid'].nunique():,} SNe")
    print(f"  SNR range: {df['snr'].min():.1f} - {df['snr'].max():.0f}")
    print(f"  True error range: {df['flux_err_true'].min():.2e} - {df['flux_err_true'].max():.2e} Jy")
    return df


def load_salt2_hd():
    """Load SALT2-reduced Hubble diagram for cross-validation."""
    hd = pd.read_csv(HD_FILE)
    hd['CID'] = hd['CID'].astype(str)
    print(f"Loaded SALT2 HD: {len(hd)} SNe, z = {hd['zHD'].min():.3f}-{hd['zHD'].max():.3f}")
    return hd


# ============================================================
# STAGE 2: HSIAO TEMPLATE FITTING
# ============================================================

# Pre-load Hsiao source (shared across all fits)
HSIAO_SOURCE = sncosmo.get_source('hsiao')


def fit_sn_hsiao(sn_data, z):
    """Fit Hsiao SN Ia template to multi-band light curve.

    Uses sncosmo.fit_lc with the Hsiao+ (2007) spectral template.
    The template handles K-corrections internally — no separate
    K-correction step needed.

    Parameters fitted: t0 (peak time), amplitude (encodes distance)
    Fixed: z (from spectroscopy)

    Returns dict with: peak_mag, t0, amplitude, chi2_dof, n_obs, n_bands, success
    """
    # Filter to valid griz observations
    valid = sn_data['band'].isin(['g', 'r', 'i', 'z']) & (sn_data['flux_nu_jy'] > 0)
    sv = sn_data[valid]

    if len(sv) < 5:
        return None

    sn_bands = [BAND_MAP[b] for b in sv['band'].values]
    flux = sv['flux_nu_jy'].values
    flux_err = sv['flux_err_true'].values

    # sncosmo data table: zp=8.9 converts Jy to AB magnitudes
    data = Table({
        'time': sv['mjd'].values,
        'band': sn_bands,
        'flux': flux,
        'fluxerr': flux_err,
        'zp': np.full(len(sv), 8.9),
        'zpsys': ['ab'] * len(sv),
    })

    # Set up Hsiao model at fixed z
    model = sncosmo.Model(source=HSIAO_SOURCE)
    model.set(z=z)

    # Initial guesses
    idx_peak = np.argmax(data['flux'])
    t0_guess = float(data['time'][idx_peak])
    peak_band = data['band'][idx_peak]

    # Estimate amplitude from peak flux
    ref_model = sncosmo.Model(source=HSIAO_SOURCE)
    ref_model.set(z=z, t0=0, amplitude=1.0)
    ref_flux = ref_model.bandflux(peak_band, 0.0, zp=8.9, zpsys='ab')
    amp_guess = float(data['flux'][idx_peak]) / ref_flux if ref_flux > 0 else 1e-10

    model.set(t0=t0_guess, amplitude=amp_guess)

    try:
        result, fitted_model = sncosmo.fit_lc(
            data, model, ['t0', 'amplitude'],
            bounds={
                't0': (float(data['time'].min()) - 30,
                       float(data['time'].max()) + 30),
                'amplitude': (amp_guess * 0.001, amp_guess * 1000),
            },
            minsnr=0.0,
        )

        # Peak rest-frame B magnitude (includes distance via amplitude)
        m_B = fitted_model.source_peakmag('bessellb', 'ab')
        chi2_dof = result.chisq / max(result.ndof, 1)
        n_bands = len(set(data['band']))

        return {
            'peak_mag': m_B,
            't0': fitted_model.get('t0'),
            'amplitude': fitted_model.get('amplitude'),
            'chi2_dof': chi2_dof,
            'n_obs': len(data),
            'n_bands': n_bands,
            'success': True,
        }

    except Exception:
        return None


def gaussian_template(t, A, t0, sigma):
    """Gaussian light curve template (fallback)."""
    return A * np.exp(-0.5 * ((t - t0) / sigma)**2)


def fit_sn_gaussian_fallback(sn_data, z):
    """Gaussian fit fallback for SNe where Hsiao fails.

    Fits Gaussian to the best band (closest to rest-frame B at 440 nm),
    returns peak magnitude without K-correction (less accurate but robust).
    """
    band_lambda = {'g': 472.0, 'r': 641.5, 'i': 783.5, 'z': 926.0}

    # Pick band closest to rest-frame B
    best_band = min(band_lambda.keys(),
                    key=lambda b: abs(band_lambda[b] / (1 + z) - 440.0))

    bd = sn_data[sn_data['band'] == best_band]
    if len(bd) < 3:
        # Try any band
        for b in ['r', 'i', 'g', 'z']:
            bd = sn_data[sn_data['band'] == b]
            if len(bd) >= 3:
                best_band = b
                break
        else:
            return None

    mjd = bd['mjd'].values
    flux = bd['flux_nu_jy'].values
    flux_err = bd['flux_err_true'].values

    idx_max = np.argmax(flux)
    A0 = max(flux[idx_max], 1e-15)
    t0_0 = mjd[idx_max]

    if A0 <= 0:
        return None

    try:
        bounds = ([0, mjd.min() - 50, 3.0], [A0 * 10, mjd.max() + 50, 150.0])
        popt, _ = curve_fit(gaussian_template, mjd, flux,
                            p0=[A0, t0_0, 15.0 * (1 + z)],
                            sigma=flux_err, absolute_sigma=True,
                            bounds=bounds, maxfev=2000)
        A, t0, sigma = popt

        if A <= 0:
            return None

        m_obs = -2.5 * np.log10(A / 3631.0)

        model = gaussian_template(mjd, *popt)
        resid = (flux - model) / flux_err
        chi2 = np.sum(resid**2)
        dof = max(len(mjd) - 3, 1)

        n_bands_total = sn_data['band'].isin(['g', 'r', 'i', 'z']).sum()
        n_unique_bands = sn_data[sn_data['band'].isin(['g', 'r', 'i', 'z'])]['band'].nunique()

        return {
            'peak_mag': m_obs,
            't0': t0,
            'amplitude': A,
            'chi2_dof': chi2 / dof,
            'n_obs': len(bd),
            'n_bands': n_unique_bands,
            'success': True,
        }
    except Exception:
        return None


def process_all_sne(df, verbose=True):
    """Fit Hsiao template to all SNe, extract peak B magnitudes.

    Primary: sncosmo Hsiao template (multi-band simultaneous fit)
    Fallback: Gaussian fit on best single band (for failed Hsiao fits)

    Returns DataFrame with one row per SN.
    """
    sne = df.groupby('snid')
    results = []
    n_total = len(sne)
    n_hsiao = 0
    n_gaussian = 0
    n_failed = 0
    t_start = time.time()

    for i, (snid, sn_data) in enumerate(sne):
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  {i+1}/{n_total} ({rate:.0f} SN/s, ETA {eta:.0f}s)  "
                  f"[Hsiao:{n_hsiao} Gauss:{n_gaussian} fail:{n_failed}]")

        z = sn_data['z'].iloc[0]
        if z <= 0:
            continue

        # Try Hsiao template first
        fit = fit_sn_hsiao(sn_data, z)
        method = 'hsiao'

        if fit is None:
            # Fallback to Gaussian
            fit = fit_sn_gaussian_fallback(sn_data, z)
            method = 'gaussian'

        if fit is None:
            n_failed += 1
            continue

        if method == 'hsiao':
            n_hsiao += 1
        else:
            n_gaussian += 1

        results.append({
            'snid': snid,
            'z': z,
            'peak_mag': fit['peak_mag'],
            'chi2_dof': fit['chi2_dof'],
            'n_obs': fit['n_obs'],
            'n_bands': fit['n_bands'],
            'method': method,
        })

    elapsed = time.time() - t_start
    out = pd.DataFrame(results)
    print(f"\nFitted {len(out)} / {n_total} SNe in {elapsed:.1f}s ({n_total/elapsed:.0f} SN/s)")
    print(f"  Hsiao template: {n_hsiao} ({100*n_hsiao/max(len(out),1):.1f}%)")
    print(f"  Gaussian fallback: {n_gaussian} ({100*n_gaussian/max(len(out),1):.1f}%)")
    print(f"  Failed: {n_failed}")
    return out


# ============================================================
# STAGE 3: QUALITY CUTS
# ============================================================

def apply_quality_cuts(sne_df, chi2_max=50.0, n_obs_min=5,
                       n_bands_min=2, z_min=0.02):
    """Apply quality cuts to the fitted SN sample.

    More permissive than v1/v2 since Hsiao template is more robust:
    - chi2/dof < 50 (Hsiao chi2 can be large due to real photometric errors)
    - n_obs >= 5 (Hsiao needs fewer points than Gaussian)
    - n_bands >= 2 (multi-band for color information)
    - z > 0.02 (peculiar velocity contamination)
    - peak_mag finite and reasonable (15 < m < 30)
    """
    n_before = len(sne_df)

    mask = (
        (sne_df['chi2_dof'] < chi2_max) &
        (sne_df['n_obs'] >= n_obs_min) &
        (sne_df['n_bands'] >= n_bands_min) &
        (sne_df['z'] > z_min) &
        np.isfinite(sne_df['peak_mag']) &
        (sne_df['peak_mag'] > 15) &
        (sne_df['peak_mag'] < 30)
    )

    out = sne_df[mask].copy().reset_index(drop=True)
    print(f"\nQuality cuts: {n_before} → {len(out)} SNe")
    print(f"  chi2/dof < {chi2_max}: removed {(sne_df['chi2_dof'] >= chi2_max).sum()}")
    print(f"  n_obs >= {n_obs_min}: removed {(sne_df['n_obs'] < n_obs_min).sum()}")
    print(f"  n_bands >= {n_bands_min}: removed {(sne_df['n_bands'] < n_bands_min).sum()}")
    print(f"  z > {z_min}: removed {(sne_df['z'] <= z_min).sum()}")
    n_hsiao = (out['method'] == 'hsiao').sum()
    print(f"  After cuts: {n_hsiao} Hsiao ({100*n_hsiao/len(out):.1f}%), "
          f"{len(out)-n_hsiao} Gaussian ({100*(len(out)-n_hsiao)/len(out):.1f}%)")

    return out


# ============================================================
# STAGE 4: HUBBLE DIAGRAM
# ============================================================

def fit_hubble_diagram(sne_df, label=""):
    """Fit v2 Kelvin model to the Hubble diagram.

    Fits only the absolute magnitude M_0 (median offset).
    No width or stretch correction — Hsiao template already
    captures the light curve shape.

    Returns dict with fit results.
    """
    z = sne_df['z'].values
    m = sne_df['peak_mag'].values

    # v2 Kelvin prediction (M=0 for now)
    mu_model = mu_v2k(z, M=0.0)
    valid = np.isfinite(mu_model) & np.isfinite(m)
    z, m, mu_model = z[valid], m[valid], mu_model[valid]

    # Simple offset (no stretch/color correction for Hsiao)
    M_0 = np.median(m - mu_model)
    resid = m - mu_model - M_0

    sigma = np.std(resid)
    mad = median_abs_deviation(resid)

    # Outlier-clipped stats (3σ)
    clip = np.abs(resid) < 3 * sigma
    sigma_clipped = np.std(resid[clip]) if clip.sum() > 10 else sigma
    n_outliers = (~clip).sum()

    # z-slope
    slope, _ = np.polyfit(z, resid, 1) if len(z) > 10 else (0.0, 0.0)

    print(f"\n{'='*60}")
    print(f"HUBBLE DIAGRAM — v2 Kelvin (0 free physics params) {label}")
    print(f"{'='*60}")
    print(f"  SNe used:           {len(z)}")
    print(f"  M_0 (calibration):  {M_0:.4f} mag")
    print(f"  RMS scatter:        {sigma:.3f} mag")
    print(f"  MAD scatter:        {mad:.3f} mag")
    print(f"  3σ-clipped RMS:     {sigma_clipped:.3f} mag")
    print(f"  3σ outliers:        {n_outliers} ({100*n_outliers/len(z):.1f}%)")
    print(f"  z-slope:            {slope:+.3f} mag/z")

    return {
        'z': z, 'mu_obs': m - M_0, 'mu_model': mu_model,
        'resid': resid, 'M_0': M_0,
        'sigma': sigma, 'sigma_clipped': sigma_clipped,
        'mad': mad, 'slope': slope,
        'n_sne': len(z), 'n_outliers': n_outliers,
    }


# ============================================================
# STAGE 5: CROSS-VALIDATION AGAINST SALT2
# ============================================================

def cross_validate_salt2(sne_df, hd_df, fit_result):
    """Compare Hsiao pipeline results to SALT2-reduced Hubble diagram."""
    sne_df = sne_df.copy()
    sne_df['snid_str'] = sne_df['snid'].astype(str)
    merged = sne_df.merge(hd_df[['CID', 'zHD', 'MU', 'MUERR_FINAL', 'mB', 'x1', 'c']],
                          left_on='snid_str', right_on='CID', how='inner')

    if len(merged) == 0:
        print("\nNo cross-match with SALT2 HD")
        return None

    z = merged['z'].values
    mu_raw = merged['peak_mag'].values - fit_result['M_0']
    mu_salt2 = merged['MU'].values
    mB_salt2 = merged['mB'].values
    mu_v2k_pred = mu_v2k(z, M=0.0)

    # Raw pipeline vs v2K
    resid_raw = mu_raw - mu_v2k_pred
    valid = np.isfinite(resid_raw)
    sigma_raw = np.std(resid_raw[valid])

    # SALT2 vs v2K (same SNe)
    resid_salt2 = mu_salt2 - mu_v2k_pred
    valid2 = np.isfinite(resid_salt2)
    sigma_salt2 = np.std(resid_salt2[valid2])

    # Direct comparison: Hsiao m_B vs SALT2 mB
    delta_mB = merged['peak_mag'].values - mB_salt2
    valid_mB = np.isfinite(delta_mB)
    sigma_mB = np.std(delta_mB[valid_mB])
    offset_mB = np.median(delta_mB[valid_mB])

    # z-slopes
    z_v = z[valid]
    r_v = resid_raw[valid]
    slope, _ = np.polyfit(z_v, r_v, 1) if len(z_v) > 10 else (0.0, 0.0)

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: Hsiao Pipeline vs SALT2")
    print(f"{'='*60}")
    print(f"  Matched SNe:           {len(merged)}")
    print(f"  Hsiao vs v2K RMS:      {sigma_raw:.3f} mag")
    print(f"  SALT2 vs v2K RMS:      {sigma_salt2:.3f} mag")
    print(f"  SALT2 improvement:     {sigma_raw/sigma_salt2:.1f}×")
    print(f"  Hsiao z-slope:         {slope:+.3f} mag/z")
    print(f"  Hsiao−SALT2 mB offset: {offset_mB:+.3f} mag")
    print(f"  Hsiao−SALT2 mB scatter:{sigma_mB:.3f} mag")

    # Binned residuals for matched subset
    print(f"\n  BINNED RESIDUALS (matched SNe):")
    print(f"  {'z':>8s}  {'<Δμ_H>':>8s}  {'<Δμ_S2>':>8s}  {'σ_H':>8s}  {'σ_S2':>8s}  {'N':>4s}")
    sort = np.argsort(z)
    z_s = z[sort]
    rr_s = resid_raw[sort]
    rs_s = resid_salt2[sort]
    n_bins = min(10, len(z_s) // 30)
    edges = np.linspace(z_s.min(), z_s.max(), n_bins + 1)
    for j in range(n_bins):
        mask = (z_s >= edges[j]) & (z_s < edges[j+1])
        if mask.sum() < 5:
            continue
        zc = np.mean(z_s[mask])
        print(f"  {zc:8.3f}  {np.mean(rr_s[mask]):+8.3f}  {np.mean(rs_s[mask]):+8.3f}  "
              f"{np.std(rr_s[mask]):8.3f}  {np.std(rs_s[mask]):8.3f}  {mask.sum():4d}")

    return {
        'n_matched': len(merged), 'sigma_raw': sigma_raw,
        'sigma_salt2': sigma_salt2, 'sigma_mB': sigma_mB,
        'offset_mB': offset_mB, 'slope': slope,
        'z': z, 'mu_raw': mu_raw, 'mu_salt2': mu_salt2,
        'resid_raw': resid_raw, 'resid_salt2': resid_salt2,
    }


# ============================================================
# STAGE 6: BINNED HUBBLE DIAGRAM
# ============================================================

def binned_hubble(z, resid, n_bins=15, label=""):
    """Compute binned residuals for visual inspection."""
    valid = np.isfinite(resid)
    z, resid = z[valid], resid[valid]
    sort = np.argsort(z)
    z, resid = z[sort], resid[sort]

    bin_edges = np.linspace(z.min(), z.max(), n_bins + 1)
    print(f"\n{'='*60}")
    print(f"BINNED RESIDUALS {label}")
    print(f"{'='*60}")
    print(f"  {'z':>8s}  {'<Δμ>':>8s}  {'σ':>8s}  {'N':>5s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")

    for j in range(n_bins):
        mask = (z >= bin_edges[j]) & (z < bin_edges[j+1])
        if mask.sum() < 3:
            continue
        zc = np.mean(z[mask])
        mu_mean = np.mean(resid[mask])
        mu_std = np.std(resid[mask])
        print(f"  {zc:8.3f}  {mu_mean:+8.4f}  {mu_std:8.4f}  {mask.sum():5d}")


# ============================================================
# STAGE 7: SALT2 mB (UNSTANDARDIZED) + v2 KELVIN
# ============================================================

def salt2_mB_v2kelvin(hd_df):
    """Test v2 Kelvin using SALT2's peak B-band magnitude WITHOUT standardization.

    This is the control experiment: uses SALT2's excellent peak extraction
    with v2 Kelvin's distance model. 0 free physics params.
    """
    z = hd_df['zHD'].values
    mB = hd_df['mB'].values
    x1 = hd_df['x1'].values
    c = hd_df['c'].values
    mu_salt2 = hd_df['MU'].values

    valid = (z > 0.01) & np.isfinite(mB) & np.isfinite(mu_salt2)
    z, mB, x1, c = z[valid], mB[valid], x1[valid], c[valid]
    mu_salt2 = mu_salt2[valid]

    mu_model = mu_v2k(z, M=0.0)
    vm = np.isfinite(mu_model)
    z, mB, x1, c, mu_model = z[vm], mB[vm], x1[vm], c[vm], mu_model[vm]
    mu_salt2 = mu_salt2[vm]

    print(f"\n  Using {len(z)} SNe from SALT2 HD (z = {z.min():.3f}-{z.max():.3f})")

    results = {}
    for label, m_use in [
        ("mB raw (no corrections)", mB),
        ("mB + stretch + color", mB + 0.15 * x1 - 3.1 * c),
        ("SALT2 fully standardized", mu_salt2),
    ]:
        if label == "SALT2 fully standardized":
            resid = m_use - mu_model
        else:
            M_fit = np.median(m_use - mu_model)
            resid = m_use - mu_model - M_fit

        sigma = np.std(resid)
        clip = np.abs(resid) < 3 * sigma
        sigma_clip = np.std(resid[clip]) if clip.sum() > 10 else sigma
        slope, _ = np.polyfit(z, resid, 1)

        print(f"\n  {label}:")
        print(f"    σ = {sigma:.3f} mag, σ_clip = {sigma_clip:.3f}, "
              f"z-slope = {slope:+.3f} mag/z")

        results[label] = {'sigma': sigma, 'sigma_clip': sigma_clip, 'slope': slope}

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("QFD CLEAN SNe PIPELINE v3 (Hsiao Template Fit)")
    print("From raw photometry to Hubble diagram — no legacy code")
    print("=" * 60)

    # Print derivation chain
    print(f"\n  α = 1/{1/ALPHA:.6f}")
    print(f"  β = {BETA:.10f}")
    print(f"  k = 7π/5 = {K_VORTEX:.6f}")
    print(f"  ξ = k²×5/6 = {XI_QFD:.4f}")
    print(f"  K_J = ξ×β^(3/2) = {K_J:.4f} km/s/Mpc")
    print(f"  η = π²/β² = {ETA:.4f}")
    print(f"  Template: Hsiao+ (2007) SN Ia SED (sncosmo)")
    print(f"  Free physics params: 0")

    # Stage 1: Load data
    print(f"\n--- STAGE 1: Load raw photometry ---")
    phot = load_photometry()
    hd = load_salt2_hd()

    # Stage 2: Hsiao template fitting
    print(f"\n--- STAGE 2: Hsiao template fitting ---")
    sne = process_all_sne(phot)

    # Stage 3: Quality cuts
    print(f"\n--- STAGE 3: Quality cuts ---")
    sne_clean = apply_quality_cuts(sne)

    # Separate Hsiao-only sample for clean comparison
    sne_hsiao = sne_clean[sne_clean['method'] == 'hsiao'].copy().reset_index(drop=True)

    # Stage 4: Hubble diagram
    print(f"\n--- STAGE 4: Hubble diagram ---")
    result_all = fit_hubble_diagram(sne_clean, label="[ALL]")
    result_hsiao = fit_hubble_diagram(sne_hsiao, label="[Hsiao only]")

    # Stage 5: Binned residuals
    binned_hubble(result_hsiao['z'], result_hsiao['resid'], label="(Hsiao only, v2 Kelvin)")

    # Stage 6: Cross-validation
    print(f"\n--- STAGE 5: Cross-validation ---")
    xval = cross_validate_salt2(sne_clean, hd, result_all)

    # Also cross-validate Hsiao-only
    if sne_hsiao is not None and len(sne_hsiao) > 100:
        print(f"\n  --- Hsiao-only cross-validation ---")
        xval_hsiao = cross_validate_salt2(sne_hsiao, hd, result_hsiao)

    # Stage 7: SALT2 mB control test
    print(f"\n--- STAGE 6: SALT2 mB control test ---")
    result_mB = salt2_mB_v2kelvin(hd)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw SNe processed:   {len(sne)}")
    print(f"  After quality cuts:  {len(sne_clean)} (Hsiao: {len(sne_hsiao)})")
    print(f"  z range:             {sne_clean['z'].min():.3f} - {sne_clean['z'].max():.3f}")
    print(f"")
    print(f"  ALL (Hsiao+Gauss):   σ = {result_all['sigma']:.3f} mag, "
          f"slope = {result_all['slope']:+.3f}")
    print(f"  Hsiao only:          σ = {result_hsiao['sigma']:.3f} mag, "
          f"slope = {result_hsiao['slope']:+.3f}")
    if xval:
        print(f"  Cross-matched:       Hsiao σ = {xval['sigma_raw']:.3f}, "
              f"SALT2 σ = {xval['sigma_salt2']:.3f}")
    if result_mB:
        mB_raw = result_mB.get("mB raw (no corrections)", {})
        print(f"  SALT2 mB + v2K:      σ = {mB_raw.get('sigma', 0):.3f}, "
              f"slope = {mB_raw.get('slope', 0):+.3f}")

    print(f"\n  Physics params: 0 (all from Golden Loop)")
    print(f"  Calibration:    M_0 = {result_all['M_0']:.4f}")

    return result_all, result_hsiao, xval, sne_clean, result_mB


if __name__ == '__main__':
    main()
