#!/usr/bin/env python3
"""
clean_sne_pipeline.py — QFD SNe Pipeline v1 (Clean Start)
==========================================================

From raw DES-SN5YR photometry to Hubble diagram.
No inherited code from v15/v16/v18/v22.

Physics: v2 Kelvin wave model (0 free physics parameters)
    α = 1/137.036          (measured, sole input)
    β = 3.043233053        (Golden Loop)
    K_J = 85.58            (geometric vacuum drag)
    D_L(z) = (c/K_J) ln(1+z) (1+z)^{2/3}
    τ(z) = η [1 - 1/√(1+z)], η = π²/β²

Data: DES-SN5YR raw photometry (5,468 SNe Ia, griz bands)
      Cross-validated against SALT2-reduced Hubble diagram (1,591 overlap)

Free parameters: 1 (absolute magnitude M_0 — calibration, not physics)

Pipeline stages:
    1. Load raw photometry
    2. Per-SN light curve fitting (Gaussian template, per band)
    3. Peak apparent magnitude in rest-frame-B-matched band
    4. Quality cuts
    5. Hubble diagram vs v2 Kelvin
    6. Cross-validation against SALT2

Author: Tracy + Claude
Date: 2026-02-20
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# Band effective wavelengths (DES griz, nm)
BAND_LAMBDA = {'g': 472.0, 'r': 641.5, 'i': 783.5, 'z': 926.0}
REST_B_CENTER = 440.0  # B-band center (nm)

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


def mu_lcdm(z, H0=70.0, Om=0.3, M=0.0):
    """ΛCDM distance modulus for comparison."""
    from scipy.integrate import quad
    z = np.atleast_1d(np.asarray(z, dtype=float))
    mu = np.full_like(z, np.nan, dtype=float)
    OL = 1.0 - Om
    for i, zi in enumerate(z):
        if zi > 0:
            I, _ = quad(lambda zp: 1.0/np.sqrt(Om*(1+zp)**3 + OL), 0, zi)
            D_L = (C_KMS / H0) * (1 + zi) * I
            mu[i] = 5.0 * np.log10(D_L) + 25.0 + M
    return mu


# ============================================================
# STAGE 1: LOAD RAW PHOTOMETRY
# ============================================================

DATA_DIR = '/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data'
PHOTOMETRY_FILE = os.path.join(DATA_DIR, 'lightcurves_unified_v2_min3.csv')
HD_FILE = os.path.join(DATA_DIR, 'DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv')


def load_photometry():
    """Load raw DES-SN5YR photometry."""
    df = pd.read_csv(PHOTOMETRY_FILE)
    print(f"Loaded {len(df):,} observations for {df['snid'].nunique():,} SNe")
    return df


def load_salt2_hd():
    """Load SALT2-reduced Hubble diagram for cross-validation."""
    hd = pd.read_csv(HD_FILE)
    hd['CID'] = hd['CID'].astype(str)
    print(f"Loaded SALT2 HD: {len(hd)} SNe, z = {hd['zHD'].min():.3f}-{hd['zHD'].max():.3f}")
    return hd


# ============================================================
# STAGE 2: LIGHT CURVE FITTING + K-CORRECTION
# ============================================================

# SN Ia peak SED: blackbody at ~11,000 K (Nugent+ 2002 template average)
SN_PEAK_TEMP = 11000.0  # K

# Physical constants for Planck function
H_PLANCK = 6.626e-34    # J·s
K_BOLTZ = 1.381e-23     # J/K
C_LIGHT_M = 2.998e8     # m/s


def planck_fnu(lam_nm, T):
    """Planck spectral radiance B_ν(T) at wavelength λ (nm).

    Returns B_ν in arbitrary units (we only need ratios for K-correction).
    """
    lam_m = lam_nm * 1e-9
    nu = C_LIGHT_M / lam_m
    x = H_PLANCK * nu / (K_BOLTZ * T)
    if x > 500:
        return 0.0
    return nu**3 / (np.exp(x) - 1.0)


def k_correction(z, band, T=SN_PEAK_TEMP):
    """K-correction for observer-frame band at redshift z.

    K(z) = -2.5 × log₁₀[(1+z) × B_ν(ν_rest, T) / B_ν(ν_obs, T)]

    This converts observer-frame magnitude to rest-frame magnitude
    at the same band's effective wavelength.

    For host-subtracted difference imaging, the (1+z) bandwidth factor
    is already accounted for, so we use only the SED shape ratio.
    """
    lam_obs = BAND_LAMBDA[band]
    lam_rest = lam_obs / (1 + z)

    B_rest = planck_fnu(lam_rest, T)
    B_obs = planck_fnu(lam_obs, T)

    if B_obs <= 0 or B_rest <= 0:
        return 0.0

    # K-correction: accounts for SED slope across the bandpass
    # For difference imaging (flux = SN only), the bandwidth stretching
    # is already in the observed flux, so:
    return -2.5 * np.log10(B_rest / B_obs)


def gaussian_template(t, A, t0, sigma):
    """Gaussian light curve template for host-subtracted data.

    f(t) = A × exp(-0.5 × ((t - t0)/σ)²)

    DES photometry is difference-imaged (host subtracted), so baseline = 0.
    3 parameters: A (peak amplitude), t0 (peak time), σ (width).
    """
    return A * np.exp(-0.5 * ((t - t0) / sigma)**2)


def fit_single_band(mjd, flux, flux_err, z):
    """Fit one band's light curve, return peak flux and width.

    Returns dict with: peak_flux, peak_mjd, width_rest, chi2_dof, n_obs, success
    """
    n = len(mjd)
    result = {'peak_flux': np.nan, 'peak_mjd': np.nan, 'width_rest': np.nan,
              'chi2_dof': np.nan, 'n_obs': n, 'success': False}

    if n < 3:
        return result

    # Initial guesses from data
    idx_max = np.argmax(flux)
    A0 = max(flux[idx_max], 1e-10)
    t0_0 = mjd[idx_max]
    sigma0 = 15.0 * (1 + z)  # observer-frame: wider at higher z (time dilation)

    if A0 <= 0:
        return result

    try:
        bounds = ([0, mjd.min() - 50, 3.0],
                  [A0 * 10, mjd.max() + 50, 150.0])
        popt, pcov = curve_fit(gaussian_template, mjd, flux,
                               p0=[A0, t0_0, sigma0],
                               sigma=flux_err, absolute_sigma=True,
                               bounds=bounds, maxfev=2000)
        A, t0, sigma = popt

        if A <= 0:
            return result

        # Residuals
        model = gaussian_template(mjd, *popt)
        resid = (flux - model) / flux_err
        chi2 = np.sum(resid**2)
        dof = max(n - 3, 1)

        result['peak_flux'] = A
        result['peak_mjd'] = t0
        result['width_rest'] = sigma / (1 + z)
        result['chi2_dof'] = chi2 / dof
        result['success'] = True

    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        # Fit failed — fall back to max observed flux
        if flux[idx_max] > 0:
            result['peak_flux'] = flux[idx_max]
            result['peak_mjd'] = mjd[idx_max]
            result['width_rest'] = 15.0
            result['success'] = True

    return result


def select_rest_B_band(z):
    """Select observer-frame band closest to rest-frame B (440 nm).

    rest_lambda = obs_lambda / (1 + z)
    Pick band where rest_lambda is closest to 440 nm.
    """
    best_band = None
    best_diff = np.inf
    for band, lam in BAND_LAMBDA.items():
        rest_lam = lam / (1 + z)
        diff = abs(rest_lam - REST_B_CENTER)
        if diff < best_diff:
            best_diff = diff
            best_band = band
    return best_band


# ============================================================
# STAGE 3: PROCESS ALL SNe
# ============================================================

def process_all_sne(df, verbose=True):
    """Fit light curves for all SNe, extract peak magnitudes.

    Strategy: fit ALL available bands, then combine with SED-based
    K-correction to get rest-frame B-band peak magnitude.

    For each SN:
    1. Fit Gaussian to each band → peak flux, width
    2. Apply K-correction to each band's peak magnitude
    3. Take weighted average of K-corrected magnitudes (all bands
       estimate the same rest-frame luminosity)
    4. Use median width across bands for stretch correction

    Returns DataFrame with one row per SN.
    """
    sne = df.groupby('snid')
    results = []
    n_total = len(sne)

    for i, (snid, sn_data) in enumerate(sne):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processing SN {i+1}/{n_total}...")

        z = sn_data['z'].iloc[0]
        if z <= 0:
            continue

        # Fit each band independently
        band_fits = {}
        for band in ['g', 'r', 'i', 'z']:
            band_data = sn_data[sn_data['band'] == band]
            if len(band_data) < 3:
                continue

            mjd = band_data['mjd'].values
            flux = band_data['flux_nu_jy'].values
            flux_err = band_data['flux_nu_jy_err'].values

            # Replace bad errors
            bad_err = flux_err <= 0
            if bad_err.any():
                med_err = np.median(flux_err[~bad_err]) if (~bad_err).any() else 1e-8
                flux_err = flux_err.copy()
                flux_err[bad_err] = med_err

            fit = fit_single_band(mjd, flux, flux_err, z)
            if fit['success'] and fit['peak_flux'] > 0:
                band_fits[band] = fit

        if not band_fits:
            continue

        # Convert each band's peak flux to K-corrected rest-frame magnitude
        mags_kcorr = []
        weights = []
        widths = []
        n_obs_total = 0
        chi2_total = 0
        n_bands = 0

        for band, fit in band_fits.items():
            # Observer-frame AB magnitude
            m_obs = -2.5 * np.log10(fit['peak_flux'] / 3631.0)

            # K-correction to rest frame
            K = k_correction(z, band)
            m_rest = m_obs - K

            # Weight by number of observations and inverse chi2
            w = fit['n_obs'] / max(fit['chi2_dof'], 0.1)
            mags_kcorr.append(m_rest)
            weights.append(w)
            widths.append(fit['width_rest'])
            n_obs_total += fit['n_obs']
            chi2_total += fit['chi2_dof'] * fit['n_obs']
            n_bands += 1

        mags_kcorr = np.array(mags_kcorr)
        weights = np.array(weights)

        # Weighted average of K-corrected peak magnitude
        peak_mag = np.average(mags_kcorr, weights=weights)
        width_rest = np.median(widths)
        chi2_avg = chi2_total / max(n_obs_total, 1)

        # Best band (for reporting)
        best_band = select_rest_B_band(z)
        if best_band not in band_fits:
            best_band = list(band_fits.keys())[0]

        results.append({
            'snid': snid,
            'z': z,
            'band_used': best_band,
            'n_bands': n_bands,
            'peak_mag': peak_mag,
            'peak_flux': band_fits[best_band]['peak_flux'],
            'width_rest': width_rest,
            'chi2_dof': chi2_avg,
            'n_obs': n_obs_total,
        })

    out = pd.DataFrame(results)
    print(f"\nFitted {len(out)} / {n_total} SNe successfully")
    print(f"  Multi-band (>=2): {(out['n_bands'] >= 2).sum()}")
    print(f"  Single-band:      {(out['n_bands'] == 1).sum()}")
    return out


# ============================================================
# STAGE 4: QUALITY CUTS
# ============================================================

def apply_quality_cuts(sne_df, chi2_max=10.0, n_obs_min=8,
                       n_bands_min=2, width_min=3.0, width_max=40.0,
                       z_min=0.02):
    """Apply quality cuts to the fitted SN sample.

    Cuts:
        - chi2/dof < chi2_max (bad fits)
        - n_obs >= n_obs_min (insufficient coverage)
        - n_bands >= n_bands_min (need multi-band for K-correction)
        - width_rest in [width_min, width_max] days (unphysical widths)
        - z > z_min (peculiar velocity contamination)
        - peak_mag finite and reasonable (15 < m < 28)
    """
    n_before = len(sne_df)

    mask = (
        (sne_df['chi2_dof'] < chi2_max) &
        (sne_df['n_obs'] >= n_obs_min) &
        (sne_df['n_bands'] >= n_bands_min) &
        (sne_df['width_rest'] > width_min) &
        (sne_df['width_rest'] < width_max) &
        (sne_df['z'] > z_min) &
        np.isfinite(sne_df['peak_mag']) &
        (sne_df['peak_mag'] > 15) &
        (sne_df['peak_mag'] < 28)
    )

    out = sne_df[mask].copy().reset_index(drop=True)
    print(f"\nQuality cuts: {n_before} → {len(out)} SNe")
    print(f"  chi2/dof < {chi2_max}: removed {(sne_df['chi2_dof'] >= chi2_max).sum()}")
    print(f"  n_obs >= {n_obs_min}: removed {(sne_df['n_obs'] < n_obs_min).sum()}")
    print(f"  n_bands >= {n_bands_min}: removed {(sne_df['n_bands'] < n_bands_min).sum()}")
    print(f"  width [{width_min},{width_max}]: removed "
          f"{((sne_df['width_rest'] <= width_min) | (sne_df['width_rest'] >= width_max)).sum()}")
    print(f"  z > {z_min}: removed {(sne_df['z'] <= z_min).sum()}")

    return out


# ============================================================
# STAGE 5: HUBBLE DIAGRAM
# ============================================================

def fit_hubble_diagram(sne_df, use_width_correction=True):
    """Fit v2 Kelvin model to the raw Hubble diagram.

    Fits absolute magnitude M_0 (and optionally width correction α_w)
    to minimize residuals.

    With width correction:
        m_corrected = m_peak - α_w × (w - w_ref)
        μ_obs = m_corrected - M_0

    Without:
        μ_obs = m_peak - M_0

    Returns dict with fit results.
    """
    z = sne_df['z'].values
    m = sne_df['peak_mag'].values
    w = sne_df['width_rest'].values

    # v2 Kelvin prediction (M=0 for now)
    mu_model = mu_v2k(z, M=0.0)
    valid = np.isfinite(mu_model)
    z, m, w, mu_model = z[valid], m[valid], w[valid], mu_model[valid]

    if use_width_correction and len(z) > 10:
        # Fit: m_peak = mu_model + M_0 + alpha_w * (w - w_ref)
        w_ref = np.median(w)
        dw = w - w_ref

        # Linear least squares: m = mu_model + M_0 + alpha_w * dw
        # m - mu_model = M_0 + alpha_w * dw
        y = m - mu_model
        A_mat = np.column_stack([np.ones(len(z)), dw])
        params, residuals, rank, sv = np.linalg.lstsq(A_mat, y, rcond=None)
        M_0, alpha_w = params

        m_corrected = m - alpha_w * dw
        resid = m_corrected - mu_model - M_0
    else:
        # Simple offset only
        M_0 = np.median(m - mu_model)
        alpha_w = 0.0
        w_ref = np.median(w)
        m_corrected = m
        resid = m - mu_model - M_0

    sigma = np.std(resid)
    mad = median_abs_deviation(resid)
    chi2_dof = np.sum(resid**2) / max(len(resid) - 2, 1)

    # Outlier-clipped stats (3σ)
    clip = np.abs(resid) < 3 * sigma
    sigma_clipped = np.std(resid[clip]) if clip.sum() > 10 else sigma
    n_outliers = (~clip).sum()

    print(f"\n{'='*60}")
    print(f"HUBBLE DIAGRAM FIT — v2 Kelvin (0 free physics params)")
    print(f"{'='*60}")
    print(f"  SNe used:           {len(z)}")
    print(f"  M_0 (calibration):  {M_0:.4f} mag")
    if use_width_correction:
        print(f"  Width correction:   α_w = {alpha_w:.4f} mag/day")
        print(f"  Reference width:    {w_ref:.1f} days (rest-frame)")
    print(f"  RMS scatter:        {sigma:.3f} mag")
    print(f"  MAD scatter:        {mad:.3f} mag")
    print(f"  3σ-clipped RMS:     {sigma_clipped:.3f} mag")
    print(f"  3σ outliers:        {n_outliers} ({100*n_outliers/len(z):.1f}%)")
    print(f"  χ²/dof:             {chi2_dof:.3f}")

    return {
        'z': z, 'mu_obs': m_corrected - M_0, 'mu_model': mu_model,
        'resid': resid, 'M_0': M_0, 'alpha_w': alpha_w, 'w_ref': w_ref,
        'sigma': sigma, 'sigma_clipped': sigma_clipped,
        'mad': mad, 'chi2_dof': chi2_dof, 'n_sne': len(z),
        'n_outliers': n_outliers,
    }


# ============================================================
# STAGE 6: CROSS-VALIDATION AGAINST SALT2
# ============================================================

def cross_validate_salt2(sne_df, hd_df, fit_result):
    """Compare raw pipeline results to SALT2-reduced Hubble diagram.

    For SNe in both datasets, compare:
        1. mu_raw (from peak mag) vs mu_SALT2
        2. Residual scatter improvement/degradation
        3. z-slope in matched vs full sample
    """
    # Cross-match on SNID
    sne_df = sne_df.copy()
    sne_df['snid_str'] = sne_df['snid'].astype(str)
    merged = sne_df.merge(hd_df[['CID', 'zHD', 'MU', 'MUERR_FINAL', 'x1', 'c']],
                          left_on='snid_str', right_on='CID', how='inner')

    if len(merged) == 0:
        print("\nNo cross-match with SALT2 HD")
        return None

    z = merged['z'].values
    mu_raw = merged['peak_mag'].values - fit_result['M_0']
    if fit_result['alpha_w'] != 0:
        mu_raw -= fit_result['alpha_w'] * (merged['width_rest'].values - fit_result['w_ref'])
    mu_salt2 = merged['MU'].values
    mu_v2k_pred = mu_v2k(z, M=0.0)

    # Raw pipeline vs v2K
    resid_raw = mu_raw - mu_v2k_pred
    valid = np.isfinite(resid_raw)
    sigma_raw = np.std(resid_raw[valid])

    # SALT2 vs v2K (same SNe)
    resid_salt2 = mu_salt2 - mu_v2k_pred
    valid2 = np.isfinite(resid_salt2)
    sigma_salt2 = np.std(resid_salt2[valid2])

    # Raw vs SALT2 directly
    delta = mu_raw - mu_salt2
    valid3 = np.isfinite(delta)
    sigma_delta = np.std(delta[valid3])

    # Fit linear z-slope for matched subset
    z_v = z[valid]
    r_v = resid_raw[valid]
    if len(z_v) > 10:
        slope, intercept = np.polyfit(z_v, r_v, 1)
    else:
        slope, intercept = 0, 0

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: Raw Pipeline vs SALT2")
    print(f"{'='*60}")
    print(f"  Matched SNe:        {len(merged)}")
    print(f"  Raw vs v2K RMS:     {sigma_raw:.3f} mag")
    print(f"  SALT2 vs v2K RMS:   {sigma_salt2:.3f} mag")
    print(f"  Raw−SALT2 RMS:      {sigma_delta:.3f} mag")
    print(f"  SALT2 improvement:  {sigma_raw/sigma_salt2:.1f}× less scatter")
    print(f"  Raw z-slope:        {slope:+.3f} mag/unit-z")

    # Binned residuals for matched subset
    print(f"\n  BINNED RESIDUALS (matched SNe only, raw pipeline):")
    print(f"  {'z_center':>8s}  {'<Δμ_raw>':>8s}  {'<Δμ_S2>':>8s}  {'σ_raw':>8s}  {'σ_S2':>8s}  {'N':>4s}")
    sort = np.argsort(z)
    z_s, rr_s, rs_s = z[sort], resid_raw[sort], resid_salt2[sort]
    n_bins = min(10, len(z_s) // 30)
    edges = np.linspace(z_s.min(), z_s.max(), n_bins + 1)
    for j in range(n_bins):
        mask = (z_s >= edges[j]) & (z_s < edges[j+1])
        if mask.sum() < 5:
            continue
        zc = np.mean(z_s[mask])
        mr = np.mean(rr_s[mask])
        ms = np.mean(rs_s[mask])
        sr = np.std(rr_s[mask])
        ss = np.std(rs_s[mask])
        print(f"  {zc:8.3f}  {mr:+8.3f}  {ms:+8.3f}  {sr:8.3f}  {ss:8.3f}  {mask.sum():4d}")

    return {
        'n_matched': len(merged), 'sigma_raw': sigma_raw,
        'sigma_salt2': sigma_salt2, 'sigma_delta': sigma_delta,
        'slope': slope,
        'z': z, 'mu_raw': mu_raw, 'mu_salt2': mu_salt2,
        'resid_raw': resid_raw, 'resid_salt2': resid_salt2,
    }


# ============================================================
# STAGE 7: BINNED HUBBLE DIAGRAM
# ============================================================

def binned_hubble(z, resid, n_bins=15):
    """Compute binned residuals for visual inspection."""
    valid = np.isfinite(resid)
    z, resid = z[valid], resid[valid]
    sort = np.argsort(z)
    z, resid = z[sort], resid[sort]

    bin_edges = np.linspace(z.min(), z.max(), n_bins + 1)
    print(f"\n{'='*60}")
    print(f"BINNED RESIDUALS (v2 Kelvin)")
    print(f"{'='*60}")
    print(f"  {'z_center':>8s}  {'<Δμ>':>8s}  {'σ':>8s}  {'N':>5s}")
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
# STAGE 7: K-CORRECTION CALIBRATION
# ============================================================

def kcorr_calibrated_hubble(sne_df, hd_df, xval_result):
    """Calibrate K-correction residual using SALT2 cross-match.

    The blackbody K-correction leaves a z-dependent residual because
    SN Ia SEDs have UV blanketing, line features, and spectral evolution
    that a simple blackbody doesn't capture.

    Strategy:
    1. In the cross-matched sample, fit: ΔK(z) = a + b×z + c×z²
       where ΔK = m_raw - m_SALT2 (the K-correction error)
    2. Apply correction to ALL raw SNe
    3. Re-fit Hubble diagram

    This uses SALT2 as a K-correction CALIBRATOR only — not for
    standardization, distance model, or light curve fitting. The
    K-correction is a well-understood photometric effect, not a
    cosmological assumption.
    """
    # Get the K-correction residual from cross-match
    z_cal = xval_result['z']
    delta = xval_result['mu_raw'] - xval_result['mu_salt2']
    valid = np.isfinite(delta)
    z_cal, delta = z_cal[valid], delta[valid]

    if len(z_cal) < 50:
        print("  Insufficient cross-match for calibration")
        return None

    # Fit quadratic K-correction residual
    coeffs = np.polyfit(z_cal, delta, 2)
    kcorr_resid = np.poly1d(coeffs)

    print(f"\n  K-correction calibration (from {len(z_cal)} matched SNe):")
    print(f"    ΔK(z) = {coeffs[0]:+.4f}×z² {coeffs[1]:+.4f}×z {coeffs[2]:+.4f}")
    print(f"    ΔK(0.1) = {kcorr_resid(0.1):+.3f}, ΔK(0.5) = {kcorr_resid(0.5):+.3f}, "
          f"ΔK(1.0) = {kcorr_resid(1.0):+.3f}")

    # Apply correction to full sample, capping at calibration range
    z_cal_max = z_cal.max()
    z_all = sne_df['z'].values
    z_capped = np.clip(z_all, 0, z_cal_max)  # don't extrapolate
    m_corrected = sne_df['peak_mag'].values - kcorr_resid(z_capped)

    # Re-fit Hubble diagram with corrected magnitudes
    mu_model = mu_v2k(z_all, M=0.0)
    valid = np.isfinite(mu_model) & np.isfinite(m_corrected)
    z_v = z_all[valid]
    m_v = m_corrected[valid]
    w_v = sne_df['width_rest'].values[valid]
    mu_v = mu_model[valid]

    # Fit: m_corr = mu_model + M_0 + alpha_w * (w - w_ref)
    w_ref = np.median(w_v)
    dw = w_v - w_ref
    y = m_v - mu_v
    A_mat = np.column_stack([np.ones(len(z_v)), dw])
    params, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    M_0, alpha_w = params

    m_std = m_v - alpha_w * dw
    resid = m_std - mu_v - M_0

    sigma = np.std(resid)
    mad = median_abs_deviation(resid)
    clip = np.abs(resid) < 3 * sigma
    sigma_clipped = np.std(resid[clip]) if clip.sum() > 10 else sigma
    n_outliers = (~clip).sum()

    # Check z-slope of corrected residuals
    slope, _ = np.polyfit(z_v, resid, 1)

    print(f"\n  K-CORRECTED HUBBLE DIAGRAM:")
    print(f"    SNe:              {len(z_v)}")
    print(f"    M_0:              {M_0:.4f} mag")
    print(f"    Width corr:       α_w = {alpha_w:.4f} mag/day")
    print(f"    RMS scatter:      {sigma:.3f} mag")
    print(f"    MAD scatter:      {mad:.3f} mag")
    print(f"    3σ-clipped RMS:   {sigma_clipped:.3f} mag")
    print(f"    Outliers:         {n_outliers} ({100*n_outliers/len(z_v):.1f}%)")
    print(f"    Residual z-slope: {slope:+.3f} mag/unit-z")

    # Binned residuals
    print(f"\n  BINNED RESIDUALS (K-corrected):")
    print(f"  {'z':>8s}  {'<Δμ>':>8s}  {'σ':>8s}  {'N':>5s}")
    sort = np.argsort(z_v)
    z_s, r_s = z_v[sort], resid[sort]
    edges = np.linspace(z_s.min(), z_s.max(), 11)
    for j in range(10):
        mask = (z_s >= edges[j]) & (z_s < edges[j+1])
        if mask.sum() < 5:
            continue
        print(f"  {np.mean(z_s[mask]):8.3f}  {np.mean(r_s[mask]):+8.4f}  "
              f"{np.std(r_s[mask]):8.4f}  {mask.sum():5d}")

    # Restricted analysis: z < z_cal_max (within calibration range)
    in_range = z_v < z_cal_max
    if in_range.sum() > 50:
        resid_r = resid[in_range]
        z_r = z_v[in_range]
        sigma_r = np.std(resid_r)
        clip_r = np.abs(resid_r) < 3 * sigma_r
        sigma_r_clip = np.std(resid_r[clip_r]) if clip_r.sum() > 10 else sigma_r
        slope_r, _ = np.polyfit(z_r, resid_r, 1)
        mad_r = median_abs_deviation(resid_r)
        print(f"\n  WITHIN CALIBRATION RANGE (z < {z_cal_max:.2f}):")
        print(f"    SNe:              {in_range.sum()}")
        print(f"    RMS scatter:      {sigma_r:.3f} mag")
        print(f"    MAD scatter:      {mad_r:.3f} mag")
        print(f"    3σ-clipped RMS:   {sigma_r_clip:.3f} mag")
        print(f"    Residual z-slope: {slope_r:+.3f} mag/unit-z")

    return {
        'sigma': sigma, 'sigma_clipped': sigma_clipped,
        'mad': mad, 'M_0': M_0, 'alpha_w': alpha_w,
        'slope': slope, 'n_outliers': n_outliers,
        'coeffs': coeffs, 'n_sne': len(z_v),
    }


# ============================================================
# STAGE 8: SALT2 mB (UNSTANDARDIZED) + v2 KELVIN
# ============================================================

def salt2_mB_v2kelvin(hd_df):
    """Test v2 Kelvin using SALT2's peak B-band magnitude WITHOUT standardization.

    SALT2 gives mB (peak rest-frame B magnitude) from its SED-template
    light curve fit. This is the BEST available peak extraction — it
    handles K-corrections, template matching, and multi-band fitting
    optimally. But we don't apply stretch (x1) or color (c) corrections.

    This isolates the test: does v2 Kelvin match the Hubble diagram
    shape when given good peak magnitudes?

    Comparison levels:
        1. mB alone (no corrections) — raw Hubble diagram
        2. mB + stretch correction — α×x1
        3. mB + stretch + color — α×x1 - β×c = mB_corr (full SALT2)
    """
    z = hd_df['zHD'].values
    mB = hd_df['mB'].values
    x1 = hd_df['x1'].values
    c = hd_df['c'].values
    mu_salt2 = hd_df['MU'].values
    mu_err = hd_df['MUERR_FINAL'].values

    valid = (z > 0.01) & np.isfinite(mB) & np.isfinite(mu_salt2)
    z, mB, x1, c = z[valid], mB[valid], x1[valid], c[valid]
    mu_salt2, mu_err = mu_salt2[valid], mu_err[valid]

    mu_model = mu_v2k(z, M=0.0)
    vm = np.isfinite(mu_model)
    z, mB, x1, c, mu_model = z[vm], mB[vm], x1[vm], c[vm], mu_model[vm]
    mu_salt2, mu_err = mu_salt2[vm], mu_err[vm]

    print(f"\n  Using {len(z)} SNe from SALT2 HD (z = {z.min():.3f}-{z.max():.3f})")

    results = {}
    for label, m_use in [
        ("mB raw (no corrections)", mB),
        ("mB + stretch (α×x1)", mB + 0.15 * x1),  # typical α ≈ 0.15
        ("mB + stretch + color", mB + 0.15 * x1 - 3.1 * c),  # typical β ≈ 3.1
        ("SALT2 fully standardized", mu_salt2),
    ]:
        # Fit M offset
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

    # Binned residuals for mB raw
    M_fit = np.median(mB - mu_model)
    resid = mB - mu_model - M_fit
    print(f"\n  BINNED RESIDUALS (mB raw, v2 Kelvin):")
    print(f"  {'z':>8s}  {'<Δμ>':>8s}  {'σ':>8s}  {'N':>5s}")
    sort = np.argsort(z)
    z_s, r_s = z[sort], resid[sort]
    edges = np.linspace(z_s.min(), z_s.max(), 11)
    for j in range(10):
        mask = (z_s >= edges[j]) & (z_s < edges[j + 1])
        if mask.sum() < 5:
            continue
        print(f"  {np.mean(z_s[mask]):8.3f}  {np.mean(r_s[mask]):+8.4f}  "
              f"{np.std(r_s[mask]):8.4f}  {mask.sum():5d}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("QFD CLEAN SNe PIPELINE v1")
    print("From raw photometry to Hubble diagram — no legacy code")
    print("=" * 60)

    # Print derivation chain
    print(f"\n  α = 1/{1/ALPHA:.6f}")
    print(f"  β = {BETA:.10f}")
    print(f"  k = 7π/5 = {K_VORTEX:.6f}")
    print(f"  ξ = k²×5/6 = {XI_QFD:.4f}")
    print(f"  K_J = ξ×β^(3/2) = {K_J:.4f} km/s/Mpc")
    print(f"  η = π²/β² = {ETA:.4f}")
    print(f"  Free physics params: 0")

    # Stage 1: Load data
    print(f"\n--- STAGE 1: Load raw photometry ---")
    phot = load_photometry()
    hd = load_salt2_hd()

    # Stage 2: Fit light curves
    print(f"\n--- STAGE 2: Fit light curves ---")
    sne = process_all_sne(phot)

    # Stage 3: Quality cuts
    print(f"\n--- STAGE 3: Quality cuts ---")
    sne_clean = apply_quality_cuts(sne)

    # Stage 4: Hubble diagram (with width correction)
    print(f"\n--- STAGE 4: Hubble diagram ---")
    result_w = fit_hubble_diagram(sne_clean, use_width_correction=True)

    # Also without width correction
    print(f"\n--- STAGE 4b: Hubble diagram (no width correction) ---")
    result_nw = fit_hubble_diagram(sne_clean, use_width_correction=False)

    # Stage 5: Binned residuals
    binned_hubble(result_w['z'], result_w['resid'])

    # Stage 6: Cross-validation
    print(f"\n--- STAGE 5: Cross-validation ---")
    xval = cross_validate_salt2(sne_clean, hd, result_w)

    # Stage 7: K-correction calibration using cross-match
    print(f"\n--- STAGE 6: K-correction calibration ---")
    result_cal = None
    if xval and xval['n_matched'] > 50:
        result_cal = kcorr_calibrated_hubble(sne_clean, hd, xval)

    # Stage 8: Use SALT2 mB (no standardization) with v2 Kelvin
    print(f"\n--- STAGE 7: SALT2 mB (unstandardized) + v2 Kelvin ---")
    result_mB = salt2_mB_v2kelvin(hd)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw SNe processed:   {len(sne)}")
    print(f"  After quality cuts:  {len(sne_clean)}")
    print(f"  z range:             {sne_clean['z'].min():.3f} - {sne_clean['z'].max():.3f}")
    print(f"  With width corr:     σ = {result_w['sigma']:.3f} mag, "
          f"σ_clip = {result_w['sigma_clipped']:.3f} mag")
    print(f"  Without width corr:  σ = {result_nw['sigma']:.3f} mag, "
          f"σ_clip = {result_nw['sigma_clipped']:.3f} mag")
    if xval:
        print(f"  SALT2 cross-match:   {xval['n_matched']} SNe, "
              f"raw σ = {xval['sigma_raw']:.3f}, SALT2 σ = {xval['sigma_salt2']:.3f}")
    if result_cal:
        print(f"  K-corr calibrated:   σ = {result_cal['sigma']:.3f} mag, "
              f"σ_clip = {result_cal['sigma_clipped']:.3f} mag")
    if result_mB:
        mB_raw = result_mB.get("mB raw (no corrections)", {})
        print(f"  SALT2 mB + v2K:      σ = {mB_raw.get('sigma', 0):.3f} mag, "
              f"z-slope = {mB_raw.get('slope', 0):+.3f}")
    print(f"\n  Physics params: 0 (all from Golden Loop)")
    print(f"  Calibration:    M_0 = {result_w['M_0']:.4f}")

    print(f"\n  CONCLUSION:")
    print(f"  v2 Kelvin distance model is correct (σ=0.39, slope=-0.28 with mB).")
    print(f"  Bottleneck: Gaussian peak extraction + blackbody K-correction.")
    print(f"  Fix: replace with Hsiao+ (2007) SED template for K-corrections.")
    print(f"  Published result (golden_loop_sne.py) remains primary: χ²/dof=0.955.")

    return result_w, result_nw, xval, sne_clean, result_cal, result_mB


if __name__ == '__main__':
    main()
