#!/usr/bin/env python3
"""
qfd_transfer_pipeline.py — QFD Vacuum Transfer Function SNe Pipeline
=====================================================================

SALT2-FREE supernova distance pipeline using QFD physics only.

Forward model: The QFD vacuum distorts intrinsic SN Ia light via two mechanisms:
    Prong 1 (achromatic): (1+z)^{1/3} kinematic time dilation (f=2 vortex ring)
    Prong 2 (chromatic):  exp[-η τ (λ_ref/λ)^{1/2}] selective blue erosion

    F_obs(t, λ) = A × I₀(t/(1+z)^{1/3}, λ/(1+z)) × exp[-η τ(z) (λ_B/λ)^{1/2}]

    where τ(z) = 1 - 1/√(1+z), η = π²/β² = 1.066, λ_B = 4400 Å

All physics parameters locked from Golden Loop (0 free):
    α → β → K_J, η, time dilation exponent 1/3, chromatic exponent 1/2

Fit parameters per SN: t0 (peak time), amplitude (encodes distance)
Calibration: M_0 (absolute magnitude, 1 free param total)

Template: Hsiao+ (2007) SN Ia spectral time series (sncosmo)
Data: DES-SN5YR raw photometry (5,468 SNe, griz, host-subtracted)

Author: Tracy + Claude
Date: 2026-02-20
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
import sncosmo
from astropy.table import Table
import os
import time as timer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS (Golden Loop — zero free parameters)
# ============================================================

PI = np.pi
ALPHA = 1.0 / 137.035999084
C_KMS = 299792.458


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
K_VORTEX = 7.0 * PI / 5.0
XI_QFD = K_VORTEX**2 * (5.0 / 6.0)
K_J = XI_QFD * BETA**1.5
ETA = PI**2 / BETA**2                     # 1.0657 — locked scattering opacity
K_MAG = 5.0 / np.log(10.0)               # 2.172 — magnitude convention
TIME_DIL_EXP = 1.0 / 3.0                 # QFD: (1+z)^{1/3}, NOT (1+z)
CHROM_EXP = 0.5                           # σ_nf ∝ λ^{-1/2}
LAM_REF = 4400.0                          # B-band reference wavelength (Å)

BAND_MAP = {'g': 'desg', 'r': 'desr', 'i': 'desi', 'z': 'desz'}

# ============================================================
# v2 KELVIN DISTANCE MODEL
# ============================================================


def mu_v2k(z, M=0.0, include_chromatic=True):
    """v2 Kelvin distance modulus.

    If include_chromatic=False, returns ACHROMATIC distance modulus only.
    Use achromatic when the fitter already applies per-band chromatic extinction.
    """
    z = np.asarray(z, dtype=float)
    D_prop = (C_KMS / K_J) * np.log1p(z)
    D_L = D_prop * (1.0 + z)**(2.0 / 3.0)
    mu = np.full_like(z, np.nan, dtype=float)
    mask = D_L > 0
    mu[mask] = 5.0 * np.log10(D_L[mask]) + 25.0 + M
    if include_chromatic:
        tau = 1.0 - 1.0 / np.sqrt(1.0 + z)
        mu[mask] += K_MAG * ETA * tau[mask]
    return mu


# ============================================================
# QFD TRANSFER FUNCTION MODEL
# ============================================================

HSIAO_SOURCE = sncosmo.get_source('hsiao')


def build_qfd_model_lcs(z, phase_min=-20, phase_max=80, n_phase=200):
    """Pre-compute QFD model light curves for a SN at redshift z.

    Returns dict of {band: flux_array} indexed by rest-frame phase grid.
    The flux includes the spectral redshift and bandwidth compression
    but NOT the chromatic extinction (applied separately during fitting).

    The time dilation enters during fitting when we map t_obs → phase_rest.
    """
    model = sncosmo.Model(source=HSIAO_SOURCE)
    model.set(z=z, t0=0, amplitude=1.0)

    phase_grid = np.linspace(phase_min, phase_max, n_phase)
    # sncosmo expects observer-frame time: t = phase × (1+z)
    t_obs_grid = phase_grid * (1 + z)

    band_lcs = {}
    for band_name in ['desg', 'desr', 'desi', 'desz']:
        fluxes = np.zeros(n_phase)
        for j, t in enumerate(t_obs_grid):
            try:
                fluxes[j] = model.bandflux(band_name, t, zp=8.9, zpsys='ab')
            except Exception:
                fluxes[j] = 0.0
        band_lcs[band_name] = fluxes

    return phase_grid, band_lcs


def chromatic_factor(z, lam_eff_angstrom):
    """QFD chromatic extinction factor for a band at effective wavelength λ.

    chromatic = 10^{-0.4 × K_MAG × η × τ(z) × (λ_B/λ)^{1/2}}

    Returns the FLUX multiplication factor (< 1 for extinction).
    """
    tau = 1.0 - 1.0 / np.sqrt(1.0 + z)
    delta_m = K_MAG * ETA * tau * (LAM_REF / lam_eff_angstrom)**CHROM_EXP
    return 10.0**(-0.4 * delta_m)


def get_band_chromatic(z):
    """Pre-compute chromatic extinction factors for DES bands at redshift z."""
    factors = {}
    for band_name in ['desg', 'desr', 'desi', 'desz']:
        bp = sncosmo.get_bandpass(band_name)
        factors[band_name] = chromatic_factor(z, bp.wave_eff)
    return factors


# ============================================================
# DATA LOADING
# ============================================================

DATA_DIR = '/home/tracy/development/SupernovaSrc/qfd-supernova-v15/data'
PHOTOMETRY_FILE = os.path.join(DATA_DIR, 'lightcurves_unified_v2_min3.csv')
HD_FILE = os.path.join(DATA_DIR, 'DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv')


def load_data():
    """Load photometry and SALT2 HD (for cross-validation only)."""
    phot = pd.read_csv(PHOTOMETRY_FILE)
    phot['flux_err_true'] = phot['flux_nu_jy'] / phot['snr']
    min_err = phot.groupby('snid')['flux_err_true'].transform('median') * 0.1
    phot['flux_err_true'] = np.maximum(phot['flux_err_true'], min_err)
    phot['flux_err_true'] = np.maximum(phot['flux_err_true'], phot['flux_nu_jy'] * 0.01)

    hd = pd.read_csv(HD_FILE)
    hd['CID'] = hd['CID'].astype(str)

    print(f"  Photometry: {len(phot):,} obs, {phot['snid'].nunique():,} SNe")
    print(f"  SALT2 HD:   {len(hd)} SNe (cross-validation only)")
    return phot, hd


# ============================================================
# QFD FITTER
# ============================================================

def fit_sn_qfd(sn_data, z, phase_grid, band_lcs, band_chrom):
    """Fit QFD vacuum transfer function to raw multi-band photometry.

    The QFD rest-frame phase is:
        φ = (t_obs - t0) / (1+z)^{1/3}    [NOT (1+z)]

    The predicted flux at observation (t, band) is:
        F_pred = amplitude × LC(φ, band) × chromatic(band, z)

    where LC is the Hsiao template light curve and chromatic is the
    wavelength-dependent extinction from Kelvin wave scattering.

    Fitted: t0, amplitude (2 params)
    Fixed: z, η, 1/3, 1/2, λ_ref (all from Golden Loop)
    """
    # Exclude g-band at z > 0.5 (probes rest-frame UV < 3200 Å, unreliable)
    allowed_bands = ['r', 'i', 'z'] if z > 0.5 else ['g', 'r', 'i', 'z']
    valid = sn_data['band'].isin(allowed_bands) & (sn_data['flux_nu_jy'] > 0)
    sv = sn_data[valid]
    if len(sv) < 5:
        return None

    t_data = sv['mjd'].values.astype(float)
    f_data = sv['flux_nu_jy'].values.astype(float)
    f_err = sv['flux_err_true'].values.astype(float)
    bands = [BAND_MAP[b] for b in sv['band'].values]

    # Apply per-band chromatic factors during fitting (QFD λ^{-1/2} extinction)
    # Use ACHROMATIC mu for comparison to avoid double-counting
    chrom = np.array([band_chrom[b] for b in bands])
    band_indices = {}
    for b in set(bands):
        band_indices[b] = np.array([i for i, bb in enumerate(bands) if bb == b])

    def compute_chi2(t0, amplitude):
        """Vectorized chi² computation."""
        # QFD rest-frame phase: (t - t0) / (1+z)^{1/3}
        phase = (t_data - t0) / (1.0 + z)**TIME_DIL_EXP

        # Interpolate model light curve for each band
        f_model = np.zeros(len(t_data))
        for b, idx in band_indices.items():
            f_model[idx] = np.interp(phase[idx], phase_grid, band_lcs[b],
                                     left=0.0, right=0.0)

        # Predicted flux: amplitude × model × chromatic
        f_pred = amplitude * f_model * chrom
        return np.sum(((f_data - f_pred) / f_err)**2)

    def neg_log_likelihood(params):
        t0, log_amp = params
        return compute_chi2(t0, np.exp(log_amp))

    # Initial guess: t0 from peak flux, amplitude from peak matching
    idx_peak = np.argmax(f_data)
    t0_guess = t_data[idx_peak]

    peak_band = bands[idx_peak]
    f_model_peak = np.interp(0.0, phase_grid, band_lcs[peak_band])
    amp_guess = f_data[idx_peak] / (f_model_peak * band_chrom[peak_band]) \
        if f_model_peak > 0 else 1e-10
    amp_guess = max(amp_guess, 1e-15)

    # Grid search on t0 (±15 days), analytical amplitude at each
    best_chi2 = np.inf
    best_t0 = t0_guess
    best_amp = amp_guess

    for dt in np.linspace(-15, 15, 31):
        t0_try = t0_guess + dt
        phase = (t_data - t0_try) / (1.0 + z)**TIME_DIL_EXP

        f_model = np.zeros(len(t_data))
        for b, idx in band_indices.items():
            f_model[idx] = np.interp(phase[idx], phase_grid, band_lcs[b],
                                     left=0.0, right=0.0)

        # Analytical amplitude: A = Σ(d×m×c/σ²) / Σ(m²×c²/σ²)
        mc = f_model * chrom
        w = mc / f_err**2
        denom = np.sum(mc * w)
        if denom <= 0:
            continue
        A_opt = np.sum(f_data * w) / denom

        if A_opt <= 0:
            continue

        f_pred = A_opt * mc
        chi2 = np.sum(((f_data - f_pred) / f_err)**2)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_t0 = t0_try
            best_amp = A_opt

    # Refine with Nelder-Mead
    try:
        result = minimize(neg_log_likelihood,
                          [best_t0, np.log(best_amp)],
                          method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 0.01, 'maxiter': 300})
        if result.fun < best_chi2:
            best_t0 = result.x[0]
            best_amp = np.exp(result.x[1])
            best_chi2 = result.fun
    except Exception:
        pass

    ndof = max(len(t_data) - 2, 1)

    # Peak rest-frame B magnitude from fitted amplitude
    model = sncosmo.Model(source=HSIAO_SOURCE)
    model.set(z=z, t0=0, amplitude=best_amp)
    try:
        m_B = model.source_peakmag('bessellb', 'ab')
    except Exception:
        m_B = -2.5 * np.log10(best_amp) + 20.0  # rough fallback

    return {
        'peak_mag': m_B,
        't0': best_t0,
        'amplitude': best_amp,
        'chi2_dof': best_chi2 / ndof,
        'n_obs': len(t_data),
        'n_bands': len(set(bands)),
    }


# ============================================================
# CHROMATIC DIAGNOSTIC
# ============================================================

def chromatic_diagnostic(sne_df, phase_grids, band_lc_cache, band_chrom_cache, phot):
    """Measure per-band residuals to test the λ^{-1/2} chromatic prediction.

    For each SN, compute the residual in each band independently.
    If QFD is correct, the g-band should show MORE dimming than z-band,
    with the ratio fixed by (λ_z/λ_g)^{1/2} = 1.40.
    """
    print(f"\n{'='*60}")
    print(f"CHROMATIC DIAGNOSTIC — λ^{{-1/2}} test")
    print(f"{'='*60}")

    band_resids = {b: [] for b in ['desg', 'desr', 'desi', 'desz']}
    band_z = {b: [] for b in ['desg', 'desr', 'desi', 'desz']}

    for _, row in sne_df.iterrows():
        snid = row['snid']
        z = row['z']
        t0 = row['t0']
        amp = row['amplitude']

        sn = phot[phot['snid'] == snid]
        z_key = round(z, 3)
        if z_key not in phase_grids:
            continue

        phase_grid = phase_grids[z_key]
        blcs = band_lc_cache[z_key]
        bchrom = band_chrom_cache[z_key]

        for band_letter in ['g', 'r', 'i', 'z']:
            band_name = BAND_MAP[band_letter]
            bd = sn[sn['band'] == band_letter]
            if len(bd) < 3:
                continue

            t = bd['mjd'].values.astype(float)
            f = bd['flux_nu_jy'].values.astype(float)
            fe = bd['flux_err_true'].values.astype(float)

            phase = (t - t0) / (1.0 + z)**TIME_DIL_EXP
            f_model = np.interp(phase, phase_grid, blcs[band_name], left=0, right=0)
            f_pred = amp * f_model * bchrom[band_name]

            # Per-band magnitude residual (data - model)
            mask = (f > 0) & (f_pred > 0)
            if mask.sum() < 2:
                continue

            m_data = -2.5 * np.log10(np.mean(f[mask]))
            m_pred = -2.5 * np.log10(np.mean(f_pred[mask]))
            band_resids[band_name].append(m_data - m_pred)
            band_z[band_name].append(z)

    # Report
    print(f"\n  Per-band mean residuals (data - QFD model, mag):")
    print(f"  If QFD chromatic is correct, all bands should be ~0.")
    print(f"  If achromatic only, g should be positive (under-corrected), z negative.")
    print(f"  {'Band':>6s}  {'λ_eff':>6s}  {'(λ_B/λ)^½':>8s}  {'<Δm>':>8s}  {'σ':>8s}  {'N':>5s}")

    for band_name, letter in [('desg','g'), ('desr','r'), ('desi','i'), ('desz','z')]:
        r = np.array(band_resids[band_name])
        if len(r) > 5:
            bp = sncosmo.get_bandpass(band_name)
            chrom_factor = (LAM_REF / bp.wave_eff)**0.5
            print(f"  {letter:>6s}  {bp.wave_eff:6.0f}  {chrom_factor:8.3f}  "
                  f"{np.mean(r):+8.4f}  {np.std(r):8.4f}  {len(r):5d}")

    # Binned by z
    print(f"\n  Color excess (g-z residual) vs z:")
    print(f"  QFD predicts: Δ(g-z) increases with z (blue eroded more at higher z)")
    print(f"  {'z':>6s}  {'Δm_g':>8s}  {'Δm_z':>8s}  {'Δ(g-z)':>8s}  {'N':>5s}")

    z_g = np.array(band_z['desg'])
    r_g = np.array(band_resids['desg'])
    z_z = np.array(band_z['desz'])
    r_z = np.array(band_resids['desz'])

    if len(z_g) > 20 and len(z_z) > 20:
        edges = np.linspace(0.1, 1.2, 8)
        for j in range(len(edges) - 1):
            mg = r_g[(z_g >= edges[j]) & (z_g < edges[j+1])]
            mz = r_z[(z_z >= edges[j]) & (z_z < edges[j+1])]
            if len(mg) > 3 and len(mz) > 3:
                zc = (edges[j] + edges[j+1]) / 2
                print(f"  {zc:6.2f}  {np.mean(mg):+8.4f}  {np.mean(mz):+8.4f}  "
                      f"{np.mean(mg)-np.mean(mz):+8.4f}  {min(len(mg),len(mz)):5d}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("QFD VACUUM TRANSFER FUNCTION — SNe PIPELINE")
    print("SALT2-free. Zero free physics parameters.")
    print("=" * 60)

    print(f"\n  LOCKED CONSTANTS (from Golden Loop):")
    print(f"    α = 1/{1/ALPHA:.6f}")
    print(f"    β = {BETA:.10f}")
    print(f"    K_J = {K_J:.4f} km/s/Mpc")
    print(f"    η = π²/β² = {ETA:.4f}")
    print(f"    Time dilation: (1+z)^{{1/3}}  [f=2 vortex]")
    print(f"    Chromatic: σ_nf ∝ λ^{{-1/2}}  [Kelvin wave]")
    print(f"    λ_ref = {LAM_REF:.0f} Å (B-band)")
    print(f"    Free physics params: 0")

    # Chromatic factors at sample redshifts
    print(f"\n  CHROMATIC EXTINCTION (flux factor per band):")
    print(f"  {'z':>5s}  {'g':>8s}  {'r':>8s}  {'i':>8s}  {'z_band':>8s}  {'Δ(g-z)':>8s}")
    for z_ex in [0.1, 0.3, 0.5, 0.7, 1.0]:
        bc = get_band_chromatic(z_ex)
        dg = -2.5 * np.log10(bc['desg'])
        dz = -2.5 * np.log10(bc['desz'])
        print(f"  {z_ex:5.1f}  {bc['desg']:8.4f}  {bc['desr']:8.4f}  "
              f"{bc['desi']:8.4f}  {bc['desz']:8.4f}  {dg-dz:+8.4f} mag")

    # Load data
    print(f"\n--- STAGE 1: Load raw photometry ---")
    phot, hd = load_data()

    # Process all SNe
    print(f"\n--- STAGE 2: QFD transfer function fitting ---")
    sne_groups = phot.groupby('snid')
    n_total = len(sne_groups)

    results = []
    phase_grids = {}
    band_lc_cache = {}
    band_chrom_cache = {}
    n_success = 0
    n_fail = 0
    t_start = timer.time()

    for i, (snid, sn_data) in enumerate(sne_groups):
        if (i + 1) % 500 == 0:
            elapsed = timer.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  {i+1}/{n_total} ({rate:.0f} SN/s, ETA {eta:.0f}s)  "
                  f"[ok:{n_success} fail:{n_fail}]")

        z = sn_data['z'].iloc[0]
        if z <= 0.01 or z > 1.6:
            continue

        # Cache model light curves per unique z (rounded to 0.001)
        z_key = round(z, 3)
        if z_key not in phase_grids:
            pg, blc = build_qfd_model_lcs(z_key)
            phase_grids[z_key] = pg
            band_lc_cache[z_key] = blc
            band_chrom_cache[z_key] = get_band_chromatic(z_key)

        fit = fit_sn_qfd(sn_data, z,
                         phase_grids[z_key],
                         band_lc_cache[z_key],
                         band_chrom_cache[z_key])

        if fit is None:
            n_fail += 1
            continue

        n_success += 1
        results.append({
            'snid': snid, 'z': z,
            'peak_mag': fit['peak_mag'],
            't0': fit['t0'],
            'amplitude': fit['amplitude'],
            'chi2_dof': fit['chi2_dof'],
            'n_obs': fit['n_obs'],
            'n_bands': fit['n_bands'],
        })

    elapsed = timer.time() - t_start
    sne_df = pd.DataFrame(results)
    print(f"\n  Fitted {len(sne_df)} / {n_total} SNe in {elapsed:.0f}s ({n_total/elapsed:.0f} SN/s)")
    print(f"  Failed: {n_fail}")
    print(f"  Unique z values cached: {len(phase_grids)}")

    # Quality cuts
    print(f"\n--- STAGE 3: Quality cuts ---")
    n_before = len(sne_df)
    mask = (
        (sne_df['chi2_dof'] < 50) &
        (sne_df['n_obs'] >= 5) &
        (sne_df['n_bands'] >= 2) &
        (sne_df['z'] > 0.02) &
        np.isfinite(sne_df['peak_mag']) &
        (sne_df['peak_mag'] > 15) &
        (sne_df['peak_mag'] < 30)
    )
    sne_clean = sne_df[mask].copy().reset_index(drop=True)
    print(f"  {n_before} → {len(sne_clean)} SNe")

    # Hubble diagram — ACHROMATIC mu (chromatic already in per-band fitter)
    print(f"\n--- STAGE 4: Hubble diagram ---")
    z = sne_clean['z'].values
    m = sne_clean['peak_mag'].values

    # Chromatic already applied per-band in fitter → use achromatic mu
    mu_model = mu_v2k(z, M=0.0, include_chromatic=False)
    valid = np.isfinite(mu_model) & np.isfinite(m)
    z, m, mu_model = z[valid], m[valid], mu_model[valid]

    M_0 = np.median(m - mu_model)
    resid = m - mu_model - M_0

    def report_hubble(label, zz, rr):
        sigma = np.std(rr)
        mad = median_abs_deviation(rr)
        clip = np.abs(rr) < 3 * sigma
        sigma_clip = np.std(rr[clip]) if clip.sum() > 10 else sigma
        slope, _ = np.polyfit(zz, rr, 1) if len(zz) > 10 else (0, 0)
        print(f"\n  {'='*55}")
        print(f"  {label}")
        print(f"  {'='*55}")
        print(f"  SNe:              {len(zz)}")
        print(f"  M_0:              {M_0:.4f}")
        print(f"  σ:                {sigma:.3f} mag")
        print(f"  MAD:              {mad:.3f} mag")
        print(f"  σ_clip:           {sigma_clip:.3f} mag")
        print(f"  z-slope:          {slope:+.3f} mag/z")
        print(f"  Outliers (3σ):    {(~clip).sum()} ({100*(~clip).sum()/len(zz):.1f}%)")
        return sigma, sigma_clip, slope

    sigma, sigma_clip, slope = report_hubble(
        "QFD HUBBLE — (1+z)^{1/3} + chromatic λ^{-1/2}", z, resid)

    # Binned residuals
    print(f"\n  BINNED RESIDUALS:")
    print(f"  {'z':>8s}  {'<Δμ>':>8s}  {'σ':>8s}  {'N':>5s}")
    sort = np.argsort(z)
    z_s, r_s = z[sort], resid[sort]
    edges = np.linspace(z_s.min(), z_s.max(), 16)
    for j in range(len(edges) - 1):
        bmask = (z_s >= edges[j]) & (z_s < edges[j+1])
        if bmask.sum() < 3:
            continue
        print(f"  {np.mean(z_s[bmask]):8.3f}  {np.mean(r_s[bmask]):+8.4f}  "
              f"{np.std(r_s[bmask]):8.4f}  {bmask.sum():5d}")

    # CONTROL: full mu (chromatic in BOTH fitter and mu — double-count test)
    mu_full = mu_v2k(z, M=0.0, include_chromatic=True)
    M_0_full = np.median(m - mu_full)
    resid_full = m - mu_full - M_0_full
    report_hubble(
        "CONTROL — chromatic in fit AND μ (double-count)", z, resid_full)

    # Cross-validation against SALT2
    print(f"\n--- STAGE 5: Cross-validation (SALT2 as reference only) ---")
    sne_clean_cv = sne_clean.copy()
    sne_clean_cv['snid_str'] = sne_clean_cv['snid'].astype(str)
    merged = sne_clean_cv.merge(hd[['CID', 'zHD', 'MU', 'mB']],
                                 left_on='snid_str', right_on='CID', how='inner')

    if len(merged) > 10:
        z_m = merged['z'].values
        mu_qfd = merged['peak_mag'].values - M_0
        mu_salt2 = merged['MU'].values
        mu_pred = mu_v2k(z_m, M=0.0, include_chromatic=False)
        mu_pred_s2 = mu_v2k(z_m, M=0.0, include_chromatic=True)

        resid_qfd = mu_qfd - mu_pred
        resid_s2 = mu_salt2 - mu_pred_s2
        valid_q = np.isfinite(resid_qfd)
        valid_s = np.isfinite(resid_s2)

        sigma_qfd = np.std(resid_qfd[valid_q])
        sigma_s2 = np.std(resid_s2[valid_s])

        slope_q, _ = np.polyfit(z_m[valid_q], resid_qfd[valid_q], 1)
        slope_s, _ = np.polyfit(z_m[valid_s], resid_s2[valid_s], 1)

        # mB comparison
        delta_mB = merged['peak_mag'].values - merged['mB'].values
        valid_mb = np.isfinite(delta_mB)

        print(f"  Matched: {len(merged)} SNe")
        print(f"  QFD vs v2K (achrom μ):    σ = {sigma_qfd:.3f}, slope = {slope_q:+.3f}")
        print(f"  SALT2 vs v2K (full μ):    σ = {sigma_s2:.3f}, slope = {slope_s:+.3f}")
        print(f"  Gap: {sigma_qfd/sigma_s2:.1f}×")
        print(f"  QFD−SALT2 mB: offset = {np.median(delta_mB[valid_mb]):+.3f}, "
              f"σ = {np.std(delta_mB[valid_mb]):.3f}")

        # Binned
        print(f"\n  {'z':>8s}  {'<Δμ_QFD>':>8s}  {'<Δμ_S2>':>8s}  {'N':>4s}")
        sort_m = np.argsort(z_m)
        z_ms, rq_s, rs_s = z_m[sort_m], resid_qfd[sort_m], resid_s2[sort_m]
        n_bins = min(10, len(z_ms) // 20)
        edges_m = np.linspace(z_ms.min(), z_ms.max(), n_bins + 1)
        for j in range(n_bins):
            bmask = (z_ms >= edges_m[j]) & (z_ms < edges_m[j+1])
            if bmask.sum() < 5:
                continue
            print(f"  {np.mean(z_ms[bmask]):8.3f}  {np.mean(rq_s[bmask]):+8.4f}  "
                  f"{np.mean(rs_s[bmask]):+8.4f}  {bmask.sum():4d}")

    # Chromatic diagnostic
    print(f"\n--- STAGE 6: Chromatic diagnostic ---")
    chromatic_diagnostic(sne_clean, phase_grids, band_lc_cache, band_chrom_cache, phot)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — QFD Transfer Function Pipeline")
    print(f"{'='*60}")
    print(f"  Model: F = A × I₀(t/(1+z)^{{1/3}}, λ/(1+z)) × exp[-ητ(λ_B/λ)^{{1/2}}]")
    print(f"  SNe fitted:       {len(sne_clean)}")
    print(f"  z range:          {sne_clean['z'].min():.3f} - {sne_clean['z'].max():.3f}")
    print(f"  σ (achromatic μ): {sigma:.3f} mag")
    print(f"  σ_clip:           {sigma_clip:.3f} mag")
    print(f"  z-slope:          {slope:+.3f} mag/z")
    print(f"  M_0:              {M_0:.4f}")
    print(f"  Free physics:     0")
    print(f"  SALT2 used:       NO (cross-validation only)")

    return sne_clean, resid, z, M_0


if __name__ == '__main__':
    main()
