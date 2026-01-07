#!/usr/bin/env python3
"""
SNe Ia Photon Model Comparison: RAW Light Curve Data

Uses raw light curve data WITHOUT SALT processing to avoid ΛCDM bias.
Tests if the ψ field coupling is real or a processing artifact.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.constants import c as c_mps

C_KM_S = c_mps / 1000.0
BETA_GOLDEN = 3.058230856

# =============================================================================
# MODELS
# =============================================================================

class ModelB_Lean4:
    """Basic Lean4: ln(1+z) = κD"""
    name = "B: Lean4"

    def __init__(self, H0=70.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S
        self.n_params = 0

    def distance_modulus(self, z):
        z = np.atleast_1d(z)
        D = np.log(1 + z) / self.kappa
        D_L = D * (1 + z)
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


class ModelE_PhotonPsi:
    """Photon + ψ field coupling"""
    name = "E: Photon+ψ"

    def __init__(self, H0=70.0, g_psi=0.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S
        self.g_psi = g_psi
        self.n_params = 1 if g_psi != 0 else 0

    def distance_modulus(self, z):
        z = np.atleast_1d(z)
        D = np.log(1 + z) / self.kappa
        D_L = D * (1 + z)
        mu = 5 * np.log10(np.maximum(D_L, 1e-10)) + 25
        # ψ field correction
        if self.g_psi != 0:
            mu += self.g_psi * np.log(1 + z)
        return mu


class ModelD_LCDM:
    """ΛCDM reference"""
    name = "D: ΛCDM"

    def __init__(self, H0=70.0, Omega_m=0.3):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m
        self.n_params = 1

    def _E(self, z):
        return np.sqrt(self.Omega_m * (1+z)**3 + self.Omega_L)

    def distance_modulus(self, z):
        from scipy.integrate import quad
        z = np.atleast_1d(z)
        D_L = np.zeros_like(z, dtype=float)
        for i, zi in enumerate(z):
            if zi > 0:
                integral, _ = quad(lambda zp: 1/self._E(zp), 0, zi)
                D_L[i] = (C_KM_S / self.H0) * (1 + zi) * integral
            else:
                D_L[i] = 1e-10
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


# =============================================================================
# RAW DATA LOADING
# =============================================================================

def load_raw_lightcurves():
    """Load raw light curve data."""
    # Try multiple locations
    lc_paths = [
        Path("/home/tracy/development/QFD_SpectralGap/projects/astrophysics/V21 Supernova Analysis package/data/lightcurves_all_transients.csv"),
        Path("/home/tracy/development/QFD_SpectralGap/data/lightcurves_unified_v2_min3.csv"),
    ]

    lc_file = None
    for path in lc_paths:
        if path.exists():
            lc_file = path
            break

    if lc_file is None:
        raise FileNotFoundError(f"Raw light curves not found")

    print(f"Loading raw light curves: {lc_file}")
    df = pd.read_csv(lc_file)
    print(f"  {len(df)} observations")
    return df


def estimate_peak_magnitude(group):
    """Estimate peak magnitude from light curve."""
    # Use g, r, i, z bands
    bands = ['g', 'r', 'i', 'z']

    best_mag = None
    best_err = None

    for band in bands:
        band_data = group[group['BAND'].str.strip() == band]
        if len(band_data) < 3:
            continue

        # Find brightest point (lowest magnitude)
        flux = band_data['FLUXCAL'].values
        flux_err = band_data['FLUXCALERR'].values

        if np.all(flux <= 0):
            continue

        # Convert flux to magnitude at peak
        valid = flux > 0
        if not np.any(valid):
            continue

        mags = -2.5 * np.log10(flux[valid]) + 27.5  # Approximate zeropoint
        mag_errs = 2.5 / np.log(10) * flux_err[valid] / flux[valid]

        # Peak is minimum magnitude
        peak_idx = np.argmin(mags)
        peak_mag = mags[peak_idx]
        peak_err = mag_errs[peak_idx]

        if best_mag is None or peak_mag < best_mag:
            best_mag = peak_mag
            best_err = peak_err

    return best_mag, best_err if best_err else 0.3


def process_raw_data(df, min_obs=5):
    """Process raw light curves to get peak magnitudes."""
    print("Processing raw light curves...")

    # Handle different column name conventions
    snid_col = 'snid' if 'snid' in df.columns else 'SNID'
    z_col = 'z' if 'z' in df.columns else 'REDSHIFT_FINAL'
    flux_col = 'flux_jy' if 'flux_jy' in df.columns else 'FLUXCAL'
    flux_err_col = 'flux_err_jy' if 'flux_err_jy' in df.columns else 'FLUXCALERR'
    band_col = 'band' if 'band' in df.columns else 'BAND'

    print(f"  Columns: snid={snid_col}, z={z_col}, flux={flux_col}")

    # Get unique SNe
    sn_ids = df[snid_col].unique()
    print(f"  {len(sn_ids)} unique transients")

    results = []

    for snid in sn_ids:
        group = df[df[snid_col] == snid]

        if len(group) < min_obs:
            continue

        # Get redshift
        z_vals = group[z_col].values
        z = z_vals[~np.isnan(z_vals)]
        if len(z) == 0 or z[0] <= 0:
            continue
        z = z[0]

        # Estimate peak magnitude from flux
        flux = group[flux_col].values
        flux_err = group[flux_err_col].values if flux_err_col in group.columns else np.abs(flux) * 0.1

        valid = (flux > 0) & np.isfinite(flux)
        if not np.any(valid):
            continue

        # Convert max flux to magnitude (approximate)
        max_flux = np.max(flux[valid])
        max_flux_err = flux_err[valid][np.argmax(flux[valid])] if len(flux_err) > 0 else max_flux * 0.1

        # Jy to AB magnitude: m = -2.5*log10(flux_Jy) + 8.9
        peak_mag = -2.5 * np.log10(max_flux) + 8.9
        peak_err = 2.5 / np.log(10) * max_flux_err / max_flux

        if np.isnan(peak_mag):
            continue

        results.append({
            'SNID': snid,
            'z': z,
            'peak_mag': peak_mag,
            'peak_err': max(peak_err, 0.1)
        })

    result_df = pd.DataFrame(results)
    print(f"  {len(result_df)} SNe with valid peaks")

    return result_df


# =============================================================================
# FITTING
# =============================================================================

def fit_offset(model, z, mu_obs, sigma):
    """Fit magnitude offset."""
    def chi2(M):
        mu_pred = model.distance_modulus(z) + M
        return np.sum(((mu_obs - mu_pred) / sigma)**2)
    result = minimize(chi2, x0=0.0, method='BFGS')
    return result.x[0]


def fit_psi(z, mu_obs, sigma, H0=70.0):
    """Fit ψ coupling."""
    def objective(params):
        g_psi, M = params
        model = ModelE_PhotonPsi(H0=H0, g_psi=g_psi)
        mu_pred = model.distance_modulus(z) + M
        return np.sum(((mu_obs - mu_pred) / sigma)**2)
    result = minimize(objective, x0=[0.0, 0.0], method='Nelder-Mead')
    return result.x[0], result.x[1]


def compute_metrics(model, z, mu_obs, sigma, M_offset):
    """Compute fit metrics."""
    mu_pred = model.distance_modulus(z) + M_offset
    residuals = mu_obs - mu_pred
    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals / sigma)**2)
    dof = max(len(z) - model.n_params - 1, 1)
    return {
        'model': model.name,
        'rms': rms,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': chi2 / dof,
        'n_params': model.n_params,
        'M_offset': M_offset
    }


# =============================================================================
# MAIN
# =============================================================================

def run_raw_comparison():
    """Run comparison on raw data."""
    print("=" * 70)
    print("SNe Ia PHOTON MODEL: RAW LIGHT CURVE COMPARISON")
    print("=" * 70)
    print()
    print("Using raw photometry WITHOUT SALT processing")
    print("(Avoids ΛCDM bias in standardization)")
    print()

    # Load and process raw data
    df_raw = load_raw_lightcurves()
    df_sne = process_raw_data(df_raw)

    z = df_sne['z'].values
    mu_obs = df_sne['peak_mag'].values
    sigma = df_sne['peak_err'].values

    print(f"\nData: {len(z)} SNe, z ∈ [{z.min():.3f}, {z.max():.3f}]")
    print()

    # Fit ψ coupling
    print("Fitting ψ field coupling...")
    g_psi_fit, M_psi = fit_psi(z, mu_obs, sigma)
    print(f"  Best-fit g_ψ = {g_psi_fit:.4f}")
    print()

    # Define models
    models = [
        ModelB_Lean4(),
        ModelE_PhotonPsi(g_psi=g_psi_fit),
        ModelD_LCDM()
    ]
    models[1].name = f"E: Photon+ψ (g={g_psi_fit:.3f})"
    models[1].n_params = 1

    results = []

    print("FITTING MODELS:")
    print("-" * 70)

    for model in models:
        if 'g=' in model.name:
            M_offset = M_psi
        else:
            M_offset = fit_offset(model, z, mu_obs, sigma)

        metrics = compute_metrics(model, z, mu_obs, sigma, M_offset)
        results.append(metrics)

        print(f"{model.name:30s}: RMS={metrics['rms']:.4f}, "
              f"χ²/dof={metrics['reduced_chi2']:.3f}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    results.sort(key=lambda x: x['rms'])

    print(f"{'Model':30s} {'RMS':>10} {'χ²/dof':>10} {'Params':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:30s} {r['rms']:>10.4f} {r['reduced_chi2']:>10.3f} {r['n_params']:>8d}")

    print()
    print(f"WINNER: {results[0]['model']}")
    print()

    # Compare to SALT results
    print("=" * 70)
    print("COMPARISON: RAW vs SALT-PROCESSED")
    print("=" * 70)
    print()
    print("SALT-processed (1,829 SNe, z ≤ 1.12):")
    print("  ΛCDM:     RMS = 0.2634, χ²/dof = 0.941")
    print("  Lean4:    RMS = 0.2695, χ²/dof = 1.022")
    print("  Photon+ψ: RMS = 0.2644, χ²/dof = 0.926")
    print()
    print(f"RAW data ({len(z)} SNe, z ≤ {z.max():.2f}):")
    for r in results:
        print(f"  {r['model']:12s}: RMS = {r['rms']:.4f}, χ²/dof = {r['reduced_chi2']:.3f}")
    print()

    if g_psi_fit > 0.1:
        print("ψ field coupling is STRONGER in raw data!")
        print("  → SALT processing may be absorbing the ψ effect")
    elif g_psi_fit < -0.1:
        print("ψ field coupling is NEGATIVE in raw data")
        print("  → Opposite sign suggests systematic difference")
    else:
        print("ψ field coupling is similar in both datasets")
        print("  → Effect appears robust to processing")

    return results


if __name__ == "__main__":
    run_raw_comparison()
