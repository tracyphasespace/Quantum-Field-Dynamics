#!/usr/bin/env python3
"""
SNe Ia Model Comparison: RAW Light Curve Data (No SALT Processing)

Uses raw flux measurements to fit peak brightness directly,
avoiding ΛCDM assumptions embedded in SALT processing.

Data: 8,277 SNe with 770,634 observations (z up to 6.84)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, curve_fit
from scipy.constants import c as c_mps
from collections import defaultdict

# Physical constants
C_KM_S = c_mps / 1000.0  # km/s

# =============================================================================
# LIGHT CURVE FITTING (No SALT assumptions)
# =============================================================================

def simple_sn_template(t, t0, A, tau_rise, tau_fall):
    """
    Simple SN Ia light curve template.

    Not SALT - just a phenomenological rise-fall model.
    This avoids ΛCDM assumptions in the light curve fitting.
    """
    dt = t - t0
    # Rising phase
    rise = np.where(dt < 0, np.exp(dt / tau_rise), 1.0)
    # Falling phase
    fall = np.where(dt >= 0, np.exp(-dt / tau_fall), 1.0)
    return A * rise * fall


def fit_light_curve(mjd, flux, flux_err, z):
    """
    Fit a single light curve to extract peak flux.

    Returns peak flux and uncertainty.
    """
    # Initial guesses
    t0_guess = mjd[np.argmax(flux)]
    A_guess = np.max(flux)

    try:
        # Fit template
        popt, pcov = curve_fit(
            simple_sn_template,
            mjd, flux,
            p0=[t0_guess, A_guess, 5.0, 20.0],
            sigma=np.abs(flux_err) + 1e-10,
            maxfev=1000,
            bounds=([mjd.min()-50, 0, 0.1, 1], [mjd.max()+50, np.inf, 50, 100])
        )

        t0, A_peak, tau_rise, tau_fall = popt
        A_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else A_peak * 0.1

        # Compute chi2
        model = simple_sn_template(mjd, *popt)
        chi2 = np.sum(((flux - model) / (flux_err + 1e-10))**2)
        chi2_dof = chi2 / (len(flux) - 4) if len(flux) > 4 else chi2

        return {
            't0': t0,
            'A_peak': A_peak,
            'A_err': A_err,
            'tau_rise': tau_rise,
            'tau_fall': tau_fall,
            'chi2_dof': chi2_dof,
            'n_obs': len(flux),
            'fit_ok': True
        }
    except Exception as e:
        return {
            'A_peak': np.nan,
            'A_err': np.nan,
            'chi2_dof': np.nan,
            'n_obs': len(flux),
            'fit_ok': False
        }


def flux_to_magnitude(flux_jy):
    """Convert flux in Jy to AB magnitude."""
    # AB magnitude: m = -2.5 * log10(flux_Jy) + 8.9
    return -2.5 * np.log10(np.maximum(flux_jy, 1e-30)) + 8.9


def peak_mag_to_distance_modulus(m_peak, M_abs=-19.3):
    """
    Convert peak apparent magnitude to distance modulus.
    μ = m - M
    """
    return m_peak - M_abs


# =============================================================================
# MODEL DEFINITIONS (Same as before)
# =============================================================================

class ModelA_Phenomenological:
    """Old phenomenological model: Δm = α × z^β"""
    name = "A: Phenomenological"

    def __init__(self, H0=70.0, alpha=0.85, beta_power=0.6):
        self.H0 = H0
        self.alpha = alpha
        self.beta_power = beta_power

    def distance_modulus(self, z):
        z = np.atleast_1d(z)
        D_L = (2 * C_KM_S / self.H0) * (1 - 1/np.sqrt(1+z)) * (1+z)
        mu_geo = 5 * np.log10(np.maximum(D_L, 1e-10)) + 25
        delta_m = self.alpha * z**self.beta_power
        return mu_geo + delta_m


class ModelB_Lean4Derived:
    """Lean4-derived model: ln(1+z) = κ×D, κ = H0/c"""
    name = "B: Lean4-Derived"

    def __init__(self, H0=70.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S

    def distance_modulus(self, z):
        z = np.atleast_1d(z)
        D = np.log(1 + z) / self.kappa
        D_L = D * (1 + z)
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


class ModelD_LCDM:
    """Standard ΛCDM reference model"""
    name = "D: ΛCDM"

    def __init__(self, H0=70.0, Omega_m=0.3):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m

    def distance_modulus(self, z):
        from scipy.integrate import quad
        z = np.atleast_1d(z)
        D_L = np.zeros_like(z, dtype=float)
        for i, zi in enumerate(z):
            if zi > 0:
                E = lambda zp: np.sqrt(self.Omega_m * (1+zp)**3 + self.Omega_L)
                integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
                D_L[i] = (C_KM_S / self.H0) * (1 + zi) * integral
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_raw_data():
    """Load raw light curve data."""
    filepath = Path("/home/tracy/development/QFD_SpectralGap/projects/astrophysics/V21 Supernova Analysis package/data/lightcurves_all_transients.csv")
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total observations: {len(df)}")
    print(f"  Unique SNe: {df['snid'].nunique()}")
    return df


def process_light_curves(df, band='r', min_obs=10, max_chi2=5.0, z_min=0.01, z_max=3.0):
    """
    Process raw light curves to extract peak brightness.

    Parameters:
    -----------
    df : DataFrame
        Raw light curve data
    band : str
        Photometric band to use
    min_obs : int
        Minimum observations per SN
    max_chi2 : float
        Maximum chi2/dof for good fit
    z_min, z_max : float
        Redshift range
    """
    print(f"\nProcessing {band}-band light curves...")

    # Filter by band
    df_band = df[df['band'] == band].copy()
    print(f"  Observations in {band}-band: {len(df_band)}")

    results = []
    snids = df_band['snid'].unique()

    for i, snid in enumerate(snids):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(snids)}...")

        sn = df_band[df_band['snid'] == snid]
        z = sn['z'].iloc[0]

        # Apply cuts
        if z < z_min or z > z_max:
            continue
        if len(sn) < min_obs:
            continue

        # Fit light curve
        mjd = sn['mjd'].values
        flux = sn['flux_jy'].values
        flux_err = sn['flux_err_jy'].values

        fit = fit_light_curve(mjd, flux, flux_err, z)

        if fit['fit_ok'] and fit['chi2_dof'] < max_chi2 and fit['A_peak'] > 0:
            # Convert to magnitude
            m_peak = flux_to_magnitude(fit['A_peak'] * 1e-6)  # Convert to Jy
            mu_obs = peak_mag_to_distance_modulus(m_peak)

            results.append({
                'snid': snid,
                'z': z,
                'A_peak': fit['A_peak'],
                'A_err': fit['A_err'],
                'm_peak': m_peak,
                'mu_obs': mu_obs,
                'chi2_dof': fit['chi2_dof'],
                'n_obs': fit['n_obs']
            })

    df_results = pd.DataFrame(results)
    print(f"  Good fits: {len(df_results)}")
    return df_results


def run_comparison_raw():
    """Run model comparison on raw data."""
    print("="*70)
    print("SNe Ia MODEL COMPARISON - RAW DATA (No SALT)")
    print("="*70)

    # Load and process data
    df_raw = load_raw_data()

    # Process light curves (use r-band as primary)
    df_fits = process_light_curves(df_raw, band='r', min_obs=10, max_chi2=10.0, z_max=5.0)

    if len(df_fits) < 50:
        print("WARNING: Too few good fits. Using looser cuts...")
        df_fits = process_light_curves(df_raw, band='r', min_obs=5, max_chi2=20.0, z_max=5.0)

    z = df_fits['z'].values
    mu_obs = df_fits['mu_obs'].values
    sigma = df_fits['A_err'].values / df_fits['A_peak'].values * 2.5 / np.log(10) + 0.15

    print(f"\nData: {len(z)} SNe, z range: [{z.min():.3f}, {z.max():.3f}]")

    # Models
    models = [
        ModelA_Phenomenological(),
        ModelB_Lean4Derived(),
        ModelD_LCDM()
    ]

    results = []

    print("\nFITTING MODELS:")
    print("-"*70)

    for model in models:
        # Fit offset
        def chi2(M):
            mu_pred = model.distance_modulus(z) + M
            return np.sum(((mu_obs - mu_pred) / sigma)**2)

        from scipy.optimize import minimize_scalar
        result = minimize_scalar(chi2, bounds=(-10, 10), method='bounded')
        M_offset = result.x

        # Compute metrics
        mu_pred = model.distance_modulus(z) + M_offset
        residuals = mu_obs - mu_pred
        rms = np.sqrt(np.mean(residuals**2))
        chi2_val = np.sum((residuals / sigma)**2)
        dof = len(z) - 1

        results.append({
            'model': model.name,
            'rms': rms,
            'chi2': chi2_val,
            'reduced_chi2': chi2_val / dof,
            'M_offset': M_offset,
            'residuals': residuals,
            'mu_pred': mu_pred
        })

        print(f"{model.name:25s}: RMS={rms:.4f} mag, χ²/dof={chi2_val/dof:.3f}, M={M_offset:+.3f}")

    # Redshift binned analysis
    print("\n" + "="*70)
    print("BINNED ANALYSIS BY REDSHIFT")
    print("="*70)

    bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 5.0)]

    for z_lo, z_hi in bins:
        mask = (z >= z_lo) & (z < z_hi)
        n_bin = mask.sum()
        if n_bin < 5:
            continue

        print(f"\nz = {z_lo:.1f} - {z_hi:.1f}: {n_bin} SNe")
        for r in results:
            rms_bin = np.sqrt(np.mean(r['residuals'][mask]**2))
            print(f"  {r['model']:25s}: RMS={rms_bin:.4f} mag")

    # Create plots
    create_raw_plots(z, mu_obs, sigma, results, df_fits)

    return results, df_fits


def create_raw_plots(z, mu_obs, sigma, results, df_fits, output_dir='results'):
    """Create comparison plots for raw data analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sort by z
    idx = np.argsort(z)
    z_sorted = z[idx]

    # Plot 1: Hubble diagram
    ax1 = axes[0, 0]
    ax1.errorbar(z, mu_obs, yerr=sigma, fmt='k.', alpha=0.3, markersize=2, label='Raw data')

    colors = ['red', 'blue', 'green']
    for i, r in enumerate(results):
        mu_pred = r['mu_pred'][idx]
        ax1.plot(z_sorted, mu_pred, colors[i], lw=2, label=f"{r['model']} (RMS={r['rms']:.3f})")

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ')
    ax1.set_title(f'Hubble Diagram: RAW Data ({len(z)} SNe)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(z.max()*1.1, 5))

    # Plot 2: Residuals vs z
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        ax2.scatter(z, r['residuals'], c=colors[i], alpha=0.4, s=5, label=r['model'])
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residual (mag)')
    ax2.set_title('Model Residuals vs Redshift')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 3)

    # Plot 3: High-z comparison
    ax3 = axes[1, 0]
    z_high = z > 0.7
    if z_high.sum() > 10:
        for i, r in enumerate(results):
            ax3.scatter(z[z_high], r['residuals'][z_high], c=colors[i], alpha=0.5, s=20, label=r['model'])
        ax3.axhline(0, color='k', linestyle='--')
        ax3.set_xlabel('Redshift z')
        ax3.set_ylabel('Residual (mag)')
        ax3.set_title(f'High-z Residuals (z > 0.7, n={z_high.sum()})')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient high-z data', ha='center', va='center', transform=ax3.transAxes)

    # Plot 4: Redshift distribution
    ax4 = axes[1, 1]
    ax4.hist(z, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Count')
    ax4.set_title('Redshift Distribution (Raw Sample)')
    ax4.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"N = {len(z)}\nz_median = {np.median(z):.2f}\nz_max = {z.max():.2f}"
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('SNe Ia Model Comparison - RAW Light Curves (No SALT)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / 'model_comparison_raw.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    results, df_fits = run_comparison_raw()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nThis analysis uses RAW light curve data without SALT processing.")
    print("Peak brightness extracted with simple rise-fall template.")
    print("No ΛCDM assumptions in light curve fitting.")
