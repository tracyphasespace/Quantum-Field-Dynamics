#!/usr/bin/env python3
"""
Validate QFD CMB Model Against Planck Power Spectra

Compares:
1. ΛCDM predictions
2. QFD photon-ψ model
3. Observed Planck data
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/tracy/development/QFD_SpectralGap/projects/astrophysics/redshift-analysis/RedShift')

from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.projector import project_limber

# =============================================================================
# LOAD PLANCK DATA
# =============================================================================

def load_planck_data():
    """Load mock Planck power spectra."""
    path = "/home/tracy/development/QFD_SpectralGap/data/raw/planck_2018_power_spectra_mock.csv"
    df = pd.read_csv(path)
    return df


# =============================================================================
# QFD MODEL
# =============================================================================

def qfd_power_spectrum(ells, chi_star=14065.0, sigma_chi=250.0,
                       ns=0.96, rpsi=147.0, Aosc=0.55, sigma_osc=0.025):
    """
    QFD power spectrum using photon-ψ coupling model.

    Parameters match fit_planck.py defaults.
    """
    # chi grid centered on last scattering surface
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)

    # Visibility function
    W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)

    # Primordial power spectrum with oscillations
    Pk = lambda k: oscillatory_psik(k, A=1.0, ns=ns, rpsi=rpsi,
                                    Aosc=Aosc, sigma_osc=sigma_osc)

    # Limber projection
    C_ell = project_limber(ells.astype(float), Pk, W_chi, chi_grid)

    return C_ell


# =============================================================================
# COMPARISON METRICS
# =============================================================================

def compute_chi2(observed, model, sigma):
    """Compute chi-squared."""
    residuals = observed - model
    chi2 = np.sum((residuals / sigma)**2)
    return chi2


def compute_rms(observed, model):
    """Compute RMS residual."""
    return np.sqrt(np.mean((observed - model)**2))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QFD CMB VALIDATION AGAINST PLANCK")
    print("=" * 70)
    print()

    # Load data
    df = load_planck_data()
    print(f"Loaded {len(df)} multipoles from Planck mock data")
    print()

    ells = df['ell'].values
    C_TT_obs = df['C_ell_TT_obs'].values
    C_TT_lcdm = df['C_ell_TT_LCDM'].values
    sigma_TT = df['sigma_TT'].values

    # Limit to ell < 1000 for this quick validation
    mask = ells < 1000
    ells = ells[mask]
    C_TT_obs = C_TT_obs[mask]
    C_TT_lcdm = C_TT_lcdm[mask]
    sigma_TT = sigma_TT[mask]

    print(f"Using multipoles: {ells[0]} to {ells[-1]} ({len(ells)} values)")
    print()

    # QFD model
    print("Computing QFD power spectrum...")
    C_TT_qfd = qfd_power_spectrum(ells)

    # Normalize QFD to match observed amplitude
    amp_ratio = np.mean(C_TT_obs[10:100]) / np.mean(C_TT_qfd[10:100])
    C_TT_qfd_norm = C_TT_qfd * amp_ratio

    print(f"QFD normalization factor: {amp_ratio:.2f}")
    print()

    # Compute metrics
    print("-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)
    print()

    # ΛCDM vs Observed
    chi2_lcdm = compute_chi2(C_TT_obs, C_TT_lcdm, sigma_TT)
    rms_lcdm = compute_rms(C_TT_obs, C_TT_lcdm)
    reduced_chi2_lcdm = chi2_lcdm / len(ells)

    print(f"ΛCDM:")
    print(f"  χ² = {chi2_lcdm:.1f}")
    print(f"  χ²/dof = {reduced_chi2_lcdm:.3f}")
    print(f"  RMS = {rms_lcdm:.1f} μK²")
    print()

    # QFD vs Observed
    chi2_qfd = compute_chi2(C_TT_obs, C_TT_qfd_norm, sigma_TT)
    rms_qfd = compute_rms(C_TT_obs, C_TT_qfd_norm)
    reduced_chi2_qfd = chi2_qfd / len(ells)

    print(f"QFD (photon-ψ coupling):")
    print(f"  χ² = {chi2_qfd:.1f}")
    print(f"  χ²/dof = {reduced_chi2_qfd:.3f}")
    print(f"  RMS = {rms_qfd:.1f} μK²")
    print()

    # Improvement
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print()
    if chi2_qfd < chi2_lcdm:
        improvement = (chi2_lcdm - chi2_qfd) / chi2_lcdm * 100
        print(f"QFD model is BETTER by {improvement:.1f}%")
    else:
        degradation = (chi2_qfd - chi2_lcdm) / chi2_lcdm * 100
        print(f"ΛCDM model is better by {degradation:.1f}%")
    print()

    print("Note: This is a preliminary validation using default QFD parameters.")
    print("Full fitting (via fit_planck.py) would optimize parameters.")
    print()

    # Physical interpretation
    print("-" * 70)
    print("PHYSICAL INTERPRETATION")
    print("-" * 70)
    print()
    print("QFD model parameters:")
    print("  chi_star = 14065 Mpc (comoving distance to 'last scattering')")
    print("  sigma_chi = 250 Mpc (width of visibility function)")
    print("  rpsi = 147 Mpc (BAO/oscillation scale)")
    print()
    print("In QFD eternal universe interpretation:")
    print("  - chi_star represents typical decay distance to CMB thermalization")
    print("  - rpsi represents ψ field coherence length")
    print("  - Oscillations from photon-ψ coupling resonances")
    print()

    return df, C_TT_qfd_norm


if __name__ == "__main__":
    df, C_qfd = main()
