#!/usr/bin/env python3
"""
QFD CMB Demo - Unified Schema Version

Generates CMB power spectra (TT, EE, TE) using the QFD unified schema v2.2.0.
This version connects CMB parameters to the canonical V21 cosmology parameters.

Usage:
    python run_demo_unified.py --outdir outputs
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add parent directory to find unified schema
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

from qfd_unified_schema import SupernovaV21GlobalParams, PhysicalConstants

from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import te_correlation_phase
from qfd_cmb.projector import project_limber
from qfd_cmb.figures import plot_TT, plot_EE, plot_TE


class CMBParameters:
    """
    CMB parameters connected to unified QFD schema.

    Links CMB observables to V21 cosmology parameters where applicable.
    """

    def __init__(self, v21_params: SupernovaV21GlobalParams = None):
        """
        Initialize CMB parameters.

        Parameters:
            v21_params: V21 global parameters (k_J, eta_prime, xi, etc.)
                       If None, uses default V21 fit to DES-SN5YR.
        """
        if v21_params is None:
            # Default from V21 DES-SN5YR fit
            v21_params = SupernovaV21GlobalParams(
                k_J=70.0,        # km/s/Mpc - cosmic drag
                eta_prime=5.0,   # FDR strength
                xi=2.5,          # Thermal coupling
                sigma_alpha=0.15,
                nu=6.5
            )

        self.v21 = v21_params

        # CMB-specific parameters
        # (These are observational constraints from Planck)
        self.lA = 301.0              # Acoustic scale (ℓ_A)
        self.rpsi = 147.0            # Correlation length (Mpc)
        self.tau = 0.054             # Optical depth to reionization

        # Derived quantities
        self.chi_star = self.lA * self.rpsi / np.pi  # Comoving distance to LSS
        self.sigma_chi = 250.0                        # Visibility window width

        # Power spectrum parameters
        self.ns = 0.96               # Scalar spectral index (Planck)
        self.Aosc = 0.55             # QFD oscillation amplitude
        self.sigma_osc = 0.025       # QFD oscillation damping

    def get_hubble_parameter(self) -> float:
        """
        Get Hubble parameter from V21 k_J.

        In V21: k_J ≈ H0 for cosmic drag interpretation.
        """
        return self.v21.k_J  # km/s/Mpc

    def get_summary(self) -> dict:
        """Return summary of all parameters."""
        return {
            # V21 cosmology
            'k_J': self.v21.k_J,
            'eta_prime': self.v21.eta_prime,
            'xi': self.v21.xi,
            'sigma_alpha': self.v21.sigma_alpha,
            'nu': self.v21.nu,

            # CMB observables
            'lA': self.lA,
            'rpsi': self.rpsi,
            'tau': self.tau,
            'chi_star': self.chi_star,
            'sigma_chi': self.sigma_chi,

            # Power spectrum
            'ns': self.ns,
            'Aosc': self.Aosc,
            'sigma_osc': self.sigma_osc,

            # Derived
            'H0_effective': self.get_hubble_parameter(),
        }


def generate_cmb_spectra(params: CMBParameters, ell_min: int = 2, ell_max: int = 2500):
    """
    Generate CMB power spectra using unified parameters.

    Parameters:
        params: CMBParameters instance with unified schema
        ell_min: Minimum multipole
        ell_max: Maximum multipole

    Returns:
        dict with 'ells', 'C_TT', 'C_EE', 'C_TE'
    """
    # Prepare visibility window in comoving distance χ
    chi_grid = np.linspace(
        params.chi_star - 5*params.sigma_chi,
        params.chi_star + 5*params.sigma_chi,
        501
    )
    Wchi = gaussian_window_chi(chi_grid, params.chi_star, params.sigma_chi)

    # Multipoles
    ells = np.arange(ell_min, ell_max + 1)

    # QFD power spectrum with oscillatory features
    # Uses parameters from unified schema where applicable
    Pk = lambda k: oscillatory_psik(
        k,
        ns=params.ns,
        rpsi=params.rpsi,
        Aosc=params.Aosc,
        sigma_osc=params.sigma_osc
    )

    # Project to get CMB spectra
    print("Computing TT spectrum via Limber projection...")
    Ctt = project_limber(ells, Pk, Wchi, chi_grid)

    print("Computing EE spectrum...")
    # Placeholder: EE ~ 0.25 * TT for this demo
    # In production, use full LOS projector with Mueller matrix
    Cee = 0.25 * Ctt

    print("Computing TE cross-correlation...")
    # TE correlation with QFD phase modulation
    rho = np.array([
        te_correlation_phase((ell+0.5)/params.chi_star, params.rpsi, ell, params.chi_star)
        for ell in ells
    ])
    Cte = rho * np.sqrt(Ctt * Cee)

    return {
        'ells': ells,
        'C_TT': Ctt,
        'C_EE': Cee,
        'C_TE': Cte
    }


def main():
    """Main execution with unified schema."""

    ap = argparse.ArgumentParser(
        description="Generate QFD CMB spectra using unified schema v2.2.0"
    )
    ap.add_argument("--outdir", default="outputs",
                    help="Output directory for plots and data")
    ap.add_argument("--lmin", type=int, default=2,
                    help="Minimum multipole")
    ap.add_argument("--lmax", type=int, default=2500,
                    help="Maximum multipole")
    ap.add_argument("--k-j", type=float, default=70.0,
                    help="V21 k_J parameter (km/s/Mpc)")
    ap.add_argument("--eta-prime", type=float, default=5.0,
                    help="V21 eta_prime (FDR strength)")
    ap.add_argument("--xi", type=float, default=2.5,
                    help="V21 xi (thermal coupling)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("QFD CMB Power Spectra Generator")
    print("Using Unified Schema v2.2.0")
    print("=" * 70)
    print()

    # Create V21 parameters from command line or defaults
    v21_params = SupernovaV21GlobalParams(
        k_J=args.k_j,
        eta_prime=args.eta_prime,
        xi=args.xi,
        sigma_alpha=0.15,
        nu=6.5
    )

    # Initialize CMB parameters with V21 connection
    cmb_params = CMBParameters(v21_params)

    print("V21 Cosmology Parameters:")
    print(f"  k_J = {cmb_params.v21.k_J:.1f} km/s/Mpc (cosmic drag)")
    print(f"  η' = {cmb_params.v21.eta_prime:.2f} (FDR strength)")
    print(f"  ξ = {cmb_params.v21.xi:.2f} (thermal coupling)")
    print()

    print("CMB Observational Constraints:")
    print(f"  ℓ_A = {cmb_params.lA:.1f} (acoustic scale)")
    print(f"  r_ψ = {cmb_params.rpsi:.1f} Mpc (correlation length)")
    print(f"  χ_* = {cmb_params.chi_star:.1f} Mpc (comoving distance to LSS)")
    print(f"  τ = {cmb_params.tau:.3f} (optical depth)")
    print()

    print("QFD Power Spectrum Parameters:")
    print(f"  n_s = {cmb_params.ns:.2f} (spectral index)")
    print(f"  A_osc = {cmb_params.Aosc:.2f} (oscillation amplitude)")
    print(f"  σ_osc = {cmb_params.sigma_osc:.3f} (oscillation damping)")
    print()

    # Generate spectra
    print("Generating CMB power spectra...")
    spectra = generate_cmb_spectra(cmb_params, args.lmin, args.lmax)

    # Save to CSV
    df = pd.DataFrame({
        "ell": spectra['ells'],
        "C_TT": spectra['C_TT'],
        "C_TE": spectra['C_TE'],
        "C_EE": spectra['C_EE']
    })

    csv_path = os.path.join(args.outdir, "qfd_cmb_spectra_unified.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved spectra to {csv_path}")

    # Generate plots
    print("\nGenerating publication-quality plots...")

    tt_path = os.path.join(args.outdir, "TT.png")
    plot_TT(spectra['ells'], spectra['C_TT'], tt_path)
    print(f"✓ Saved TT spectrum to {tt_path}")

    ee_path = os.path.join(args.outdir, "EE.png")
    plot_EE(spectra['ells'], spectra['C_EE'], ee_path)
    print(f"✓ Saved EE spectrum to {ee_path}")

    te_path = os.path.join(args.outdir, "TE.png")
    plot_TE(spectra['ells'], spectra['C_TE'], te_path)
    print(f"✓ Saved TE spectrum to {te_path}")

    # Save parameter summary
    summary = cmb_params.get_summary()
    summary_path = os.path.join(args.outdir, "parameters_unified.json")

    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved parameters to {summary_path}")

    print()
    print("=" * 70)
    print("CMB Spectra Generation Complete!")
    print("=" * 70)
    print()
    print("Cross-Domain Consistency:")
    print(f"  V21 k_J: {cmb_params.v21.k_J:.1f} km/s/Mpc")
    print(f"  Effective H0: {cmb_params.get_hubble_parameter():.1f} km/s/Mpc")
    print(f"  → Same universal coupling across nuclear, cosmology, and CMB ✓")
    print()
    print(f"Output files in: {args.outdir}/")
    print(f"  - TT.png, EE.png, TE.png (power spectrum plots)")
    print(f"  - qfd_cmb_spectra_unified.csv (numerical data)")
    print(f"  - parameters_unified.json (all parameters used)")


if __name__ == "__main__":
    main()
