#!/usr/bin/env python3
"""
Basic Usage Example for QFD CMB Module

This script demonstrates the fundamental usage of the QFD CMB Module for computing
CMB angular power spectra using photon-photon scattering models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.projector import project_limber
from qfd_cmb.figures import plot_TT, plot_EE, plot_TE
from qfd_cmb.kernels import te_correlation_phase


def basic_power_spectrum_example():
    """Demonstrate basic power spectrum calculation."""
    print("=" * 60)
    print("Basic Power Spectrum Example")
    print("=" * 60)
    
    # Define wavenumber range (1/Mpc)
    k = np.logspace(-4, 1, 200)
    
    # Compute power spectrum with default parameters
    Pk_default = oscillatory_psik(k)
    
    # Compute power spectrum with custom parameters
    Pk_custom = oscillatory_psik(
        k,
        A=1.2,           # Increased amplitude
        ns=0.965,        # Slightly different spectral index
        rpsi=150.0,      # Different oscillation scale
        Aosc=0.6,        # Stronger oscillations
        sigma_osc=0.03   # Different damping
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.loglog(k, Pk_default, 'b-', linewidth=2, label='Default Parameters')
    plt.loglog(k, Pk_custom, 'r--', linewidth=2, label='Custom Parameters')
    plt.xlabel('k [1/Mpc]', fontsize=12)
    plt.ylabel('P(k)', fontsize=12)
    plt.title('QFD Oscillatory Power Spectrum', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/basic_power_spectrum.png', dpi=200)
    plt.show()
    
    print(f"Power spectrum range: {Pk_default.min():.2e} to {Pk_default.max():.2e}")
    print("Plot saved to: outputs/basic_power_spectrum.png")
    print()


def cmb_spectrum_calculation():
    """Demonstrate CMB angular power spectrum calculation."""
    print("=" * 60)
    print("CMB Angular Power Spectrum Calculation")
    print("=" * 60)
    
    # Physical parameters (Planck-anchored)
    lA = 301.0                    # Acoustic scale parameter
    rpsi = 147.0                  # Oscillation scale (Mpc)
    chi_star = lA * rpsi / np.pi  # Comoving distance to last scattering
    sigma_chi = 250.0             # Width of last scattering surface
    
    print(f"Physical parameters:")
    print(f"  Acoustic scale lA: {lA}")
    print(f"  Oscillation scale rpsi: {rpsi} Mpc")
    print(f"  Distance to last scattering: {chi_star:.1f} Mpc")
    print(f"  Last scattering width: {sigma_chi} Mpc")
    print()
    
    # Define comoving distance grid
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 300)
    
    # Create visibility window
    W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
    
    # Define multipole range
    ells = np.arange(2, 2500)
    
    # Define power spectrum function
    def Pk_func(k):
        return oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
    
    # Compute TT spectrum
    print("Computing TT spectrum...")
    Ctt = project_limber(ells, Pk_func, W_chi, chi_grid)
    
    # Compute EE spectrum (simplified model)
    print("Computing EE spectrum...")
    def Pk_EE(k):
        return 0.25 * Pk_func(k)  # EE is typically ~25% of TT amplitude
    
    Cee = project_limber(ells, Pk_EE, W_chi, chi_grid)
    
    # Compute TE spectrum with correlation
    print("Computing TE spectrum...")
    rho = np.array([
        te_correlation_phase((ell + 0.5)/chi_star, rpsi, ell, chi_star) 
        for ell in ells
    ])
    Cte = rho * np.sqrt(Ctt * Cee)
    
    # Create plots
    print("Creating plots...")
    plot_TT(ells, Ctt, 'outputs/basic_tt_spectrum.png')
    plot_EE(ells, Cee, 'outputs/basic_ee_spectrum.png')
    plot_TE(ells, Cte, 'outputs/basic_te_spectrum.png')
    
    # Create combined plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.loglog(ells, ells*(ells+1)*Ctt, 'r-', linewidth=2)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
    plt.title('TT Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.loglog(ells, ells*(ells+1)*Cee, 'b-', linewidth=2)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{EE}$')
    plt.title('EE Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    sign = np.sign(Cte + 1e-30)
    plt.semilogx(ells, sign * ells*(ells+1)*np.abs(Cte), 'g-', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'sign×$\ell(\ell+1)|C_\ell^{TE}|$')
    plt.title('TE Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/basic_combined_spectra.png', dpi=200)
    plt.show()
    
    # Print some statistics
    print(f"TT spectrum peak: {(ells*(ells+1)*Ctt).max():.2e} at ell={ells[np.argmax(ells*(ells+1)*Ctt)]}")
    print(f"EE spectrum peak: {(ells*(ells+1)*Cee).max():.2e} at ell={ells[np.argmax(ells*(ells+1)*Cee)]}")
    print(f"TE correlation range: {rho.min():.3f} to {rho.max():.3f}")
    print("Individual plots saved to: outputs/basic_*_spectrum.png")
    print("Combined plot saved to: outputs/basic_combined_spectra.png")
    print()


def parameter_sensitivity_study():
    """Demonstrate parameter sensitivity analysis."""
    print("=" * 60)
    print("Parameter Sensitivity Study")
    print("=" * 60)
    
    # Setup basic calculation
    chi_star = 14065.0
    sigma_chi = 250.0
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 200)
    W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
    ells = np.arange(2, 1000)
    
    # Study rpsi dependence
    rpsi_values = [140, 147, 154]
    
    plt.figure(figsize=(14, 10))
    
    # rpsi sensitivity
    plt.subplot(2, 2, 1)
    for rpsi in rpsi_values:
        Pk_func = lambda k: oscillatory_psik(k, rpsi=rpsi, Aosc=0.55)
        Ctt = project_limber(ells, Pk_func, W_chi, chi_grid)
        plt.loglog(ells, ells*(ells+1)*Ctt, linewidth=2, 
                   label=f'$r_\\psi = {rpsi}$ Mpc')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
    plt.title('Oscillation Scale Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aosc sensitivity
    plt.subplot(2, 2, 2)
    Aosc_values = [0.0, 0.3, 0.55, 0.8]
    for Aosc in Aosc_values:
        Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=Aosc)
        Ctt = project_limber(ells, Pk_func, W_chi, chi_grid)
        plt.loglog(ells, ells*(ells+1)*Ctt, linewidth=2, 
                   label=f'$A_{{osc}} = {Aosc}$')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
    plt.title('Oscillation Amplitude Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Spectral index sensitivity
    plt.subplot(2, 2, 3)
    ns_values = [0.94, 0.96, 0.98]
    for ns in ns_values:
        Pk_func = lambda k: oscillatory_psik(k, ns=ns, rpsi=147.0, Aosc=0.55)
        Ctt = project_limber(ells, Pk_func, W_chi, chi_grid)
        plt.loglog(ells, ells*(ells+1)*Ctt, linewidth=2, 
                   label=f'$n_s = {ns}$')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
    plt.title('Spectral Index Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison with no oscillations
    plt.subplot(2, 2, 4)
    Pk_standard = lambda k: oscillatory_psik(k, Aosc=0.0)
    Pk_qfd = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
    
    Ctt_standard = project_limber(ells, Pk_standard, W_chi, chi_grid)
    Ctt_qfd = project_limber(ells, Pk_qfd, W_chi, chi_grid)
    
    plt.loglog(ells, ells*(ells+1)*Ctt_standard, 'k-', linewidth=2, 
               label='Standard (no osc.)')
    plt.loglog(ells, ells*(ells+1)*Ctt_qfd, 'r-', linewidth=2, 
               label='QFD Model')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
    plt.title('QFD vs Standard Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/basic_parameter_sensitivity.png', dpi=200)
    plt.show()
    
    print("Parameter sensitivity study completed.")
    print("Plot saved to: outputs/basic_parameter_sensitivity.png")
    print()


def main():
    """Run all basic usage examples."""
    print("QFD CMB Module - Basic Usage Examples")
    print("=====================================")
    print()
    
    # Run examples
    basic_power_spectrum_example()
    cmb_spectrum_calculation()
    parameter_sensitivity_study()
    
    print("All basic examples completed successfully!")
    print("Check the 'outputs' directory for generated plots.")


if __name__ == "__main__":
    main()