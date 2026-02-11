#!/usr/bin/env python3
"""
V₆ Higher-Order Geometric Correction

GOAL: Derive V₆ coefficient that matches QED C₃ = +1.18123

QED series:
  a = (α/2π) [1 + C₂(α/π) + C₃(α/π)² + ...]
  C₂ = -0.32848 (vertex + vacuum polarization)
  C₃ = +1.18123 (light-by-light scattering)

QFD series:
  a = (α/2π) [1 + V₄(α/π) + V₆(α/π)² + ...]
  V₄ = -0.327 (compression + circulation, O(v²))
  V₆ = ? (higher-order geometric effects, O(v⁴))

HYPOTHESIS: V₆ comes from higher-order circulation integrals
  - V₄ ~ ∫ (v_φ)² · (dρ/dr)² dV  (leading order)
  - V₆ ~ ∫ (v_φ)⁴ · (dρ/dr)⁴ dV  (next order)

Or from cross-terms:
  - V₆ ~ ∫ (v_φ)² · (d²ρ/dr²)² dV  (curvature correction)
  - V₆ ~ ∫ (v_r)² · (v_φ)² dV       (mixed velocity terms)

Physical interpretation:
  - C₃ (QED): Light-by-light scattering, photon self-interaction
  - V₆ (QFD): Nonlinear vortex effects, higher-order flow corrections
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA

# Constants
HBARC = 197.3269804  # MeV·fm

# QFD parameters (from Golden Loop via shared_constants)
XI = 1.0      # From mass fit

# QED coefficients
C2_QED = -0.32848
C3_QED = +1.18123

# Experimental g-2
a_electron = 0.00115965218
a_muon = 0.00116592059
a_schwinger = ALPHA / (2 * np.pi)

# Lepton Compton wavelengths
R_electron = HBARC / 0.511  # 386.16 fm
R_muon = HBARC / 105.66     # 1.87 fm


def hill_vortex_velocity_azimuthal(r, theta, R, U=0.5):
    """
    Azimuthal velocity component of Hill vortex.

    v_φ = U · sin(θ) · (3r/2R - r³/2R³) for r < R
    """
    if r < R:
        x = r / R
        v_phi = U * np.sin(theta) * (1.5 * x - 0.5 * x**3)
    else:
        v_phi = U * np.sin(theta) * (R / r)**3 / 2
    return v_phi


def hill_vortex_velocity_radial(r, theta, R, U=0.5):
    """
    Radial velocity component of Hill vortex.

    v_r = U · cos(θ) · (3r/R - 2r³/R³) for r < R

    This is needed for mixed velocity terms in V₆.
    """
    if r < R:
        x = r / R
        v_r = U * np.cos(theta) * (3 * x - 2 * x**3)
    else:
        v_r = -U * np.cos(theta) * (R / r)**3
    return v_r


def density_gradient(r, R, rho_vac=1.0):
    """
    Hill vortex density gradient.

    ρ(r) = ρ_vac + 2(1 - (r/R)²)²
    dρ/dr = -8x(1-x²)/R where x = r/R
    """
    if r < R:
        x = r / R
        drho_dr = -8 * x * (1 - x**2) / R
    else:
        drho_dr = 0.0
    return drho_dr


def density_curvature(r, R, rho_vac=1.0):
    """
    Second derivative of Hill vortex density.

    d²ρ/dr² needed for curvature corrections to V₆.

    ρ(r) = ρ_vac + 2(1 - x²)² where x = r/R
    dρ/dx = -8x(1-x²)
    d²ρ/dx² = -8(1-x²) + 16x²
            = -8 + 8x² + 16x²
            = -8 + 24x²
    d²ρ/dr² = (d²ρ/dx²) · (1/R²)
    """
    if r < R:
        x = r / R
        d2rho_dr2 = (-8 + 24 * x**2) / R**2
    else:
        d2rho_dr2 = 0.0
    return d2rho_dr2


def calculate_v6_fourth_power(R, U=0.5):
    """
    V₆ from fourth power of velocity.

    V₆_4 = ∫∫∫ (v_φ)⁴ · (dρ/dr)⁴ · r² sin(θ) dr dθ dφ

    Physical interpretation:
    - Higher-order relativistic correction
    - (v/c)⁴ ~ α² for typical vortex velocities
    """

    def integrand(r, theta):
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        drho_dr = density_gradient(r, R)
        return (v_phi)**4 * (drho_dr)**4 * r**2 * np.sin(theta)

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    I_v6 = 2 * np.pi * I_r

    # Normalize: [v⁴ × length³] / [v⁴ × length³]
    I_v6_normalized = I_v6 / (U**4 * R**3)

    return I_v6_normalized


def calculate_v6_curvature(R, U=0.5):
    """
    V₆ from density curvature correction.

    V₆_curv = ∫∫∫ (v_φ)² · (d²ρ/dr²)² · r² sin(θ) dr dθ dφ

    Physical interpretation:
    - Correction from density profile curvature
    - Sharp boundaries (high curvature) contribute more
    """

    def integrand(r, theta):
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        d2rho_dr2 = density_curvature(r, R)
        return (v_phi)**2 * (d2rho_dr2)**2 * r**2 * np.sin(theta)

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    I_v6 = 2 * np.pi * I_r

    # Normalize
    I_v6_normalized = I_v6 / (U**2 * R**3)

    return I_v6_normalized


def calculate_v6_mixed_velocity(R, U=0.5):
    """
    V₆ from mixed radial-azimuthal velocity terms.

    V₆_mix = ∫∫∫ (v_r)² · (v_φ)² · (dρ/dr)² · r² sin(θ) dr dθ dφ

    Physical interpretation:
    - Cross-coupling between radial and azimuthal flow
    - Important for 3D vortex structure
    """

    def integrand(r, theta):
        v_r = hill_vortex_velocity_radial(r, theta, R, U)
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        drho_dr = density_gradient(r, R)
        return (v_r)**2 * (v_phi)**2 * (drho_dr)**2 * r**2 * np.sin(theta)

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    I_v6 = 2 * np.pi * I_r

    # Normalize
    I_v6_normalized = I_v6 / (U**4 * R**3)

    return I_v6_normalized


def calculate_v6_gradient_squared(R, U=0.5):
    """
    V₆ from gradient squared term.

    V₆_grad = ∫∫∫ (v_φ)² · (dρ/dr)² · (∇²ρ)² · r² sin(θ) dr dθ dφ

    This combines gradient and curvature effects.
    """

    def integrand(r, theta):
        v_phi = hill_vortex_velocity_azimuthal(r, theta, R, U)
        drho_dr = density_gradient(r, R)
        d2rho_dr2 = density_curvature(r, R)

        # Laplacian in spherical coords (azimuthally symmetric)
        if r > 1e-10:
            laplacian_rho = d2rho_dr2 + (2/r) * drho_dr
        else:
            laplacian_rho = 0.0

        return (v_phi)**2 * (drho_dr)**2 * (laplacian_rho)**2 * r**2 * np.sin(theta)

    def integrate_theta(r_val):
        result, _ = quad(lambda theta: integrand(r_val, theta), 0, np.pi)
        return result

    I_r, _ = quad(integrate_theta, 0, 10 * R, limit=100)
    I_v6 = 2 * np.pi * I_r

    # Normalize
    I_v6_normalized = I_v6 / (U**2 * R**3)

    return I_v6_normalized


def extract_v6_from_experiment():
    """
    Extract required V₆ from experimental g-2 data.

    a_exp = a_schwinger + V₄·(α/π)² + V₆·(α/π)⁴

    V₆ = (a_exp - a_schwinger - V₄·(α/π)²) / (α/π)⁴
    """

    V4_electron = -0.327  # From our derivation
    V4_muon = +0.836

    alpha_over_pi_sq = (ALPHA / np.pi)**2
    alpha_over_pi_4 = (ALPHA / np.pi)**4

    # Electron
    delta_a_e = a_electron - a_schwinger
    v4_contrib_e = V4_electron * alpha_over_pi_sq
    v6_contrib_e = delta_a_e - v4_contrib_e
    V6_electron = v6_contrib_e / alpha_over_pi_4

    # Muon
    delta_a_mu = a_muon - a_schwinger
    v4_contrib_mu = V4_muon * alpha_over_pi_sq
    v6_contrib_mu = delta_a_mu - v4_contrib_mu
    V6_muon = v6_contrib_mu / alpha_over_pi_4

    return V6_electron, V6_muon


def scan_v6_contributions(R_min=0.1, R_max=1000, n_points=30):
    """
    Scan different V₆ contributions vs radius.

    Test which term(s) give the correct scale dependence.
    """
    R_values = np.logspace(np.log10(R_min), np.log10(R_max), n_points)

    V6_fourth = []
    V6_curv = []
    V6_mixed = []
    V6_grad = []

    print("Scanning V₆ contributions...")
    print(f"R range: {R_min:.2f} fm to {R_max:.2f} fm")
    print()

    for i, R in enumerate(R_values):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_points} ({100*i/n_points:.0f}%)")

        I4 = calculate_v6_fourth_power(R)
        Ic = calculate_v6_curvature(R)
        Im = calculate_v6_mixed_velocity(R)
        Ig = calculate_v6_gradient_squared(R)

        V6_fourth.append(I4)
        V6_curv.append(Ic)
        V6_mixed.append(Im)
        V6_grad.append(Ig)

    return R_values, V6_fourth, V6_curv, V6_mixed, V6_grad


def test_electron_muon_v6():
    """
    Calculate V₆ integrals for electron and muon.
    Compare to required values from experiment.
    """
    print("="*80)
    print("V₆ CALCULATION: ELECTRON AND MUON")
    print("="*80)
    print()

    # Required V₆ from experiment
    V6_electron_req, V6_muon_req = extract_v6_from_experiment()

    print("Required V₆ from experimental g-2:")
    print(f"  Electron: V₆ = {V6_electron_req:.4f}")
    print(f"  Muon:     V₆ = {V6_muon_req:.4f}")
    print(f"  QED C₃:   C₃ = {C3_QED:.5f}")
    print()

    # Calculate geometric V₆ contributions
    print("Calculating geometric V₆ contributions...")
    print()

    leptons = [
        ("Electron", R_electron),
        ("Muon", R_muon)
    ]

    results = {}

    for name, R in leptons:
        print(f"{name} (R = {R:.2f} fm):")

        I4 = calculate_v6_fourth_power(R)
        Ic = calculate_v6_curvature(R)
        Im = calculate_v6_mixed_velocity(R)
        Ig = calculate_v6_gradient_squared(R)

        print(f"  Fourth power:    I₄ = {I4:.6e}")
        print(f"  Curvature:       Ic = {Ic:.6e}")
        print(f"  Mixed velocity:  Im = {Im:.6e}")
        print(f"  Gradient²:       Ig = {Ig:.6e}")
        print()

        results[name] = {
            'R': R,
            'I4': I4,
            'Ic': Ic,
            'Im': Im,
            'Ig': Ig,
            'V6_required': V6_electron_req if name == "Electron" else V6_muon_req
        }

    return results


def calibrate_v6_coefficients(results):
    """
    Determine coupling constants for V₆ contributions.

    V₆ = α₄·I₄ + αc·Ic + αm·Im + αg·Ig

    Fit to match electron and muon requirements.
    """
    print("="*80)
    print("CALIBRATION: V₆ COUPLING CONSTANTS")
    print("="*80)
    print()

    # Extract values
    I4_e = results['Electron']['I4']
    Ic_e = results['Electron']['Ic']
    Im_e = results['Electron']['Im']
    Ig_e = results['Electron']['Ig']
    V6_e = results['Electron']['V6_required']

    I4_mu = results['Muon']['I4']
    Ic_mu = results['Muon']['Ic']
    Im_mu = results['Muon']['Im']
    Ig_mu = results['Muon']['Ig']
    V6_mu = results['Muon']['V6_required']

    # Test single-term hypotheses
    print("Single-term hypotheses:")
    print()

    # Hypothesis 1: Only fourth power
    if abs(I4_e) > 1e-10:
        alpha4_from_e = V6_e / I4_e
        V6_mu_pred = alpha4_from_e * I4_mu
        error = abs(V6_mu_pred - V6_mu) / abs(V6_mu)
        print(f"H1: V₆ = α₄ · I₄")
        print(f"  α₄ = {alpha4_from_e:.6f} (from electron)")
        print(f"  Muon prediction: {V6_mu_pred:.4f} vs required {V6_mu:.4f}")
        print(f"  Error: {100*error:.1f}%")
        print()

    # Hypothesis 2: Only curvature
    if abs(Ic_e) > 1e-10:
        alphac_from_e = V6_e / Ic_e
        V6_mu_pred = alphac_from_e * Ic_mu
        error = abs(V6_mu_pred - V6_mu) / abs(V6_mu)
        print(f"H2: V₆ = αc · Ic")
        print(f"  αc = {alphac_from_e:.6f} (from electron)")
        print(f"  Muon prediction: {V6_mu_pred:.4f} vs required {V6_mu:.4f}")
        print(f"  Error: {100*error:.1f}%")
        print()

    # Hypothesis 3: Mixed velocity
    if abs(Im_e) > 1e-10:
        alpham_from_e = V6_e / Im_e
        V6_mu_pred = alpham_from_e * Im_mu
        error = abs(V6_mu_pred - V6_mu) / abs(V6_mu)
        print(f"H3: V₆ = αm · Im")
        print(f"  αm = {alpham_from_e:.6f} (from electron)")
        print(f"  Muon prediction: {V6_mu_pred:.4f} vs required {V6_mu:.4f}")
        print(f"  Error: {100*error:.1f}%")
        print()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("V₆ HIGHER-ORDER CORRECTION FROM HILL VORTEX GEOMETRY")
    print("Testing if V₆ ≈ C₃(QED) = +1.18")
    print("="*80)
    print()

    # Test electron and muon
    results = test_electron_muon_v6()

    # Calibrate coupling constants
    calibrate_v6_coefficients(results)

    # Scan full range
    print("="*80)
    print("FULL RADIUS SCAN")
    print("="*80)
    print()

    R_values, V6_fourth, V6_curv, V6_mixed, V6_grad = scan_v6_contributions(
        R_min=0.1, R_max=1000, n_points=30
    )

    # Plot results
    os.makedirs('../results', exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Fourth power
    ax1.loglog(R_values, np.abs(V6_fourth), 'b-', linewidth=2)
    ax1.axvline(R_electron, color='g', alpha=0.3, label='Electron')
    ax1.axvline(R_muon, color='r', alpha=0.3, label='Muon')
    ax1.set_xlabel('R (fm)')
    ax1.set_ylabel('|I₄|')
    ax1.set_title('Fourth Power: (v_φ)⁴ · (dρ/dr)⁴')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Curvature
    ax2.loglog(R_values, np.abs(V6_curv), 'r-', linewidth=2)
    ax2.axvline(R_electron, color='g', alpha=0.3, label='Electron')
    ax2.axvline(R_muon, color='r', alpha=0.3, label='Muon')
    ax2.set_xlabel('R (fm)')
    ax2.set_ylabel('|Ic|')
    ax2.set_title('Curvature: (v_φ)² · (d²ρ/dr²)²')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Mixed velocity
    ax3.loglog(R_values, np.abs(V6_mixed), 'g-', linewidth=2)
    ax3.axvline(R_electron, color='g', alpha=0.3, label='Electron')
    ax3.axvline(R_muon, color='r', alpha=0.3, label='Muon')
    ax3.set_xlabel('R (fm)')
    ax3.set_ylabel('|Im|')
    ax3.set_title('Mixed Velocity: v_r² · v_φ² · (dρ/dr)²')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Gradient squared
    ax4.loglog(R_values, np.abs(V6_grad), 'm-', linewidth=2)
    ax4.axvline(R_electron, color='g', alpha=0.3, label='Electron')
    ax4.axvline(R_muon, color='r', alpha=0.3, label='Muon')
    ax4.set_xlabel('R (fm)')
    ax4.set_ylabel('|Ig|')
    ax4.set_title('Gradient²: (v_φ)² · (dρ/dr)² · (∇²ρ)²')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    fig.savefig('../results/v6_contributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot: results/v6_contributions.png")
    print()

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("V₆ represents next-order geometric correction:")
    print("  - If single term matches both e and μ → universal V₆")
    print("  - If combination needed → generation-dependent V₆(R)")
    print("  - If V₆ ≈ C₃ = +1.18 → QED fully emergent from geometry!")
    print()
    print("="*80)
