#!/usr/bin/env python3
"""
Radius Constraint Functions

Purpose: Compute RMS radius and gradient proxy from density profiles.
         These provide a second, non-EM geometric observable to break
         the (R,U,A) degeneracy and test whether β_eff → β_Golden.

Physical motivation:
- RMS radius captures void volume and boundary-layer geometry
- Gradient proxy G quantifies curvature energy (missing in current closure)
- Scaling law m·R_rms ≈ κ tests cross-lepton geometric consistency
"""

import numpy as np
from typing import Tuple, Dict


def compute_R_rms(delta_rho: np.ndarray, r_grid: np.ndarray) -> float:
    """
    Compute deficit-weighted RMS radius.

    R_rms = sqrt(∫ r⁴ (Δρ)² dr / ∫ r² (Δρ)² dr)

    where Δρ = ρ_vac - ρ(r) is the density deficit (≥ 0 in void core).

    Physical interpretation:
    - Measures the "size" of the vacuum displacement
    - Weighted by deficit squared (same as stabilization energy)
    - Differentiates "same deficit volume, different boundary thickness"

    Args:
        delta_rho: Density deficit profile Δρ(r) = ρ_vac - ρ(r)
        r_grid: Radial grid points

    Returns:
        R_rms: RMS radius in code units

    Note: The spherical 4πr² volume element cancels between numerator
          and denominator, so we integrate r⁴(Δρ)² and r²(Δρ)² directly.
    """
    # Ensure delta_rho is non-negative (should be by construction)
    delta_rho = np.abs(delta_rho)

    # Numerator: ∫ r⁴ (Δρ)² dr
    integrand_num = r_grid**4 * delta_rho**2
    numerator = np.trapz(integrand_num, r_grid)

    # Denominator: ∫ r² (Δρ)² dr
    integrand_den = r_grid**2 * delta_rho**2
    denominator = np.trapz(integrand_den, r_grid)

    # Avoid division by zero (should not happen for physical solutions)
    if denominator < 1e-20:
        return 0.0

    R_rms = np.sqrt(numerator / denominator)
    return float(R_rms)


def compute_gradient_proxy(rho: np.ndarray, r_grid: np.ndarray) -> float:
    """
    Compute gradient energy proxy (diagnostic, not a constraint).

    G = ∫ (dρ/dr)² 4πr² dr

    Physical interpretation:
    - Proxy for curvature/gradient energy E_∇ ∝ ∫ |∇ρ|² dV
    - Penalizes sharp boundaries and steep transitions
    - Expected to be larger for tau (sharper void boundary)

    Args:
        rho: Density profile ρ(r)
        r_grid: Radial grid points

    Returns:
        G: Gradient proxy in code units

    Note: This is a DIAGNOSTIC only. It exposes whether tau has
          pathologically steep boundaries, but is not included in χ²
          on the first pass (may be added later if needed).
    """
    # Compute dρ/dr using centered finite differences
    drho_dr = np.gradient(rho, r_grid)

    # Integrate (dρ/dr)² with spherical volume element 4πr²
    integrand = drho_dr**2 * 4 * np.pi * r_grid**2
    G = np.trapz(integrand, r_grid)

    return float(G)


def profile_kappa_analytically(x_leptons: Dict[str, float],
                               f_sigma: float = 0.03) -> Tuple[float, float, float]:
    """
    Analytically profile out the shared scale κ.

    Given per-lepton x_ℓ = m_ℓ · R_rms,ℓ, find the optimal κ that
    minimizes the radius constraint χ² term.

    Method (Tracy's κ₀ reference trick):
    1. Compute reference: κ₀ = mean(x_ℓ)
    2. Set σ_κ = f · κ₀  (relative error, but with fixed reference)
    3. Optimal κ = mean(x_ℓ)  (simple unweighted mean)
    4. χ²_R = Σ [(x_ℓ - κ_opt) / σ_κ]²

    This keeps profiling strictly quadratic and numerically stable,
    while preserving "3% relative" interpretation.

    Args:
        x_leptons: Dict of {lepton_name: m·R_rms} values
        f_sigma: Relative uncertainty on κ (default 3%)

    Returns:
        kappa_opt: Optimal shared scale
        sigma_kappa: Absolute uncertainty σ_κ = f · κ₀
        chi2_radius: Radius constraint χ² at optimum
    """
    # Extract x values in consistent order
    x_e = x_leptons['electron']
    x_mu = x_leptons['muon']
    x_tau = x_leptons['tau']

    # Reference scale (unweighted mean)
    kappa_0 = (x_e + x_mu + x_tau) / 3.0

    # Optimal κ (same as κ₀ for unweighted case)
    kappa_opt = kappa_0

    # Absolute uncertainty
    sigma_kappa = f_sigma * kappa_0

    # χ² from radius constraints
    chi2_radius = (
        ((x_e - kappa_opt) / sigma_kappa)**2 +
        ((x_mu - kappa_opt) / sigma_kappa)**2 +
        ((x_tau - kappa_opt) / sigma_kappa)**2
    )

    return kappa_opt, sigma_kappa, chi2_radius


def analyze_radius_consistency(results: Dict[str, Dict]) -> Dict:
    """
    Analyze cross-lepton radius scaling consistency.

    Args:
        results: Dict with per-lepton fit results containing:
                 {lepton: {'mass': m, 'R_rms': R, 'G': G, ...}}

    Returns:
        analysis: Dict with:
            - kappa_opt: Optimal shared scale
            - kappa_spread: Max fractional deviation from mean
            - G_ratios: Gradient proxy ratios (tau/electron, muon/electron)
            - radius_chi2: χ² from radius constraint
    """
    # Compute x_ℓ = m_ℓ · R_rms,ℓ for each lepton
    x_leptons = {}
    for lepton in ['electron', 'muon', 'tau']:
        if lepton in results:
            m = results[lepton]['mass']
            R_rms = results[lepton]['R_rms']
            x_leptons[lepton] = m * R_rms

    # Profile κ analytically
    kappa_opt, sigma_kappa, chi2_radius = profile_kappa_analytically(x_leptons)

    # Compute fractional spread
    x_values = list(x_leptons.values())
    kappa_spread = (max(x_values) - min(x_values)) / kappa_opt

    # Gradient proxy ratios (if available)
    G_ratios = {}
    if all(lepton in results for lepton in ['electron', 'muon', 'tau']):
        G_e = results['electron'].get('G', None)
        G_mu = results['muon'].get('G', None)
        G_tau = results['tau'].get('G', None)

        if G_e and G_e > 0:
            G_ratios['muon_to_electron'] = G_mu / G_e
            G_ratios['tau_to_electron'] = G_tau / G_e

    return {
        'kappa_opt': kappa_opt,
        'sigma_kappa': sigma_kappa,
        'kappa_spread': kappa_spread,
        'x_leptons': x_leptons,
        'G_ratios': G_ratios,
        'radius_chi2': chi2_radius
    }


# Unit tests
if __name__ == "__main__":
    print("Testing radius constraint functions...\n")

    # Test 1: Simple Gaussian-like deficit
    r = np.linspace(0.01, 5.0, 200)
    delta_rho_gaussian = np.exp(-r**2 / 0.5**2)

    R_rms_test = compute_R_rms(delta_rho_gaussian, r)
    print(f"Test 1: Gaussian deficit (σ=0.5)")
    print(f"  R_rms = {R_rms_test:.4f}")
    print(f"  Expected: ~0.61 (sqrt(3/2)·σ for Gaussian)")

    # Test 2: Gradient proxy
    rho_test = 1.0 - delta_rho_gaussian  # ρ_vac = 1.0
    G_test = compute_gradient_proxy(rho_test, r)
    print(f"\nTest 2: Gradient proxy")
    print(f"  G = {G_test:.4e}")

    # Test 3: Analytic profiling
    x_test = {
        'electron': 1.00,
        'muon': 1.05,
        'tau': 0.95
    }
    kappa, sigma, chi2 = profile_kappa_analytically(x_test, f_sigma=0.03)
    print(f"\nTest 3: Analytic κ profiling")
    print(f"  x_e = {x_test['electron']:.3f}")
    print(f"  x_μ = {x_test['muon']:.3f}")
    print(f"  x_τ = {x_test['tau']:.3f}")
    print(f"  κ_opt = {kappa:.4f}")
    print(f"  σ_κ = {sigma:.4f}")
    print(f"  χ²_R = {chi2:.2f}")
    print(f"  Spread: {(max(x_test.values())-min(x_test.values()))/kappa*100:.1f}%")

    print("\n✓ All tests complete")
