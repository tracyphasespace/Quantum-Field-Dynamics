"""
QFD Cosmology Adapter: Unified Transport (Spherical Scattering)
Module: qfd.adapters.cosmology.cmb_boltzmann_transport

REFINED UNIFIED MODEL:
Merges Boltzmann Transport (frequency-dependent scattering) with
Vacuum Soliton geometry (spherical form factor).

Physics:
1. Scattering Cross-Section: σ(k) ∝ |F_sphere(k)|²
2. Form Factor: Spherical Bessel j₁(kr) for bubble/void geometry
3. Frequency Scaling: "Blue Interaction" - higher ν → more scattering
4. Proper Normalization: Includes ℓ(ℓ+1)/(2π) CMB convention

Key Insights:
- Geometry dictates transport: Universe made of SPHERES not slits
- Scattering kernel must match scatterer shape (spherical_jn)
- 274 Mpc = DIAMETER, 147 Mpc = RADIUS (factor of 2)
- Transport and Geometry are dual views of same vacuum structure

Radius vs Diameter Test:
This model tests whether we measure domain RADIUS (r ~ 147 Mpc)
or DIAMETER (d ~ 274 Mpc) by fitting the scale directly.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.special import spherical_jn


def spherical_scattering_kernel(k, r, sharpness=4.0):
    """
    Scattering cross-section of spherical vacuum domains with non-linear sharpening.

    This is the structure factor S(k) = |F(k)|² where F(k) is the
    Fourier transform of a uniform sphere, modified by the quartic
    potential V(ρ) = β ρ⁴.

    Parameters
    ----------
    k : array-like
        Wavenumber (ℓ/D_M in Mpc⁻¹)
    r : float
        Domain radius in Mpc (NOT diameter)
    sharpness : float
        Non-linearity parameter from V(ρ) = β ρ⁴ (beta_wall)
        Higher → sharper domain boundaries → more pronounced peaks
        The effective form factor becomes: (|F|²)^(1/sharpness)

    Returns
    -------
    array-like
        Non-linearly sharpened form factor (scattering intensity)

    Notes
    -----
    For a uniform sphere of radius r:
        F(k) = 3 * j₁(kr) / (kr)
    where j₁ is the spherical Bessel function of first kind, order 1.

    The non-linear sharpening implements:
        F_nonlinear = (|F|²)^(1/β)

    This is analogous to the Mexican hat potential sharpening the
    transition between inside/outside the soliton core.

    Peak structure:
    - First peak at kr ≈ 5.76 (constructive interference)
    - Zeros at kr = nπ for n = 1, 2, 3, ... (destructive interference)
    - Sharpness modulates peak heights and widths
    """
    x = k * r

    # Use scipy's spherical Bessel for numerical stability
    # spherical_jn(n=1, z) computes j₁(z)
    # Add small epsilon to avoid division by zero at k=0
    x_safe = np.where(np.abs(x) > 1e-9, x, 1e-9)

    # Form factor for sphere: F(x) = 3 * j₁(x) / x
    form_factor = 3.0 * spherical_jn(1, x_safe) / x_safe

    # Square to get intensity (power spectrum)
    F_squared = form_factor**2

    # Apply non-linear sharpening from quartic potential
    # This makes domain walls sharper (β > 1) or softer (β < 1)
    # CRITICAL: This is what the geometric model does!
    F_nonlinear = np.power(np.abs(F_squared) + 1e-20, 1.0 / sharpness)

    return F_nonlinear


def predict_cmb_transport_spectrum(
    df: pd.DataFrame,
    params: Dict[str, float],
    config: Dict[str, Any] = None
) -> np.ndarray:
    """
    Predicts CMB power spectrum using unified transport-geometry model.

    Model Equation:
    ---------------
    D_ℓ = A × |F_sphere(kr)|² × [ℓ(ℓ+1)/(2π)] × (ℓ/ℓ₀)^α

    where:
    - F_sphere(kr) = 3j₁(kr)/(kr) is spherical form factor
    - ℓ(ℓ+1)/(2π) is standard CMB normalization
    - (ℓ/ℓ₀)^α is spectral tilt (Blue Interaction)
    - α ≈ 0: pure geometry (no frequency dependence)
    - α > 0: blue tilt (high ℓ enhanced)
    - α < 0: red tilt (low ℓ enhanced)

    Parameters
    ----------
    r_psi : float (Mpc)
        Domain RADIUS (not diameter)
        Hypothesis: r ~ 137-150 Mpc
        If fit finds r ~ 274 Mpc, we're measuring diameter

    beta_wall : float (dimensionless)
        Domain wall sharpness from non-linear potential V(ρ) = β ρ⁴
        Higher → sharper boundaries → more pronounced peaks
        This is the CRITICAL parameter that the geometric model uses!

    A_norm : float (μK²)
        Overall amplitude normalization

    alpha : float (dimensionless)
        Spectral tilt exponent (Blue Interaction strength)
        Expected: |α| < 0.2 (small perturbation to geometry)

    chi_star : float (Mpc)
        Comoving distance to last scattering (fixed at 14156)
    """
    # 1. Extract multipole moments
    ell = df.get("ell", df.get("L")).astype(float).to_numpy()

    # 2. Extract parameters
    r_radius = params.get('r_psi', 147.0)      # Domain radius (Mpc)
    beta_wall = params.get('beta_wall', 4.0)   # Sharpness (CRITICAL!)
    A_norm = params.get('A_norm', 1.0)         # Amplitude
    alpha = params.get('alpha', 0.0)           # Spectral tilt
    D_M = params.get('chi_star', 14156.0)      # Distance to z=1090

    # Optional: Test diameter vs radius hypothesis
    # Set scale_factor = 1.0 to fit radius directly
    # Set scale_factor = 2.0 to convert radius to diameter
    scale_factor = params.get('scale_factor', 1.0)
    r_effective = r_radius * scale_factor

    # 3. Compute wavenumber from multipole
    # Limber approximation: k ≈ (ℓ + 1/2) / D_M
    k = (ell + 0.5) / D_M  # Mpc⁻¹

    # 4. Spherical scattering resonance (the "wiggles")
    # This encodes the vacuum domain structure
    # INCLUDES NON-LINEAR SHARPENING - the key to matching geometric model!
    resonance = spherical_scattering_kernel(k, r_effective, sharpness=beta_wall)

    # 5. Standard CMB normalization
    # D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) is the standard convention
    # This was MISSING in previous version!
    ell_factor = ell * (ell + 1.0) / (2.0 * np.pi)

    # 6. Spectral tilt (Blue Interaction)
    # Normalized at ℓ = 100 for numerical stability
    # α > 0: blue photons scatter more (high ℓ enhanced)
    # α < 0: red photons scatter more (low ℓ enhanced)
    ell_pivot = 100.0
    tilt_modulation = (ell / ell_pivot)**alpha

    # 7. Combine all components
    # Amplitude × Resonance × CMB_Normalization × Frequency_Modulation
    D_ell_predicted = A_norm * resonance * ell_factor * tilt_modulation

    return D_ell_predicted

