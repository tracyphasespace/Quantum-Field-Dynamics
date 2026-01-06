"""
Energy functional implementations for QFD lepton model.

Implements hierarchical functionals:
- V22 baseline: E = ∫ β(δρ)² dV
- Stage 1: E = ∫ [½ξ|∇ρ|² + β(δρ)²] dV
- Stage 2: E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV
"""

import numpy as np
from typing import Tuple, Callable
from scipy.integrate import simpson


def gradient_energy_functional(
    ρ: np.ndarray,
    r: np.ndarray,
    ξ: float,
    β: float,
    ρ_vac: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute energy with gradient density term.

    E = ∫ [½ξ|∇ρ|² + β(δρ)²] · 4πr² dr

    Parameters
    ----------
    ρ : array
        Density profile ρ(r)
    r : array
        Radial grid
    ξ : float
        Gradient stiffness parameter (dimensionless)
    β : float
        Vacuum stiffness parameter (dimensionless)
    ρ_vac : float
        Vacuum density baseline (default: 1.0)

    Returns
    -------
    E_total : float
        Total energy
    E_gradient : float
        Gradient contribution ∫ ½ξ|∇ρ|² dV
    E_compression : float
        Compression contribution ∫ β(δρ)² dV
    """
    # Compute gradient |∇ρ| = |dρ/dr| in spherical symmetry
    dρ_dr = np.gradient(ρ, r)
    grad_ρ_squared = dρ_dr**2

    # Density perturbation from vacuum
    δρ = ρ - ρ_vac

    # Energy densities
    ε_gradient = 0.5 * ξ * grad_ρ_squared
    ε_compression = β * δρ**2

    # Integrate in spherical coordinates: ∫ ε · 4πr² dr
    E_gradient = simpson(ε_gradient * 4 * np.pi * r**2, r)
    E_compression = simpson(ε_compression * 4 * np.pi * r**2, r)

    E_total = E_gradient + E_compression

    return E_total, E_gradient, E_compression


def temporal_energy_functional(
    ρ: np.ndarray,
    r: np.ndarray,
    ξ: float,
    β: float,
    τ: float,
    dρ_dt: np.ndarray = None,
    ρ_vac: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Compute energy with gradient and temporal terms.

    E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] · 4πr² dr

    Parameters
    ----------
    ρ : array
        Density profile ρ(r)
    r : array
        Radial grid
    ξ : float
        Gradient stiffness
    β : float
        Vacuum stiffness
    τ : float
        Temporal stiffness
    dρ_dt : array, optional
        Time derivative ∂ρ/∂t. If None, assume static solution (∂ρ/∂t = 0)
    ρ_vac : float
        Vacuum density baseline

    Returns
    -------
    E_total, E_gradient, E_compression, E_temporal
    """
    # Gradient and compression terms (same as before)
    E_total, E_gradient, E_compression = gradient_energy_functional(
        ρ, r, ξ, β, ρ_vac
    )

    # Temporal term
    if dρ_dt is None:
        # Static solution: ∂ρ/∂t = 0
        E_temporal = 0.0
    else:
        ε_temporal = τ * dρ_dt**2
        E_temporal = simpson(ε_temporal * 4 * np.pi * r**2, r)

    E_total += E_temporal

    return E_total, E_gradient, E_compression, E_temporal


def compute_mass_from_energy(E_total: float, c: float = 299792458.0) -> float:
    """
    Convert total energy to effective mass via E = mc².

    Parameters
    ----------
    E_total : float
        Total energy (in Joules or natural units)
    c : float
        Speed of light (default: SI units m/s)

    Returns
    -------
    mass : float
        Effective mass in kg (or natural units if c=1)
    """
    return E_total / c**2


def v22_baseline_functional(
    ρ: np.ndarray,
    r: np.ndarray,
    β: float,
    ρ_vac: float = 1.0
) -> float:
    """
    V22 baseline functional without gradient term.

    E = ∫ β(δρ)² · 4πr² dr

    This is the simplified model that gave β ≈ 3.15.
    Should reproduce V22 results when ξ=0.

    Parameters
    ----------
    ρ : array
        Density profile
    r : array
        Radial grid
    β : float
        Vacuum stiffness
    ρ_vac : float
        Vacuum density

    Returns
    -------
    E_total : float
        Total energy
    """
    δρ = ρ - ρ_vac
    ε = β * δρ**2
    return simpson(ε * 4 * np.pi * r**2, r)


def euler_lagrange_residual(
    ρ: np.ndarray,
    r: np.ndarray,
    ξ: float,
    β: float,
    ρ_vac: float = 1.0
) -> np.ndarray:
    """
    Compute residual of Euler-Lagrange equation.

    Variational derivative: δE/δρ = 0
    → -ξ∇²ρ + 2β(ρ - ρ_vac) = 0

    In spherical coordinates:
    ∇²ρ = (1/r²) d/dr(r² dρ/dr)

    Parameters
    ----------
    ρ : array
        Trial density profile
    r : array
        Radial grid
    ξ : float
        Gradient stiffness
    β : float
        Vacuum stiffness
    ρ_vac : float
        Vacuum density

    Returns
    -------
    residual : array
        Left-hand side of Euler-Lagrange equation
        (should be ~0 for true solution)
    """
    # First derivative
    dρ_dr = np.gradient(ρ, r)

    # Laplacian in spherical coordinates
    # ∇²ρ = (1/r²) d/dr(r² dρ/dr)
    r_sq_dρ_dr = r**2 * dρ_dr
    d_r_sq_dρ_dr = np.gradient(r_sq_dρ_dr, r)
    laplacian_ρ = d_r_sq_dρ_dr / r**2

    # Handle singularity at r=0 using L'Hôpital
    # lim_{r→0} (1/r²) d/dr(r² dρ/dr) = 3 d²ρ/dr²|_{r=0}
    if r[0] == 0 or r[0] < 1e-10:
        # Estimate second derivative at origin
        d2ρ_dr2_0 = (ρ[2] - 2*ρ[1] + ρ[0]) / (r[1]**2)
        laplacian_ρ[0] = 3 * d2ρ_dr2_0

    # Euler-Lagrange equation
    residual = -ξ * laplacian_ρ + 2 * β * (ρ - ρ_vac)

    return residual


def check_boundary_conditions(
    ρ: np.ndarray,
    r: np.ndarray,
    R_core: float,
    ρ_vac: float = 1.0
) -> Tuple[float, float]:
    """
    Check boundary conditions for density profile.

    Required:
    1. ρ(R_core) matches Hill vortex
    2. ρ(r→∞) → ρ_vac

    Parameters
    ----------
    ρ : array
        Density profile
    r : array
        Radial grid
    R_core : float
        Vortex core radius
    ρ_vac : float
        Vacuum density

    Returns
    -------
    bc_core_error : float
        Error in core boundary condition
    bc_infinity_error : float
        Error in asymptotic boundary condition
    """
    # Find index closest to R_core
    idx_core = np.argmin(np.abs(r - R_core))
    ρ_at_core = ρ[idx_core]

    # Expected from Hill vortex (placeholder - should match actual profile)
    ρ_expected_core = ρ_vac + 1.0  # TODO: Use actual Hill vortex value

    bc_core_error = np.abs(ρ_at_core - ρ_expected_core)

    # Asymptotic behavior
    ρ_at_infinity = ρ[-1]
    bc_infinity_error = np.abs(ρ_at_infinity - ρ_vac)

    return bc_core_error, bc_infinity_error
