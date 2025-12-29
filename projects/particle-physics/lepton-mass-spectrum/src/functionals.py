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
