"""
Variational solvers for Euler-Lagrange equations.

Solves for equilibrium density profile ρ(r) that minimizes energy functional.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson, odeint
from typing import Tuple, Optional
import warnings


def hill_vortex_profile(r: np.ndarray, R: float, U: float, A: float = 1.0) -> np.ndarray:
    """
    Hill's spherical vortex density profile.

    ρ(r) = ρ_vac + A·f(r/R)  for r < R
    ρ(r) = ρ_vac              for r ≥ R

    where f is the Hill vortex shape function.

    Parameters
    ----------
    r : array
        Radial coordinates
    R : float
        Vortex radius
    U : float
        Circulation velocity
    A : float
        Amplitude normalization

    Returns
    -------
    ρ : array
        Density profile (normalized to ρ_vac = 1.0)
    """
    ρ = np.ones_like(r)  # Start with vacuum density = 1.0

    # Hill vortex interior (r < R)
    mask_interior = r < R
    x = r[mask_interior] / R

    # Hill vortex shape: f(x) = (1 - x²)² for simple model
    # This is approximate - full Hill vortex has more complex profile
    f = (1 - x**2)**2

    ρ[mask_interior] += A * f * (U / 0.5)**2  # Scale with velocity

    return ρ


def solve_euler_lagrange(
    ξ: float,
    β: float,
    R: float,
    U: float,
    A: float = 1.0,
    r_max: float = 10.0,
    n_points: int = 500,
    method: str = 'relaxation'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Euler-Lagrange equation for density profile.

    Equation: -ξ∇²ρ + 2β(ρ - ρ_vac) = 0

    Boundary conditions:
    - ρ(R) from Hill vortex
    - ρ(∞) → ρ_vac = 1.0
    - dρ/dr(0) = 0 (regularity at origin)

    Parameters
    ----------
    ξ : float
        Gradient stiffness
    β : float
        Vacuum stiffness
    R : float
        Vortex core radius
    U : float
        Circulation velocity
    A : float
        Amplitude normalization
    r_max : float
        Maximum radius (R_max >> R)
    n_points : int
        Number of radial grid points
    method : str
        Solver method: 'relaxation' (iterative) or 'shooting' (ODE)

    Returns
    -------
    r : array
        Radial grid
    ρ : array
        Equilibrium density profile
    """
    # Create radial grid
    r = np.linspace(0, r_max * R, n_points)

    if method == 'relaxation':
        return _solve_relaxation(ξ, β, R, U, A, r)
    elif method == 'shooting':
        return _solve_shooting(ξ, β, R, U, A, r)
    else:
        raise ValueError(f"Unknown method: {method}")


def _solve_relaxation(
    ξ: float,
    β: float,
    R: float,
    U: float,
    A: float,
    r: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve via iterative relaxation.

    Algorithm:
    1. Initialize with Hill vortex profile
    2. Compute Laplacian ∇²ρ
    3. Update: ρ_new = ρ_vac + (ξ/2β)∇²ρ
    4. Relax: ρ ← (1-α)ρ + α·ρ_new
    5. Repeat until convergence
    """
    ρ_vac = 1.0
    dr = r[1] - r[0]

    # Initialize with Hill vortex
    ρ = hill_vortex_profile(r, R, U, A)

    for iteration in range(max_iter):
        ρ_old = ρ.copy()

        # Compute Laplacian in spherical coordinates
        dρ_dr = np.gradient(ρ, r)
        r_sq_dρ_dr = r**2 * dρ_dr
        d_r_sq_dρ_dr = np.gradient(r_sq_dρ_dr, r)
        laplacian_ρ = d_r_sq_dρ_dr / (r**2 + 1e-10)  # Avoid division by zero

        # Handle origin using L'Hôpital
        if r[0] < 1e-10:
            d2ρ_dr2_0 = (ρ[2] - 2*ρ[1] + ρ[0]) / (dr**2)
            laplacian_ρ[0] = 3 * d2ρ_dr2_0

        # Update from Euler-Lagrange: 2β(ρ - ρ_vac) = ξ∇²ρ
        ρ_new = ρ_vac + (ξ / (2 * β)) * laplacian_ρ

        # Enforce boundary conditions
        # BC1: ρ(∞) = ρ_vac
        ρ_new[-1] = ρ_vac

        # BC2: Match Hill vortex at core (keep interior fixed)
        mask_core = r < R
        ρ_new[mask_core] = ρ[mask_core]

        # Relaxation (α = 0.5 for stability)
        α = 0.5
        ρ = (1 - α) * ρ_old + α * ρ_new

        # Check convergence
        residual = np.max(np.abs(ρ - ρ_old))
        if residual < tol:
            break

    if iteration == max_iter - 1:
        warnings.warn(f"Relaxation did not converge after {max_iter} iterations")

    return r, ρ


def _solve_shooting(
    ξ: float,
    β: float,
    R: float,
    U: float,
    A: float,
    r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve via shooting method (ODE integration).

    Convert 2nd order ODE to system of 1st order:
    ρ' = v
    v' = (1/r²) d/dr(r² v) = v/r + (2β/ξ)(ρ - ρ_vac)

    Integrate outward from r=0 with initial conditions.
    """
    ρ_vac = 1.0

    def ode_system(y, r_val):
        """ODE system [ρ, v=dρ/dr]"""
        ρ_val, v = y

        if r_val < 1e-10:
            # At origin: regularity requires v=0, v'=0
            return [0.0, 0.0]

        # Equation: ξ(v'/r + v/r²) = 2β(ρ - ρ_vac)
        # → v' = (2β/ξ)(ρ - ρ_vac)·r - v/r
        dvdr = (2 * β / ξ) * (ρ_val - ρ_vac) * r_val - v / r_val

        return [v, dvdr]

    # Initial conditions at origin
    # ρ(0) from Hill vortex central density
    ρ_0 = 1.0 + A * (U / 0.5)**2  # Approximate central density
    v_0 = 0.0  # Regularity: dρ/dr(0) = 0

    # Integrate
    solution = odeint(ode_system, [ρ_0, v_0], r)
    ρ = solution[:, 0]

    # Enforce boundary: decay to ρ_vac at infinity
    # (May need adjustment for better boundary matching)

    return r, ρ


def integrate_energy(
    ξ: float,
    β: float,
    ρ: np.ndarray,
    r: np.ndarray,
    ρ_vac: float = 1.0
) -> float:
    """
    Integrate total energy for given density profile.

    E = ∫ [½ξ|∇ρ|² + β(δρ)²] · 4πr² dr

    Parameters
    ----------
    ξ : float
        Gradient stiffness
    β : float
        Vacuum stiffness
    ρ : array
        Density profile
    r : array
        Radial grid
    ρ_vac : float
        Vacuum density

    Returns
    -------
    E_total : float
        Total energy
    """
    from .functionals import gradient_energy_functional

    E_total, _, _ = gradient_energy_functional(ρ, r, ξ, β, ρ_vac)
    return E_total


def compute_stability_eigenvalue(
    ξ: float,
    β: float,
    τ: float,
    ρ: np.ndarray,
    r: np.ndarray,
    mode: str = 'breathing'
) -> float:
    """
    Compute lowest stability eigenvalue (breathing mode frequency).

    Linearize around equilibrium:
    ρ = ρ_eq + ε·δρ·exp(iωt)

    Eigenvalue problem:
    -τω²·δρ = -ξ∇²(δρ) + 2β·δρ

    For breathing mode (spherically symmetric):
    ω² = (2β - ξk²) / τ  where k is radial wavenumber

    Parameters
    ----------
    ξ, β, τ : float
        Functional parameters
    ρ : array
        Equilibrium density
    r : array
        Radial grid
    mode : str
        Perturbation mode ('breathing' or 'quadrupole')

    Returns
    -------
    ω : float
        Angular frequency of lowest mode (imaginary if unstable)
    """
    # Simplified estimate: use characteristic wavenumber k ~ 1/R
    R_eff = r[len(r)//2]  # Approximate vortex size
    k = 1.0 / R_eff

    # Breathing mode dispersion
    ω_squared = (2 * β - ξ * k**2) / τ

    if ω_squared < 0:
        warnings.warn("Unstable mode detected (ω² < 0)")
        return np.sqrt(-ω_squared) * 1j  # Imaginary frequency
    else:
        return np.sqrt(ω_squared)


def compute_mass_from_functional(
    ξ: float,
    β: float,
    R: float,
    U: float,
    A: float = 1.0,
    r_max: float = 10.0,
    n_points: int = 500
) -> float:
    """
    Complete pipeline: Solve EL equation → integrate energy → compute mass.

    Parameters
    ----------
    ξ : float
        Gradient stiffness
    β : float
        Vacuum stiffness
    R : float
        Vortex radius (meters)
    U : float
        Velocity (fraction of c)
    A : float
        Amplitude
    r_max : float
        Integration domain (× R)
    n_points : int
        Grid resolution

    Returns
    -------
    mass : float
        Effective mass in MeV/c²
    """
    # Solve for equilibrium density
    r, ρ = solve_euler_lagrange(ξ, β, R, U, A, r_max, n_points)

    # Integrate energy
    E_total = integrate_energy(ξ, β, ρ, r)

    # Convert to mass (E = mc²)
    # TODO: Handle unit conversion properly
    # For now, return in arbitrary energy units
    mass_arbitrary = E_total  # Placeholder

    return mass_arbitrary
