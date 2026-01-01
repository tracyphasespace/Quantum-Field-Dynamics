"""
Reverse-engineer the quartic potential parameters for the lepton isomer ladder.

We model the scalar potential as:

    V(ρ) = -μ² ρ + λ ρ² + κ ρ³ + β ρ⁴ ,

with β fixed by the Golden-Loop constraint (β = 3.058230856).
Demanding that this potential admit *two* degenerate minima at radii
proportional to the electron and muon masses results in a fully
determined set of coefficients.

Let the lighter minimum be normalised to ρₑ = 1 and the heavier
minimum be ρ_μ = m_μ / mₑ ≈ 206.77. Enforcing:

    dV/dρ (ρₑ) = dV/dρ (ρ_μ) = 0,
    V(ρₑ) = V(ρ_μ),

leads to the closed-form solution

    κ   = -2 β (ρₑ + ρ_μ)
    λ   =  β (ρₑ² + 4 ρₑρ_μ + ρ_μ²)
    μ² =  2 β ρₑ ρ_μ (ρₑ + ρ_μ)

This script evaluates those expressions for the measured masses and
verifies that the resulting potential indeed has the desired twin
minima (positive second derivative and the correct ratio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


ELECTRON_MASS_MEV = 0.510_998_95
MUON_MASS_MEV = 105.658_375_5
GOLDEN_LOOP_BETA = 3.058_230_856


@dataclass(frozen=True)
class QuarticPotentialParameters:
    beta: float
    mu_sq: float
    lam: float
    kappa: float
    rho_e: float
    rho_mu: float


def derive_potential_parameters(
    beta: float = GOLDEN_LOOP_BETA,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    muon_mass_mev: float = MUON_MASS_MEV,
) -> QuarticPotentialParameters:
    """
    Solve for μ², λ, κ so that V(ρ) has two degenerate minima located at
    ρₑ ∝ mₑ and ρ_μ ∝ m_μ.
    """
    rho_e = 1.0
    rho_mu = muon_mass_mev / electron_mass_mev

    kappa = -2.0 * beta * (rho_e + rho_mu)
    lam = beta * (rho_e**2 + 4.0 * rho_e * rho_mu + rho_mu**2)
    mu_sq = 2.0 * beta * rho_e * rho_mu * (rho_e + rho_mu)

    return QuarticPotentialParameters(
        beta=beta,
        mu_sq=mu_sq,
        lam=lam,
        kappa=kappa,
        rho_e=rho_e,
        rho_mu=rho_mu,
    )


def potential(rho: float, params: QuarticPotentialParameters) -> float:
    """Evaluate V(ρ) for the derived parameters."""
    return (
        -params.mu_sq * rho
        + params.lam * rho**2
        + params.kappa * rho**3
        + params.beta * rho**4
    )


def d_potential(rho: float, params: QuarticPotentialParameters) -> float:
    """First derivative dV/dρ."""
    return (
        -params.mu_sq
        + 2.0 * params.lam * rho
        + 3.0 * params.kappa * rho**2
        + 4.0 * params.beta * rho**3
    )


def d2_potential(rho: float, params: QuarticPotentialParameters) -> float:
    """Second derivative d²V/dρ²."""
    return (
        2.0 * params.lam
        + 6.0 * params.kappa * rho
        + 12.0 * params.beta * rho**2
    )


def stationary_points(
    params: QuarticPotentialParameters,
    tol: float = 1e-9,
) -> List[float]:
    """
    Find real, positive stationary points by solving dV/dρ = 0.
    """
    coeffs = [
        4.0 * params.beta,
        3.0 * params.kappa,
        2.0 * params.lam,
        -params.mu_sq,
    ]
    roots = np.roots(coeffs)
    real_roots = [
        r.real for r in roots if abs(r.imag) < tol and r.real > 0.0
    ]
    return sorted(real_roots)


def main() -> None:
    params = derive_potential_parameters()
    points = stationary_points(params)

    print("=== Reverse Potential Solver ===")
    print(f"Electron mass (MeV): {ELECTRON_MASS_MEV:.9f}")
    print(f"Muon mass     (MeV): {MUON_MASS_MEV:.9f}")
    print(f"Mass ratio μ/e:      {params.rho_mu:.6f}")
    print()
    print("Derived coefficients (β fixed by Golden Loop):")
    print(f"  β      = {params.beta:.9f}")
    print(f"  κ      = {params.kappa:.6f}")
    print(f"  λ      = {params.lam:.6f}")
    print(f"  μ²     = {params.mu_sq:.6f}")
    print(f"  μ      = {np.sqrt(params.mu_sq):.6f}")
    print()

    if len(points) < 2:
        print("Failed to find two positive minima in the potential.")
        return

    print("Stationary points (sorted, units of normalised mass):")
    for idx, rho in enumerate(points, start=1):
        dv = d2_potential(rho, params)
        kind = "minimum" if dv > 0 else "maximum"
        print(
            f"  ρ_{idx} = {rho:.6f}  |  d²V/dρ² = {dv:.6f} ({kind})"
        )

    ratio = points[-1] / points[0]
    print()
    print(f"Large/small minimum ratio: {ratio:.6f}")
    print(
        "Verification (target 206.768): "
        f"Δ = {abs(ratio - params.rho_mu):.6e}"
    )

    # Demonstrate degeneracy.
    v_small = potential(points[0], params)
    v_large = potential(points[-1], params)
    print(f"Potential at ρ_small : {v_small:.6f}")
    print(f"Potential at ρ_large : {v_large:.6f}")
    print(f"Energy difference    : {abs(v_large - v_small):.6e}")


if __name__ == "__main__":
    main()
