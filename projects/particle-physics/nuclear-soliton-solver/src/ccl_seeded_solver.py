#!/usr/bin/env python3
"""
CCL-Seeded Soliton Solver

Uses Core Compression Law Q(A) = c1·A^(2/3) + c2·A to:
1. Predict expected charge distribution
2. Initialize density profiles realistically
3. Seed optimization with physically-guided parameters

This should dramatically improve convergence vs. random initialization.
"""
import sys
import os
import numpy as np
import torch
from typing import Dict, Tuple

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import C1_SURFACE, C2_VOLUME

# CCL parameters (from shared_constants, originally CoreCompressionLaw.lean)
CCL_C1 = C1_SURFACE  # Surface term
CCL_C2 = C2_VOLUME   # Volume term

def backbone_charge(A: int) -> float:
    """
    Core Compression Law: Q(A) = c1·A^(2/3) + c2·A

    Predicts the "optimal" charge for mass number A.
    This is the charge that minimizes elastic stress.

    Args:
        A: Mass number

    Returns:
        Q: Predicted charge (may be non-integer)

    Reference: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean:593-595
    """
    return CCL_C1 * (A ** (2.0/3.0)) + CCL_C2 * A


def charge_stress(Z: int, A: int) -> float:
    """
    Elastic stress: |Z - Q_backbone(A)|

    Measures how far actual charge Z deviates from CCL prediction.

    - Stress < 1: Stable isotope
    - Stress > 3: Unstable, will decay

    Reference: projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean:612-614
    """
    Q = backbone_charge(A)
    return abs(Z - Q)


def predict_optimal_radius(A: int) -> float:
    """
    Estimate nuclear radius from mass number.

    Uses empirical formula: r_0 ≈ 1.2 fm × A^(1/3)

    This provides initial scale for density profiles.
    """
    return 1.2 * (A ** (1.0/3.0))  # fm


def seed_density_parameters(Z: int, A: int) -> Dict[str, float]:
    """
    Use CCL to seed nuclear potential parameters.

    Strategy:
    1. Compute Q_backbone(A) - expected charge
    2. Compute stress = |Z - Q| - deviation from backbone
    3. Adjust parameters based on stress:
       - Low stress → use default parameters
       - High stress → adjust c_v2 terms to accommodate deviation

    Args:
        Z: Proton number
        A: Mass number

    Returns:
        Dictionary of seeded parameter values
    """
    Q_ccl = backbone_charge(A)
    stress = abs(Z - Q_ccl)
    N = A - Z

    # Base parameters (from initial guess)
    params = {
        'c_v2_base': 2.201711,
        'c_v2_iso': 0.027035,
        'c_v2_mass': -0.000205,
        'c_v4_base': 5.282364,
        'c_v4_size': -0.085018,
        'alpha_e_scale': 1.007419,
        'beta_e_scale': 0.504312,
        'c_sym': 25.0,
        'kappa_rho': 0.029816
    }

    # Adjust based on CCL predictions

    # 1. Isospin adjustment: CCL tells us about N-Z balance
    iso_fraction = abs(N - Z) / A
    params['c_v2_iso'] = 0.027 * (1.0 + iso_fraction)  # Stronger for asymmetric nuclei

    # 2. Mass-dependent compression: Use CCL c2 to guide
    # CCL: Q(A) = c1·A^(2/3) + c2·A
    # c2 encodes how charge scales linearly with A
    # This should correlate with bulk compression strength
    params['c_v2_mass'] = -0.0002 * (CCL_C2 / 0.32)  # Scale by CCL c2

    # 3. Surface term: Use CCL c1 to guide
    # CCL: c1·A^(2/3) is surface contribution
    # Should correlate with surface energy coefficient
    surface_scale = CCL_C1 / 0.5  # Normalize by typical c1
    params['c_v4_size'] = -0.085 * surface_scale

    # 4. Symmetry energy: Adjust for high-stress isotopes
    # High stress → unstable → likely due to poor N/Z balance
    # → Need stronger symmetry energy penalty
    if stress > 2.0:
        params['c_sym'] = 28.0  # Increase from 25.0
    elif stress < 0.5:
        params['c_sym'] = 23.0  # Decrease for stable isotopes

    # 5. Surface tension: Guide by predicted radius
    r0 = predict_optimal_radius(A)
    # Larger nuclei need more surface tension to hold together
    params['kappa_rho'] = 0.03 * (r0 / 6.0)  # Scale by radius/6fm

    return params


def seed_initial_density(A: int, Z: int, grid_size: int = 32,
                         box_size: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create physically-motivated initial density guess using CCL.

    Strategy:
    1. Use r_0 = 1.2·A^(1/3) for nuclear radius
    2. Create parabolic density profile ρ(r) = ρ_0·(1 - (r/R)²)
    3. Normalize to ∫ ρ dV = A (mass conservation)

    Args:
        A: Mass number
        Z: Proton number
        grid_size: Number of grid points per dimension
        box_size: Box size in fm

    Returns:
        (rho_N, rho_Z): Initial nucleon and charge densities (3D arrays)
    """
    r0 = predict_optimal_radius(A)
    N = A - Z

    # Create 3D grid
    dx = box_size / grid_size
    x = np.linspace(-box_size/2, box_size/2, grid_size)
    X, Y, Z_grid = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z_grid**2)

    # Parabolic density profile (smooth, physically realistic)
    # ρ(r) = ρ_0 · max(1 - (r/R_cutoff)², 0)
    R_cutoff = r0 * 1.5  # Extend slightly beyond r0
    rho_profile = np.maximum(1.0 - (R / R_cutoff)**2, 0.0)

    # Normalize nucleon density to A
    total_nucleons = rho_profile.sum() * (dx**3)
    rho_N = rho_profile * (A / total_nucleons)

    # Charge density (protons only)
    # Assume protons have similar distribution but scaled by Z/A
    rho_Z = rho_N * (Z / A)

    return rho_N, rho_Z


def ccl_guided_optimization_bounds(Z: int, A: int) -> Dict[str, Tuple[float, float]]:
    """
    Tighten parameter bounds using CCL guidance.

    Instead of wide bounds that allow unphysical regions,
    use CCL to constrain search space around physically realistic values.

    Args:
        Z: Proton number
        A: Mass number

    Returns:
        Dictionary of parameter bounds (name → (lower, upper))
    """
    stress = charge_stress(Z, A)

    # Base bounds (conservative)
    bounds = {
        'c_v2_base': (2.0, 2.5),
        'c_v2_iso': (0.020, 0.035),
        'c_v2_mass': (-0.0005, 0.0),
        'c_v4_base': (4.5, 6.0),
        'c_v4_size': (-0.10, -0.07),
        'alpha_e_scale': (0.95, 1.05),
        'beta_e_scale': (0.45, 0.55),
        'c_sym': (22.0, 28.0),
        'kappa_rho': (0.02, 0.04)
    }

    # Tighten bounds for low-stress (stable) isotopes
    # These should be close to optimal → narrow search
    if stress < 1.0:
        bounds['c_v2_base'] = (2.15, 2.35)  # ±10% instead of ±25%
        bounds['c_sym'] = (23.0, 27.0)
        bounds['kappa_rho'] = (0.025, 0.035)

    # For high-stress isotopes, allow wider exploration
    # These are far from backbone → may need unusual parameters
    elif stress > 3.0:
        bounds['c_sym'] = (20.0, 30.0)  # Allow more variation

    return bounds


if __name__ == "__main__":
    # Test CCL seeding for Pb-208
    Z, A = 82, 208

    print("=" * 70)
    print("CCL-SEEDED PARAMETER INITIALIZATION")
    print("=" * 70)
    print()

    print(f"Isotope: {Z}-{A} (Lead-208)")
    print()

    # CCL predictions
    Q_ccl = backbone_charge(A)
    stress = charge_stress(Z, A)
    r0 = predict_optimal_radius(A)

    print(f"CCL Predictions:")
    print(f"  Q_backbone(A={A}) = {Q_ccl:.2f}")
    print(f"  Z_experimental    = {Z}")
    print(f"  Stress |Z - Q|    = {stress:.2f}")
    print(f"  Status            = {'Stable' if stress < 1.5 else 'Unstable'}")
    print(f"  Predicted r_0     = {r0:.2f} fm")
    print()

    # Seeded parameters
    params = seed_density_parameters(Z, A)
    print(f"Seeded Parameters:")
    for name, value in params.items():
        print(f"  {name:20s} = {value:.6f}")
    print()

    # Seeded bounds
    bounds = ccl_guided_optimization_bounds(Z, A)
    print(f"Tightened Bounds:")
    for name, (lower, upper) in bounds.items():
        print(f"  {name:20s}: [{lower:.4f}, {upper:.4f}]")
    print()

    print("✓ CCL seeding complete - ready for optimization")
