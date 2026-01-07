#!/usr/bin/env python3
"""
Derive β from α (Fine Structure Constant)

Two approaches:
1. Golden Loop: e^β/β = (α⁻¹ × c₁) / π²
2. LEVEL4 formula: 1/α = π² · exp(β) · (c₂/c₁)

Both should give consistent β values.
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# EMPIRICAL CONSTANTS
# =============================================================================

# Fine structure constant (CODATA 2018)
ALPHA_INV = 137.035999084
ALPHA = 1.0 / ALPHA_INV

# NuBase 2020 coefficients (from GoldenLoop.lean)
C1_SURFACE = 0.496297  # Surface coefficient
C2_VOLUME = 0.32704    # Volume coefficient

# Alternative coefficients (from FineStructure.lean)
C1_FINE = 0.529251
C2_FINE = 0.316743

PI_SQ = np.pi**2

# =============================================================================
# METHOD 1: GOLDEN LOOP (Transcendental Equation)
# =============================================================================

def golden_loop_K(alpha_inv, c1):
    """K = (α⁻¹ × c₁) / π²"""
    return (alpha_inv * c1) / PI_SQ

def transcendental_f(beta):
    """f(β) = e^β / β"""
    return np.exp(beta) / beta

def solve_golden_loop(K):
    """Solve e^β/β = K for β > 1"""
    def objective(beta):
        return transcendental_f(beta) - K
    return brentq(objective, 1.5, 10.0)

# =============================================================================
# METHOD 2: LEVEL4 FORMULA (Direct Exponential)
# =============================================================================

def level4_beta(alpha_inv, c2_c1_ratio):
    """
    From: 1/α = π² · exp(β) · (c₂/c₁)
    Solve: β = ln(α⁻¹) - ln(π²) - ln(c₂/c₁)
    """
    return np.log(alpha_inv) - np.log(PI_SQ) - np.log(c2_c1_ratio)

def level4_alpha_inv(beta, c2_c1_ratio):
    """Predict α⁻¹ from β and c₂/c₁ ratio"""
    return PI_SQ * np.exp(beta) * c2_c1_ratio

# =============================================================================
# ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("DERIVING β FROM α (Fine Structure Constant)")
    print("=" * 70)
    print()

    print("EMPIRICAL INPUTS:")
    print(f"  α⁻¹ = {ALPHA_INV} (CODATA 2018)")
    print(f"  c₁  = {C1_SURFACE} (NuBase 2020, surface)")
    print(f"  c₂  = {C2_VOLUME} (NuBase 2020, volume)")
    print(f"  π²  = {PI_SQ:.10f}")
    print()

    # Method 1: Golden Loop
    print("=" * 70)
    print("METHOD 1: GOLDEN LOOP (Transcendental)")
    print("=" * 70)
    print()
    print("Equation: e^β/β = K where K = (α⁻¹ × c₁) / π²")
    print()

    K = golden_loop_K(ALPHA_INV, C1_SURFACE)
    beta_golden = solve_golden_loop(K)

    print(f"  K = ({ALPHA_INV} × {C1_SURFACE}) / {PI_SQ:.4f}")
    print(f"  K = {K:.6f}")
    print()
    print(f"  Solving e^β/β = {K:.6f}...")
    print(f"  β = {beta_golden:.6f}")
    print()
    print(f"  Verification: e^{beta_golden:.4f}/{beta_golden:.4f} = {transcendental_f(beta_golden):.6f}")
    print()

    # Method 2: LEVEL4 Formula
    print("=" * 70)
    print("METHOD 2: LEVEL4 FORMULA (Direct)")
    print("=" * 70)
    print()
    print("Equation: 1/α = π² · exp(β) · (c₂/c₁)")
    print("Inverted: β = ln(α⁻¹) - ln(π²) - ln(c₂/c₁)")
    print()

    c2_c1_nubase = C2_VOLUME / C1_SURFACE
    c2_c1_fine = C2_FINE / C1_FINE

    print(f"  NuBase ratio: c₂/c₁ = {C2_VOLUME}/{C1_SURFACE} = {c2_c1_nubase:.6f}")
    print(f"  FineStruct ratio: c₂/c₁ = {C2_FINE}/{C1_FINE} = {c2_c1_fine:.6f}")
    print()

    beta_level4_nubase = level4_beta(ALPHA_INV, c2_c1_nubase)
    beta_level4_fine = level4_beta(ALPHA_INV, c2_c1_fine)

    print(f"  Using NuBase ratio:")
    print(f"    β = ln({ALPHA_INV}) - ln({PI_SQ:.4f}) - ln({c2_c1_nubase:.4f})")
    print(f"    β = {np.log(ALPHA_INV):.4f} - {np.log(PI_SQ):.4f} - ({np.log(c2_c1_nubase):.4f})")
    print(f"    β = {beta_level4_nubase:.6f}")
    print()

    print(f"  Using FineStructure ratio:")
    print(f"    β = {beta_level4_fine:.6f}")
    print()

    # Verification
    alpha_inv_check = level4_alpha_inv(beta_level4_nubase, c2_c1_nubase)
    print(f"  Verification: π² × exp({beta_level4_nubase:.4f}) × {c2_c1_nubase:.4f} = {alpha_inv_check:.4f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY: β DERIVED FROM α")
    print("=" * 70)
    print()
    print(f"{'Method':<40} {'β value':>12} {'c₂ = 1/β':>12}")
    print("-" * 70)
    print(f"{'Golden Loop (e^β/β = K)':<40} {beta_golden:>12.6f} {1/beta_golden:>12.6f}")
    print(f"{'LEVEL4 (NuBase c₂/c₁ = 0.659)':<40} {beta_level4_nubase:>12.6f} {1/beta_level4_nubase:>12.6f}")
    print(f"{'LEVEL4 (FineStruct c₂/c₁ = 0.598)':<40} {beta_level4_fine:>12.6f} {1/beta_level4_fine:>12.6f}")
    print(f"{'Current GoldenLoop.lean':<40} {'3.058231':>12} {'0.326986':>12}")
    print()

    print("=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print()
    print(f"When β is DERIVED from α using the Golden Loop equation:")
    print(f"  β = {beta_golden:.4f}")
    print()
    print(f"This predicts c₂ = 1/β = {1/beta_golden:.5f}")
    print(f"vs empirical c₂ = {C2_VOLUME:.5f}")
    print(f"Error: {abs(1/beta_golden - C2_VOLUME)/C2_VOLUME * 100:.2f}%")
    print()
    print("The 0.5% tension is between:")
    print(f"  β_from_α = {beta_golden:.4f} (true root of transcendental)")
    print(f"  β_from_c₂ = {1/C2_VOLUME:.4f} (from nuclear volume coefficient)")
    print()

    # What if there's a correction factor?
    print("=" * 70)
    print("CORRECTION FACTOR ANALYSIS")
    print("=" * 70)
    print()

    # From LEVEL4: k_EM ≈ 1.27 is mentioned
    k_EM = 4.3813 / 3.45  # empirical scaling factor
    c2_c1_eff = c2_c1_fine / k_EM
    beta_corrected = level4_beta(ALPHA_INV, c2_c1_eff)

    print(f"If dimensional projection correction k_EM = {k_EM:.3f} is applied:")
    print(f"  (c₂/c₁)_eff = {c2_c1_fine:.4f} / {k_EM:.3f} = {c2_c1_eff:.4f}")
    print(f"  β_corrected = {beta_corrected:.4f}")
    print()

    # What k would make β = 3.058?
    beta_target = 3.058230856
    # exp(β) = α⁻¹ / (π² × (c₂/c₁)_eff)
    # (c₂/c₁)_eff = α⁻¹ / (π² × exp(β))
    c2_c1_needed = ALPHA_INV / (PI_SQ * np.exp(beta_target))
    k_needed = c2_c1_fine / c2_c1_needed

    print(f"To get β = 3.058 from LEVEL4 formula:")
    print(f"  Need (c₂/c₁)_eff = {c2_c1_needed:.6f}")
    print(f"  Requires k = {c2_c1_fine:.4f} / {c2_c1_needed:.4f} = {k_needed:.4f}")
    print()

    print("=" * 70)
    print("ANSWER TO USER'S QUESTION")
    print("=" * 70)
    print()
    print("When we derive β from α (Fine Structure Constant):")
    print()
    print(f"  EXPECTED β = {beta_golden:.4f}")
    print()
    print("This is the TRUE ROOT of the transcendental equation")
    print("e^β/β = (α⁻¹ × c₁) / π²")
    print()
    print(f"The current β = 3.058 in GoldenLoop.lean differs by ~0.5%")
    print("because it was optimized for c₂ prediction rather than")
    print("strict satisfaction of the transcendental equation.")
    print()

if __name__ == "__main__":
    main()
