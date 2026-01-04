#!/usr/bin/env python3
"""
Test: Fine structure constant α universality across QFD sectors.

Goal: Check if all sectors predict the same α from β = 3.058.

Sectors:
1. Nuclear: α⁻¹ = π² · exp(β) · (c₂/c₁)
2. Photon: α = e²/(4πε₀ℏc) (measured)
3. Lepton: (TBD - from vortex coupling?)
4. Cosmology: (TBD - from CMB physics?)

Status: Nuclear vs Photon comparison ready.
"""

import numpy as np

# Physical constants
e = 1.602176634e-19  # C
hbar = 1.054571817e-34  # J⋅s
c = 299792458  # m/s
epsilon_0 = 8.854187817e-12  # F/m
alpha_measured = 1 / 137.035999084  # CODATA 2018

# QFD parameters
beta = 3.058  # Vacuum stiffness
c2_over_c1 = 6.42  # Nuclear coupling ratio (empirical)


def test_alpha_universality():
    """
    Test if nuclear and photon sectors give same α.
    """

    print("=" * 80)
    print("TEST: FINE STRUCTURE CONSTANT UNIVERSALITY")
    print("=" * 80)

    # Nuclear sector prediction
    alpha_inv_nuclear = np.pi**2 * np.exp(beta) * c2_over_c1
    alpha_nuclear = 1 / alpha_inv_nuclear

    print(f"\n1. NUCLEAR SECTOR")
    print(f"   Formula: α⁻¹ = π² · exp(β) · (c₂/c₁)")
    print(f"   Parameters:")
    print(f"     β = {beta}")
    print(f"     c₂/c₁ = {c2_over_c1}")
    print(f"   Prediction:")
    print(f"     α⁻¹ = {alpha_inv_nuclear:.6f}")
    print(f"     α = {alpha_nuclear:.10f}")

    # Photon sector (measured)
    alpha_photon = alpha_measured

    print(f"\n2. PHOTON SECTOR")
    print(f"   Formula: α = e²/(4πε₀ℏc)")
    print(f"   Measured:")
    print(f"     α = {alpha_photon:.10f}")

    # Comparison
    diff_abs = alpha_nuclear - alpha_photon
    diff_rel = diff_abs / alpha_photon * 100

    print(f"\n3. COMPARISON")
    print(f"   Nuclear:  α = {alpha_nuclear:.10f}")
    print(f"   Photon:   α = {alpha_photon:.10f}")
    print(f"   Difference: {diff_abs:.10f} ({diff_rel:+.4f}%)")

    # Test result
    threshold = 5.0  # 5% tolerance
    passed = abs(diff_rel) < threshold

    print(f"\n4. TEST RESULT")
    print(f"   Tolerance: {threshold}%")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        print(f"\n   ⚠ SECTORS DISAGREE! QFD CONSISTENCY VIOLATED!")
        print(f"   Possible causes:")
        print(f"     1. c₂/c₁ = {c2_over_c1} is wrong (empirically fitted)")
        print(f"     2. β = {beta} is wrong")
        print(f"     3. Nuclear formula is incorrect")
        print(f"     4. Photon and nuclear sectors use different physics")
    else:
        print(f"\n   ✅ Sectors agree within {threshold}%!")
        print(f"   This supports β = {beta} as universal parameter.")
        print(f"   But: c₂/c₁ = {c2_over_c1} is still empirical.")
        print(f"   Next: Derive c₂/c₁ from first principles!")

    return passed, diff_rel


def reverse_engineer_c2_c1():
    """
    What c₂/c₁ would make nuclear α exactly match photon α?
    """

    print("\n" + "=" * 80)
    print("REVERSE ENGINEERING: What c₂/c₁ matches α exactly?")
    print("=" * 80)

    # Required: α⁻¹ = π² · exp(β) · (c₂/c₁)
    # Solving for c₂/c₁:
    alpha_inv_target = 1 / alpha_measured
    c2_c1_required = alpha_inv_target / (np.pi**2 * np.exp(beta))

    print(f"\nTarget: α⁻¹ = {alpha_inv_target:.10f}")
    print(f"Given:  β = {beta}")
    print(f"\nSolving: (c₂/c₁) = α⁻¹ / (π² · exp(β))")
    print(f"         (c₂/c₁) = {alpha_inv_target:.6f} / {np.pi**2 * np.exp(beta):.6f}")
    print(f"         (c₂/c₁) = {c2_c1_required:.10f}")

    print(f"\nComparison:")
    print(f"  Required: c₂/c₁ = {c2_c1_required:.10f}")
    print(f"  Current:  c₂/c₁ = {c2_over_c1:.10f}")
    print(f"  Ratio: {c2_c1_required / c2_over_c1:.6f}")

    diff_pct = (c2_c1_required - c2_over_c1) / c2_over_c1 * 100
    print(f"  Difference: {diff_pct:+.4f}%")

    print(f"\n⚠ CHALLENGE:")
    print(f"  Can we derive c₂/c₁ = {c2_c1_required:.6f} from Cl(3,3) geometry?")
    print(f"  Or is c₂/c₁ = {c2_over_c1} from nuclear fits the 'true' value?")
    print(f"  If latter, what explains {diff_pct:.2f}% discrepancy?")

    return c2_c1_required


def sensitivity_analysis():
    """
    How sensitive is α to variations in β and c₂/c₁?
    """

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Partial derivatives
    # α⁻¹ = π² · exp(β) · (c₂/c₁)
    # ∂(α⁻¹)/∂β = π² · exp(β) · (c₂/c₁) = α⁻¹
    # ∂(α⁻¹)/∂(c₂/c₁) = π² · exp(β)

    alpha_inv = 1 / alpha_measured

    d_alpha_inv_d_beta = alpha_inv  # = α⁻¹
    d_alpha_inv_d_c2c1 = np.pi**2 * np.exp(beta)

    # Convert to relative sensitivities
    # (∂α/α) / (∂β/β) = -β (because α = 1/α⁻¹)
    # (∂α/α) / (∂(c₂/c₁)/(c₂/c₁)) = -1

    print(f"\nRelative sensitivities:")
    print(f"  (Δα/α) ≈ -β · (Δβ/β)")
    print(f"  (Δα/α) ≈ -(Δ(c₂/c₁)/(c₂/c₁))")

    print(f"\nWith β = {beta}:")
    print(f"  1% error in β → {beta:.2f}% error in α")
    print(f"  1% error in c₂/c₁ → 1% error in α")

    # Example: What β gives α exactly?
    alpha_inv_target = 1 / alpha_measured
    c2_c1_current = c2_over_c1
    beta_required = np.log(alpha_inv_target / (np.pi**2 * c2_c1_current))

    print(f"\nAlternative: Fix c₂/c₁ = {c2_c1_current}, solve for β:")
    print(f"  Required: β = {beta_required:.10f}")
    print(f"  Current:  β = {beta:.10f}")
    print(f"  Difference: {(beta_required - beta):.10f}")

    diff_pct = (beta_required - beta) / beta * 100
    print(f"  Relative: {diff_pct:+.4f}%")


if __name__ == "__main__":
    # Run tests
    passed, error = test_alpha_universality()
    c2_c1_exact = reverse_engineer_c2_c1()
    sensitivity_analysis()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Test status: {'PASS' if passed else 'FAIL'}")
    print(f"   Nuclear vs photon α: {error:+.4f}% difference")
    print(f"\n⚠ CRITICAL ISSUE:")
    print(f"   c₂/c₁ = {c2_over_c1} is empirically fitted to nuclear data.")
    print(f"   Without first-principles derivation, this is CIRCULAR!")
    print(f"\n🎯 NEXT STEP:")
    print(f"   Derive c₂/c₁ from Cl(3,3) geometric algebra.")
    print(f"   If derivation gives c₂/c₁ ≈ {c2_c1_exact:.6f}, α universality proven!")
    print(f"   If derivation fails, QFD photon-nuclear unification fails.")
    print("=" * 80)
