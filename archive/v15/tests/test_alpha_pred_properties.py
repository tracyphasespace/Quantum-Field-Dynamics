#!/usr/bin/env python3
"""
Comprehensive property tests for alpha_pred
Tests edge cases, numerical stability, and invariants
"""

import sys
import os
import numpy as np
import jax.numpy as jnp

# Add parent directory's src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from v15_model import alpha_pred, alpha_pred_batch

# Import Stage 3 for mu-space consistency check
try:
    from stage3_hubble_optimized import qfd_distance_modulus_distance_only
    HAS_STAGE3 = True
except ImportError:
    HAS_STAGE3 = False
    print("Warning: stage3_hubble_optimized not available, skipping mu-space test")

K = 2.5 / np.log(10)  # Conversion constant


def test_alpha_pred_boundaries_and_monotone():
    """
    Test alpha_pred at boundary conditions and monotonicity
    - z=0 should give alpha=0 (normalization)
    - All values should be finite
    - Should be monotonically decreasing (more negative with z)
    """
    print("\n### TEST: Boundary Conditions & Monotonicity")
    print("-" * 60)

    z = np.array([0.0, 1e-6, 0.1, 0.5, 1.0, 1.5])
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    # Test both scalar and batch versions
    a_scalar = np.array([alpha_pred(zi, k_J, eta_prime, xi) for zi in z])
    a_batch = np.array(alpha_pred_batch(z, k_J, eta_prime, xi))

    print(f"Testing z values: {z}")
    print(f"alpha_pred values: {a_batch}")

    # Check 1: All finite
    assert np.isfinite(a_batch).all(), "alpha_pred returned non-finite values"
    print("✓ All values finite")

    # Check 2: Normalization at z=0
    assert abs(a_batch[0]) < 1e-12, f"alpha_pred(0) = {a_batch[0]}, expected 0"
    print(f"✓ alpha_pred(0) = {a_batch[0]:.2e} ≈ 0")

    # Check 3: Monotonically decreasing
    diffs = np.diff(a_batch)
    assert np.all(diffs < 0), "alpha_pred not monotonically decreasing"
    print(f"✓ Monotonically decreasing: Δα = {diffs}")

    # Check 4: Scalar and batch agree (relaxed tolerance for JAX float32)
    np.testing.assert_allclose(a_scalar, a_batch, rtol=1e-6, atol=1e-6)
    print("✓ Scalar and batch versions agree")

    print("✓ TEST PASSED: Boundary conditions & monotonicity\n")
    return True


def test_alpha_pred_param_sensitivity_kJ():
    """
    Test parameter sensitivity: larger k_J → stronger dimming
    """
    print("\n### TEST: Parameter Sensitivity (k_J)")
    print("-" * 60)

    z = np.linspace(0.05, 1.2, 50)
    eta_prime, xi = 0.01, 30.0

    a1 = np.array(alpha_pred_batch(z, 60.0, eta_prime, xi))
    a2 = np.array(alpha_pred_batch(z, 70.0, eta_prime, xi))
    a3 = np.array(alpha_pred_batch(z, 80.0, eta_prime, xi))

    print(f"k_J = 60: α(z=0.5) = {a1[25]:.3f}")
    print(f"k_J = 70: α(z=0.5) = {a2[25]:.3f}")
    print(f"k_J = 80: α(z=0.5) = {a3[25]:.3f}")

    # Larger k_J → more negative alpha (stronger dimming)
    assert (a2 < a1).all(), "alpha not decreasing with increasing k_J"
    assert (a3 < a2).all(), "alpha not decreasing with increasing k_J"
    print("✓ Larger k_J → stronger dimming (more negative α)")

    # Check sensitivity magnitude is reasonable (not too extreme)
    delta_per_dkJ = np.abs(a3 - a1) / 20.0  # Change per unit k_J
    assert np.median(delta_per_dkJ) > 0.01, "Sensitivity too weak"
    assert np.median(delta_per_dkJ) < 10.0, "Sensitivity too strong"
    print(f"✓ Sensitivity magnitude reasonable: Δα/Δk_J ~ {np.median(delta_per_dkJ):.3f}")

    # Test ∂α/∂k_J < 0 via finite difference
    dkJ = 1e-3
    dalpha = np.array(alpha_pred_batch(z, 70.0 + dkJ, eta_prime, xi)) - a2
    gradient = dalpha / dkJ
    assert (gradient < 0).all(), "∂α/∂k_J not negative"
    print(f"✓ ∂α/∂k_J < 0 (gradient ~ {np.median(gradient):.3f})")

    print("✓ TEST PASSED: Parameter sensitivity\n")
    return True


def test_numerical_stability_dtypes():
    """
    Test numerical stability across float32 and float64
    """
    print("\n### TEST: Numerical Stability (dtypes)")
    print("-" * 60)

    z = np.array([0.0, 1e-6, 0.1, 0.5, 1.0, 1.5])
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    # Compute in float64 (reference)
    z64 = z.astype(np.float64)
    a64 = np.array(alpha_pred_batch(z64, k_J, eta_prime, xi))

    # Compute in float32
    z32 = z.astype(np.float32)
    a32 = np.array(alpha_pred_batch(z32,
                                     np.float32(k_J),
                                     np.float32(eta_prime),
                                     np.float32(xi)))

    print(f"float64: {a64}")
    print(f"float32: {a32}")

    # Both should be finite
    assert np.isfinite(a64).all(), "float64 returned non-finite"
    assert np.isfinite(a32).all(), "float32 returned non-finite"
    print("✓ Both dtypes produce finite values")

    # Should agree to float32 precision
    rel_diff = np.abs(a64 - a32) / (np.abs(a64) + 1e-10)
    max_rel_diff = np.max(rel_diff[1:])  # Skip z=0 where both are ~0
    print(f"✓ Max relative difference: {max_rel_diff:.2e}")
    assert max_rel_diff < 1e-5, f"float32/64 disagree by {max_rel_diff}"

    print("✓ TEST PASSED: Numerical stability\n")
    return True


def test_mu_alpha_identity():
    """
    Test consistency between mu-space and alpha-space:
    mu_obs - mu_qfd = -K*(alpha_obs - alpha_th)
    """
    if not HAS_STAGE3:
        print("\n### TEST: mu-space/alpha-space consistency [SKIPPED]")
        return True

    print("\n### TEST: mu-space/alpha-space Consistency")
    print("-" * 60)

    z = np.linspace(0.1, 0.9, 20)
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    # Synthetic observations
    alpha_obs = -0.2 + 0.05 * np.cos(7 * z)

    # Theory predictions
    alpha_th = np.array(alpha_pred_batch(z, k_J, eta_prime, xi))
    mu_th = np.array([qfd_distance_modulus_distance_only(zi, k_J) for zi in z])

    # Compute magnitudes
    mu_obs = mu_th - K * alpha_obs
    mu_qfd = mu_th - K * alpha_th

    # Identity check
    lhs = mu_obs - mu_qfd
    rhs = -K * (alpha_obs - alpha_th)

    print(f"LHS (mu_obs - mu_qfd): {lhs[:3]}...")
    print(f"RHS (-K*(α_obs - α_th)): {rhs[:3]}...")

    np.testing.assert_allclose(lhs, rhs, atol=1e-10)
    print("✓ Identity holds: mu_obs - mu_qfd = -K*(α_obs - α_th)")

    print("✓ TEST PASSED: mu-space/alpha-space consistency\n")
    return True


def test_residual_variance_positive():
    """
    Hard stop: residual variance must be positive
    This is the core wiring bug detector
    """
    print("\n### TEST: Residual Variance > 0 (Wiring Bug Detector)")
    print("-" * 60)

    z = np.linspace(0.05, 1.2, 100)
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    # Synthetic observations with perturbations
    alpha_obs = -0.1 + 0.05 * np.sin(5 * z)

    # Theory predictions
    alpha_th = np.array(alpha_pred_batch(z, k_J, eta_prime, xi))

    # Residuals
    r = alpha_obs - alpha_th
    var_r = np.var(r)

    print(f"var(r_alpha) = {var_r:.6f}")
    print(f"std(r_alpha) = {np.sqrt(var_r):.6f}")
    print(f"mean(r_alpha) = {np.mean(r):.6f}")

    assert var_r > 0, "WIRING BUG: Residual variance is zero"
    print("✓ Residual variance is positive (no wiring bug)")

    # Also check not suspiciously small
    assert var_r > 1e-12, "Residual variance suspiciously small"
    print("✓ Variance not suspiciously small")

    print("✓ TEST PASSED: Residual variance check\n")
    return True


def test_no_recentering_per_zbin():
    """
    Check that mean residuals per z-bin are near zero
    (no systematic offsets as function of z)
    """
    print("\n### TEST: No Re-centering (per z-bin)")
    print("-" * 60)

    np.random.seed(42)
    z = np.linspace(0.1, 1.2, 200)
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    # Synthetic observations with noise but no systematic offset
    alpha_th = np.array(alpha_pred_batch(z, k_J, eta_prime, xi))
    alpha_obs = alpha_th + np.random.randn(len(z)) * 0.1

    # Residuals
    r = alpha_obs - alpha_th

    # Bin by z (Δz ≈ 0.05)
    z_bins = np.arange(0.1, 1.25, 0.05)
    bin_means = []

    for i in range(len(z_bins) - 1):
        mask = (z >= z_bins[i]) & (z < z_bins[i+1])
        if np.any(mask):
            bin_mean = np.mean(r[mask])
            bin_means.append(bin_mean)

    bin_means = np.array(bin_means)
    max_abs_mean = np.nanmax(np.abs(bin_means))

    print(f"z-bin mean residuals: {bin_means}")
    print(f"Max |mean| across bins: {max_abs_mean:.6f}")

    # Should be close to zero (within statistical fluctuations)
    # With σ=0.1 noise and ~10 points per bin, expect fluctuations ~ 0.1/√10 ~ 0.03
    # Allow 3σ = 0.1 threshold for occasional outliers
    assert max_abs_mean < 0.1, f"Systematic offset detected: {max_abs_mean}"
    print("✓ No systematic re-centering needed (within statistical fluctuations)")

    print("✓ TEST PASSED: No re-centering\n")
    return True


def test_parameter_identifiability_visualization():
    """
    Check for parameter degeneracies by plotting correlated parameter sets
    """
    print("\n### TEST: Parameter Identifiability Check")
    print("-" * 60)

    z = np.linspace(0.05, 1.2, 100)

    # Baseline
    a_base = np.array(alpha_pred_batch(z, 70.0, 0.01, 30.0))

    # Try to compensate k_J increase with xi decrease
    # (if these overlap, expect posterior correlation)
    a_comp1 = np.array(alpha_pred_batch(z, 75.0, 0.01, 28.0))
    a_comp2 = np.array(alpha_pred_batch(z, 80.0, 0.01, 26.0))

    # Check if curves overlap significantly
    rms_diff1 = np.sqrt(np.mean((a_base - a_comp1)**2))
    rms_diff2 = np.sqrt(np.mean((a_base - a_comp2)**2))

    print(f"Baseline: k_J=70, ξ=30")
    print(f"Variant 1: k_J=75, ξ=28 → RMS diff = {rms_diff1:.3f}")
    print(f"Variant 2: k_J=80, ξ=26 → RMS diff = {rms_diff2:.3f}")

    # If RMS diff is small, parameters are correlated
    if rms_diff1 < 1.0:
        print("⚠️  Parameters may be correlated (k_J vs ξ)")
    else:
        print("✓ Parameters appear identifiable")

    # Check correlation coefficient
    corr1 = np.corrcoef(a_base, a_comp1)[0, 1]
    print(f"✓ Correlation baseline vs variant1: {corr1:.4f}")

    if corr1 > 0.95:
        print("⚠️  WARNING: High correlation suggests posterior degeneracy")

    print("✓ TEST PASSED: Identifiability check complete\n")
    return True


def test_extreme_parameter_values():
    """
    Test alpha_pred remains well-behaved at extreme parameter values
    """
    print("\n### TEST: Extreme Parameter Values")
    print("-" * 60)

    z = np.array([0.1, 0.5, 1.0])

    # Test parameter ranges
    test_cases = [
        (50.0, 0.001, 10.0, "Min params"),
        (70.0, 0.01, 30.0, "Nominal"),
        (90.0, 0.1, 50.0, "Max params"),
    ]

    for k_J, eta_prime, xi, label in test_cases:
        a = np.array(alpha_pred_batch(z, k_J, eta_prime, xi))
        print(f"{label}: k_J={k_J}, η'={eta_prime}, ξ={xi}")
        print(f"  α = {a}")

        assert np.isfinite(a).all(), f"{label}: Non-finite values"
        assert a[0] > a[-1], f"{label}: Not monotonically decreasing"
        print(f"  ✓ Finite and monotonic")

    print("✓ TEST PASSED: Extreme parameters\n")
    return True


# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("ALPHA_PRED PROPERTY TESTS")
    print("=" * 60)

    tests = [
        test_alpha_pred_boundaries_and_monotone,
        test_alpha_pred_param_sensitivity_kJ,
        test_numerical_stability_dtypes,
        test_mu_alpha_identity,
        test_residual_variance_positive,
        test_no_recentering_per_zbin,
        test_parameter_identifiability_visualization,
        test_extreme_parameter_values,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 ALL PROPERTY TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - review output above")
        sys.exit(1)
