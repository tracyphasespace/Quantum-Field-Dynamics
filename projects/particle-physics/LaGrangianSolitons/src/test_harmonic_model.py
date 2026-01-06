#!/usr/bin/env python3
"""
Unit tests for harmonic_model.py

Tests:
  - Basic model functions (Z_predicted, N_hat, epsilon)
  - Parameter validation
  - Numerical consistency
  - Edge cases
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from harmonic_model import (
    FamilyParams,
    Z_baseline,
    delta_Z,
    Z_predicted,
    N_hat,
    epsilon,
    residual,
    score_nuclide,
    score_best_family,
    classify_by_epsilon,
    dc3_comparison,
    validate_params,
)


def test_family_params_basic():
    """Test FamilyParams creation and validation."""
    print("Testing FamilyParams...")

    # Valid params
    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )
    assert params.name == "A"
    assert params.dc3 == -0.865

    # to_dict / from_dict roundtrip
    params_dict = params.to_dict()
    params_reloaded = FamilyParams.from_dict(params_dict)
    assert params_reloaded.name == params.name
    assert params_reloaded.dc3 == params.dc3

    print("  ✓ FamilyParams creation and serialization")


def test_Z_baseline():
    """Test baseline Z calculation."""
    print("Testing Z_baseline...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    # Single value
    A = 100
    Z_0 = Z_baseline(A, params)
    expected = 1.5 * 100**(2/3) + 0.4 * 100 + 10.0
    assert np.isclose(Z_0, expected), f"Expected {expected}, got {Z_0}"

    # Array
    A_arr = np.array([50, 100, 150])
    Z_0_arr = Z_baseline(A_arr, params)
    assert len(Z_0_arr) == 3
    assert Z_0_arr[1] == Z_0  # Second element should match single-value test

    print("  ✓ Z_baseline calculation")


def test_delta_Z():
    """Test mode spacing calculation."""
    print("Testing delta_Z...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    A = 100
    dZ = delta_Z(A, params)
    expected = -0.05 * 100**(2/3) - 0.01 * 100 - 0.865
    assert np.isclose(dZ, expected), f"Expected {expected}, got {dZ}"

    # Should be negative for typical params (Z decreases with mode)
    assert dZ < 0, "Expected negative mode spacing"

    print("  ✓ delta_Z calculation")


def test_Z_predicted():
    """Test predicted Z for given (A, N)."""
    print("Testing Z_predicted...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    A = 100
    N = 0
    Z_pred_N0 = Z_predicted(A, N, params)
    Z_0 = Z_baseline(A, params)
    assert np.isclose(Z_pred_N0, Z_0), "Z_pred(N=0) should equal Z_baseline"

    # Check mode spacing
    N = 1
    Z_pred_N1 = Z_predicted(A, N, params)
    dZ = delta_Z(A, params)
    expected_Z_N1 = Z_0 + 1 * dZ
    assert np.isclose(Z_pred_N1, expected_Z_N1), f"Expected {expected_Z_N1}, got {Z_pred_N1}"

    # Check N=2
    N = 2
    Z_pred_N2 = Z_predicted(A, N, params)
    expected_Z_N2 = Z_0 + 2 * dZ
    assert np.isclose(Z_pred_N2, expected_Z_N2), f"Expected {expected_Z_N2}, got {Z_pred_N2}"

    print("  ✓ Z_predicted calculation")


def test_N_hat():
    """Test continuous mode estimation."""
    print("Testing N_hat...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    # If Z is exactly on baseline, N_hat should be 0
    A = 100
    Z = Z_baseline(A, params)
    Nhat = N_hat(A, Z, params)
    assert np.isclose(Nhat, 0.0), f"Expected N_hat=0 at baseline, got {Nhat}"

    # If Z is at N=1 line, N_hat should be 1
    Z_N1 = Z_predicted(A, 1, params)
    Nhat = N_hat(A, Z_N1, params)
    assert np.isclose(Nhat, 1.0), f"Expected N_hat=1, got {Nhat}"

    # Midpoint between N=0 and N=1 should give N_hat=0.5
    Z_mid = (Z_baseline(A, params) + Z_N1) / 2
    Nhat_mid = N_hat(A, Z_mid, params)
    assert np.isclose(Nhat_mid, 0.5, atol=0.01), f"Expected N_hat≈0.5, got {Nhat_mid}"

    print("  ✓ N_hat calculation")


def test_epsilon():
    """Test dissonance metric."""
    print("Testing epsilon...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    A = 100

    # Exactly on harmonic (N=0): ε = 0
    Z = Z_baseline(A, params)
    eps = epsilon(A, Z, params)
    assert np.isclose(eps, 0.0, atol=1e-6), f"Expected ε=0 at harmonic, got {eps}"

    # Exactly on harmonic (N=1): ε = 0
    Z_N1 = Z_predicted(A, 1, params)
    eps = epsilon(A, Z_N1, params)
    assert np.isclose(eps, 0.0, atol=1e-6), f"Expected ε=0 at harmonic, got {eps}"

    # Midpoint: ε = 0.5 (maximum dissonance)
    Z_mid = (Z_baseline(A, params) + Z_N1) / 2
    eps_mid = epsilon(A, Z_mid, params)
    assert np.isclose(eps_mid, 0.5, atol=0.01), f"Expected ε=0.5 at midpoint, got {eps_mid}"

    # ε should always be in [0, 0.5]
    for Z_test in np.linspace(30, 60, 50):
        eps_test = epsilon(A, Z_test, params)
        assert 0.0 <= eps_test <= 0.5, f"ε out of bounds: {eps_test}"

    print("  ✓ epsilon calculation")


def test_residual():
    """Test Z residual calculation."""
    print("Testing residual...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    A = 100

    # Exactly on harmonic: residual = 0
    Z = Z_baseline(A, params)
    resid = residual(A, Z, params)
    assert np.isclose(resid, 0.0, atol=1e-6), f"Expected residual=0, got {resid}"

    # At N=1 harmonic: residual = 0
    Z_N1 = Z_predicted(A, 1, params)
    resid = residual(A, Z_N1, params)
    assert np.isclose(resid, 0.0, atol=1e-6), f"Expected residual=0, got {resid}"

    # Midpoint: residual should be ±ΔZ/2 (depending on rounding)
    dZ = delta_Z(A, params)
    Z_mid = Z_baseline(A, params) + 0.5 * dZ
    resid_mid = residual(A, Z_mid, params)
    # round(0.5) could be 0 or 1 depending on rounding mode
    # Expected residual magnitude ≈ |ΔZ|/2
    assert np.isclose(abs(resid_mid), abs(dZ) / 2, atol=0.1), \
        f"Expected |residual|≈{abs(dZ)/2}, got {abs(resid_mid)}"

    print("  ✓ residual calculation")


def test_score_nuclide():
    """Test nuclide scoring."""
    print("Testing score_nuclide...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    A = 100
    Z = Z_baseline(A, params)

    score = score_nuclide(A, Z, params)

    assert score['family'] == "A"
    assert np.isclose(score['N_hat'], 0.0, atol=1e-6)
    assert score['N_best'] == 0
    assert np.isclose(score['epsilon'], 0.0, atol=1e-6)
    assert np.isclose(score['residual'], 0.0, atol=1e-6)

    print("  ✓ score_nuclide")


def test_score_best_family():
    """Test best-family scoring."""
    print("Testing score_best_family...")

    # Create three families with different baseline Z
    families = {
        'A': FamilyParams(name="A", c1_0=1.5, c2_0=0.4, c3_0=10.0,
                         dc1=-0.05, dc2=-0.01, dc3=-0.865),
        'B': FamilyParams(name="B", c1_0=1.6, c2_0=0.38, c3_0=8.0,
                         dc1=-0.048, dc2=-0.012, dc3=-0.860),
        'C': FamilyParams(name="C", c1_0=1.4, c2_0=0.42, c3_0=12.0,
                         dc1=-0.052, dc2=-0.009, dc3=-0.870),
    }

    A = 100
    # Choose Z exactly on family B baseline
    Z = Z_baseline(A, families['B'])

    score = score_best_family(A, Z, families)

    # Best family should be B (ε=0)
    assert score['best_family'] == 'B', f"Expected best family B, got {score['best_family']}"
    assert np.isclose(score['epsilon_best'], 0.0, atol=1e-6), \
        f"Expected ε=0 for family B, got {score['epsilon_best']}"

    # Should have epsilon for all families
    assert 'epsilon_A' in score
    assert 'epsilon_B' in score
    assert 'epsilon_C' in score

    print("  ✓ score_best_family")


def test_classify_by_epsilon():
    """Test epsilon classification."""
    print("Testing classify_by_epsilon...")

    assert classify_by_epsilon(0.01) == 'harmonic'
    assert classify_by_epsilon(0.04999) == 'harmonic'
    assert classify_by_epsilon(0.05) == 'near_harmonic'
    assert classify_by_epsilon(0.10) == 'near_harmonic'
    assert classify_by_epsilon(0.1499) == 'near_harmonic'
    assert classify_by_epsilon(0.15) == 'dissonant'
    assert classify_by_epsilon(0.30) == 'dissonant'
    assert classify_by_epsilon(0.50) == 'dissonant'

    print("  ✓ classify_by_epsilon")


def test_dc3_comparison():
    """Test dc3 universality check."""
    print("Testing dc3_comparison...")

    families = {
        'A': FamilyParams(name="A", c1_0=1.5, c2_0=0.4, c3_0=10.0,
                         dc1=-0.05, dc2=-0.01, dc3=-0.865),
        'B': FamilyParams(name="B", c1_0=1.6, c2_0=0.38, c3_0=8.0,
                         dc1=-0.048, dc2=-0.012, dc3=-0.860),
        'C': FamilyParams(name="C", c1_0=1.4, c2_0=0.42, c3_0=12.0,
                         dc1=-0.052, dc2=-0.009, dc3=-0.870),
    }

    comparison = dc3_comparison(families)

    assert 'dc3_values' in comparison
    assert 'dc3_mean' in comparison
    assert 'dc3_std' in comparison
    assert 'dc3_range' in comparison
    assert 'dc3_relative_std' in comparison

    # Check values
    assert comparison['dc3_values']['A'] == -0.865
    assert comparison['dc3_values']['B'] == -0.860
    assert comparison['dc3_values']['C'] == -0.870

    # Mean should be around -0.865
    assert np.isclose(comparison['dc3_mean'], -0.865, atol=0.01)

    # Std should be small (universality check)
    assert comparison['dc3_std'] < 0.01, \
        f"Expected small dc3 std deviation, got {comparison['dc3_std']}"

    # Relative std should be < 2% for "universal" dc3
    assert comparison['dc3_relative_std'] < 0.02, \
        f"Expected relative std < 2%, got {comparison['dc3_relative_std']:.1%}"

    print("  ✓ dc3_comparison")


def test_validate_params():
    """Test parameter validation."""
    print("Testing validate_params...")

    # Realistic params (checked over reasonable A range)
    params_good = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=-5.0,  # Negative intercept for physical Z_0 < A
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    validation = validate_params(params_good, A_range=(10, 300))  # Skip A=1-9
    # May have warnings for small A, but dc3 should be fine
    assert not any('dc3' in w and 'positive' in w.lower() for w in validation['warnings']), \
        f"Should not warn about dc3 sign for negative dc3"

    # Params with positive dc3 (should warn)
    params_bad_dc3 = FamilyParams(
        name="BadDc3",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=-5.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=0.5,  # POSITIVE (wrong sign)
    )

    validation_bad = validate_params(params_bad_dc3, A_range=(10, 300))
    assert validation_bad['valid'] == False
    assert any('dc3' in w for w in validation_bad['warnings'])

    print("  ✓ validate_params")


def test_numerical_consistency():
    """Test that forward and inverse operations are consistent."""
    print("Testing numerical consistency...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    # Test: Z_pred(A, round(N_hat(A, Z))) should approximately equal Z
    A = 100
    for Z_test in np.linspace(30, 60, 20):
        Nhat = N_hat(A, Z_test, params)
        N_rounded = int(np.round(Nhat))
        Z_reconstructed = Z_predicted(A, N_rounded, params)

        # Residual should be small (within one mode spacing)
        error = abs(Z_test - Z_reconstructed)
        dZ = abs(delta_Z(A, params))
        assert error <= dZ / 2 + 0.1, \
            f"Reconstruction error too large: {error} > {dZ/2} for Z={Z_test}"

    print("  ✓ Numerical consistency (forward-inverse)")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    params = FamilyParams(
        name="A",
        c1_0=1.5,
        c2_0=0.4,
        c3_0=10.0,
        dc1=-0.05,
        dc2=-0.01,
        dc3=-0.865,
    )

    # Very small A
    A_small = 1
    Z_small = Z_baseline(A_small, params)
    eps_small = epsilon(A_small, Z_small, params)
    assert np.isfinite(eps_small), "epsilon should be finite for A=1"

    # Very large A
    A_large = 300
    Z_large = Z_baseline(A_large, params)
    eps_large = epsilon(A_large, Z_large, params)
    assert np.isfinite(eps_large), "epsilon should be finite for A=300"
    assert np.isclose(eps_large, 0.0, atol=1e-6), "Should be on harmonic"

    # Arrays with mixed sizes
    A_arr = np.array([50, 100, 150])
    Z_arr = np.array([22, 45, 68])
    eps_arr = epsilon(A_arr, Z_arr, params)
    assert len(eps_arr) == 3, "Should handle array inputs"
    assert all(0 <= e <= 0.5 for e in eps_arr), "All epsilons in valid range"

    print("  ✓ Edge cases")


def run_all_tests():
    """Run all unit tests."""
    print("="*80)
    print("HARMONIC MODEL UNIT TESTS")
    print("="*80)
    print()

    try:
        test_family_params_basic()
        test_Z_baseline()
        test_delta_Z()
        test_Z_predicted()
        test_N_hat()
        test_epsilon()
        test_residual()
        test_score_nuclide()
        test_score_best_family()
        test_classify_by_epsilon()
        test_dc3_comparison()
        test_validate_params()
        test_numerical_consistency()
        test_edge_cases()

        print()
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return True

    except AssertionError as e:
        print()
        print("="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        return False
    except Exception as e:
        print()
        print("="*80)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
