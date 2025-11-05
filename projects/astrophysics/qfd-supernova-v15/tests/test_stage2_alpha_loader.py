#!/usr/bin/env python3
"""
Unit tests for Stage 2 alpha loading and conversion.

Guards against sign/units regressions when loading alpha from Stage 1.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stage2_mcmc_numpyro import _ensure_alpha_natural, K_MAG_PER_LN


def test_loader_converts_mag_to_nat():
    """
    Test that magnitude-space alpha is correctly converted to natural-log.

    When alpha values look like magnitudes (large positive values), they should
    be converted: α_nat = -α_mag / K
    """
    # Simulate Stage 1 output in magnitude space (typical range 15-30)
    a_mag = np.array([15.0, 20.0, 25.0, 30.0])

    # Convert
    a_nat = _ensure_alpha_natural(a_mag)

    # Expected: α_nat = -α_mag / K_MAG_PER_LN
    expected = -a_mag / K_MAG_PER_LN

    np.testing.assert_allclose(a_nat, expected, rtol=1e-10)

    # Check sign: should be negative
    assert np.all(a_nat < 0), f"Converted alpha should be negative: {a_nat}"

    # Check magnitude: should be in range [-30, -10] roughly
    assert np.all(a_nat >= -30), f"Converted alpha too negative: {a_nat}"
    assert np.all(a_nat <= 0), f"Converted alpha should be negative: {a_nat}"

    print("✓ test_loader_converts_mag_to_nat PASSED")
    print(f"  Input (mag):  {a_mag}")
    print(f"  Output (nat): {a_nat}")


def test_loader_preserves_natural_alpha():
    """
    Test that natural-log alpha values are preserved (not converted).

    When alpha values are already small (median |α| < 5), they should
    be passed through unchanged.
    """
    # Simulate Stage 1 output already in natural-log space
    a_nat_original = np.array([-10.0, -5.0, -2.0, -0.5])

    # Should be unchanged
    a_nat_out = _ensure_alpha_natural(a_nat_original)

    np.testing.assert_allclose(a_nat_out, a_nat_original, rtol=1e-10)

    print("✓ test_loader_preserves_natural_alpha PASSED")
    print(f"  Input:  {a_nat_original}")
    print(f"  Output: {a_nat_out}")


def test_loader_heuristic_threshold():
    """
    Test that the heuristic threshold (median |α| > 5) works correctly.
    """
    # Edge case: median |α| slightly below threshold
    a_below = np.array([-4.9, -4.8, -4.7, -4.6])
    a_out_below = _ensure_alpha_natural(a_below)
    np.testing.assert_allclose(a_out_below, a_below, rtol=1e-10)

    # Edge case: median |α| slightly above threshold
    a_above = np.array([5.1, 5.2, 5.3, 5.4])
    a_out_above = _ensure_alpha_natural(a_above)
    expected_above = -a_above / K_MAG_PER_LN
    np.testing.assert_allclose(a_out_above, expected_above, rtol=1e-10)

    print("✓ test_loader_heuristic_threshold PASSED")


def test_loader_rejects_invalid_input():
    """
    Test that loader raises errors for invalid inputs.
    """
    # NaN should raise
    try:
        _ensure_alpha_natural(np.array([1.0, np.nan, 3.0]))
        assert False, "Should have raised ValueError for NaN"
    except ValueError as e:
        assert "NaN/inf" in str(e)

    # Inf should raise
    try:
        _ensure_alpha_natural(np.array([1.0, np.inf, 3.0]))
        assert False, "Should have raised ValueError for inf"
    except ValueError as e:
        assert "NaN/inf" in str(e)

    print("✓ test_loader_rejects_invalid_input PASSED")


def test_conversion_constant():
    """
    Test that K_MAG_PER_LN has the correct value.
    """
    expected_K = 2.5 / np.log(10.0)  # ≈ 1.0857362
    assert abs(K_MAG_PER_LN - expected_K) < 1e-10, \
        f"K_MAG_PER_LN = {K_MAG_PER_LN} != {expected_K}"

    print("✓ test_conversion_constant PASSED")
    print(f"  K = 2.5 / ln(10) = {K_MAG_PER_LN:.7f}")


if __name__ == "__main__":
    test_loader_converts_mag_to_nat()
    test_loader_preserves_natural_alpha()
    test_loader_heuristic_threshold()
    test_loader_rejects_invalid_input()
    test_conversion_constant()
    print("\n✓ All alpha loader tests PASSED!")
