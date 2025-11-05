#!/usr/bin/env python3
"""
Unit tests for alpha sign and scale conventions.

Guards against sign/units regressions in alpha_pred.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v15_model import alpha_pred_batch


def test_alpha_pred_is_negative_and_decreasing():
    """
    Test that alpha_pred is negative at high z and decreases with increasing z.

    Canonical convention:
    - α ≡ ln(A) where A is amplitude scaling factor
    - Higher z → dimmer → smaller A → more negative α
    """
    z = np.array([0.0, 0.1, 0.5, 1.0, 1.5])
    a = np.array(alpha_pred_batch(z, 70.0, 0.01, 30.0))

    # All values should be finite
    assert np.all(np.isfinite(a)), f"alpha_pred contains non-finite values: {a}"

    # Should be non-increasing (decreasing or flat)
    diffs = np.diff(a)
    assert np.all(diffs <= 0), f"alpha_pred is not monotone decreasing: diffs={diffs}"

    # Should be negative at high z
    assert a[-1] < 0, f"alpha_pred({z[-1]}) = {a[-1]} should be negative"

    # At z=0, should be exactly 0 (normalization)
    assert abs(a[0]) < 1e-10, f"alpha_pred(0) = {a[0]} should be ≈ 0"

    print("✓ test_alpha_pred_is_negative_and_decreasing PASSED")
    print(f"  alpha_pred(z): {a}")


def test_alpha_pred_scale():
    """
    Test that alpha_pred has reasonable magnitude (not absurdly large).

    For typical cosmological parameters and z ∈ [0, 1.5], alpha should
    be in range [-100, 0], not thousands.
    """
    z = np.linspace(0, 1.5, 100)
    a = np.array(alpha_pred_batch(z, 70.0, 0.01, 30.0))

    # Should be bounded
    assert np.all(a >= -200), f"alpha_pred has absurdly negative values: min={a.min()}"
    assert np.all(a <= 10), f"alpha_pred has absurdly positive values: max={a.max()}"

    # Typical range at z=1 should be O(-50) not O(-1000) or O(+50)
    a_at_1 = float(alpha_pred_batch(np.array([1.0]), 70.0, 0.01, 30.0)[0])
    assert -100 < a_at_1 < 0, f"alpha_pred(z=1) = {a_at_1} is outside typical range"

    print("✓ test_alpha_pred_scale PASSED")
    print(f"  alpha_pred(z=1) = {a_at_1:.2f}")


def test_alpha_pred_parameter_sensitivity():
    """
    Test that alpha_pred actually depends on parameters (not a constant).
    """
    z = np.array([0.5, 1.0])

    # Different k_J should give different alpha
    a1 = np.array(alpha_pred_batch(z, 50.0, 0.01, 30.0))
    a2 = np.array(alpha_pred_batch(z, 90.0, 0.01, 30.0))
    assert not np.allclose(a1, a2), "alpha_pred doesn't depend on k_J"

    # Different eta_prime should give different alpha
    a3 = np.array(alpha_pred_batch(z, 70.0, 0.001, 30.0))
    a4 = np.array(alpha_pred_batch(z, 70.0, 0.1, 30.0))
    assert not np.allclose(a3, a4), "alpha_pred doesn't depend on eta_prime"

    # Different xi should give different alpha
    a5 = np.array(alpha_pred_batch(z, 70.0, 0.01, 10.0))
    a6 = np.array(alpha_pred_batch(z, 70.0, 0.01, 50.0))
    assert not np.allclose(a5, a6), "alpha_pred doesn't depend on xi"

    print("✓ test_alpha_pred_parameter_sensitivity PASSED")


if __name__ == "__main__":
    test_alpha_pred_is_negative_and_decreasing()
    test_alpha_pred_scale()
    test_alpha_pred_parameter_sensitivity()
    print("\n✓ All alpha sign/scale tests PASSED!")
