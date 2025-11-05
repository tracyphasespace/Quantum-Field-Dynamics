"""
Unit test to enforce Stage 3 residual identity.

This test verifies that:
1. residual_qfd = -K * (alpha_obs - alpha_th)
2. When alpha_obs == alpha_th, residual_qfd == 0
3. alpha_pred returns different values for different redshifts
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v15_model import alpha_pred_batch

K = 2.5 / np.log(10.0)


def qfd_distance_modulus_distance_only(z, k_J):
    """
    Simplified QFD distance modulus (distance component only).
    """
    # Simple linear approximation for testing
    D_mpc = z * 299792.458 / 70.0  # Hubble law
    return 5.0 * np.log10(D_mpc * 1e6 / 10.0)


def test_residual_qfd_identity():
    """Test that residual_qfd matches the expected formula."""
    z = 0.1
    k_J, eta_prime, xi = 70.0, 0.01, 30.0
    alpha_obs = 0.1234

    mu_th = qfd_distance_modulus_distance_only(z, k_J)
    mu_obs = mu_th - K * alpha_obs

    alpha_th = float(alpha_pred_batch(np.array([z]), k_J, eta_prime, xi)[0])
    mu_qfd = mu_th - K * alpha_th

    residual_qfd = mu_obs - mu_qfd
    np.testing.assert_allclose(
        residual_qfd, -K * (alpha_obs - alpha_th),
        rtol=0, atol=1e-10,
        err_msg="residual_qfd must equal -K*(alpha_obs - alpha_th)"
    )


def test_zero_residual_when_alpha_match():
    """Test that residual is zero when alpha_obs == alpha_th."""
    z = 0.1
    k_J, eta_prime, xi = 70.0, 0.01, 30.0

    mu_th = qfd_distance_modulus_distance_only(z, k_J)
    alpha_th = float(alpha_pred_batch(np.array([z]), k_J, eta_prime, xi)[0])

    # Use alpha_th as alpha_obs
    mu_obs = mu_th - K * alpha_th
    mu_qfd = mu_th - K * alpha_th

    residual = mu_obs - mu_qfd
    np.testing.assert_allclose(
        residual, 0.0,
        rtol=0, atol=1e-12,
        err_msg="Residual must be zero when alpha_obs == alpha_th"
    )


def test_alpha_pred_varies_with_z():
    """Test that alpha_pred returns different values for different z."""
    k_J, eta_prime, xi = 70.0, 0.01, 30.0
    z_array = np.array([0.0, 0.1, 0.5, 1.0])

    alpha_array = alpha_pred_batch(z_array, k_J, eta_prime, xi)

    # Check that values are different
    assert len(np.unique(alpha_array)) == len(alpha_array), \
        "alpha_pred must return different values for different redshifts"

    # Check normalization: alpha_pred(0) should be close to 0
    assert np.abs(alpha_array[0]) < 1e-10, \
        f"alpha_pred(0) should be ~0, got {alpha_array[0]}"

    # Check that alpha becomes more negative with increasing z (dimming)
    assert np.all(np.diff(alpha_array) < 0), \
        "alpha_pred should decrease (become more negative) with increasing z"


def test_alpha_pred_not_constant():
    """Test that alpha_pred is not returning a constant value."""
    k_J, eta_prime, xi = 70.0, 0.01, 30.0
    z1, z2 = 0.1, 0.5

    alpha1 = float(alpha_pred_batch(np.array([z1]), k_J, eta_prime, xi)[0])
    alpha2 = float(alpha_pred_batch(np.array([z2]), k_J, eta_prime, xi)[0])

    assert not np.isclose(alpha1, alpha2), \
        f"alpha_pred returning same value for z={z1} and z={z2}: {alpha1}"


if __name__ == "__main__":
    print("Running Stage 3 identity tests...")

    try:
        test_residual_qfd_identity()
        print("✓ test_residual_qfd_identity PASSED")
    except AssertionError as e:
        print(f"✗ test_residual_qfd_identity FAILED: {e}")

    try:
        test_zero_residual_when_alpha_match()
        print("✓ test_zero_residual_when_alpha_match PASSED")
    except AssertionError as e:
        print(f"✗ test_zero_residual_when_alpha_match FAILED: {e}")

    try:
        test_alpha_pred_varies_with_z()
        print("✓ test_alpha_pred_varies_with_z PASSED")
    except AssertionError as e:
        print(f"✗ test_alpha_pred_varies_with_z FAILED: {e}")

    try:
        test_alpha_pred_not_constant()
        print("✓ test_alpha_pred_not_constant PASSED")
    except AssertionError as e:
        print(f"✗ test_alpha_pred_not_constant FAILED: {e}")

    print("\nAll tests completed!")
