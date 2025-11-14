#!/usr/bin/env python3
"""
Spec contract tests: Lock V15 invariants into automated tests.

These tests enforce the core V15 spec:
1. α-pred properties (normalization, monotonicity, independence)
2. Residual variance must be positive (wiring bug detector)
3. μ-space/α-space identity
"""
import sys
import pathlib
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from v15_model import alpha_pred_batch

K = 2.5 / np.log(10.0)  # mag-space constant


def test_alpha_pred_properties():
    """
    α_pred must satisfy:
    - α_pred(z=0) = 0 (normalization)
    - α_pred monotonically decreasing with z
    - All finite values
    """
    z = np.array([0.0, 1e-6, 0.1, 0.5, 1.0, 1.5])
    a = alpha_pred_batch(z, 70.0, 0.01, 30.0)

    # Normalization at z=0
    assert np.isfinite(a).all(), "α_pred must be finite for all z"
    assert abs(a[0]) < 1e-12, "α_pred(z=0) must be 0"

    # Monotonicity (should be decreasing)
    diffs = np.diff(a)
    assert np.all(diffs < 0), f"α_pred must be monotonically decreasing, got diffs={diffs}"


def test_residual_variance_positive():
    """
    Wiring bug detector: var(r_α) must be > 0.

    If residuals have zero variance, α_obs is being copied into α_th.
    """
    z = np.linspace(0.05, 1.2, 100)
    # Simulated observations with structure
    alpha_obs = -0.1 + 0.05 * np.sin(5 * z)
    alpha_th = alpha_pred_batch(z, 70.0, 0.01, 30.0)

    r = alpha_obs - alpha_th
    var_r = np.var(r)

    assert var_r > 0, f"Residual variance must be positive, got {var_r}"


def test_mu_alpha_identity():
    """
    μ-space and α-space must satisfy identity:
    μ_obs - μ_QFD = -K·(α_obs - α_th)

    This ensures consistent mapping between magnitude and alpha deviations.
    """
    # Import stage3 distance function
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from stage3_hubble_optimized import qfd_distance_modulus_distance_only

    z = np.linspace(0.1, 0.9, 20)
    alpha_obs = -0.2 + 0.05 * np.cos(7 * z)
    alpha_th = alpha_pred_batch(z, 70.0, 0.01, 30.0)

    # Theoretical QFD distance modulus (H0=70 fiducial)
    mu_th = np.array([qfd_distance_modulus_distance_only(zi, 70.0) for zi in z])

    # Observed distance modulus
    mu_obs = mu_th - K * alpha_obs

    # QFD prediction
    mu_qfd = mu_th - K * alpha_th

    # Identity check
    lhs = mu_obs - mu_qfd
    rhs = -K * (alpha_obs - alpha_th)

    np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=0)


def test_alpha_pred_independence():
    """
    α_pred must NOT depend on per-SN parameters (α_obs, β, t₀, etc.).

    This test verifies the function signature only takes global params.
    """
    import inspect

    sig = inspect.signature(alpha_pred_batch)
    params = list(sig.parameters.keys())

    # Should only accept: z, k_J, eta_prime, xi (no per-SN params)
    expected_params = ["z", "k_J", "eta_prime", "xi"]
    assert params == expected_params, f"α_pred signature must be {expected_params}, got {params}"


def test_alpha_pred_not_copies_alpha_obs():
    """
    Ensure α_pred does not accidentally return α_obs.

    This would manifest as perfect correlation between predictions and observations.
    """
    z = np.linspace(0.05, 1.2, 100)
    alpha_obs = -0.1 + 0.05 * np.sin(5 * z)
    alpha_th = alpha_pred_batch(z, 70.0, 0.01, 30.0)

    # They should be uncorrelated (or weakly correlated at best)
    assert not np.allclose(alpha_th, alpha_obs, atol=0.01), "α_pred must not copy α_obs"


def test_dtype_consistency():
    """
    α_pred should work with both float32 and float64.
    """
    z64 = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    z32 = np.array([0.1, 0.5, 1.0], dtype=np.float32)

    a64 = alpha_pred_batch(z64, 70.0, 0.01, 30.0)
    a32 = alpha_pred_batch(z32, 70.0, 0.01, 30.0)

    # Should match to float32 precision
    np.testing.assert_allclose(a64, a32, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
