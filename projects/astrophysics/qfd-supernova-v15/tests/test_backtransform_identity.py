#!/usr/bin/env python3
"""
Test that backtransform_physical() correctly inverts the standardization.

The transformation is:
    α_pred = α₀ + Σᵢ cᵢ·φᵢ   where φᵢ = (ϕᵢ - μᵢ)/σᵢ
           = α₀ + Σᵢ cᵢ·(ϕᵢ - μᵢ)/σᵢ
           = [α₀ - Σᵢ cᵢ·μᵢ/σᵢ] + Σᵢ (cᵢ/σᵢ)·ϕᵢ
           = α₀_phys + Σᵢ kᵢ_phys·ϕᵢ

Where:
    kᵢ_phys = cᵢ/σᵢ
    α₀_phys = α₀ - Σᵢ cᵢ·μᵢ/σᵢ

This test verifies that predictions match before and after transformation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

import numpy as np
from write_stage2_summary import backtransform_physical, Standardizer


def test_backtransform_identity():
    """Verify that back-transformation preserves alpha predictions."""
    np.random.seed(42)

    # Mock standardizer (from some redshift data)
    std = Standardizer(
        means=np.array([0.4, 0.5, 0.3]),
        scales=np.array([0.2, 0.3, 0.1])
    )

    # Mock posterior samples
    n_samples = 100
    c_samples = np.random.randn(n_samples, 3)  # standardized coefficients
    alpha0_samples = np.random.randn(n_samples) * 5.0  # standardized offset

    # Mock standardized features for test data
    n_test = 50
    Phi_test = np.random.randn(n_test, 3)  # standardized features [N, 3]

    # Predictions using standardized parameters
    # For each sample i: alpha_pred[i, j] = alpha0[i] + Phi_test[j, :] @ c[i, :]
    # Shape: (n_samples, N_test)
    alpha_pred_std = alpha0_samples[:, None] + c_samples @ Phi_test.T

    # Back-transform to physical parameters
    phys = backtransform_physical(c_samples, alpha0_samples, std)
    k_J_phys = phys['k_J']
    eta_prime_phys = phys['eta_prime']
    xi_phys = phys['xi']
    alpha0_phys = phys['alpha0_phys']

    # Reconstruct raw features from standardized
    phi_raw = Phi_test * std.scales + std.means  # [N, 3]

    # Predictions using physical parameters
    # For each sample i: alpha_pred[i, j] = alpha0_phys[i] + sum_k (phi_raw[j, k] * k_phys[i, k])
    # Shape: (n_samples, N_test)
    k_phys_matrix = np.column_stack([k_J_phys, eta_prime_phys, xi_phys])  # [n_samples, 3]
    alpha_pred_phys = alpha0_phys[:, None] + k_phys_matrix @ phi_raw.T

    # Check that predictions match
    diff = np.abs(alpha_pred_std - alpha_pred_phys)
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))

    print(f"Max difference: {max_diff:.3e}")
    print(f"RMS difference: {rms_diff:.3e}")

    # Tolerance: numerical precision limit
    assert max_diff < 1e-10, f"Max diff {max_diff:.3e} exceeds tolerance"
    assert rms_diff < 1e-11, f"RMS diff {rms_diff:.3e} exceeds tolerance"

    print("✅ Back-transformation identity test PASSED")


if __name__ == '__main__':
    test_backtransform_identity()
