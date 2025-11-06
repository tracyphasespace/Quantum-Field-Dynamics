#!/usr/bin/env python3
"""
Monotonicity tests for V15 QFD model.

Tests that α_pred(z) is monotone non-increasing and μ_pred(z) is monotone non-decreasing.
"""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v15_model import alpha_pred_batch
from v15_metrics import monotonicity_violations, check_monotonicity_detailed

# Test grid configuration
Z_MIN, Z_MAX = 0.01, 1.50
N_Z = 1500
Z_GRID = np.linspace(Z_MIN, Z_MAX, N_Z, dtype=np.float64)

# Best-fit JSON path (corrected from cloud.txt)
BESTFIT_JSON = Path(__file__).parent.parent / "results/v15_production/stage2/best_fit.json"


def _load_bestfit_or_fallback():
    """
    Load best-fit globals; if unavailable, use a safe, typical set.
    """
    try:
        with open(BESTFIT_JSON, "r") as f:
            d = json.load(f)
        # Keys from actual file
        k_J       = float(d["k_J"])
        eta_prime = float(d["eta_prime"])
        xi        = float(d["xi"])
        print(f"Loaded best-fit: k_J={k_J:.3f}, eta_prime={eta_prime:.3f}, xi={xi:.3f}")
    except Exception as e:
        # Fallback: values consistent with latest runs
        k_J, eta_prime, xi = 10.0, -8.0, -7.0
        print(f"Using fallback params (file not found: {e})")
    return k_J, eta_prime, xi


@pytest.mark.monotone
def test_alpha_pred_monotone_nonincreasing_bestfit():
    """
    α_pred(z) must be monotone non-increasing for the best-fit globals.
    """
    k_J, eta_prime, xi = _load_bestfit_or_fallback()
    alpha = alpha_pred_batch(Z_GRID, k_J, eta_prime, xi)

    # Use relaxed tolerance for now (1e-8) pending empirical verification
    vio = monotonicity_violations(alpha, increasing=False, tol=1e-8)

    if vio > 0:
        # Provide diagnostic information
        detail = check_monotonicity_detailed(Z_GRID, alpha, increasing=False, tol=1e-8)
        print(f"\n=== MONOTONICITY VIOLATION DETAILS ===")
        print(f"Number of violations: {detail['n_violations']}")
        print(f"Max violation magnitude: {detail['max_violation']:.6e}")
        if len(detail['violation_z_ranges']) > 0:
            print(f"First 5 violation z-ranges:")
            for z_lo, z_hi in detail['violation_z_ranges'][:5]:
                print(f"  z ∈ [{z_lo:.6f}, {z_hi:.6f}]")

    assert vio == 0, f"alpha_pred not monotone non-increasing; violations={vio}"


@pytest.mark.monotone
def test_mu_pred_monotone_nondecreasing_affine_from_alpha():
    """
    μ_pred(z) = μ0 - K*α(z). For any K>0, monotonicity flips sign relative to α.
    We don't need μ0 or the exact K for monotonicity; K=1 is sufficient.
    """
    k_J, eta_prime, xi = _load_bestfit_or_fallback()
    alpha = alpha_pred_batch(Z_GRID, k_J, eta_prime, xi)
    mu_pred = -alpha  # K=1, μ0 dropped: monotonicity preserved

    vio = monotonicity_violations(mu_pred, increasing=True, tol=1e-8)

    if vio > 0:
        detail = check_monotonicity_detailed(Z_GRID, mu_pred, increasing=True, tol=1e-8)
        print(f"\n=== MU MONOTONICITY VIOLATION DETAILS ===")
        print(f"Number of violations: {detail['n_violations']}")
        print(f"Max violation magnitude: {detail['max_violation']:.6e}")
        if len(detail['violation_z_ranges']) > 0:
            print(f"First 5 violation z-ranges:")
            for z_lo, z_hi in detail['violation_z_ranges'][:5]:
                print(f"  z ∈ [{z_lo:.6f}, {z_hi:.6f}]")

    assert vio == 0, f"mu_pred not monotone non-decreasing; violations={vio}"


@pytest.mark.monotone
def test_alpha_pred_random_perturbations():
    """
    Optional robustness: small random jitters around the best-fit should
    retain monotonicity. This catches fragile algebra/precision issues.
    """
    rng = np.random.default_rng(0)
    k_J, eta_prime, xi = _load_bestfit_or_fallback()

    for trial in range(8):
        kJ  = k_J       + rng.normal(scale=0.3 * max(1.0, abs(k_J)))
        et  = eta_prime + rng.normal(scale=0.3 * max(1.0, abs(eta_prime)))
        xii = xi        + rng.normal(scale=0.3 * max(1.0, abs(xi)))

        alpha = alpha_pred_batch(Z_GRID, kJ, et, xii)
        vio = monotonicity_violations(alpha, increasing=False, tol=1e-8)

        if vio > 0:
            detail = check_monotonicity_detailed(Z_GRID, alpha, increasing=False, tol=1e-8)
            print(f"\n=== PERTURBATION TRIAL {trial} FAILED ===")
            print(f"Params: (k_J={kJ:.3f}, eta'={et:.3f}, xi={xii:.3f})")
            print(f"Violations: {detail['n_violations']}, max={detail['max_violation']:.6e}")

        assert vio == 0, (
            f"alpha_pred monotonicity broke for trial {trial}: "
            f"(kJ,eta',xi)=({kJ:.3f},{et:.3f},{xii:.3f}), violations={vio}"
        )
