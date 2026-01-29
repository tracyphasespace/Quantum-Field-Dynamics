#!/usr/bin/env python3
"""
BBH Lensing Parameter Initialization Strategies

Provides intelligent initial guesses for A_lens optimization based on
residual magnitudes and physical constraints.

PROBLEM:
The current Stage 3 uses arbitrary starting values for A_lens, leading to
96% optimization failure rate. L-BFGS-B is a local optimizer that requires
good initialization.

SOLUTION:
Use residual magnitude to estimate A_lens scale, then grid search around
that estimate to find a starting point in the basin of attraction.
"""

import numpy as np
from typing import Tuple, Optional
import jax.numpy as jnp


def estimate_alens_from_residual(
    residual_mag: float,
    z: float,
    too_dark: bool = True
) -> float:
    """
    Estimate initial A_lens from residual magnitude.

    Physical intuition:
    - If SN is too dark by Δm magnitudes, lensing scattered light away
    - If SN is too bright by Δm, lensing focused light toward us
    - Magnitude residual ≈ 2.5 * log10(1 + A_lens) for weak lensing

    Args:
        residual_mag: Residual in magnitudes (QFD - observed)
        z: Redshift (affects lensing strength)
        too_dark: True if SN is dimmer than expected (BBH scatter)

    Returns:
        Initial A_lens estimate (positive for magnification, negative for scattering)
    """
    # Convert magnitude residual to flux ratio
    # Δm = -2.5 log10(F_obs / F_pred)
    # F_obs / F_pred = 10^(-Δm / 2.5)
    # A_lens ≈ (F_obs / F_pred) - 1

    flux_ratio = 10 ** (-residual_mag / 2.5)
    A_lens_estimate = flux_ratio - 1.0

    # Physical bounds: lensing can't create infinite magnification
    # or complete darkness. Clamp to reasonable range.
    if too_dark:
        # Scattering: A_lens should be negative
        A_lens_estimate = np.clip(A_lens_estimate, -0.9, -0.01)
    else:
        # Magnification: A_lens should be positive
        A_lens_estimate = np.clip(A_lens_estimate, 0.01, 5.0)

    # Redshift scaling: higher-z lensing typically requires larger A_lens
    # due to geometric projection effects
    z_factor = 1.0 + 0.5 * z  # Empirical scaling
    A_lens_estimate *= z_factor

    return float(A_lens_estimate)


def grid_search_alens_init(
    lc_data,
    stage1_params: np.ndarray,
    global_params: dict,
    residual_mag: float,
    too_dark: bool,
    n_grid: int = 5
) -> Tuple[float, float]:
    """
    Grid search around estimated A_lens to find best starting point.

    Strategy:
    1. Estimate A_lens from residual
    2. Create small grid around estimate
    3. Evaluate chi² at each grid point
    4. Return grid point with lowest chi²

    Args:
        lc_data: Supernova light curve data
        stage1_params: Best-fit params from Stage 1 (t0, ln_A, A_plasma, beta)
        global_params: Global QFD params (k_J_correction, eta_prime, xi)
        residual_mag: QFD residual magnitude
        too_dark: True if BBH scatter candidate
        n_grid: Number of grid points to try

    Returns:
        (best_A_lens, best_chi2)
    """
    # FIX 2025-01-16: Use simplified V18 BBH model
    from v18_bbh_model import compute_chi2_simple

    # Get initial estimate
    A_lens_center = estimate_alens_from_residual(
        residual_mag, lc_data.z, too_dark
    )

    # Create grid (log-spaced for better coverage)
    if too_dark:
        # For scattering: grid from -0.9 to -0.01
        A_lens_grid = -np.logspace(
            np.log10(0.01), np.log10(0.9), n_grid
        )
    else:
        # For magnification: grid from 0.01 to 5.0
        A_lens_grid = np.logspace(
            np.log10(0.01), np.log10(5.0), n_grid
        )

    # Center grid around estimate (if it's in range)
    if too_dark and -0.9 < A_lens_center < -0.01:
        A_lens_grid = A_lens_center * np.linspace(0.5, 2.0, n_grid)
        A_lens_grid = np.clip(A_lens_grid, -0.9, -0.01)
    elif not too_dark and 0.01 < A_lens_center < 5.0:
        A_lens_grid = A_lens_center * np.linspace(0.5, 2.0, n_grid)
        A_lens_grid = np.clip(A_lens_grid, 0.01, 5.0)

    # Evaluate chi² at each grid point
    best_chi2 = np.inf
    best_A_lens = A_lens_center

    for A_lens_trial in A_lens_grid:
        # Fixed parameters for this trial (6 params - xi removed)
        params_trial = np.concatenate([
            stage1_params[:1],  # t0
            [A_lens_trial],     # A_lens (trial value)
            stage1_params[1:],  # ln_A, A_plasma, beta
            [global_params['eta_prime']],  # FDR opacity parameter only
        ])

        try:
            # Compute chi² using V18 BBH model
            chi2 = compute_chi2_simple(
                lc_data,
                params_trial,
                global_params['k_J_correction']
            )

            if chi2 < best_chi2 and np.isfinite(chi2):
                best_chi2 = chi2
                best_A_lens = A_lens_trial

        except:
            # If model evaluation fails, skip this grid point
            continue

    return best_A_lens, best_chi2


def get_smart_bounds(residual_mag: float, too_dark: bool) -> Tuple[float, float]:
    """
    Get adaptive bounds for A_lens based on residual magnitude.

    Tighter bounds improve convergence by restricting search space.

    Args:
        residual_mag: Absolute residual magnitude
        too_dark: True for BBH scatter, False for magnify

    Returns:
        (lower_bound, upper_bound) for A_lens
    """
    if too_dark:
        # Scattering: more negative for larger residuals
        mag_scale = min(abs(residual_mag) / 5.0, 0.8)
        return (-0.9, -mag_scale * 0.05)
    else:
        # Magnification: higher for larger residuals
        mag_scale = min(abs(residual_mag) / 5.0, 4.0)
        return (0.01, mag_scale * 1.5)


# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        (-8.0, 0.3, True),   # Too dark by 8 mag at z=0.3
        (+7.0, 1.2, False),  # Too bright by 7 mag at z=1.2
        (-5.0, 0.5, True),   # Moderate dark outlier
    ]

    print("Testing A_lens initialization:")
    print("-" * 60)
    for residual, z, dark in test_cases:
        A_lens = estimate_alens_from_residual(residual, z, dark)
        bounds = get_smart_bounds(residual, dark)
        print(f"Residual: {residual:+.1f} mag, z={z:.2f}, "
              f"{'DARK' if dark else 'BRIGHT'}")
        print(f"  → A_lens estimate: {A_lens:+.4f}")
        print(f"  → Bounds: [{bounds[0]:+.2f}, {bounds[1]:+.2f}]")
        print()
