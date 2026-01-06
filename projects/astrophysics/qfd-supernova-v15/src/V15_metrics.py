#!/usr/bin/env python3
"""
Metrics and validation utilities for V15 QFD model.

Includes monotonicity checks, residual diagnostics, and posterior predictive utilities.
"""

import numpy as np
from typing import Optional, Tuple, Dict


def monotonicity_violations(
    y: np.ndarray,
    *,
    increasing: bool,
    tol: float = 1e-9
) -> int:
    """
    Counts pairwise monotonicity violations in y.

    Parameters
    ----------
    y : np.ndarray
        1D array to check for monotonicity
    increasing : bool
        If True, require y[i+1] >= y[i] - tol (non-decreasing)
        If False, require y[i+1] <= y[i] + tol (non-increasing)
    tol : float, optional
        Numerical tolerance for violations (default: 1e-9)

    Returns
    -------
    int
        Number of pairwise violations detected
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")

    if len(y) < 2:
        return 0

    dy = np.diff(y)

    if increasing:
        violations = np.count_nonzero(dy < -tol)
    else:
        violations = np.count_nonzero(dy > tol)

    return int(violations)


def check_monotonicity_detailed(
    z: np.ndarray,
    y: np.ndarray,
    *,
    increasing: bool,
    tol: float = 1e-9
) -> Dict[str, any]:
    """
    Detailed monotonicity check with diagnostic information.

    Parameters
    ----------
    z : np.ndarray
        Independent variable (e.g., redshift)
    y : np.ndarray
        Dependent variable (e.g., alpha_pred or mu_pred)
    increasing : bool
        Expected monotonicity direction
    tol : float, optional
        Numerical tolerance

    Returns
    -------
    dict
        Contains:
        - 'is_monotone': bool
        - 'n_violations': int
        - 'violation_indices': np.ndarray (indices where violations occur)
        - 'violation_z_ranges': list of tuples (z[i], z[i+1])
        - 'max_violation': float (largest violation magnitude)
    """
    z = np.asarray(z)
    y = np.asarray(y)

    if len(z) != len(y):
        raise ValueError(f"z and y must have same length: {len(z)} vs {len(y)}")

    dy = np.diff(y)

    if increasing:
        violation_mask = dy < -tol
    else:
        violation_mask = dy > tol

    violation_indices = np.where(violation_mask)[0]
    n_violations = len(violation_indices)

    violation_z_ranges = []
    if n_violations > 0:
        for idx in violation_indices:
            violation_z_ranges.append((z[idx], z[idx + 1]))

        if increasing:
            max_violation = np.abs(np.min(dy))
        else:
            max_violation = np.max(dy)
    else:
        max_violation = 0.0

    return {
        'is_monotone': n_violations == 0,
        'n_violations': n_violations,
        'violation_indices': violation_indices,
        'violation_z_ranges': violation_z_ranges,
        'max_violation': max_violation,
    }


def compute_residual_slope(
    z: np.ndarray,
    residuals: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Compute weighted linear slope of residuals vs redshift.

    A flat (slope ≈ 0) residual pattern indicates good model calibration
    across the redshift range.

    Parameters
    ----------
    z : np.ndarray
        Redshift values
    residuals : np.ndarray
        Residual values (observed - predicted)
    weights : np.ndarray, optional
        Weights for each point (e.g., 1/σ²)

    Returns
    -------
    slope : float
        Best-fit slope (residuals per unit redshift)
    slope_err : float
        Standard error on the slope
    """
    z = np.asarray(z)
    residuals = np.asarray(residuals)

    if len(z) != len(residuals):
        raise ValueError(f"z and residuals must have same length")

    if weights is None:
        weights = np.ones_like(z)
    else:
        weights = np.asarray(weights)

    # Weighted least squares: residuals = a + b*z
    W = np.sum(weights)
    Wz = np.sum(weights * z)
    Wr = np.sum(weights * residuals)
    Wzz = np.sum(weights * z * z)
    Wzr = np.sum(weights * z * residuals)

    denom = W * Wzz - Wz * Wz
    if abs(denom) < 1e-10:
        return np.nan, np.nan

    # Slope
    slope = (W * Wzr - Wz * Wr) / denom

    # Intercept (for residual calculation)
    intercept = (Wr - slope * Wz) / W

    # Standard error on slope
    fit_residuals = residuals - (intercept + slope * z)
    chi2 = np.sum(weights * fit_residuals**2)
    n_dof = len(z) - 2

    if n_dof > 0:
        reduced_chi2 = chi2 / n_dof
        var_slope = reduced_chi2 * W / denom
        slope_err = np.sqrt(var_slope)
    else:
        slope_err = np.nan

    return float(slope), float(slope_err)