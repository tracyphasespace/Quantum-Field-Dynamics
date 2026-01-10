#!/usr/bin/env python3
"""
Harmonic family model for nuclear structure.

Implements:
  - Z_pred(A, N, family_params): Predicted Z for given A, mode N, family
  - epsilon(A, Z, family_params): Dissonance metric (distance to nearest harmonic)
  - Nhat(A, Z, family_params): Continuous mode estimate

Model form:
    Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)

Where:
    - c1_0, c2_0, c3_0: baseline coefficients (N=0 line)
    - dc1, dc2, dc3: per-mode increments
    - N: integer mode index

Dissonance (ε):
    ε = |N_hat - round(N_hat)| ∈ [0, 0.5]

Where:
    N_hat = (Z - Z_0(A)) / ΔZ(A)
    Z_0(A) = c1_0·A^(2/3) + c2_0·A + c3_0
    ΔZ(A) = dc1·A^(2/3) + dc2·A + dc3

Implements EXPERIMENT_PLAN.md §1.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class FamilyParams:
    """
    Parameters for a single harmonic family.

    Attributes:
        name: Family identifier (e.g., "A", "B", "C")
        c1_0: Baseline A^(2/3) coefficient
        c2_0: Baseline A coefficient
        c3_0: Baseline constant term
        dc1: Per-mode increment for A^(2/3) term
        dc2: Per-mode increment for A term
        dc3: Per-mode increment for constant term (the "clock step")
    """
    name: str
    c1_0: float
    c2_0: float
    c3_0: float
    dc1: float
    dc2: float
    dc3: float

    def __post_init__(self):
        """Validate parameters."""
        if not self.name:
            raise ValueError("Family name must be non-empty")

        # dc3 is the "clock step" and should be negative (Z decreases with mode)
        if self.dc3 > 0:
            import warnings
            warnings.warn(f"Family {self.name}: dc3={self.dc3} is positive (expected negative)")

    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'name': self.name,
            'c1_0': self.c1_0,
            'c2_0': self.c2_0,
            'c3_0': self.c3_0,
            'dc1': self.dc1,
            'dc2': self.dc2,
            'dc3': self.dc3,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FamilyParams':
        """Create from dictionary (for JSON deserialization)."""
        return cls(**data)


def Z_baseline(A: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate baseline Z (N=0 line) for given A.

    Z_0(A) = c1_0·A^(2/3) + c2_0·A + c3_0

    Args:
        A: Mass number (can be scalar or array)
        params: Family parameters

    Returns:
        Baseline Z values
    """
    A = np.asarray(A, dtype=float)
    return params.c1_0 * A**(2/3) + params.c2_0 * A + params.c3_0


def delta_Z(A: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate mode spacing ΔZ(A) for given A.

    ΔZ(A) = dc1·A^(2/3) + dc2·A + dc3

    This is the Z-increment per mode step.

    Args:
        A: Mass number (can be scalar or array)
        params: Family parameters

    Returns:
        Mode spacing values
    """
    A = np.asarray(A, dtype=float)
    return params.dc1 * A**(2/3) + params.dc2 * A + params.dc3


def Z_predicted(A: np.ndarray, N: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate predicted Z for given (A, N).

    Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)
                 = Z_0(A) + N·ΔZ(A)

    Args:
        A: Mass number (can be scalar or array)
        N: Mode index (can be scalar or array)
        params: Family parameters

    Returns:
        Predicted Z values
    """
    A = np.asarray(A, dtype=float)
    N = np.asarray(N, dtype=float)

    # Expand model
    term_A23 = (params.c1_0 + N * params.dc1) * A**(2/3)
    term_A = (params.c2_0 + N * params.dc2) * A
    term_const = params.c3_0 + N * params.dc3

    return term_A23 + term_A + term_const


def N_hat(A: np.ndarray, Z: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate continuous mode estimate N_hat for given (A, Z).

    N_hat = (Z - Z_0(A)) / ΔZ(A)

    This inverts the model to estimate which mode a nuclide is near.

    Args:
        A: Mass number (can be scalar or array)
        Z: Atomic number (can be scalar or array)
        params: Family parameters

    Returns:
        Continuous mode estimates
    """
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)

    Z_0 = Z_baseline(A, params)
    dZ = delta_Z(A, params)

    # Avoid division by zero
    dZ = np.where(np.abs(dZ) < 1e-10, np.nan, dZ)

    return (Z - Z_0) / dZ


def epsilon(A: np.ndarray, Z: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate dissonance ε for given (A, Z).

    ε = |N_hat - round(N_hat)| ∈ [0, 0.5]

    This measures distance to the nearest harmonic mode.

    Args:
        A: Mass number (can be scalar or array)
        Z: Atomic number (can be scalar or array)
        params: Family parameters

    Returns:
        Dissonance values in [0, 0.5]
    """
    Nhat = N_hat(A, Z, params)
    Nhat_rounded = np.round(Nhat)
    eps = np.abs(Nhat - Nhat_rounded)

    # Should be in [0, 0.5] by definition
    # (round gives nearest integer, so max distance is 0.5)
    return eps


def residual(A: np.ndarray, Z: np.ndarray, params: FamilyParams) -> np.ndarray:
    """
    Calculate Z residual (Z_obs - Z_pred) for given (A, Z).

    This uses the nearest integer mode:
        N_best = round(N_hat)
        residual = Z - Z_pred(A, N_best)

    Args:
        A: Mass number (can be scalar or array)
        Z: Atomic number (can be scalar or array)
        params: Family parameters

    Returns:
        Z residuals
    """
    Nhat = N_hat(A, Z, params)
    N_best = np.round(Nhat)
    Z_pred = Z_predicted(A, N_best, params)

    return Z - Z_pred


def score_nuclide(
    A: float,
    Z: float,
    params: FamilyParams
) -> Dict:
    """
    Score a single nuclide against a family.

    Returns all relevant quantities:
        - N_hat: Continuous mode estimate
        - N_best: Nearest integer mode
        - epsilon: Dissonance
        - Z_pred: Predicted Z at N_best
        - residual: Z - Z_pred

    Args:
        A: Mass number
        Z: Atomic number
        params: Family parameters

    Returns:
        Dictionary with scoring results
    """
    Nhat = N_hat(A, Z, params)
    N_best = np.round(Nhat)
    eps = epsilon(A, Z, params)
    Z_pred = Z_predicted(A, N_best, params)
    resid = Z - Z_pred

    return {
        'family': params.name,
        'N_hat': float(Nhat),
        'N_best': int(N_best),
        'epsilon': float(eps),
        'Z_pred': float(Z_pred),
        'residual': float(resid),
    }


def score_best_family(
    A: float,
    Z: float,
    families: Dict[str, FamilyParams]
) -> Dict:
    """
    Score a nuclide against all families and return best match.

    "Best" is defined as minimum epsilon.

    Args:
        A: Mass number
        Z: Atomic number
        families: Dictionary of family_name -> FamilyParams

    Returns:
        Dictionary with:
            - best_family: Name of best-matching family
            - epsilon_best: Dissonance for best family
            - epsilon_{A,B,C,...}: Dissonance for each family
            - N_hat_best, N_best, Z_pred_best, residual_best
    """
    scores = {}

    for name, params in families.items():
        score = score_nuclide(A, Z, params)
        scores[name] = score

    # Find best family (minimum epsilon)
    best_name = min(scores, key=lambda k: scores[k]['epsilon'])
    best_score = scores[best_name]

    result = {
        'best_family': best_name,
        'epsilon_best': best_score['epsilon'],
        'N_hat_best': best_score['N_hat'],
        'N_best': best_score['N_best'],
        'Z_pred_best': best_score['Z_pred'],
        'residual_best': best_score['residual'],
    }

    # Add per-family epsilons
    for name, score in scores.items():
        result[f'epsilon_{name}'] = score['epsilon']

    return result


def classify_by_epsilon(eps: float) -> str:
    """
    Classify nuclide by dissonance level.

    Pre-registered thresholds (EXPERIMENT_PLAN.md §1.2):
        - "harmonic": ε < 0.05
        - "near_harmonic": 0.05 ≤ ε < 0.15
        - "dissonant": ε ≥ 0.15

    Args:
        eps: Dissonance value

    Returns:
        Classification string
    """
    if eps < 0.05:
        return 'harmonic'
    elif eps < 0.15:
        return 'near_harmonic'
    else:
        return 'dissonant'


def dc3_comparison(families: Dict[str, FamilyParams]) -> Dict:
    """
    Compare dc3 values across families.

    dc3 is the "A-independent clock step" and should be nearly universal
    across families if the harmonic model is valid.

    Args:
        families: Dictionary of family_name -> FamilyParams

    Returns:
        Dictionary with:
            - dc3_values: dict of family_name -> dc3
            - dc3_mean: mean across families
            - dc3_std: standard deviation
            - dc3_range: max - min
            - dc3_relative_std: std / |mean|
    """
    dc3_values = {name: params.dc3 for name, params in families.items()}
    dc3_array = np.array(list(dc3_values.values()))

    return {
        'dc3_values': dc3_values,
        'dc3_mean': float(np.mean(dc3_array)),
        'dc3_std': float(np.std(dc3_array)),
        'dc3_range': float(np.ptp(dc3_array)),
        'dc3_relative_std': float(np.std(dc3_array) / np.abs(np.mean(dc3_array))),
    }


def validate_params(params: FamilyParams, A_range=(1, 300)) -> Dict:
    """
    Validate family parameters for physical consistency.

    Checks:
        - ΔZ(A) should be non-zero (avoid singularities)
        - Z_0(A) should be in valid range [0, A]
        - dc3 sign (should be negative)

    Args:
        params: Family parameters to validate
        A_range: Range of A to check (min, max)

    Returns:
        Dictionary with validation results and warnings
    """
    A_test = np.arange(A_range[0], A_range[1] + 1)
    Z_0 = Z_baseline(A_test, params)
    dZ = delta_Z(A_test, params)

    warnings = []

    # Check ΔZ is non-zero
    if np.any(np.abs(dZ) < 1e-10):
        warnings.append("ΔZ(A) approaches zero for some A (singularity risk)")

    # Check Z_0 is in valid range
    if np.any(Z_0 < 0):
        warnings.append(f"Z_0(A) < 0 for some A (unphysical)")
    if np.any(Z_0 > A_test):
        warnings.append(f"Z_0(A) > A for some A (unphysical)")

    # Check dc3 sign
    if params.dc3 > 0:
        warnings.append(f"dc3={params.dc3} > 0 (expected negative for Z-decreasing modes)")

    return {
        'family': params.name,
        'valid': len(warnings) == 0,
        'warnings': warnings,
        'Z_0_range': (float(np.min(Z_0)), float(np.max(Z_0))),
        'delta_Z_range': (float(np.min(dZ)), float(np.max(dZ))),
    }
