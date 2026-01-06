#!/usr/bin/env python3
"""
Fit harmonic family parameters to training data.

Fits the model:
    Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)

Training protocols:
    - stable: Fit on stable nuclides only (285 nuclides)
    - longlived: Fit on nuclides with half_life > threshold
    - all: Fit on all nuclides (for comparison only, causes leakage)

Outputs:
    - JSON file with fitted FamilyParams for each family
    - Fit diagnostics (χ², residuals, covariance)

Implements EXPERIMENT_PLAN.md §3.2 (fit family parameters).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from scipy.optimize import minimize, least_squares
from typing import Dict, List, Tuple, Optional

from harmonic_model import FamilyParams, Z_predicted, epsilon, validate_params

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(
    nuclides_file: str,
    train_set: str = 'stable',
    min_half_life_s: Optional[float] = None
) -> pd.DataFrame:
    """
    Load training set from NUBASE data.

    Args:
        nuclides_file: Path to nuclides_all.parquet
        train_set: Training set selector
            - 'stable': Only stable nuclides (is_stable=True)
            - 'longlived': Nuclides with half_life_s > min_half_life_s
            - 'all': All nuclides (WARNING: causes leakage in Exp 1)
        min_half_life_s: Minimum half-life for 'longlived' mode

    Returns:
        pd.DataFrame: Training set with columns A, Z, N
    """
    df = pd.read_parquet(nuclides_file)
    logger.info(f"Loaded {len(df)} nuclides from {nuclides_file}")

    if train_set == 'stable':
        df_train = df[df['is_stable']].copy()
        logger.info(f"Training on {len(df_train)} stable nuclides")

    elif train_set == 'longlived':
        if min_half_life_s is None:
            raise ValueError("Must specify min_half_life_s for longlived training set")
        df_train = df[df['half_life_s'] > min_half_life_s].copy()
        logger.info(f"Training on {len(df_train)} nuclides with t₁/₂ > {min_half_life_s:.0e} s")

    elif train_set == 'all':
        logger.warning("Training on ALL nuclides (causes leakage in Experiment 1)")
        df_train = df.copy()
        logger.info(f"Training on all {len(df_train)} nuclides")

    else:
        raise ValueError(f"Unknown train_set: {train_set}")

    # Ensure required columns exist
    required_cols = ['A', 'Z']
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute N if not present
    if 'N' not in df_train.columns:
        df_train['N'] = df_train['A'] - df_train['Z']

    return df_train[['A', 'Z', 'N']].copy()


def fit_single_family_lsq(
    A: np.ndarray,
    Z: np.ndarray,
    initial_params: Optional[Dict] = None,
    family_name: str = "Family"
) -> Tuple[FamilyParams, Dict]:
    """
    Fit harmonic family parameters using least-squares.

    Model:
        Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)

    We fit on the assumption that nuclides cluster near integer modes.
    The fit minimizes residuals after assigning each nuclide to its nearest mode.

    Args:
        A: Mass numbers (training set)
        Z: Atomic numbers (training set)
        initial_params: Initial guess for parameters (dict with c1_0, c2_0, ...)
        family_name: Name for this family

    Returns:
        (params, diagnostics)
        params: Fitted FamilyParams
        diagnostics: Dict with χ², residuals, covariance, etc.
    """
    n_data = len(A)
    logger.info(f"Fitting {family_name} to {n_data} nuclides...")

    # Initial guess (reasonable defaults)
    if initial_params is None:
        initial_params = {
            'c1_0': 1.5,      # ~A^(2/3) surface term
            'c2_0': 0.4,      # ~A volume term
            'c3_0': -5.0,     # Constant offset
            'dc1': -0.05,     # Mode spacing (surface)
            'dc2': -0.01,     # Mode spacing (volume)
            'dc3': -0.865,    # Mode spacing (constant, "clock step")
        }

    x0 = np.array([
        initial_params['c1_0'],
        initial_params['c2_0'],
        initial_params['c3_0'],
        initial_params['dc1'],
        initial_params['dc2'],
        initial_params['dc3'],
    ])

    def residual_function(x):
        """
        Residual function for least-squares.

        For each nuclide (A, Z):
        1. Estimate N̂ from current parameters
        2. Round to nearest integer N
        3. Compute Z_pred(A, N) with current parameters
        4. Return residual = Z - Z_pred
        """
        c1_0, c2_0, c3_0, dc1, dc2, dc3 = x

        # Create temporary params
        params_temp = FamilyParams(
            name=family_name,
            c1_0=c1_0,
            c2_0=c2_0,
            c3_0=c3_0,
            dc1=dc1,
            dc2=dc2,
            dc3=dc3,
        )

        # For each nuclide, find nearest mode and compute residual
        residuals = np.zeros(n_data)
        for i in range(n_data):
            # Estimate mode
            Z_0 = c1_0 * A[i]**(2/3) + c2_0 * A[i] + c3_0
            dZ = dc1 * A[i]**(2/3) + dc2 * A[i] + dc3

            # Avoid division by zero
            if abs(dZ) < 1e-10:
                residuals[i] = 1e6  # Large penalty
                continue

            N_hat = (Z[i] - Z_0) / dZ
            N_best = int(np.round(N_hat))

            # Predict Z at this mode
            Z_pred = Z_predicted(A[i], N_best, params_temp)
            residuals[i] = Z[i] - Z_pred

        return residuals

    # Run least-squares optimization
    result = least_squares(
        residual_function,
        x0,
        method='lm',  # Levenberg-Marquardt
        verbose=0,
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")

    # Extract fitted parameters
    c1_0, c2_0, c3_0, dc1, dc2, dc3 = result.x

    params = FamilyParams(
        name=family_name,
        c1_0=c1_0,
        c2_0=c2_0,
        c3_0=c3_0,
        dc1=dc1,
        dc2=dc2,
        dc3=dc3,
    )

    # Compute diagnostics
    residuals = result.fun
    chi2 = np.sum(residuals**2)
    chi2_reduced = chi2 / (n_data - 6)  # 6 parameters
    rmse = np.sqrt(np.mean(residuals**2))

    # Covariance (from Jacobian)
    try:
        # Covariance = (J^T J)^{-1} * σ²
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * chi2_reduced
        param_errors = np.sqrt(np.diag(cov))
    except:
        logger.warning("Could not compute parameter covariance (singular Jacobian)")
        cov = None
        param_errors = None

    diagnostics = {
        'chi2': float(chi2),
        'chi2_reduced': float(chi2_reduced),
        'rmse': float(rmse),
        'n_data': int(n_data),
        'n_params': 6,
        'residuals_mean': float(np.mean(residuals)),
        'residuals_std': float(np.std(residuals)),
        'residuals_max': float(np.max(np.abs(residuals))),
        'converged': result.success,
        'iterations': result.nfev,
    }

    if param_errors is not None:
        diagnostics['param_errors'] = {
            'c1_0': float(param_errors[0]),
            'c2_0': float(param_errors[1]),
            'c3_0': float(param_errors[2]),
            'dc1': float(param_errors[3]),
            'dc2': float(param_errors[4]),
            'dc3': float(param_errors[5]),
        }

    logger.info(f"  χ² = {chi2:.2e}, χ²_red = {chi2_reduced:.2e}, RMSE = {rmse:.3f}")
    logger.info(f"  dc3 = {dc3:.4f} ± {param_errors[5]:.4f}" if param_errors is not None else f"  dc3 = {dc3:.4f}")

    return params, diagnostics


def fit_multiple_families(
    df_train: pd.DataFrame,
    n_families: int = 3,
    family_names: Optional[List[str]] = None,
    initial_params_list: Optional[List[Dict]] = None
) -> Tuple[Dict[str, FamilyParams], Dict]:
    """
    Fit multiple harmonic families simultaneously.

    Strategy:
        1. Cluster nuclides by residual from smooth baseline
        2. Assign each cluster to a family candidate
        3. Fit each family independently
        4. Iterate to refine cluster assignments

    Args:
        df_train: Training data with columns A, Z
        n_families: Number of families to fit
        family_names: Names for families (default: A, B, C, ...)
        initial_params_list: List of initial parameter dicts (one per family)

    Returns:
        (families, diagnostics)
        families: Dict of family_name -> FamilyParams
        diagnostics: Dict with overall fit quality
    """
    if family_names is None:
        family_names = [chr(65 + i) for i in range(n_families)]  # A, B, C, ...

    logger.info(f"Fitting {n_families} families: {family_names}")

    A = df_train['A'].values
    Z = df_train['Z'].values

    # Simple approach: fit a single "average" family first,
    # then perturb parameters to create multiple families
    logger.info("Step 1: Fit average family to all data")
    params_avg, diag_avg = fit_single_family_lsq(A, Z, family_name="Average")

    # Create initial guesses for each family by perturbing average
    if initial_params_list is None:
        initial_params_list = []
        for i in range(n_families):
            # Perturb c1_0, c2_0, c3_0 slightly
            perturbation = (i - n_families/2) * 0.05  # e.g., -0.05, 0, +0.05 for 3 families
            params_init = {
                'c1_0': params_avg.c1_0 + perturbation * 0.1,
                'c2_0': params_avg.c2_0 + perturbation * 0.02,
                'c3_0': params_avg.c3_0 + perturbation * 2.0,
                'dc1': params_avg.dc1,
                'dc2': params_avg.dc2,
                'dc3': params_avg.dc3,
            }
            initial_params_list.append(params_init)

    # Fit each family
    logger.info("Step 2: Fit individual families with perturbed initial conditions")
    families = {}
    diagnostics_all = {}

    for name, params_init in zip(family_names, initial_params_list):
        params, diag = fit_single_family_lsq(A, Z, params_init, family_name=name)
        families[name] = params
        diagnostics_all[name] = diag

    # Overall diagnostics
    overall = {
        'n_families': n_families,
        'n_training': len(df_train),
        'families': diagnostics_all,
    }

    return families, overall


def save_params_json(
    families: Dict[str, FamilyParams],
    diagnostics: Dict,
    output_file: str
):
    """
    Save fitted parameters to JSON file.

    Args:
        families: Dict of family_name -> FamilyParams
        diagnostics: Fit diagnostics
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'families': {name: params.to_dict() for name, params in families.items()},
        'diagnostics': diagnostics,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved parameters to: {output_file}")


def load_params_json(input_file: str) -> Dict[str, FamilyParams]:
    """
    Load fitted parameters from JSON file.

    Args:
        input_file: Path to JSON file created by save_params_json

    Returns:
        Dict of family_name -> FamilyParams
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    families = {
        name: FamilyParams.from_dict(params_dict)
        for name, params_dict in data['families'].items()
    }

    logger.info(f"Loaded {len(families)} families from {input_file}")
    return families


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fit harmonic family parameters to training data'
    )
    parser.add_argument(
        '--nuclides',
        required=True,
        help='Path to nuclides_all.parquet'
    )
    parser.add_argument(
        '--train_set',
        default='stable',
        choices=['stable', 'longlived', 'all'],
        help='Training set selector (default: stable)'
    )
    parser.add_argument(
        '--min_half_life_s',
        type=float,
        help='Minimum half-life for longlived mode (e.g., 86400 for 1 day)'
    )
    parser.add_argument(
        '--n_families',
        type=int,
        default=3,
        help='Number of families to fit (default: 3)'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Path to output JSON file (e.g., reports/fits/family_params_stable.json)'
    )

    args = parser.parse_args()

    # Load training data
    df_train = load_training_data(
        args.nuclides,
        train_set=args.train_set,
        min_half_life_s=args.min_half_life_s
    )

    # Fit families
    families, diagnostics = fit_multiple_families(df_train, n_families=args.n_families)

    # Validate fitted parameters
    logger.info("\n" + "="*80)
    logger.info("PARAMETER VALIDATION")
    logger.info("="*80)
    for name, params in families.items():
        validation = validate_params(params, A_range=(1, 300))
        if validation['valid']:
            logger.info(f"Family {name}: ✓ Valid")
        else:
            logger.info(f"Family {name}: ⚠ Warnings:")
            for w in validation['warnings']:
                logger.info(f"  - {w}")

    # Check dc3 universality
    from harmonic_model import dc3_comparison
    comparison = dc3_comparison(families)
    logger.info("\n" + "="*80)
    logger.info("dc3 UNIVERSALITY CHECK")
    logger.info("="*80)
    for name, dc3 in comparison['dc3_values'].items():
        logger.info(f"Family {name}: dc3 = {dc3:.4f}")
    logger.info(f"Mean: {comparison['dc3_mean']:.4f}")
    logger.info(f"Std: {comparison['dc3_std']:.5f}")
    logger.info(f"Relative std: {comparison['dc3_relative_std']:.2%}")

    if comparison['dc3_relative_std'] < 0.02:
        logger.info("✓ dc3 is universal (<2% variation)")
    else:
        logger.info("⚠ dc3 varies significantly across families")

    # Save results
    save_params_json(families, diagnostics, args.out)

    print("\n" + "="*80)
    print("FAMILY FITTING COMPLETE")
    print("="*80)
    print(f"Families: {list(families.keys())}")
    print(f"Training set: {args.train_set} ({diagnostics['n_training']} nuclides)")
    print(f"Output: {args.out}")
    print("="*80)


if __name__ == '__main__':
    main()
