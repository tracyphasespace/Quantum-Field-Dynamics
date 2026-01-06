#!/usr/bin/env python3
"""
Generate null models and candidate universe for Experiment 1.

Creates:
  1. Candidate universe: All plausible (A, Z) combinations
  2. Smooth baseline: Polynomial/spline fit to valley of stability
  3. Baseline scores: Distance from smooth valley for comparison

Implements EXPERIMENT_PLAN.md §4.2 (candidate generation).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_candidates_for_A(
    A: int,
    valley_Z: Optional[float] = None,
    valley_width: float = 0.25,
    use_physics_bounds: bool = True
) -> np.ndarray:
    """
    Generate candidate Z values for given A.

    Args:
        A: Mass number
        valley_Z: Center of valley of stability for this A (if known)
        valley_width: Width of physics band (fraction of A)
        use_physics_bounds: If True, restrict to valley band

    Returns:
        Array of candidate Z values
    """
    # Hard bounds: 1 ≤ Z ≤ A-1
    Z_min_hard = 1
    Z_max_hard = A - 1

    if not use_physics_bounds or valley_Z is None or np.isnan(valley_Z):
        # Full enumeration (if no physics bounds or valley_Z is invalid)
        Z_candidates = np.arange(Z_min_hard, Z_max_hard + 1)
    else:
        # Restrict to valley band: |Z - valley_Z| < valley_width * A
        band_width = valley_width * A
        Z_min_physics = max(Z_min_hard, int(np.floor(valley_Z - band_width)))
        Z_max_physics = min(Z_max_hard, int(np.ceil(valley_Z + band_width)))
        Z_candidates = np.arange(Z_min_physics, Z_max_physics + 1)

    return Z_candidates


def fit_smooth_valley(df_observed: pd.DataFrame, method: str = 'spline') -> callable:
    """
    Fit smooth curve to valley of stability.

    Args:
        df_observed: DataFrame with observed nuclides (A, Z)
        method: 'spline', 'polynomial', or 'sqrt'
            - 'spline': Cubic spline (flexible, smooth)
            - 'polynomial': Polynomial fit (degree 3)
            - 'sqrt': Physics-motivated Z ~ A/(2 + c·A^(1/3))

    Returns:
        Function Z_smooth(A) that predicts Z from A
    """
    A = df_observed['A'].values
    Z = df_observed['Z'].values

    logger.info(f"Fitting smooth valley to {len(df_observed)} observed nuclides (method={method})")

    if method == 'spline':
        # Cubic spline (smoothing spline with automatic lambda selection)
        # Use stable nuclides if available for better valley estimate
        if 'is_stable' in df_observed.columns:
            stable = df_observed[df_observed['is_stable']]
            if len(stable) > 10:
                # Take median Z for each A (to handle multiple stable isotopes)
                A_Z_stable = stable.groupby('A')['Z'].median().reset_index()
                A_fit = A_Z_stable['A'].values
                Z_fit = A_Z_stable['Z'].values
                logger.info(f"  Using {len(stable)} stable nuclides ({len(A_Z_stable)} unique A values)")
            else:
                # Take median Z for each A from all nuclides
                A_Z_all = df_observed.groupby('A')['Z'].median().reset_index()
                A_fit = A_Z_all['A'].values
                Z_fit = A_Z_all['Z'].values
        else:
            # Take median Z for each A
            A_Z_all = df_observed.groupby('A')['Z'].median().reset_index()
            A_fit = A_Z_all['A'].values
            Z_fit = A_Z_all['Z'].values

        # Already sorted by A from groupby, but ensure it
        sort_idx = np.argsort(A_fit)
        A_fit = A_fit[sort_idx]
        Z_fit = Z_fit[sort_idx]

        # Fit spline with extrapolation handling
        spline = UnivariateSpline(A_fit, Z_fit, s=len(A_fit), k=3, ext=0)  # ext=0 for extrapolation

        # Wrapper to handle edge cases
        A_min_fit = A_fit.min()
        A_max_fit = A_fit.max()
        def Z_smooth(A_val):
            # Clip to valid range to avoid extrapolation issues
            A_clipped = np.clip(A_val, A_min_fit, A_max_fit)
            return float(spline(A_clipped))

        logger.info(f"  Spline fit complete (knots: {len(spline.get_knots())})")
        logger.info(f"  Valid A range: [{A_min_fit}, {A_max_fit}]")

    elif method == 'polynomial':
        # Polynomial fit (degree 3)
        degree = 3
        coeffs = np.polyfit(A, Z, degree)
        poly = np.poly1d(coeffs)
        Z_smooth = poly
        logger.info(f"  Polynomial fit complete (degree {degree})")
        logger.info(f"    Coefficients: {coeffs}")

    elif method == 'sqrt':
        # Physics-motivated: Z ~ A/(2 + c·A^(1/3))
        # This captures Coulomb repulsion vs strong force balance
        def Z_theory(A, c):
            return A / (2.0 + c * A**(1/3))

        # Fit parameter c
        popt, _ = curve_fit(Z_theory, A, Z, p0=[0.015])
        c_fit = popt[0]
        Z_smooth = lambda A_val: Z_theory(A_val, c_fit)
        logger.info(f"  Physics-motivated fit complete")
        logger.info(f"    c = {c_fit:.6f}")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Test fit quality on observed data
    Z_pred = np.array([Z_smooth(a) for a in A])
    residuals = Z - Z_pred
    rmse = np.sqrt(np.mean(residuals**2))
    logger.info(f"  Fit RMSE: {rmse:.3f} protons")

    return Z_smooth


def generate_candidate_universe(
    df_observed: pd.DataFrame,
    A_range: Optional[Tuple[int, int]] = None,
    use_physics_bounds: bool = True,
    valley_method: str = 'spline',
    valley_width: float = 0.25
) -> pd.DataFrame:
    """
    Generate candidate universe for Experiment 1.

    For each A, enumerate all plausible Z values (either full range or physics band).

    Args:
        df_observed: DataFrame with observed nuclides
        A_range: (A_min, A_max) to generate candidates for (default: from observed)
        use_physics_bounds: If True, restrict to valley band
        valley_method: Method for fitting smooth valley ('spline', 'polynomial', 'sqrt')
        valley_width: Width of physics band (fraction of A)

    Returns:
        DataFrame with candidate (A, Z) pairs
    """
    if A_range is None:
        A_min = df_observed['A'].min()
        A_max = df_observed['A'].max()
    else:
        A_min, A_max = A_range

    logger.info(f"Generating candidate universe for A ∈ [{A_min}, {A_max}]")
    logger.info(f"  Physics bounds: {use_physics_bounds}")
    if use_physics_bounds:
        logger.info(f"  Valley method: {valley_method}")
        logger.info(f"  Valley width: ±{valley_width:.2f}·A")

    # Fit smooth valley
    if use_physics_bounds:
        Z_smooth = fit_smooth_valley(df_observed, method=valley_method)
    else:
        Z_smooth = None

    # Generate candidates for each A
    candidates = []
    n_candidates_total = 0

    for A in range(A_min, A_max + 1):
        # Get valley center for this A
        valley_Z = Z_smooth(A) if Z_smooth is not None else None

        # Generate candidate Z values
        Z_candidates = generate_candidates_for_A(A, valley_Z, valley_width, use_physics_bounds)

        # Add to list
        for Z in Z_candidates:
            candidates.append({'A': int(A), 'Z': int(Z)})

        n_candidates_total += len(Z_candidates)

        # Progress logging
        if A % 50 == 0:
            logger.info(f"  Generated candidates for A={A} ({len(Z_candidates)} candidates)")

    df_candidates = pd.DataFrame(candidates)
    df_candidates['N'] = df_candidates['A'] - df_candidates['Z']

    logger.info(f"Generated {len(df_candidates)} total candidates")
    logger.info(f"  Average candidates per A: {len(df_candidates) / (A_max - A_min + 1):.1f}")

    return df_candidates


def compute_smooth_baseline_scores(
    df: pd.DataFrame,
    Z_smooth: callable
) -> pd.DataFrame:
    """
    Compute distance from smooth valley baseline.

    For baseline comparison in Experiment 1.

    Args:
        df: DataFrame with columns A, Z
        Z_smooth: Function Z_smooth(A) from fit_smooth_valley

    Returns:
        DataFrame with added columns:
            - Z_smooth: Smooth valley prediction
            - residual_smooth: Z - Z_smooth
            - distance_smooth: |Z - Z_smooth| (absolute distance)
    """
    df = df.copy()

    # Predict Z from smooth valley (vectorized)
    Z_smooth_vals = np.array([Z_smooth(a) for a in df['A'].values])
    df['Z_smooth'] = Z_smooth_vals

    # Compute residuals and distance
    df['residual_smooth'] = df['Z'] - df['Z_smooth']
    df['distance_smooth'] = np.abs(df['residual_smooth'])

    return df


def flag_observed_nuclides(
    df_candidates: pd.DataFrame,
    df_observed: pd.DataFrame
) -> pd.DataFrame:
    """
    Flag which candidates are actually observed.

    Args:
        df_candidates: Candidate universe
        df_observed: Observed nuclides

    Returns:
        DataFrame with added column 'is_observed' (bool)
    """
    df_candidates = df_candidates.copy()

    # Create set of observed (A, Z) pairs
    observed_set = set(zip(df_observed['A'], df_observed['Z']))

    # Flag candidates
    df_candidates['is_observed'] = df_candidates.apply(
        lambda row: (row['A'], row['Z']) in observed_set,
        axis=1
    )

    n_observed = df_candidates['is_observed'].sum()
    n_null = (~df_candidates['is_observed']).sum()

    logger.info(f"Flagged candidates:")
    logger.info(f"  Observed: {n_observed}")
    logger.info(f"  Null: {n_null}")

    return df_candidates


def create_null_model_dataset(
    nuclides_file: str,
    output_file: str,
    use_physics_bounds: bool = True,
    valley_method: str = 'spline',
    valley_width: float = 0.25,
    A_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Create complete null model dataset.

    Args:
        nuclides_file: Path to nuclides_all.parquet (observed nuclides)
        output_file: Path to output parquet file
        use_physics_bounds: If True, restrict to valley band
        valley_method: Method for smooth valley ('spline', 'polynomial', 'sqrt')
        valley_width: Width of physics band (fraction of A)
        A_range: (A_min, A_max) for candidates (default: from observed)

    Returns:
        DataFrame with candidates, flagged by is_observed
    """
    # Load observed nuclides
    logger.info(f"Loading observed nuclides from {nuclides_file}")
    df_observed = pd.read_parquet(nuclides_file)
    logger.info(f"Loaded {len(df_observed)} observed nuclides")

    # Generate candidate universe
    df_candidates = generate_candidate_universe(
        df_observed,
        A_range=A_range,
        use_physics_bounds=use_physics_bounds,
        valley_method=valley_method,
        valley_width=valley_width
    )

    # Flag observed nuclides
    df_candidates = flag_observed_nuclides(df_candidates, df_observed)

    # Compute smooth baseline scores
    logger.info("Computing smooth baseline scores...")
    Z_smooth = fit_smooth_valley(df_observed, method=valley_method)
    df_candidates = compute_smooth_baseline_scores(df_candidates, Z_smooth)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_candidates.to_parquet(output_path, index=False)
    logger.info(f"Saved candidates to: {output_file}")
    logger.info(f"  Rows: {len(df_candidates)}")
    logger.info(f"  Columns: {list(df_candidates.columns)}")

    return df_candidates


def summarize_null_model(df_candidates: pd.DataFrame):
    """
    Print summary statistics for null model.

    Args:
        df_candidates: Candidate universe with is_observed flag
    """
    logger.info("\n" + "="*80)
    logger.info("NULL MODEL SUMMARY")
    logger.info("="*80)

    # Overall counts
    n_total = len(df_candidates)
    n_observed = df_candidates['is_observed'].sum()
    n_null = (~df_candidates['is_observed']).sum()

    logger.info(f"\nTotal candidates: {n_total}")
    logger.info(f"  Observed: {n_observed} ({100*n_observed/n_total:.1f}%)")
    logger.info(f"  Null: {n_null} ({100*n_null/n_total:.1f}%)")

    # A range
    A_min = df_candidates['A'].min()
    A_max = df_candidates['A'].max()
    logger.info(f"\nA range: [{A_min}, {A_max}]")

    # Candidates per A
    candidates_per_A = df_candidates.groupby('A').size()
    logger.info(f"\nCandidates per A:")
    logger.info(f"  Mean: {candidates_per_A.mean():.1f}")
    logger.info(f"  Median: {candidates_per_A.median():.1f}")
    logger.info(f"  Min: {candidates_per_A.min()}")
    logger.info(f"  Max: {candidates_per_A.max()}")

    # Smooth baseline statistics
    if 'distance_smooth' in df_candidates.columns:
        logger.info(f"\nSmooth baseline (distance from valley):")

        # Observed nuclides
        dist_obs = df_candidates[df_candidates['is_observed']]['distance_smooth']
        logger.info(f"  Observed:")
        logger.info(f"    Mean: {dist_obs.mean():.3f}")
        logger.info(f"    Median: {dist_obs.median():.3f}")
        logger.info(f"    Std: {dist_obs.std():.3f}")

        # Null candidates
        dist_null = df_candidates[~df_candidates['is_observed']]['distance_smooth']
        logger.info(f"  Null:")
        logger.info(f"    Mean: {dist_null.mean():.3f}")
        logger.info(f"    Median: {dist_null.median():.3f}")
        logger.info(f"    Std: {dist_null.std():.3f}")

        # Effect size
        mean_diff = dist_obs.mean() - dist_null.mean()
        logger.info(f"  Difference (obs - null): {mean_diff:.3f}")
        if mean_diff < 0:
            logger.info(f"    ✓ Observed nuclides closer to valley (baseline works!)")
        else:
            logger.info(f"    ✗ Observed nuclides farther from valley (unexpected)")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate null models and candidate universe for Experiment 1'
    )
    parser.add_argument(
        '--nuclides',
        required=True,
        help='Path to nuclides_all.parquet (observed nuclides)'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Path to output parquet file (e.g., data/derived/candidates_by_A.parquet)'
    )
    parser.add_argument(
        '--physics_bounds',
        action='store_true',
        default=True,
        help='Use physics bounds (restrict to valley band)'
    )
    parser.add_argument(
        '--no_physics_bounds',
        action='store_false',
        dest='physics_bounds',
        help='Disable physics bounds (enumerate all Z for each A)'
    )
    parser.add_argument(
        '--valley_method',
        default='spline',
        choices=['spline', 'polynomial', 'sqrt'],
        help='Method for fitting smooth valley (default: spline)'
    )
    parser.add_argument(
        '--valley_width',
        type=float,
        default=0.25,
        help='Width of physics band (fraction of A, default: 0.25)'
    )
    parser.add_argument(
        '--A_min',
        type=int,
        help='Minimum A for candidates (default: from observed)'
    )
    parser.add_argument(
        '--A_max',
        type=int,
        help='Maximum A for candidates (default: from observed)'
    )

    args = parser.parse_args()

    # Parse A range
    A_range = None
    if args.A_min is not None and args.A_max is not None:
        A_range = (args.A_min, args.A_max)

    # Generate null model dataset
    df = create_null_model_dataset(
        args.nuclides,
        args.out,
        use_physics_bounds=args.physics_bounds,
        valley_method=args.valley_method,
        valley_width=args.valley_width,
        A_range=A_range
    )

    # Summary
    summarize_null_model(df)

    print("\n" + "="*80)
    print("NULL MODEL GENERATION COMPLETE")
    print("="*80)
    print(f"Candidates generated: {len(df)}")
    print(f"Output: {args.out}")
    print("="*80)


if __name__ == '__main__':
    main()
