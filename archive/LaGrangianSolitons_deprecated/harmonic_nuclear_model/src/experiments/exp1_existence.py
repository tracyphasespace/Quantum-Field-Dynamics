#!/usr/bin/env python3
"""
Experiment 1: Out-of-sample existence prediction.

Tests hypothesis H1: Observed nuclides have significantly lower ε than null
candidates at the same A.

This is the PRIMARY FALSIFIER of the harmonic model.

Implements EXPERIMENT_PLAN.md §4.

Metrics:
  1. Mean ε separation (observed vs null)
  2. AUC (existence classifier)
  3. Calibration curve P(exists | ε)
  4. Permutation test

Baselines:
  - Smooth valley (distance from valley of stability)
  - Random (uniform score)

Pass criterion:
  AUC_ε > AUC_smooth + 0.05 and permutation p < 1e-4
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harmonic_model import score_best_family, classify_by_epsilon
from fit_families import load_params_json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def score_candidates_with_harmonic_model(
    df_candidates: pd.DataFrame,
    families: Dict
) -> pd.DataFrame:
    """
    Score all candidates with harmonic model.

    Args:
        df_candidates: Candidate universe (A, Z, is_observed)
        families: Fitted family parameters

    Returns:
        DataFrame with added harmonic scoring columns
    """
    n_total = len(df_candidates)
    logger.info(f"Scoring {n_total} candidates with harmonic model...")

    scores = []
    for idx, row in df_candidates.iterrows():
        score = score_best_family(row['A'], row['Z'], families)
        score['category'] = classify_by_epsilon(score['epsilon_best'])
        scores.append(score)

        # Progress logging
        if (idx + 1) % 5000 == 0:
            logger.info(f"  Scored {idx + 1}/{n_total} candidates...")

    df_scores = pd.DataFrame(scores)

    # Merge with candidates
    df_full = pd.concat([df_candidates.reset_index(drop=True), df_scores], axis=1)

    logger.info(f"Scoring complete: {len(df_full)} candidates scored")
    return df_full


def compute_mean_separation(
    df: pd.DataFrame,
    score_col: str = 'epsilon_best'
) -> Dict:
    """
    Compute mean ε separation between observed and null.

    Args:
        df: DataFrame with is_observed flag and score column
        score_col: Column name for score (default: epsilon_best)

    Returns:
        Dictionary with statistics
    """
    obs = df[df['is_observed']][score_col].values
    null = df[~df['is_observed']][score_col].values

    mean_obs = np.mean(obs)
    mean_null = np.mean(null)
    separation = mean_obs - mean_null

    median_obs = np.median(obs)
    median_null = np.median(null)

    # Bootstrap confidence intervals for mean separation
    n_bootstrap = 1000
    bootstrap_seps = []
    for _ in range(n_bootstrap):
        # Resample by A to preserve structure
        A_values = df['A'].unique()
        A_resample = np.random.choice(A_values, size=len(A_values), replace=True)

        obs_boot = []
        null_boot = []
        for a in A_resample:
            df_a = df[df['A'] == a]
            obs_boot.extend(df_a[df_a['is_observed']][score_col].values)
            null_boot.extend(df_a[~df_a['is_observed']][score_col].values)

        if len(obs_boot) > 0 and len(null_boot) > 0:
            bootstrap_seps.append(np.mean(obs_boot) - np.mean(null_boot))

    ci_lower = np.percentile(bootstrap_seps, 2.5)
    ci_upper = np.percentile(bootstrap_seps, 97.5)

    return {
        'mean_observed': float(mean_obs),
        'mean_null': float(mean_null),
        'separation': float(separation),
        'median_observed': float(median_obs),
        'median_null': float(median_null),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_observed': int(len(obs)),
        'n_null': int(len(null)),
    }


def compute_auc(
    df: pd.DataFrame,
    score_col: str = 'epsilon_best',
    reverse_score: bool = False
) -> Dict:
    """
    Compute AUC for existence classifier.

    Args:
        df: DataFrame with is_observed flag and score column
        score_col: Column name for score
        reverse_score: If True, use -score (for ε where lower is better)

    Returns:
        Dictionary with AUC and related metrics
    """
    y_true = df['is_observed'].astype(int).values
    scores = df[score_col].values

    if reverse_score:
        scores = -scores  # Lower ε should predict higher existence probability

    # Compute AUC
    auc = roc_auc_score(y_true, scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    return {
        'auc': float(auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }


def compute_calibration_curve(
    df: pd.DataFrame,
    score_col: str = 'epsilon_best',
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration curve P(exists | ε-bin).

    Args:
        df: DataFrame with is_observed flag and score column
        score_col: Column name for score
        n_bins: Number of bins

    Returns:
        Dictionary with calibration data
    """
    # Bin by epsilon
    df['epsilon_bin'] = pd.cut(df[score_col], bins=n_bins)

    # Compute P(exists | bin)
    calibration = df.groupby('epsilon_bin', observed=True).agg({
        'is_observed': ['sum', 'count', 'mean']
    }).reset_index()

    calibration.columns = ['epsilon_bin', 'n_observed', 'n_total', 'p_exists']

    # Extract bin centers
    bin_centers = calibration['epsilon_bin'].apply(lambda x: x.mid).values

    return {
        'bin_centers': bin_centers.tolist(),
        'p_exists': calibration['p_exists'].values.tolist(),
        'n_observed': calibration['n_observed'].values.tolist(),
        'n_total': calibration['n_total'].values.tolist(),
    }


def permutation_test(
    df: pd.DataFrame,
    score_col: str = 'epsilon_best',
    n_permutations: int = 1000
) -> Dict:
    """
    Permutation test: shuffle is_observed within each A-bin.

    Tests if observed separation is significant vs random labeling.

    Args:
        df: DataFrame with is_observed flag and score column
        score_col: Column name for score
        n_permutations: Number of permutations

    Returns:
        Dictionary with p-value and permutation distribution
    """
    logger.info(f"Running permutation test ({n_permutations} permutations)...")

    # Observed separation
    obs_sep_true = compute_mean_separation(df, score_col)['separation']

    # Permutation distribution
    perm_seps = []
    for i in range(n_permutations):
        # Shuffle is_observed within each A
        df_perm = df.copy()
        df_perm['is_observed'] = df_perm.groupby('A')['is_observed'].transform(
            lambda x: np.random.permutation(x.values)
        )

        # Compute separation for permuted data
        sep = compute_mean_separation(df_perm, score_col)['separation']
        perm_seps.append(sep)

        if (i + 1) % 200 == 0:
            logger.info(f"  Completed {i + 1}/{n_permutations} permutations...")

    perm_seps = np.array(perm_seps)

    # P-value: fraction of permutations with |separation| >= |observed|
    p_value = np.mean(np.abs(perm_seps) >= np.abs(obs_sep_true))

    logger.info(f"Permutation test complete: p = {p_value:.4e}")

    return {
        'observed_separation': float(obs_sep_true),
        'permutation_mean': float(np.mean(perm_seps)),
        'permutation_std': float(np.std(perm_seps)),
        'p_value': float(p_value),
        'n_permutations': int(n_permutations),
    }


def compare_to_baselines(
    df: pd.DataFrame
) -> Dict:
    """
    Compare harmonic model (ε) to baseline models.

    Baselines:
      1. Smooth valley (distance_smooth)
      2. Random (uniform score)

    Args:
        df: DataFrame with epsilon_best, distance_smooth, is_observed

    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing to baseline models...")

    results = {}

    # Harmonic model (ε, reverse score since lower is better)
    results['harmonic'] = {
        'separation': compute_mean_separation(df, 'epsilon_best'),
        'auc': compute_auc(df, 'epsilon_best', reverse_score=True),
    }

    # Smooth baseline (distance from valley, reverse score)
    if 'distance_smooth' in df.columns:
        results['smooth'] = {
            'separation': compute_mean_separation(df, 'distance_smooth'),
            'auc': compute_auc(df, 'distance_smooth', reverse_score=True),
        }
    else:
        logger.warning("distance_smooth not found, skipping smooth baseline")

    # Random baseline
    df['random_score'] = np.random.uniform(0, 1, size=len(df))
    results['random'] = {
        'separation': compute_mean_separation(df, 'random_score'),
        'auc': compute_auc(df, 'random_score', reverse_score=False),
    }

    # Summary comparison
    logger.info("\nBaseline comparison:")
    for name, res in results.items():
        logger.info(f"  {name:10s}: AUC={res['auc']['auc']:.4f}, "
                   f"separation={res['separation']['separation']:+.4f}")

    return results


def ks_test_by_A_bin(
    df: pd.DataFrame,
    score_col: str = 'epsilon_best',
    A_bin_size: int = 20
) -> Dict:
    """
    Kolmogorov-Smirnov test within A-bins.

    Tests if distributions differ after controlling for A.

    Args:
        df: DataFrame with A, is_observed, score_col
        score_col: Column name for score
        A_bin_size: Width of A bins

    Returns:
        Dictionary with KS test results per bin
    """
    logger.info("Running KS test by A-bin...")

    # Create A bins
    A_min = df['A'].min()
    A_max = df['A'].max()
    A_bins = np.arange(A_min, A_max + A_bin_size, A_bin_size)

    df['A_bin'] = pd.cut(df['A'], bins=A_bins)

    results = []
    for bin_label, bin_data in df.groupby('A_bin', observed=True):
        obs = bin_data[bin_data['is_observed']][score_col].values
        null = bin_data[~bin_data['is_observed']][score_col].values

        if len(obs) > 0 and len(null) > 0:
            ks_stat, ks_pval = stats.ks_2samp(obs, null)
            results.append({
                'A_bin': str(bin_label),
                'A_mid': bin_label.mid,
                'n_observed': len(obs),
                'n_null': len(null),
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_pval),
                'mean_obs': float(np.mean(obs)),
                'mean_null': float(np.mean(null)),
            })

    df_ks = pd.DataFrame(results)

    # Overall summary
    n_significant = (df_ks['ks_p_value'] < 0.05).sum()
    logger.info(f"  {n_significant}/{len(df_ks)} A-bins have significant KS test (p < 0.05)")

    return {
        'per_bin': df_ks.to_dict('records'),
        'n_bins': len(df_ks),
        'n_significant_05': int(n_significant),
    }


def run_experiment_1(
    candidates_file: str,
    params_file: str,
    output_dir: str
) -> Dict:
    """
    Run complete Experiment 1.

    Args:
        candidates_file: Path to candidates_by_A.parquet
        params_file: Path to fitted family parameters JSON
        output_dir: Directory for output files

    Returns:
        Dictionary with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading candidates from {candidates_file}")
    df_candidates = pd.read_parquet(candidates_file)
    logger.info(f"Loaded {len(df_candidates)} candidates")
    logger.info(f"  Observed: {df_candidates['is_observed'].sum()}")
    logger.info(f"  Null: {(~df_candidates['is_observed']).sum()}")

    # Load fitted families
    logger.info(f"Loading fitted parameters from {params_file}")
    families = load_params_json(params_file)

    # Score candidates with harmonic model
    df_scored = score_candidates_with_harmonic_model(df_candidates, families)

    # Save scored candidates
    scored_file = output_path / 'candidates_scored.parquet'
    df_scored.to_parquet(scored_file, index=False)
    logger.info(f"Saved scored candidates to {scored_file}")

    # Compute all metrics
    results = {}

    logger.info("\n" + "="*80)
    logger.info("METRIC 1: Mean ε Separation")
    logger.info("="*80)
    results['mean_separation'] = compute_mean_separation(df_scored, 'epsilon_best')
    sep = results['mean_separation']
    logger.info(f"Observed: {sep['mean_observed']:.4f}")
    logger.info(f"Null:     {sep['mean_null']:.4f}")
    logger.info(f"Separation: {sep['separation']:+.4f} [{sep['ci_lower']:+.4f}, {sep['ci_upper']:+.4f}]")

    logger.info("\n" + "="*80)
    logger.info("METRIC 2: AUC (Existence Classifier)")
    logger.info("="*80)
    results['baseline_comparison'] = compare_to_baselines(df_scored)
    for name, res in results['baseline_comparison'].items():
        logger.info(f"{name:10s}: AUC = {res['auc']['auc']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("METRIC 3: Calibration Curve")
    logger.info("="*80)
    results['calibration'] = compute_calibration_curve(df_scored, 'epsilon_best', n_bins=10)
    logger.info(f"Computed calibration curve with {len(results['calibration']['bin_centers'])} bins")

    logger.info("\n" + "="*80)
    logger.info("METRIC 4: Permutation Test")
    logger.info("="*80)
    results['permutation_test'] = permutation_test(df_scored, 'epsilon_best', n_permutations=1000)
    perm = results['permutation_test']
    logger.info(f"Observed separation: {perm['observed_separation']:+.4f}")
    logger.info(f"Permutation mean:    {perm['permutation_mean']:+.4f}")
    logger.info(f"P-value:             {perm['p_value']:.4e}")

    logger.info("\n" + "="*80)
    logger.info("ADDITIONAL: KS Test by A-bin")
    logger.info("="*80)
    results['ks_by_A'] = ks_test_by_A_bin(df_scored, 'epsilon_best', A_bin_size=20)

    # Pass/Fail determination
    logger.info("\n" + "="*80)
    logger.info("PASS/FAIL DETERMINATION")
    logger.info("="*80)

    auc_harmonic = results['baseline_comparison']['harmonic']['auc']['auc']
    auc_smooth = results['baseline_comparison']['smooth']['auc']['auc']
    p_value = results['permutation_test']['p_value']

    # Criterion: AUC_ε > AUC_smooth + 0.05 and p < 1e-4
    auc_exceeds = auc_harmonic > auc_smooth + 0.05
    p_significant = p_value < 1e-4

    results['pass_fail'] = {
        'auc_harmonic': float(auc_harmonic),
        'auc_smooth': float(auc_smooth),
        'auc_difference': float(auc_harmonic - auc_smooth),
        'auc_threshold': 0.05,
        'auc_exceeds': bool(auc_exceeds),
        'p_value': float(p_value),
        'p_threshold': 1e-4,
        'p_significant': bool(p_significant),
        'overall_pass': bool(auc_exceeds and p_significant),
    }

    logger.info(f"AUC (harmonic): {auc_harmonic:.4f}")
    logger.info(f"AUC (smooth):   {auc_smooth:.4f}")
    logger.info(f"Difference:     {auc_harmonic - auc_smooth:+.4f} (threshold: +0.05)")
    logger.info(f"AUC exceeds threshold: {auc_exceeds}")
    logger.info(f"")
    logger.info(f"Permutation p-value: {p_value:.4e} (threshold: 1e-4)")
    logger.info(f"P-value significant: {p_significant}")
    logger.info(f"")
    if results['pass_fail']['overall_pass']:
        logger.info("✓ EXPERIMENT 1 PASSES")
    else:
        logger.info("✗ EXPERIMENT 1 FAILS")

    # Save results
    results_file = output_path / 'exp1_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nSaved results to {results_file}")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Experiment 1: Out-of-sample existence prediction'
    )
    parser.add_argument(
        '--candidates',
        required=True,
        help='Path to candidates_by_A.parquet'
    )
    parser.add_argument(
        '--params',
        required=True,
        help='Path to fitted family parameters JSON'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output directory for results'
    )

    args = parser.parse_args()

    results = run_experiment_1(args.candidates, args.params, args.out)

    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print("="*80)
    if results['pass_fail']['overall_pass']:
        print("RESULT: ✓ PASS")
    else:
        print("RESULT: ✗ FAIL")
    print(f"Output directory: {args.out}")
    print("="*80)


if __name__ == '__main__':
    main()
