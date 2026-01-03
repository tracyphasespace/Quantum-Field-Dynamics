#!/usr/bin/env python3
"""
Score all nuclides using fitted harmonic family parameters.

For each nuclide (A, Z):
  - Calculate epsilon (dissonance) for each family
  - Identify best-matching family (min epsilon)
  - Compute N_hat, N_best, Z_pred, residual

Outputs:
  - harmonic_scores.parquet with all scoring metrics

Implements EXPERIMENT_PLAN.md §2.2 (derived datasets).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict

from harmonic_model import FamilyParams, score_best_family, classify_by_epsilon
from fit_families import load_params_json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def score_all_nuclides(
    df_nuclides: pd.DataFrame,
    families: Dict[str, FamilyParams]
) -> pd.DataFrame:
    """
    Score all nuclides against fitted families.

    Args:
        df_nuclides: DataFrame with columns A, Z
        families: Dict of family_name -> FamilyParams

    Returns:
        DataFrame with scoring results (one row per nuclide)
    """
    n_nuclides = len(df_nuclides)
    logger.info(f"Scoring {n_nuclides} nuclides against {len(families)} families...")

    scores = []
    for idx, row in df_nuclides.iterrows():
        A = row['A']
        Z = row['Z']

        # Score against all families
        score = score_best_family(A, Z, families)

        # Add classification
        score['category'] = classify_by_epsilon(score['epsilon_best'])

        scores.append(score)

        # Progress logging
        if (idx + 1) % 500 == 0:
            logger.info(f"  Scored {idx + 1}/{n_nuclides} nuclides...")

    df_scores = pd.DataFrame(scores)
    logger.info(f"Scoring complete: {len(df_scores)} nuclides scored")

    return df_scores


def merge_with_nuclides(
    df_nuclides: pd.DataFrame,
    df_scores: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge scoring results with original nuclide data.

    Args:
        df_nuclides: Original nuclide data
        df_scores: Scoring results

    Returns:
        Merged DataFrame
    """
    # Reset index to ensure alignment
    df_nuclides_reset = df_nuclides.reset_index(drop=True)
    df_scores_reset = df_scores.reset_index(drop=True)

    # Concatenate (assume same order)
    df_merged = pd.concat([df_nuclides_reset, df_scores_reset], axis=1)

    return df_merged


def summarize_scores(df_scores: pd.DataFrame, families: Dict[str, FamilyParams]):
    """
    Print summary statistics for scoring results.

    Args:
        df_scores: DataFrame with scoring results
        families: Dict of fitted families
    """
    logger.info("\n" + "="*80)
    logger.info("HARMONIC SCORING SUMMARY")
    logger.info("="*80)

    # Overall epsilon statistics
    eps_best = df_scores['epsilon_best']
    logger.info(f"\nEpsilon (best family) statistics:")
    logger.info(f"  Mean:   {eps_best.mean():.4f}")
    logger.info(f"  Median: {eps_best.median():.4f}")
    logger.info(f"  Std:    {eps_best.std():.4f}")
    logger.info(f"  Min:    {eps_best.min():.4f}")
    logger.info(f"  Max:    {eps_best.max():.4f}")

    # Percentiles
    logger.info(f"  Percentiles:")
    for p in [5, 25, 50, 75, 95]:
        logger.info(f"    {p}%: {eps_best.quantile(p/100):.4f}")

    # Category distribution
    logger.info(f"\nCategory distribution:")
    category_counts = df_scores['category'].value_counts()
    for category in ['harmonic', 'near_harmonic', 'dissonant']:
        count = category_counts.get(category, 0)
        pct = 100 * count / len(df_scores)
        logger.info(f"  {category:15s}: {count:4d} ({pct:5.1f}%)")

    # Best family distribution
    logger.info(f"\nBest family distribution:")
    family_counts = df_scores['best_family'].value_counts()
    for family_name in sorted(families.keys()):
        count = family_counts.get(family_name, 0)
        pct = 100 * count / len(df_scores)
        logger.info(f"  Family {family_name}: {count:4d} ({pct:5.1f}%)")

    # Per-family epsilon statistics
    logger.info(f"\nPer-family epsilon statistics:")
    for family_name in sorted(families.keys()):
        eps_col = f'epsilon_{family_name}'
        if eps_col in df_scores.columns:
            eps = df_scores[eps_col]
            logger.info(f"  Family {family_name}:")
            logger.info(f"    Mean:   {eps.mean():.4f}")
            logger.info(f"    Median: {eps.median():.4f}")
            logger.info(f"    Std:    {eps.std():.4f}")


def score_harmonics(
    nuclides_file: str,
    params_file: str,
    output_file: str
) -> pd.DataFrame:
    """
    Score all nuclides and save results.

    Args:
        nuclides_file: Path to nuclides_all.parquet
        params_file: Path to fitted family parameters JSON
        output_file: Path to output parquet file

    Returns:
        DataFrame with merged nuclide data and scores
    """
    # Load nuclides
    logger.info(f"Loading nuclides from {nuclides_file}")
    df_nuclides = pd.read_parquet(nuclides_file)
    logger.info(f"Loaded {len(df_nuclides)} nuclides")

    # Load fitted parameters
    logger.info(f"Loading fitted parameters from {params_file}")
    families = load_params_json(params_file)
    logger.info(f"Loaded {len(families)} families: {list(families.keys())}")

    # Score all nuclides
    df_scores = score_all_nuclides(df_nuclides, families)

    # Merge with original data
    df_full = merge_with_nuclides(df_nuclides, df_scores)

    # Summary statistics
    summarize_scores(df_scores, families)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(output_path, index=False)
    logger.info(f"\nSaved scores to: {output_file}")
    logger.info(f"  Rows: {len(df_full)}")
    logger.info(f"  Columns: {len(df_full.columns)}")

    return df_full


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Score all nuclides using fitted harmonic family parameters'
    )
    parser.add_argument(
        '--nuclides',
        required=True,
        help='Path to nuclides_all.parquet'
    )
    parser.add_argument(
        '--params',
        required=True,
        help='Path to fitted family parameters JSON'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Path to output parquet file (e.g., data/derived/harmonic_scores.parquet)'
    )

    args = parser.parse_args()

    df = score_harmonics(args.nuclides, args.params, args.out)

    print("\n" + "="*80)
    print("HARMONIC SCORING COMPLETE")
    print("="*80)
    print(f"Nuclides scored: {len(df)}")
    print(f"Output: {args.out}")
    print(f"Columns: {list(df.columns)}")
    print("="*80)


if __name__ == '__main__':
    main()
