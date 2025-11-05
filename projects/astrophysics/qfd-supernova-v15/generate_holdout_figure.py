#!/usr/bin/env python3
"""
Generate Figure 8: Hold-out performance validation

Splits mock Stage 3 data into train/test sets and computes per-survey RMS
to demonstrate out-of-sample generalization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import the figure generation function
import sys
sys.path.append('scripts')
from make_publication_figures import figure8_holdout_performance


def compute_survey_rms(df):
    """Compute RMS by survey for a given dataset."""
    survey_stats = []

    for survey in df['survey'].unique():
        survey_df = df[df['survey'] == survey]
        rms_alpha = np.sqrt(np.mean(survey_df['residual_alpha']**2))

        survey_stats.append({
            'survey': survey,
            'N': len(survey_df),
            'rms_alpha': rms_alpha
        })

    return pd.DataFrame(survey_stats)


def main():
    # Load mock Stage 3 data
    data_path = Path("results/mock_stage3/stage3_results.csv")
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Total SNe: {len(df)}")
    print(f"Surveys: {df['survey'].unique()}")

    # Split into train/test (70/30) stratified by survey
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['survey']
    )

    print(f"\nTrain set: {len(train_df)} SNe")
    print(f"Test set: {len(test_df)} SNe")

    # Compute per-survey RMS for train and test
    print("\nComputing per-survey RMS statistics...")
    train_stats = compute_survey_rms(train_df)
    test_stats = compute_survey_rms(test_df)

    print("\nTrain RMS by survey:")
    print(train_stats)

    print("\nTest RMS by survey:")
    print(test_stats)

    # Generate Figure 8
    out_path = Path("results/mock_stage3/figures/fig8_holdout_performance.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating Figure 8: {out_path}")
    figure8_holdout_performance(train_stats, test_stats, str(out_path))

    print("\n✓ Figure 8 generation complete!")

    # Also save the statistics for reference
    train_stats.to_csv("results/mock_stage3/reports/train_rms_by_survey.csv", index=False)
    test_stats.to_csv("results/mock_stage3/reports/test_rms_by_survey.csv", index=False)
    print("✓ Saved train/test statistics to reports/")


if __name__ == "__main__":
    main()
