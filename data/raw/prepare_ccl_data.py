#!/usr/bin/env python3
"""
Prepare AME2020 data for Core Compression Law (CCL) fitting.

The Core Compression Law predicts Z (charge) from A (mass):
    Q(A) = c1·A^(2/3) + c2·A

Input: ame2020.csv
Output: ame2020_ccl.csv (A, Z ready for Grand Solver)
"""

import pandas as pd
import numpy as np

def prepare_ccl_data(input_path='data/raw/ame2020.csv',
                     output_path='data/raw/ame2020_ccl.csv'):
    """
    Prepare AME2020 for Core Compression Law fitting.

    Target: Predict Z (charge) from A (mass)
    Observable: Which (A, Z) combinations exist in nature

    Filters:
    - Remove estimated values (unreliable)
    - Keep all isotopes (both stable and unstable)
    - Require A >= 1 (include all nuclides)
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} nuclides")

    # Initial filtering
    initial_count = len(df)

    # Remove estimated values
    df = df[df['is_estimated'] == False]
    print(f"After removing estimated: {len(df)} ({initial_count - len(df)} removed)")

    # Remove entries with non-finite A or Z
    df = df[np.isfinite(df['A']) & np.isfinite(df['Z'])]
    print(f"After removing non-finite: {len(df)}")

    # For CCL, we want minimal uncertainty on Z (it's an integer)
    # Use A uncertainty as a proxy for measurement quality
    df['sigma'] = 0.1  # Small constant uncertainty (Z is discrete)

    # Select columns for Grand Solver
    output_df = df[['A', 'Z', 'N', 'element']].copy()
    output_df['target'] = df['Z']  # Target is Z (charge number)
    output_df['sigma'] = df['sigma']

    # Sort by A
    output_df = output_df.sort_values('A').reset_index(drop=True)

    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ CCL dataset saved to {output_path}")
    print(f"   Final count: {len(output_df)} nuclides")

    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Mass number (A): {output_df['A'].min():.0f} - {output_df['A'].max():.0f}")
    print(f"  Charge (Z): {output_df['Z'].min():.0f} - {output_df['Z'].max():.0f}")
    print(f"  Unique A values: {output_df['A'].nunique()}")
    print(f"  Average isotopes per A: {len(output_df) / output_df['A'].nunique():.1f}")

    # Show element distribution
    print(f"\nElement distribution:")
    element_counts = output_df['element'].value_counts().head(10)
    for elem, count in element_counts.items():
        print(f"  {elem}: {count} isotopes")

    return output_df

if __name__ == '__main__':
    df = prepare_ccl_data()

    # Quick validation
    print("\n✅ Validation checks:")
    print(f"   All A finite: {np.all(np.isfinite(df['A']))}")
    print(f"   All Z finite: {np.all(np.isfinite(df['Z']))}")
    print(f"   All sigma > 0: {np.all(df['sigma'] > 0)}")

    # Show sample
    print(f"\nSample data (first 10 rows):")
    print(df[['A', 'Z', 'element', 'target', 'sigma']].head(10))
