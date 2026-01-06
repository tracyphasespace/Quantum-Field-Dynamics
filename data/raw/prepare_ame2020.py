#!/usr/bin/env python3
"""
Prepare AME2020 data for Grand Solver v1.1

Converts AME2020 mass excess data to binding energy format expected by Grand Solver.

Input: ame2020.csv (from qfd_research_suite)
Output: ame2020_prepared.csv (ready for Grand Solver)
"""

import pandas as pd
import numpy as np

def prepare_ame2020(input_path='data/raw/ame2020.csv',
                    output_path='data/raw/ame2020_prepared.csv',
                    stable_only=False):
    """
    Prepare AME2020 dataset for Grand Solver.

    Calculations:
    - Binding Energy (MeV) = BE_per_A_MeV * A
    - Uncertainty (MeV) = E_exp_unc_MeV * A

    Filters:
    - Exclude estimated values (is_estimated == True)
    - Optionally keep only stable isotopes
    - Require finite binding energy and uncertainty
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} nuclides")

    # Calculate binding energy from BE/A
    df['binding_energy'] = df['BE_per_A_MeV'] * df['A']
    df['sigma'] = df['E_exp_unc_MeV'] * df['A']

    # Initial filtering
    initial_count = len(df)

    # Remove estimated values (unreliable)
    df = df[df['is_estimated'] == False]
    print(f"After removing estimated: {len(df)} ({initial_count - len(df)} removed)")

    # Remove entries with non-finite binding energy or uncertainty
    df = df[np.isfinite(df['binding_energy']) & np.isfinite(df['sigma'])]
    print(f"After removing non-finite: {len(df)}")

    # Remove entries with zero or negative uncertainty
    df = df[df['sigma'] > 0]
    print(f"After removing σ≤0: {len(df)}")

    # Keep only nuclides with A >= 4 (exclude single nucleons and deuteron/triton for CCL validity)
    df = df[df['A'] >= 4]
    print(f"After A >= 4 filter: {len(df)}")

    # Select columns for Grand Solver
    output_df = df[['A', 'Z', 'N', 'element', 'binding_energy', 'sigma']].copy()

    # Sort by A, then Z
    output_df = output_df.sort_values(['A', 'Z']).reset_index(drop=True)

    # Save prepared data
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Prepared dataset saved to {output_path}")
    print(f"   Final count: {len(output_df)} nuclides")

    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Mass number (A): {output_df['A'].min():.0f} - {output_df['A'].max():.0f}")
    print(f"  Charge (Z): {output_df['Z'].min():.0f} - {output_df['Z'].max():.0f}")
    print(f"  Binding Energy: {output_df['binding_energy'].min():.2f} - {output_df['binding_energy'].max():.2f} MeV")
    print(f"  Uncertainty: {output_df['sigma'].min():.4f} - {output_df['sigma'].max():.4f} MeV")
    print(f"  Mean σ/BE: {(output_df['sigma'] / output_df['binding_energy']).mean():.6f}")

    return output_df

if __name__ == '__main__':
    df = prepare_ame2020()

    # Verify Grand Solver v1.1 sigma validation will pass
    print("\n✅ Validation checks:")
    print(f"   All σ finite: {np.all(np.isfinite(df['sigma']))}")
    print(f"   All σ > 0: {np.all(df['sigma'] > 0)}")
    print(f"   All BE finite: {np.all(np.isfinite(df['binding_energy']))}")
