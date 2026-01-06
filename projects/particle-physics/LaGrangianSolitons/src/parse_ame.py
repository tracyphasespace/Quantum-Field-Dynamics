#!/usr/bin/env python3
"""
Parse AME2020 (Atomic Mass Evaluation) and compute Q-values for decay modes.

Calculates:
  - Q_alpha: Alpha decay Q-value
  - Q_beta_minus: β⁻ decay Q-value
  - Q_beta_plus: β⁺ decay Q-value
  - Q_EC: Electron capture Q-value
  - S_n: Neutron separation energy
  - S_p: Proton separation energy
  - S_2n, S_2p: Two-nucleon separation energies

Uses mass excess values from AME2020 to compute reaction energetics.

Implements EXPERIMENT_PLAN.md §2.2 data pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Standard masses (mass excess in keV, from AME2020)
MASS_EXCESS_keV = {
    'neutron': 8071.31806,      # n
    'proton': 7288.971064,      # H-1 (atomic mass includes electron)
    'electron': 510.9989461,    # e⁻
    'alpha': 2424.91587,        # He-4
    'deuteron': 13135.722895,   # H-2
    'triton': 14949.8109,       # H-3
    'He3': 14931.21888,         # He-3
}


def load_ame(input_file):
    """
    Load AME2020 CSV file.

    Args:
        input_file: Path to ame2020.csv

    Returns:
        pd.DataFrame with columns:
          Z, N, A, element, mass_excess_keV, is_estimated
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"AME file not found: {input_file}")

    logger.info(f"Loading AME2020 file: {input_file}")

    df = pd.read_csv(input_path)

    # Keep only relevant columns
    columns_to_keep = ['Z', 'N', 'A', 'element', 'mass_excess_keV',
                       'mass_excess_unc_keV', 'is_estimated']

    # Check if all columns exist
    for col in columns_to_keep:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in AME file")

    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns].copy()

    logger.info(f"Loaded {len(df)} nuclides from AME2020")

    return df


def get_mass_excess(df, Z, A):
    """
    Get mass excess for a specific (Z, A) nuclide.

    Args:
        df: DataFrame with AME data
        Z: Atomic number
        A: Mass number

    Returns:
        float: Mass excess in keV (NaN if not found)
    """
    mask = (df['Z'] == Z) & (df['A'] == A)
    matches = df[mask]

    if len(matches) == 0:
        return np.nan
    elif len(matches) > 1:
        logger.warning(f"Multiple entries for Z={Z}, A={A}, using first")

    return matches.iloc[0]['mass_excess_keV']


def calculate_q_values(df):
    """
    Calculate Q-values for all decay modes.

    Q-value formulas (using mass excess Δ):

    Alpha decay:      Q_α = Δ(A,Z) - Δ(A-4,Z-2) - Δ(He-4)
    Beta⁻ decay:      Q_β⁻ = Δ(A,Z) - Δ(A,Z+1)
    Beta⁺ decay:      Q_β⁺ = Δ(A,Z) - Δ(A,Z-1) - 2m_e
    Electron capture: Q_EC = Δ(A,Z) - Δ(A,Z-1)
    Neutron sep:      S_n = Δ(A-1,Z) + Δ(n) - Δ(A,Z)
    Proton sep:       S_p = Δ(A-1,Z-1) + Δ(H-1) - Δ(A,Z)

    Args:
        df: DataFrame with AME data (must have Z, A, mass_excess_keV)

    Returns:
        pd.DataFrame: Input data with added Q-value columns
    """
    logger.info("Calculating Q-values for all nuclides...")

    # Create a lookup dictionary for fast access
    mass_lookup = {}
    for _, row in df.iterrows():
        mass_lookup[(int(row['Z']), int(row['A']))] = row['mass_excess_keV']

    # Initialize Q-value columns
    df['Q_alpha_keV'] = np.nan
    df['Q_beta_minus_keV'] = np.nan
    df['Q_beta_plus_keV'] = np.nan
    df['Q_EC_keV'] = np.nan
    df['S_n_keV'] = np.nan
    df['S_p_keV'] = np.nan
    df['S_2n_keV'] = np.nan
    df['S_2p_keV'] = np.nan

    n_calculated = 0

    for idx, row in df.iterrows():
        Z = int(row['Z'])
        A = int(row['A'])
        N = A - Z
        mass_parent = row['mass_excess_keV']

        # Skip if parent mass is NaN
        if pd.isna(mass_parent):
            continue

        # Q_alpha = M(A,Z) - M(A-4,Z-2) - M(He-4)
        if (Z-2, A-4) in mass_lookup:
            mass_daughter = mass_lookup[(Z-2, A-4)]
            Q_alpha = mass_parent - mass_daughter - MASS_EXCESS_keV['alpha']
            df.at[idx, 'Q_alpha_keV'] = Q_alpha

        # Q_beta_minus = M(A,Z) - M(A,Z+1)
        # (Atomic mass convention: includes electrons, so no need to add m_e)
        if (Z+1, A) in mass_lookup:
            mass_daughter = mass_lookup[(Z+1, A)]
            Q_beta_minus = mass_parent - mass_daughter
            df.at[idx, 'Q_beta_minus_keV'] = Q_beta_minus

        # Q_beta_plus = M(A,Z) - M(A,Z-1) - 2m_e
        # (Atomic masses: parent has Z electrons, daughter has Z-1,
        #  so need to subtract 2m_e for positron creation + 1 less atomic e⁻)
        if (Z-1, A) in mass_lookup and Z > 0:
            mass_daughter = mass_lookup[(Z-1, A)]
            Q_beta_plus = mass_parent - mass_daughter - 2 * MASS_EXCESS_keV['electron']
            df.at[idx, 'Q_beta_plus_keV'] = Q_beta_plus

        # Q_EC = M(A,Z) - M(A,Z-1)
        # (Electron capture: atomic masses differ by one electron exactly)
        if (Z-1, A) in mass_lookup and Z > 0:
            mass_daughter = mass_lookup[(Z-1, A)]
            Q_EC = mass_parent - mass_daughter
            df.at[idx, 'Q_EC_keV'] = Q_EC

        # S_n = M(A-1,Z) + M(n) - M(A,Z)
        # (Energy to remove one neutron)
        if (Z, A-1) in mass_lookup:
            mass_n_removed = mass_lookup[(Z, A-1)]
            S_n = mass_n_removed + MASS_EXCESS_keV['neutron'] - mass_parent
            df.at[idx, 'S_n_keV'] = S_n

        # S_p = M(A-1,Z-1) + M(H-1) - M(A,Z)
        # (Energy to remove one proton; use H-1 atomic mass)
        if (Z-1, A-1) in mass_lookup and Z > 0:
            mass_p_removed = mass_lookup[(Z-1, A-1)]
            S_p = mass_p_removed + MASS_EXCESS_keV['proton'] - mass_parent
            df.at[idx, 'S_p_keV'] = S_p

        # S_2n = M(A-2,Z) + 2×M(n) - M(A,Z)
        # (Two-neutron separation energy)
        if (Z, A-2) in mass_lookup:
            mass_2n_removed = mass_lookup[(Z, A-2)]
            S_2n = mass_2n_removed + 2 * MASS_EXCESS_keV['neutron'] - mass_parent
            df.at[idx, 'S_2n_keV'] = S_2n

        # S_2p = M(A-2,Z-2) + 2×M(H-1) - M(A,Z)
        # (Two-proton separation energy)
        if (Z-2, A-2) in mass_lookup and Z >= 2:
            mass_2p_removed = mass_lookup[(Z-2, A-2)]
            S_2p = mass_2p_removed + 2 * MASS_EXCESS_keV['proton'] - mass_parent
            df.at[idx, 'S_2p_keV'] = S_2p

        n_calculated += 1

    logger.info(f"Calculated Q-values for {n_calculated} nuclides")

    # Convert keV to MeV for convenience
    for col in ['Q_alpha', 'Q_beta_minus', 'Q_beta_plus', 'Q_EC',
                'S_n', 'S_p', 'S_2n', 'S_2p']:
        df[f'{col}_MeV'] = df[f'{col}_keV'] / 1000.0

    return df


def summarize_q_values(df):
    """
    Print summary statistics for Q-values.
    """
    logger.info("\n" + "="*80)
    logger.info("Q-VALUE STATISTICS")
    logger.info("="*80)

    # For each Q-value, count how many are positive (energetically allowed)
    q_columns = ['Q_alpha_keV', 'Q_beta_minus_keV', 'Q_beta_plus_keV',
                 'Q_EC_keV', 'S_n_keV', 'S_p_keV']

    for col in q_columns:
        if col in df.columns:
            data = df[col].dropna()
            n_total = len(data)
            n_positive = (data > 0).sum()
            n_negative = (data <= 0).sum()

            if n_total > 0:
                logger.info(f"\n{col}:")
                logger.info(f"  Total calculated: {n_total}")
                logger.info(f"  Positive (allowed): {n_positive} ({100*n_positive/n_total:.1f}%)")
                logger.info(f"  Negative (forbidden): {n_negative} ({100*n_negative/n_total:.1f}%)")

                if n_positive > 0:
                    positive_data = data[data > 0]
                    logger.info(f"  Range (positive): {positive_data.min():.1f} to {positive_data.max():.1f} keV")
                    logger.info(f"  Median (positive): {positive_data.median():.1f} keV")


def parse_ame(input_file, output_file=None):
    """
    Parse AME2020 and calculate Q-values.

    Args:
        input_file: Path to ame2020.csv
        output_file: Path to output parquet file (optional)

    Returns:
        pd.DataFrame: AME data with Q-values
    """
    # Load AME data
    df = load_ame(input_file)

    # Calculate Q-values
    df = calculate_q_values(df)

    # Summary statistics
    summarize_q_values(df)

    # Save to parquet if output path provided
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"\nSaved to: {output_file}")
        logger.info(f"Rows: {len(df)}")
        logger.info(f"Columns: {len(df.columns)}")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse AME2020 and compute Q-values for all decay modes'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to ame2020.csv'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output parquet file (e.g., data/derived/ame.parquet)'
    )

    args = parser.parse_args()

    df = parse_ame(args.input, args.output)

    print("\n" + "="*80)
    print("AME PARSING COMPLETE")
    print("="*80)
    print(f"Total nuclides: {len(df)}")
    print(f"Q-values calculated for:")
    print(f"  Q_alpha:      {df['Q_alpha_keV'].notna().sum()}")
    print(f"  Q_beta_minus: {df['Q_beta_minus_keV'].notna().sum()}")
    print(f"  Q_beta_plus:  {df['Q_beta_plus_keV'].notna().sum()}")
    print(f"  S_n:          {df['S_n_keV'].notna().sum()}")
    print(f"  S_p:          {df['S_p_keV'].notna().sum()}")
    print(f"Output: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
