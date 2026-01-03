#!/usr/bin/env python3
"""
Download AME2020 nuclear data

This script provides instructions for obtaining the AME2020 database,
which is required for running the half-life prediction models.

The AME2020 (Atomic Mass Evaluation 2020) database contains experimental
mass data for ~3500 nuclei and is maintained by the Nuclear Data Section
of the IAEA.
"""

import os
import sys


def print_instructions():
    """Print instructions for obtaining AME2020 data."""
    print("="*80)
    print("AME2020 DATA DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print()
    print("The AME2020 database is required for running the half-life predictions.")
    print()
    print("Option 1: Download from IAEA Nuclear Data Services")
    print("-" * 80)
    print("1. Visit: https://www-nds.iaea.org/amdc/")
    print("2. Click on 'AME2020' or 'Atomic Mass Data Center'")
    print("3. Download the mass table (mass_1.mas20 or similar)")
    print("4. Place in ./data/ directory")
    print()
    print("Option 2: Use the provided sample dataset")
    print("-" * 80)
    print("A preprocessed version is available in this repository:")
    print("  data/ame2020_system_energies.csv")
    print()
    print("This file contains columns:")
    print("  - A: Mass number")
    print("  - Z: Atomic number")
    print("  - element: Element symbol")
    print("  - mass_excess_MeV: Mass excess in MeV")
    print("  - BE_per_A_MeV: Binding energy per nucleon in MeV")
    print()
    print("Required columns for scripts:")
    print("  ['A', 'Z', 'element', 'mass_excess_MeV', 'BE_per_A_MeV']")
    print()
    print("="*80)
    print()


def check_data_file():
    """Check if AME2020 data file exists."""
    data_path = '../data/ame2020_system_energies.csv'

    if os.path.exists(data_path):
        print(f"✓ Found: {data_path}")

        # Try to load and validate
        try:
            import pandas as pd
            df = pd.read_csv(data_path)

            required_cols = ['A', 'Z', 'element', 'mass_excess_MeV', 'BE_per_A_MeV']
            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                print(f"⚠ Warning: Missing columns: {missing}")
                return False

            print(f"✓ Valid AME2020 data file")
            print(f"  - {len(df)} nuclei")
            print(f"  - Z range: {df['Z'].min()} to {df['Z'].max()}")
            print(f"  - A range: {df['A'].min()} to {df['A'].max()}")
            return True

        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return False
    else:
        print(f"✗ Not found: {data_path}")
        print()
        print_instructions()
        return False


if __name__ == '__main__':
    print()
    if check_data_file():
        print()
        print("="*80)
        print("DATA FILE OK - Ready to run predictions!")
        print("="*80)
        sys.exit(0)
    else:
        sys.exit(1)
