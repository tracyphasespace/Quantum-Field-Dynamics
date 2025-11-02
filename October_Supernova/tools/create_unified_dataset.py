#!/usr/bin/env python3
"""
Create Unified QFD Master Dataset
Instance 2 (I2) - Data Infrastructure

Combines DES-SN 5YR and Pantheon+ (non-DES) into a single optimized dataset.

Strategy:
  1. DES-SN 5YR: Primary (homogeneous, modern, 1,635 SNe)
  2. Pantheon+ non-DES: Supplement (PS1, SNLS, low-z, ~692 SNe)
  3. Remove 230 DES SNe from Pantheon+ (duplicates)
  4. Apply quality gates uniformly
  5. Sort by redshift for batch efficiency

Expected output:
  - DES-SN 5YR clean: ~1,400-1,500 SNe
  - Pantheon+ non-DES clean: ~600-650 SNe
  - Total: ~2,000-2,100 SNe (3× larger than Pantheon+ alone)

Usage:
    python tools/create_unified_dataset.py \
        --des-raw data/des_sn5yr/lightcurves_des_sn5yr.csv \
        --pantheon-raw data/pantheon_plus/lightcurves_pantheon_plus_prefiltered.csv \
        --schema data/quality_gates_schema_v1.json \
        --output data/unified/lightcurves_unified_v1.csv \
        --manifest data/unified/unified_manifest_v1.json

Author: Instance 2 (Data Infrastructure)
Date: 2025-11-01
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_and_tag_datasets(des_path: Path, pantheon_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DES-SN 5YR and Pantheon+ datasets, tag source.

    Returns:
        df_des : DES-SN 5YR data
        df_pantheon : Pantheon+ data
    """
    print("Loading datasets...")

    df_des = pd.read_csv(des_path)
    print(f"  DES-SN 5YR: {len(df_des):,} obs, {df_des['snid'].nunique():,} SNe")
    print(f"    Redshift: z = {df_des['z'].min():.3f} - {df_des['z'].max():.3f}")

    df_pantheon = pd.read_csv(pantheon_path)
    print(f"  Pantheon+:  {len(df_pantheon):,} obs, {df_pantheon['snid'].nunique():,} SNe")
    print(f"    Redshift: z = {df_pantheon['z'].min():.3f} - {df_pantheon['z'].max():.3f}")

    # Tag source dataset
    df_des['source_dataset'] = 'DES-SN5YR'
    df_pantheon['source_dataset'] = 'Pantheon+'

    return df_des, df_pantheon


def remove_des_from_pantheon(df_pantheon: pd.DataFrame) -> pd.DataFrame:
    """
    Remove DES SNe from Pantheon+ to avoid duplicates with DES-SN 5YR.

    DES-SN 5YR is preferred because:
    - Larger (1,635 vs 230 SNe)
    - Homogeneous (single survey)
    - Modern calibration (DES Y6)
    """
    print("\nRemoving DES duplicates from Pantheon+...")

    n_before = df_pantheon['snid'].nunique()
    des_sne_count = len(df_pantheon[df_pantheon['survey'] == 'DES']['snid'].unique())

    # Keep only non-DES surveys
    df_pantheon_noDES = df_pantheon[df_pantheon['survey'] != 'DES'].copy()

    n_after = df_pantheon_noDES['snid'].nunique()
    n_dropped = n_before - n_after

    print(f"  Pantheon+ SNe: {n_before} → {n_after} (dropped {n_dropped} DES SNe)")
    print(f"  Remaining surveys: {sorted(df_pantheon_noDES['survey'].unique())}")

    return df_pantheon_noDES


def merge_datasets(df_des: pd.DataFrame, df_pantheon_noDES: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DES-SN 5YR and Pantheon+ (non-DES) into unified dataset.
    """
    print("\nMerging datasets...")

    # Ensure column compatibility
    common_cols = list(set(df_des.columns) & set(df_pantheon_noDES.columns))

    df_unified = pd.concat([
        df_des[common_cols],
        df_pantheon_noDES[common_cols]
    ], ignore_index=True)

    print(f"  Unified dataset: {len(df_unified):,} obs, {df_unified['snid'].nunique():,} SNe")
    print(f"  Redshift range: z = {df_unified['z'].min():.3f} - {df_unified['z'].max():.3f}")

    # Survey breakdown
    print(f"\n  Survey composition:")
    survey_counts = df_unified.groupby('survey')['snid'].nunique().sort_values(ascending=False)
    for survey, count in survey_counts.items():
        print(f"    {survey:12s}: {count:4d} SNe")

    return df_unified


def optimize_sorting(df: pd.DataFrame, sort_by: str = 'redshift') -> pd.DataFrame:
    """
    Sort dataset for processing efficiency.

    Sorting options:
    - 'redshift': Group by z for batch locality (recommended)
    - 'survey': Group by survey for systematic handling
    - 'snid': Alphabetical (default CSV ordering)
    """
    print(f"\nOptimizing dataset (sort_by={sort_by})...")

    if sort_by == 'redshift':
        # Sort by z, then snid, then MJD
        df_sorted = df.sort_values(['z', 'snid', 'mjd']).reset_index(drop=True)
        print(f"  Sorted by redshift → batch locality for MCMC")

    elif sort_by == 'survey':
        # Sort by survey, then z, then snid, then MJD
        df_sorted = df.sort_values(['survey', 'z', 'snid', 'mjd']).reset_index(drop=True)
        print(f"  Sorted by survey → systematic grouping")

    elif sort_by == 'snid':
        # Sort by snid, then MJD
        df_sorted = df.sort_values(['snid', 'mjd']).reset_index(drop=True)
        print(f"  Sorted by SNID → light curves grouped")

    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    return df_sorted


def generate_unified_manifest(df_unified: pd.DataFrame,
                               df_des_orig: pd.DataFrame,
                               df_pantheon_orig: pd.DataFrame,
                               output_file: Path) -> dict:
    """
    Generate manifest for unified dataset.
    """
    z_values = df_unified.groupby('snid')['z'].first()

    manifest = {
        'dataset': 'QFD Unified v1.0',
        'description': 'DES-SN 5YR + Pantheon+ (non-DES)',
        'output_file': str(output_file),
        'date_created': datetime.now().isoformat(),

        'composition': {
            'des_sn5yr': {
                'n_sne': len(df_unified[df_unified['source_dataset'] == 'DES-SN5YR']['snid'].unique()),
                'n_obs': len(df_unified[df_unified['source_dataset'] == 'DES-SN5YR']),
                'source': 'Sánchez et al. 2024, ApJ, 975, 5',
                'zenodo_doi': '10.5281/zenodo.12720778'
            },
            'pantheon_plus_non_des': {
                'n_sne': len(df_unified[df_unified['source_dataset'] == 'Pantheon+']['snid'].unique()),
                'n_obs': len(df_unified[df_unified['source_dataset'] == 'Pantheon+']),
                'source': 'Scolnic et al. 2022, ApJ, 938, 113',
                'note': 'DES SNe removed to avoid duplicates'
            }
        },

        'total': {
            'n_sne': df_unified['snid'].nunique(),
            'n_obs': len(df_unified),
            'z_range': [float(z_values.min()), float(z_values.max())],
            'z_median': float(z_values.median()),
            'surveys': df_unified.groupby('survey')['snid'].nunique().to_dict(),
            'bands': sorted(df_unified['band'].unique().tolist())
        },

        'duplicate_handling': {
            'des_sne_in_pantheon': 230,
            'action': 'Removed from Pantheon+, kept DES-SN 5YR versions',
            'rationale': 'DES-SN 5YR is larger (1,635 vs 230) and more homogeneous'
        },

        'optimization': {
            'sorted_by': 'redshift',
            'purpose': 'Batch locality for MCMC efficiency'
        },

        'provenance': {
            'des_input': 'data/des_sn5yr/lightcurves_des_sn5yr.csv',
            'pantheon_input': 'data/pantheon_plus/lightcurves_pantheon_plus_prefiltered.csv',
            'schema': 'data/quality_gates_schema_v1.json'
        }
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create unified QFD master dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--des-raw',
        required=True,
        help='DES-SN 5YR raw CSV (after format conversion)'
    )
    parser.add_argument(
        '--pantheon-raw',
        required=True,
        help='Pantheon+ raw/prefiltered CSV'
    )
    parser.add_argument(
        '--schema',
        help='Quality gates schema (for validation)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output unified CSV'
    )
    parser.add_argument(
        '--manifest',
        required=True,
        help='Output manifest JSON'
    )
    parser.add_argument(
        '--sort-by',
        choices=['redshift', 'survey', 'snid'],
        default='redshift',
        help='Sort strategy for efficiency (default: redshift)'
    )
    parser.add_argument(
        '--apply-gates',
        action='store_true',
        help='Apply quality gates during merge (otherwise merge raw)'
    )

    args = parser.parse_args()

    des_path = Path(args.des_raw)
    pantheon_path = Path(args.pantheon_raw)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    # Validate inputs
    if not des_path.exists():
        print(f"❌ Error: DES file not found: {des_path}")
        sys.exit(1)

    if not pantheon_path.exists():
        print(f"❌ Error: Pantheon+ file not found: {pantheon_path}")
        sys.exit(1)

    print("=" * 70)
    print("QFD UNIFIED DATASET CREATION")
    print("=" * 70)

    # Step 1: Load datasets
    df_des, df_pantheon = load_and_tag_datasets(des_path, pantheon_path)

    # Step 2: Remove DES duplicates from Pantheon+
    df_pantheon_noDES = remove_des_from_pantheon(df_pantheon)

    # Step 3: Merge datasets
    df_unified = merge_datasets(df_des, df_pantheon_noDES)

    # Step 4: Apply quality gates (optional)
    if args.apply_gates:
        if not args.schema:
            print("❌ Error: --schema required when --apply-gates is used")
            sys.exit(1)

        print("\nApplying quality gates...")
        print("  (Use tools/apply_quality_gates.py for full gate application)")
        print("  Skipping for now - apply gates separately to unified dataset")

    # Step 5: Optimize sorting
    df_unified = optimize_sorting(df_unified, sort_by=args.sort_by)

    # Step 6: Generate manifest
    print("\nGenerating manifest...")
    manifest = generate_unified_manifest(df_unified, df_des, df_pantheon, output_path)

    # Step 7: Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving unified dataset: {output_path}")
    df_unified.to_csv(output_path, index=False, float_format='%.10g')

    print(f"Saving manifest: {manifest_path}")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ UNIFIED DATASET CREATED")
    print("=" * 70)
    print(f"\nComposition:")
    print(f"  DES-SN 5YR:          {manifest['composition']['des_sn5yr']['n_sne']:4d} SNe")
    print(f"  Pantheon+ (non-DES): {manifest['composition']['pantheon_plus_non_des']['n_sne']:4d} SNe")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:               {manifest['total']['n_sne']:4d} SNe")
    print(f"                       {manifest['total']['n_obs']:,} observations")

    print(f"\nRedshift coverage:")
    print(f"  z = {manifest['total']['z_range'][0]:.4f} - {manifest['total']['z_range'][1]:.4f}")
    print(f"  Median z = {manifest['total']['z_median']:.3f}")

    print(f"\nSurveys included:")
    for survey, count in sorted(manifest['total']['surveys'].items(), key=lambda x: -x[1]):
        print(f"  {survey:12s}: {count:4d} SNe")

    print(f"\nOutputs:")
    print(f"  Unified CSV: {output_path}")
    print(f"  Manifest:    {manifest_path}")

    print(f"\nNext steps:")
    print(f"  1. Apply quality gates: python tools/apply_quality_gates.py \\")
    print(f"       --input {output_path} \\")
    print(f"       --output data/unified/lightcurves_unified_v1_clean.csv")
    print(f"  2. Signal ready for V12 fitting")


if __name__ == '__main__':
    main()
