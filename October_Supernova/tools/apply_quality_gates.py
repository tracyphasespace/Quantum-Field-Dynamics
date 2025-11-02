#!/usr/bin/env python3
"""
Apply QFD Quality Gates to Supernova Data
Instance 2 (I2) - Data Infrastructure

Applies pre-registered quality gates from quality_gates_schema_v1.json
to supernova light curve data, generating a clean sample and provenance manifest.

Input:  Raw or prefiltered CSV (QFD format)
Output: Clean CSV + sample_selection_manifest.json

Usage:
    python tools/apply_quality_gates.py \
        --input data/des_sn5yr/lightcurves_des_sn5yr.csv \
        --schema data/quality_gates_schema_v1.json \
        --output data/des_sn5yr/lightcurves_des_sn5yr_clean.csv \
        --manifest data/des_sn5yr/sample_selection_manifest_v1.json

Author: Instance 2 (Data Infrastructure)
Date: 2025-11-01
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_schema(schema_path: Path) -> dict:
    """Load quality gates schema from JSON."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    if schema.get('status') != 'LOCKED':
        print(f"⚠️  WARNING: Schema status is '{schema.get('status')}', expected 'LOCKED'")
        print("   For credibility, schema should be locked before fitting.")

    return schema


def apply_z_cuts(df: pd.DataFrame, z_min: float, z_max: float) -> Tuple[pd.DataFrame, int]:
    """Apply redshift cuts."""
    n_before = len(df['snid'].unique())
    df_filtered = df[(df['z'] >= z_min) & (df['z'] <= z_max)].copy()
    n_after = len(df_filtered['snid'].unique())
    n_dropped = n_before - n_after

    return df_filtered, n_dropped


def apply_obs_count_cuts(df: pd.DataFrame, min_obs: int, max_obs: int) -> Tuple[pd.DataFrame, int, int]:
    """Apply observation count cuts (per SN)."""
    # Count observations per SN
    obs_counts = df.groupby('snid').size()

    # Identify SNe to keep
    sne_to_keep_min = obs_counts[obs_counts >= min_obs].index
    sne_to_keep_max = obs_counts[obs_counts <= max_obs].index
    sne_to_keep = sne_to_keep_min.intersection(sne_to_keep_max)

    n_before = len(obs_counts)
    n_dropped_min = n_before - len(sne_to_keep_min)
    n_dropped_max = len(sne_to_keep_min) - len(sne_to_keep)

    df_filtered = df[df['snid'].isin(sne_to_keep)].copy()

    return df_filtered, n_dropped_min, n_dropped_max


def apply_band_count_cut(df: pd.DataFrame, min_bands: int) -> Tuple[pd.DataFrame, int]:
    """Require minimum number of bands per SN."""
    # Count unique bands per SN
    band_counts = df.groupby('snid')['band'].nunique()

    # Identify SNe to keep
    sne_to_keep = band_counts[band_counts >= min_bands].index

    n_before = len(band_counts)
    n_after = len(sne_to_keep)
    n_dropped = n_before - n_after

    df_filtered = df[df['snid'].isin(sne_to_keep)].copy()

    return df_filtered, n_dropped


def apply_snr_cut(df: pd.DataFrame, min_snr: float) -> Tuple[pd.DataFrame, int]:
    """
    Apply S/N cut at epoch level.

    Note: This is lenient (S/N >= 5) compared to community standard (S/N >= 12).
    QFD compensates with sigma_floor to prevent over-weighting high-S/N epochs.
    """
    n_obs_before = len(df)

    # Compute S/N
    df_filtered = df.copy()
    df_filtered['snr'] = df_filtered['flux_nu_jy'] / df_filtered['flux_nu_jy_err']

    # Keep epochs with S/N >= min_snr
    df_filtered = df_filtered[df_filtered['snr'] >= min_snr].copy()

    n_obs_after = len(df_filtered)
    n_dropped_obs = n_obs_before - n_obs_after

    # Count SNe that were completely removed
    sne_before = set(df['snid'].unique())
    sne_after = set(df_filtered['snid'].unique())
    n_dropped_sne = len(sne_before - sne_after)

    return df_filtered, n_dropped_sne


def apply_sigma_floor(df: pd.DataFrame, sigma_floor_jy: float) -> pd.DataFrame:
    """
    Add systematic uncertainty floor in quadrature.

    sigma_eff = sqrt(sigma_reported^2 + sigma_floor^2)

    This prevents over-weighting of high-S/N epochs and accounts for
    calibration systematics (~2% floor in flux).
    """
    df_floored = df.copy()

    # Add floor in quadrature
    sigma_reported = df_floored['flux_nu_jy_err'].values
    sigma_eff = np.sqrt(sigma_reported**2 + sigma_floor_jy**2)

    df_floored['flux_nu_jy_err'] = sigma_eff

    # Log the effect
    median_before = np.median(sigma_reported)
    median_after = np.median(sigma_eff)
    percent_increase = 100 * (median_after - median_before) / median_before

    print(f"  Applied sigma_floor = {sigma_floor_jy:.4f} Jy")
    print(f"    Median error before: {median_before:.4f} Jy")
    print(f"    Median error after:  {median_after:.4f} Jy ({percent_increase:+.1f}%)")

    return df_floored


def collapse_duplicate_mjds(df: pd.DataFrame, policy: str = 'best_snr') -> Tuple[pd.DataFrame, int]:
    """
    Collapse duplicate observations on same night (same MJD).

    Policies:
    - 'best_snr': Keep observation with highest S/N
    - 'average': Average flux values (not recommended - correlations)
    - 'keep_all': No collapsing
    """
    if policy == 'keep_all':
        return df, 0

    n_obs_before = len(df)

    if policy == 'best_snr':
        # Compute S/N
        df_copy = df.copy()
        df_copy['snr'] = df_copy['flux_nu_jy'] / df_copy['flux_nu_jy_err']
        df_copy['mjd_int'] = df_copy['mjd'].astype(int)

        # Group by (snid, band, mjd_int) and keep best S/N
        idx_to_keep = df_copy.groupby(['snid', 'band', 'mjd_int'])['snr'].idxmax()
        df_collapsed = df.loc[idx_to_keep].copy()

    elif policy == 'average':
        # Average fluxes (not recommended)
        df_copy = df.copy()
        df_copy['mjd_int'] = df_copy['mjd'].astype(int)

        # Average flux, errors add in quadrature / sqrt(N)
        def weighted_average(group):
            weights = 1.0 / group['flux_nu_jy_err']**2
            flux_avg = np.average(group['flux_nu_jy'], weights=weights)
            flux_err_avg = 1.0 / np.sqrt(np.sum(weights))

            # Take first entry for metadata
            result = group.iloc[0].copy()
            result['flux_nu_jy'] = flux_avg
            result['flux_nu_jy_err'] = flux_err_avg
            result['mjd'] = group['mjd'].mean()  # Average MJD

            return result

        df_collapsed = df_copy.groupby(['snid', 'band', 'mjd_int']).apply(weighted_average).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown duplicate_mjd_policy: {policy}")

    n_obs_after = len(df_collapsed)
    n_dropped_obs = n_obs_before - n_obs_after

    return df_collapsed, n_dropped_obs


def apply_survey_whitelist(df: pd.DataFrame, whitelist: List[str]) -> Tuple[pd.DataFrame, int]:
    """Keep only SNe from whitelisted surveys."""
    n_before = len(df['snid'].unique())

    df_filtered = df[df['survey'].isin(whitelist)].copy()

    n_after = len(df_filtered['snid'].unique())
    n_dropped = n_before - n_after

    return df_filtered, n_dropped


def apply_quality_gates(df: pd.DataFrame, schema: dict) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Apply all quality gates in sequence, tracking before/after counts.

    Returns:
        df_clean : Cleaned DataFrame
        gate_log : List of dicts with gate stats
    """
    gate_log = []

    # Extract gate values
    std_gates = schema['filter_categories']['standard_community_filters']['gates']
    qfd_gates = schema['filter_categories']['qfd_specific_filters']['gates']

    print("\nApplying quality gates...")
    print("=" * 60)

    # 1. Redshift cuts
    print("\n1. Redshift cuts:")
    z_min = std_gates['z_min']['value']
    z_max = std_gates['z_max']['value']

    n_sne_before = len(df['snid'].unique())
    df, n_dropped = apply_z_cuts(df, z_min, z_max)
    n_sne_after = len(df['snid'].unique())

    print(f"   z_min = {z_min}, z_max = {z_max}")
    print(f"   SNe: {n_sne_before} → {n_sne_after} (dropped {n_dropped})")

    gate_log.append({
        'gate': f'z_min={z_min}',
        'before': n_sne_before,
        'after': n_sne_after,
        'dropped': n_dropped
    })

    if n_dropped > 0:
        gate_log.append({
            'gate': f'z_max={z_max}',
            'before': n_sne_after + n_dropped,  # Approximation
            'after': n_sne_after,
            'dropped': n_dropped
        })

    # 2. Observation count cuts
    print("\n2. Observation count cuts:")
    min_obs = std_gates['min_obs']['value']
    max_obs = qfd_gates['max_obs']['value']

    n_sne_before = len(df['snid'].unique())
    df, n_dropped_min, n_dropped_max = apply_obs_count_cuts(df, min_obs, max_obs)
    n_sne_after = len(df['snid'].unique())

    print(f"   min_obs = {min_obs}, max_obs = {max_obs}")
    print(f"   SNe: {n_sne_before} → {n_sne_after}")
    print(f"     Dropped (N < {min_obs}): {n_dropped_min}")
    print(f"     Dropped (N > {max_obs}): {n_dropped_max}")

    gate_log.append({
        'gate': f'min_obs={min_obs}',
        'before': n_sne_before,
        'after': n_sne_before - n_dropped_min,
        'dropped': n_dropped_min
    })

    gate_log.append({
        'gate': f'max_obs={max_obs}',
        'before': n_sne_before - n_dropped_min,
        'after': n_sne_after,
        'dropped': n_dropped_max
    })

    # 3. Band count cut
    print("\n3. Band coverage:")
    min_bands = std_gates['min_bands']['value']

    n_sne_before = len(df['snid'].unique())
    df, n_dropped = apply_band_count_cut(df, min_bands)
    n_sne_after = len(df['snid'].unique())

    print(f"   min_bands = {min_bands}")
    print(f"   SNe: {n_sne_before} → {n_sne_after} (dropped {n_dropped})")

    gate_log.append({
        'gate': f'min_bands={min_bands}',
        'before': n_sne_before,
        'after': n_sne_after,
        'dropped': n_dropped
    })

    # 4. S/N cut (epoch level)
    print("\n4. Signal-to-noise cut:")
    min_snr = std_gates['min_snr']['value']

    n_sne_before = len(df['snid'].unique())
    df, n_dropped = apply_snr_cut(df, min_snr)
    n_sne_after = len(df['snid'].unique())

    print(f"   min_snr = {min_snr}")
    print(f"   SNe: {n_sne_before} → {n_sne_after} (dropped {n_dropped} complete SNe)")

    gate_log.append({
        'gate': f'min_snr={min_snr}',
        'before': n_sne_before,
        'after': n_sne_after,
        'dropped': n_dropped
    })

    # 5. Sigma floor (QFD-specific)
    print("\n5. Systematic uncertainty floor:")
    sigma_floor_jy = qfd_gates['sigma_floor_jy']['value']

    df = apply_sigma_floor(df, sigma_floor_jy)
    # Note: This doesn't drop SNe, just modifies uncertainties

    gate_log.append({
        'gate': f'sigma_floor={sigma_floor_jy}Jy',
        'before': len(df['snid'].unique()),
        'after': len(df['snid'].unique()),
        'dropped': 0,
        'note': 'Modifies uncertainties only'
    })

    # 6. Duplicate MJD collapse (QFD-specific)
    print("\n6. Duplicate MJD collapse:")
    mjd_policy = qfd_gates['duplicate_mjd_policy']['value']

    n_obs_before = len(df)
    df, n_dropped_obs = collapse_duplicate_mjds(df, policy=mjd_policy)
    n_obs_after = len(df)

    print(f"   policy = '{mjd_policy}'")
    print(f"   Observations: {n_obs_before} → {n_obs_after} (dropped {n_dropped_obs} duplicates)")

    gate_log.append({
        'gate': f'duplicate_mjd={mjd_policy}',
        'before': len(df['snid'].unique()),
        'after': len(df['snid'].unique()),
        'dropped': 0,
        'note': f'Dropped {n_dropped_obs} duplicate observations'
    })

    # 7. Survey whitelist (QFD-specific)
    print("\n7. Survey homogeneity:")
    survey_whitelist = qfd_gates['survey_whitelist']['value']

    n_sne_before = len(df['snid'].unique())
    df, n_dropped = apply_survey_whitelist(df, survey_whitelist)
    n_sne_after = len(df['snid'].unique())

    print(f"   whitelist = {survey_whitelist}")
    print(f"   SNe: {n_sne_before} → {n_sne_after} (dropped {n_dropped})")

    gate_log.append({
        'gate': f'survey_whitelist={",".join(survey_whitelist)}',
        'before': n_sne_before,
        'after': n_sne_after,
        'dropped': n_dropped
    })

    print("\n" + "=" * 60)
    print(f"✅ Quality gates applied successfully!")

    return df, gate_log


def generate_manifest(df_clean: pd.DataFrame, gate_log: List[dict],
                      schema: dict, input_file: Path, output_file: Path) -> dict:
    """Generate sample selection manifest with provenance."""

    # Compute final sample statistics
    z_values = df_clean.groupby('snid')['z'].first()

    manifest = {
        'dataset': 'DES-SN 5YR',  # Update if using different dataset
        'input_file': str(input_file),
        'output_file': str(output_file),
        'schema_version': schema['schema_version'],
        'schema_status': schema['status'],
        'date_created': datetime.now().isoformat(),

        'gates_applied': gate_log,

        'final_sample': {
            'n_sne': len(df_clean['snid'].unique()),
            'n_obs': len(df_clean),
            'z_range': [float(z_values.min()), float(z_values.max())],
            'z_median': float(z_values.median()),
            'surveys': df_clean.groupby('survey')['snid'].nunique().to_dict(),
            'bands': sorted(df_clean['band'].unique().tolist())
        },

        'credibility_framework': {
            'pre_registered': schema.get('credibility_framework', {}).get('pre_registration', {}).get('status') == 'LOCKED',
            'frozen_before_fitting': schema.get('locked_before_fitting', False),
            'exclusion_list': None,  # To be added after frozen-knobs audit
            'ablation_tests_required': True
        },

        'provenance': {
            'zenodo_doi': '10.5281/zenodo.12720778',
            'publication': 'Sánchez et al. 2024, ApJ, 975, 5',
            'schema_file': 'data/quality_gates_schema_v1.json'
        }
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Apply QFD quality gates to supernova data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file (QFD format)'
    )
    parser.add_argument(
        '--schema',
        required=True,
        help='Quality gates schema JSON'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output clean CSV file'
    )
    parser.add_argument(
        '--manifest',
        required=True,
        help='Output sample selection manifest JSON'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    schema_path = Path(args.schema)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    # Validate inputs
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    if not schema_path.exists():
        print(f"❌ Error: Schema file not found: {schema_path}")
        sys.exit(1)

    # Load schema
    print(f"Loading schema: {schema_path}")
    schema = load_schema(schema_path)

    # Load data
    print(f"\nLoading input data: {input_path}")
    df_raw = pd.read_csv(input_path)

    print(f"  Loaded {len(df_raw)} observations from {df_raw['snid'].nunique()} SNe")
    print(f"  Redshift range: z = {df_raw['z'].min():.3f} - {df_raw['z'].max():.3f}")
    print(f"  Surveys: {df_raw.groupby('survey')['snid'].nunique().to_dict()}")

    # Apply quality gates
    df_clean, gate_log = apply_quality_gates(df_raw, schema)

    # Generate manifest
    print("\nGenerating sample selection manifest...")
    manifest = generate_manifest(df_clean, gate_log, schema, input_path, output_path)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving clean sample: {output_path}")
    df_clean.to_csv(output_path, index=False, float_format='%.10g')

    print(f"Saving manifest: {manifest_path}")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ QUALITY FILTERING COMPLETE")
    print("=" * 60)
    print(f"\nFinal sample:")
    print(f"  SNe: {manifest['final_sample']['n_sne']}")
    print(f"  Observations: {manifest['final_sample']['n_obs']}")
    print(f"  Redshift: z = {manifest['final_sample']['z_range'][0]:.3f} - {manifest['final_sample']['z_range'][1]:.3f}")
    print(f"  Median z: {manifest['final_sample']['z_median']:.3f}")
    print(f"  Surveys: {manifest['final_sample']['surveys']}")
    print(f"  Bands: {manifest['final_sample']['bands']}")

    print(f"\nOutputs:")
    print(f"  Clean CSV: {output_path}")
    print(f"  Manifest:  {manifest_path}")

    print(f"\nNext steps:")
    print(f"  1. Signal ready for fitting: touch {output_path.parent}/.READY_FOR_FIT")
    print(f"  2. Instance 1 runs V12 fit on clean sample")
    print(f"  3. Instance 2 runs frozen-knobs audit: python tools/frozen_knobs_audit.py")


if __name__ == '__main__':
    main()
