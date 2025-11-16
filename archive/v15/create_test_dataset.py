#!/usr/bin/env python3
"""
Create a small test dataset for V16 collaboration sandbox.

Samples ~200 SNe from full Stage 1 results to create a lightweight
dataset for debugging Stage 2 MCMC.
"""

import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Paths
STAGE1_FULL = Path("../results/v15_clean/stage1_fullscale")
LIGHTCURVES_FULL = Path("data/lightcurves_unified_v2_min3.csv")
OUTPUT_DIR = Path("../results/v15_clean/test_dataset")
OUTPUT_STAGE1 = OUTPUT_DIR / "stage1_results"
OUTPUT_LIGHTCURVES = OUTPUT_DIR / "lightcurves_test.csv"

# Parameters
TARGET_COUNT = 200
QUALITY_CUT = 2000  # chi2 threshold

def load_stage1_summary():
    """Load Stage 1 results and filter by quality"""
    sn_data = []

    for sn_dir in STAGE1_FULL.iterdir():
        if not sn_dir.is_dir():
            continue

        sn_id = int(sn_dir.name)
        metrics_path = sn_dir / "metrics.json"

        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Check quality
            chi2 = metrics.get('chi2', float('inf'))
            if chi2 < QUALITY_CUT:
                sn_data.append({
                    'sn_id': sn_id,
                    'chi2': chi2,
                    'dir': sn_dir
                })
        except Exception as e:
            print(f"Warning: Could not load {sn_id}: {e}")
            continue

    return pd.DataFrame(sn_data)

def sample_by_redshift(df, lightcurves_df, target_count):
    """Sample SNe distributed across redshift bins"""
    # Merge with lightcurves to get redshift
    df_merged = df.merge(
        lightcurves_df[['snid', 'z']].drop_duplicates('snid'),
        left_on='sn_id',
        right_on='snid',
        how='inner'
    )

    # Create redshift bins
    df_merged['z_bin'] = pd.cut(df_merged['z'], bins=5, labels=False)

    # Sample proportionally from each bin
    sampled = df_merged.groupby('z_bin', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), target_count // 5), random_state=42)
    )

    # If we don't have enough, add more from best chi2
    if len(sampled) < target_count:
        remaining = target_count - len(sampled)
        excluded = df_merged[~df_merged.index.isin(sampled.index)]
        additional = excluded.nsmallest(remaining, 'chi2')
        sampled = pd.concat([sampled, additional])

    return sampled.sort_values('chi2').head(target_count)

def create_test_dataset(sampled_df, lightcurves_df):
    """Create test dataset directories and files"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_STAGE1.mkdir(exist_ok=True)

    # Copy Stage 1 results
    print(f"Copying Stage 1 results for {len(sampled_df)} SNe...")
    for _, row in sampled_df.iterrows():
        src_dir = row['dir']
        dst_dir = OUTPUT_STAGE1 / src_dir.name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    # Create lightcurves subset
    print(f"Creating lightcurves subset...")
    sn_ids = set(sampled_df['sn_id'].values)
    lightcurves_subset = lightcurves_df[lightcurves_df['snid'].isin(sn_ids)]
    lightcurves_subset.to_csv(OUTPUT_LIGHTCURVES, index=False)

    # Create summary
    summary = {
        'n_supernovae': len(sampled_df),
        'quality_cut': QUALITY_CUT,
        'chi2_range': [float(sampled_df['chi2'].min()), float(sampled_df['chi2'].max())],
        'chi2_mean': float(sampled_df['chi2'].mean()),
        'redshift_range': [float(sampled_df['z'].min()), float(sampled_df['z'].max())],
        'sn_ids': sorted([int(x) for x in sampled_df['sn_id'].values])
    }

    with open(OUTPUT_DIR / "test_dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Create README
    readme_content = f"""# Test Dataset for V16 Collaboration Sandbox

**Created**: 2025-01-12
**Size**: {len(sampled_df)} supernovae (sampled from {len(lightcurves_df['snid'].unique())} total)

## Contents

- `stage1_results/` - Stage 1 optimization results for {len(sampled_df)} SNe
- `lightcurves_test.csv` - Lightcurves data for test SNe
- `test_dataset_summary.json` - Dataset statistics

## Quality Criteria

- Chi-squared < {QUALITY_CUT}
- Distributed across redshift range z = [{summary['redshift_range'][0]:.3f}, {summary['redshift_range'][1]:.3f}]
- Mean chi-squared = {summary['chi2_mean']:.1f}

## Usage with Stage 2

```bash
python3 stages/stage2_simple.py \\
  --stage1-results test_dataset/stage1_results \\
  --lightcurves test_dataset/lightcurves_test.csv \\
  --out test_output \\
  --nchains 2 \\
  --nsamples 2000 \\
  --nwarmup 1000 \\
  --quality-cut 2000 \\
  --use-informed-priors
```

## Purpose

This small dataset allows collaborators to:
1. Test Stage 2 MCMC modifications quickly
2. Debug issues without downloading full 107 MB Stage 1 results
3. Verify their environment is working correctly

For full-scale production runs, use the complete Stage 1 results.
"""

    with open(OUTPUT_DIR / "README.md", 'w') as f:
        f.write(readme_content)

    return summary

def main():
    print("Creating test dataset for V16 collaboration sandbox...")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    df = load_stage1_summary()
    print(f"Found {len(df)} SNe with chi2 < {QUALITY_CUT}")

    # Load lightcurves
    print("Loading lightcurves...")
    lightcurves_df = pd.read_csv(LIGHTCURVES_FULL)
    print(f"Loaded {len(lightcurves_df)} lightcurve measurements")

    # Sample SNe
    print(f"Sampling {TARGET_COUNT} SNe distributed across redshift...")
    sampled_df = sample_by_redshift(df, lightcurves_df, TARGET_COUNT)
    print(f"Selected {len(sampled_df)} SNe")
    print(f"  Redshift range: z = [{sampled_df['z'].min():.3f}, {sampled_df['z'].max():.3f}]")
    print(f"  Chi2 range: [{sampled_df['chi2'].min():.1f}, {sampled_df['chi2'].max():.1f}]")
    print()

    # Create test dataset
    summary = create_test_dataset(sampled_df, lightcurves_df)

    # Report size
    import subprocess
    result = subprocess.run(['du', '-sh', str(OUTPUT_DIR)], capture_output=True, text=True)
    size = result.stdout.split()[0]

    print()
    print("=" * 60)
    print("Test dataset created successfully!")
    print("=" * 60)
    print(f"Location: {OUTPUT_DIR}")
    print(f"Size: {size}")
    print(f"SNe count: {summary['n_supernovae']}")
    print(f"Mean chi2: {summary['chi2_mean']:.1f}")
    print()
    print("Next steps:")
    print("1. Copy to V16 sandbox: cp -r ../results/v15_clean/test_dataset /path/to/V16/")
    print("2. Test Stage 2: See test_dataset/README.md for usage")
    print()

if __name__ == '__main__':
    main()
