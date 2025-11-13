# Test Dataset for V16 Collaboration Sandbox

**Created**: 2025-01-12
**Size**: 200 supernovae (sampled from 5468 total)

## Contents

- `stage1_results/` - Stage 1 optimization results for 200 SNe
- `lightcurves_test.csv` - Lightcurves data for test SNe
- `test_dataset_summary.json` - Dataset statistics

## Quality Criteria

- Chi-squared < 2000
- Distributed across redshift range z = [0.083, 1.498]
- Mean chi-squared = -246.9

## Usage with Stage 2

```bash
python3 stages/stage2_simple.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out test_output \
  --nchains 2 \
  --nsamples 2000 \
  --nwarmup 1000 \
  --quality-cut 2000 \
  --use-informed-priors
```

## Purpose

This small dataset allows collaborators to:
1. Test Stage 2 MCMC modifications quickly
2. Debug issues without downloading full 107 MB Stage 1 results
3. Verify their environment is working correctly

For full-scale production runs, use the complete Stage 1 results.
