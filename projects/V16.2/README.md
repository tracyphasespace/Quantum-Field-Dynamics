# QFD Supernova Analysis V16.2 - Prior Recovery Workspace

**Status**: Recovery and Reconstruction
**Created**: 2025-11-13 (forked from V16)
**Purpose**: Recovery workspace for reconstructing hardcoded priors lost in directory deletion incident

## ⚠️ CRITICAL ISSUE → ⚡ SMART SOLUTION

The hardcoded priors in V16 may be **incorrect** (lost derivation code).

**Instead of expensive re-derivation, we're working backwards!**

### Quick Validation Approach

Use published parameters (k_J ≈ 10.74, η' ≈ -7.97, ξ ≈ -6.95) and see how well they fit:

```bash
python3 tools/validate_published_params.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out validation_test
```

**Takes 20 seconds** (vs 12 hours for full MCMC pipeline!)

### Documentation

- **`QUICKSTART.md`** - How to validate parameters (START HERE!)
- **`RECOVERY.md`** - Full recovery plan and context
- **`tools/README_VALIDATION.md`** - Detailed validation documentation

### Goals

1. ✅ **Validate** published parameters fit the data (RMS ≈ 1.4 mag expected)
2. ✅ **Identify** outliers (BBH/lensing candidates)
3. ✅ **Confirm** hardcoded priors are correct
4. ❌ **Not conflicting** with parallel V16 development

---

# Original V16 Documentation

**Original Created**: 2025-01-12
**Original Purpose**: Collaboration sandbox for QFD supernova analysis with clean V15 implementation

## Overview

This directory contains a clean, documented implementation of the QFD (Quantum Field Dynamics) supernova analysis pipeline. It's designed as a collaboration sandbox where team members can work on improvements and extensions to the V15 codebase.

## What's Included

This sandbox is a copy of the V15 clean implementation with:

- **Stage 1**: Individual supernova optimization (`stages/stage1_optimize.py`)
- **Stage 2 (Clean)**: Global MCMC with informed priors fix (`stages/stage2_simple.py`)
- **Stage 2 (Full)**: Production MCMC with all features (`stages/stage2_mcmc_numpyro.py`)
- **Stage 3**: Hubble diagram analysis (`stages/stage3_hubble_optimized.py`)
- **Documentation**: Pseudocode and technical documentation (`documents/`)
- **Test Dataset**: Small subset of 200 SNe for quick testing (`test_dataset/`)
- **Tools**: Comparison and analysis scripts (`tools/`)

## Recent Fixes

### Dataset-Dependent Priors Bug (Fixed 2025-01-12)

The Stage 2 MCMC had a subtle statistical bug where priors defined on physical parameters were being implicitly scaled by dataset-dependent standardization statistics, making the model specification unstable.

**Fix**: Added `--use-informed-priors` flag to `stage2_simple.py` that defines priors directly on standardized coefficients c, making them data-independent:

```python
# Informed priors DIRECTLY on standardized coefficients c
c0 ~ Normal(1.857, 0.5)  # Instead of k_J ~ transformed prior
c1 ~ Normal(-2.227, 0.5) # Instead of eta' ~ transformed prior
c2 ~ Normal(-0.766, 0.3) # Instead of xi ~ transformed prior
```

This eliminates MCMC divergences and produces stable results matching the November 5, 2024 golden reference.

## Key Scripts

### `stages/stage2_simple.py`
Clean implementation of Stage 2 MCMC from pseudocode. Includes:
- Standardized feature space (not orthogonalization)
- Informed priors option (`--use-informed-priors`)
- Student-t likelihood for outlier robustness
- Holdout validation support

Usage:
```bash
python3 stages/stage2_simple.py \
  --stage1-results ../results/v15_clean/stage1_fullscale \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out ../results/v15_clean/stage2_output \
  --nchains 2 \
  --nsamples 2000 \
  --nwarmup 1000 \
  --quality-cut 2000 \
  --use-informed-priors
```

### `stages/stage2_mcmc_numpyro.py`
Production Stage 2 with additional features:
- Multiple sign constraint variants
- Ln(A) space vs magnitude space options
- Extended diagnostics
- Flexible prior configurations

## Golden Reference Results (November 5, 2024)

Target parameters for validation:
- k_J = 10.770 ± 4.567 km/s/Mpc
- eta' = -7.988 ± 1.439
- xi = -6.908 ± 3.746
- sigma_alpha = 1.398 ± 0.024
- nu = 6.522 ± 0.961

Standardized coefficients:
- c[0] = 1.857
- c[1] = -2.227
- c[2] = -0.766

## Documentation

See `documents/Supernovae_Pseudocode.md` for detailed algorithm specifications.

## Collaboration Guidelines

1. **Don't modify the original files** - Create new versions or branches
2. **Document all changes** - Update this README with new features
3. **Test thoroughly** - Compare against golden reference results
4. **Coordinate** - Communicate with team before major changes

## Getting Started

### Quick Start with Test Dataset

For quick testing and debugging Stage 2, use the included test dataset:

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

See `test_dataset/README.md` for details.

### Full Production Data

For production runs matching the November 5, 2024 golden reference, you'll need:
- Full Stage 1 results directory (~107 MB, 4727 SNe)
- Complete unified lightcurves CSV (~12 MB)

See `DATA.md` for detailed data requirements and how to obtain full datasets.

## Questions or Issues?

Contact the QFD research team or open an issue in the main repository.
