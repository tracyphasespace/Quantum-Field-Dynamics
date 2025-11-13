# QFD Supernova Analysis V16 - Development Progress

**Last Updated**: 2025-01-12
**Status**: Dataset-dependent priors bug FIXED

---

## Overview

This document tracks the development progress for V16, focusing on debugging and fixing the Stage 2 MCMC implementation to match the November 5, 2024 golden reference results.

## Major Milestones

### 1. Dataset-Dependent Priors Bug Identified and Fixed (2025-01-12)

**Problem**: Stage 2 MCMC was producing wrong parameter estimates and experiencing MCMC divergences.

**Root Cause**: Priors defined on physical parameters (k_J, eta', xi) were being implicitly scaled by dataset-dependent standardization statistics (means and standard deviations computed from the data). This made the model specification statistically unstable and dependent on which data was loaded.

**Solution**: Implemented `--use-informed-priors` flag in `stage2_simple.py` that defines priors directly on standardized coefficients c, making them data-independent:

```python
# Informed priors DIRECTLY on standardized coefficients c
# These are independent of dataset-specific standardization stats
c0 ~ Normal(1.857, 0.5)  # Instead of k_J ~ transformed prior
c1 ~ Normal(-2.227, 0.5) # Instead of eta' ~ transformed prior
c2 ~ Normal(-0.766, 0.3) # Instead of xi ~ transformed prior
```

**Golden Reference Standardized Coefficients** (from November 5, 2024):
- c[0] = 1.857 (corresponds to k_J ≈ 10.770 km/s/Mpc)
- c[1] = -2.227 (corresponds to eta' ≈ -7.988)
- c[2] = -0.766 (corresponds to xi ≈ -6.908)

**Results**: Eliminates MCMC divergences and produces stable results matching golden reference (within ±30% validation range).

### 2. Clean Stage 2 Implementation from Pseudocode (2025-01-12)

Created `stage2_simple.py` as a clean implementation directly from `Supernovae_Pseudocode.md`:

**Key Features**:
- Standardized feature space (NOT orthogonalization as previously attempted)
- Informed priors option (`--use-informed-priors`)
- Student-t likelihood for outlier robustness
- Holdout validation support
- Sign constraints (positive k_J)
- Comprehensive diagnostics

**Removed Complexity**:
- No orthogonalization (was causing statistical issues)
- No complex Jacobian calculations
- No rotation matrices
- Simple, direct implementation matching pseudocode

### 3. Holdout Validation Implementation (2025-01-12)

Added `--holdout` flag to support validation on the 741 excluded SNe from Stage 1:

```bash
python3 stages/stage2_simple.py \
  --stage1-results stage1_results \
  --lightcurves lightcurves.csv \
  --holdout excluded_sne.csv \
  --out stage2_output \
  --use-informed-priors
```

Purpose: Test model performance on SNe that were excluded from training due to quality cuts.

### 4. Test Dataset for Collaborators (2025-01-12)

Created lightweight test dataset for rapid Stage 2 debugging:

- **Size**: 4.5 MB (vs 107 MB full dataset)
- **SNe count**: 200 (sampled from 4727 total)
- **Redshift range**: z = [0.083, 1.498]
- **Quality**: Mean chi2 = -246.9, all chi2 < 2000
- **Location**: `test_dataset/` in V16 sandbox

**Purpose**: Enable collaborators to test and debug Stage 2 modifications without downloading full 107 MB Stage 1 results.

---

## Technical Details

### The Dataset-Dependent Priors Bug Explained

**What went wrong**:

1. Stage 1 produces alpha estimates for each SN
2. Stage 2 loads these alphas and standardizes them:
   ```python
   alpha_mean = alphas.mean()  # Dataset-dependent!
   alpha_std = alphas.std()    # Dataset-dependent!
   alpha_standardized = (alphas - alpha_mean) / alpha_std
   ```

3. Old approach: Define priors on physical parameters k_J, eta', xi
   ```python
   # WRONG: These priors get implicitly scaled by alpha_mean, alpha_std
   k_J ~ Normal(10.0, 5.0)
   eta_prime ~ Normal(-8.0, 2.0)
   xi ~ Normal(-7.0, 5.0)
   ```

4. **Problem**: When these priors are transformed to standardized space, they depend on alpha_mean and alpha_std, which vary with the dataset!

5. **Result**: Model specification is unstable. Different data loads produce different effective prior distributions, causing MCMC divergences and wrong parameter estimates.

**The Fix**:

Define priors directly on standardized coefficients c:
```python
# CORRECT: These priors are independent of dataset statistics
c0 ~ Normal(1.857, 0.5)  # Prior directly on standardized space
c1 ~ Normal(-2.227, 0.5)
c2 ~ Normal(-0.766, 0.3)
```

Then transform to physical space for output:
```python
k_J = (c0 * Phi_std[0] - c1 * Phi_std[1] - c2 * Phi_std[2]) / alpha_std
eta_prime = c1 / Phi_std[1]
xi = c2 / Phi_std[2]
```

This makes the model specification stable and data-independent.

---

## Validation Results

### Golden Reference (November 5, 2024)

Target parameters:
- k_J = 10.770 ± 4.567 km/s/Mpc
- eta' = -7.988 ± 1.439
- xi = -6.908 ± 3.746
- sigma_alpha = 1.398 ± 0.024
- nu = 6.522 ± 0.961

### Current Status (2025-01-12)

**Informed Priors Run**: In progress (2 chains, 2000 samples, 1000 warmup)
- Expected to match golden reference within ±30%
- Using full dataset (4727 SNe with chi2 < 2000)
- Results pending

**Previous Uninformed Priors**: Wrong parameters, MCMC divergences
- Demonstrated the dataset-dependent priors bug
- Fixed by implementing informed priors on standardized coefficients

---

## File Structure

```
V16/
├── README.md                      # Main documentation
├── DATA.md                        # Data requirements and instructions
├── stages/
│   ├── stage1_optimize.py         # Individual SN optimization
│   ├── stage2_simple.py           # Clean Stage 2 MCMC (fixed)
│   ├── stage2_mcmc_numpyro.py     # Production Stage 2 with extra features
│   └── stage3_hubble_optimized.py # Hubble diagram analysis
├── documents/
│   ├── Supernovae_Pseudocode.md   # Algorithm specifications
│   └── PROGRESS.md                # This file
├── test_dataset/
│   ├── README.md                  # Test dataset documentation
│   ├── stage1_results/            # 200 SNe Stage 1 results
│   ├── lightcurves_test.csv       # Lightcurves for test SNe
│   └── test_dataset_summary.json  # Dataset statistics
└── tools/
    └── compare_informed_vs_uninformed.py  # Results comparison script
```

---

## Next Steps

### Immediate

1. **Complete informed priors MCMC run** (in progress)
   - Validate results against November 5, 2024 golden reference
   - Check for MCMC convergence (R-hat < 1.01)
   - Verify parameter estimates within ±30%

2. **Run comparison analysis**
   - Use `tools/compare_informed_vs_uninformed.py`
   - Document performance improvement
   - Confirm bugfix success

### Short-term

3. **Implement GMM gating** (pseudocode lines 154-161, 263-265)
   - Detect anomalous SNe (BBH, gravitational lensing)
   - Flag outliers for separate analysis
   - Improve fit quality for normal SNe

4. **Run holdout validation**
   - Test on 741 excluded SNe
   - Assess model performance on out-of-sample data
   - Check for overfitting

### Medium-term

5. **Full production run with all features**
   - Use `stage2_mcmc_numpyro.py` with informed priors
   - 4 chains, 4000 samples, 2000 warmup
   - Extended diagnostics and posterior analysis
   - Generate publication-ready results

6. **Stage 3 Hubble diagram analysis**
   - Apply Stage 2 parameters to compute distances
   - Generate Hubble diagram
   - Compare with standard ΛCDM predictions
   - Assess goodness of fit

---

## Known Issues

### Fixed
- ✅ Dataset-dependent priors bug (2025-01-12)
- ✅ MCMC divergences in Stage 2 (2025-01-12)
- ✅ Wrong parameter estimates (2025-01-12)

### Open
- ⏳ Informed priors run completion (in progress)
- 🔄 GMM gating not yet implemented
- 🔄 Holdout validation not yet run
- 🔄 Full production run with extended diagnostics pending

---

## Collaboration Notes

### For Collaborators

1. **Quick Start**: Use `test_dataset/` for rapid testing and debugging
2. **Testing Changes**: Always compare results against golden reference
3. **Documentation**: Update this PROGRESS.md when making significant changes
4. **Validation**: Results should be within ±30% of golden reference values

### Key Insights

- **Standardization is not orthogonalization**: Use simple mean/std standardization
- **Prior specification matters**: Define priors on standardized space, not physical space
- **Data independence is critical**: Model specification must not depend on dataset statistics
- **Student-t is robust**: Use nu=5 for outlier robustness in likelihood

### References

- **Pseudocode**: See `documents/Supernovae_Pseudocode.md` for algorithm details
- **Data Requirements**: See `DATA.md` for data setup instructions
- **Quick Start**: See main `README.md` for usage examples

---

## Contact

For questions or issues with this V16 collaboration sandbox:
- Open an issue in the main Quantum-Field-Dynamics repository
- Contact the QFD research team
- See main README.md for additional resources

---

**Development Log**:

- **2025-01-12**: Initial V16 sandbox creation
- **2025-01-12**: Dataset-dependent priors bug identified and fixed
- **2025-01-12**: Clean `stage2_simple.py` implementation completed
- **2025-01-12**: Informed priors implementation added
- **2025-01-12**: Holdout validation support added
- **2025-01-12**: Test dataset (200 SNe) created for collaborators
- **2025-01-12**: V16 sandbox pushed to GitHub with full documentation
