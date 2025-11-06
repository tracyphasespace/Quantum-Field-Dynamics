# A/B/C Testing Framework for Sign Constraint Variants

## Overview

This document describes the A/B/C testing framework implemented to address the basis collinearity issue discovered in Stage 2 MCMC, which was causing sign ambiguity in the fitted parameters (k_J, η', ξ).

## The Problem

### Root Cause: Basis Collinearity

The three basis functions used in the QFD model are nearly perfectly correlated:

```
φ₁ = ln(1+z)
φ₂ = z
φ₃ = z/(1+z)
```

**Diagnostic findings** (see `scripts/check_basis_correlation.py`):
- Correlation matrix shows r > 0.99 between all pairs
- Condition number κ ≈ 2.1×10⁵ (should be < 100 for well-conditioned systems)
- This creates a **sign ambiguity**: multiple coefficient combinations produce nearly identical fits

### Symptom: Wrong Monotonicity

The current best-fit has:
- k_J ≈ +10.7 (positive)
- η' ≈ -8.0 (NEGATIVE)
- ξ ≈ -6.9 (NEGATIVE)

Since α(z) = -(k_J·φ₁ + η'·φ₂ + ξ·φ₃), the negative η' and ξ counteract the leading minus sign, causing α(z) to INCREASE with z instead of decrease. This violates the expected monotonicity.

## The Solution: Three Model Variants

We implement three variants to compare band-aid fixes vs root-cause solutions:

### Model A: Unconstrained (Baseline)
- **Flag**: `--constrain-signs off`
- **Description**: Current behavior - unconstrained Normal priors on standardized coefficients c
- **Purpose**: Baseline for comparison
- **Expected**: Good RMS, but fails monotonicity check

### Model B: Constrained (Symptom Fix)
- **Flag**: `--constrain-signs alpha`
- **Description**: Forces c ≤ 0 using HalfNormal priors: `c = -|c_raw|`
- **Purpose**: Guarantees α(z) is non-increasing without changing basis
- **Expected**: Passes monotonicity by construction, similar RMS

### Model C: Orthogonal (Root Cause Fix)
- **Flag**: `--constrain-signs ortho`
- **Description**: Uses QR-decomposed orthogonal basis to eliminate collinearity
- **Purpose**: Fixes the fundamental numerical instability
- **Expected**: Better conditioning, stable signs, equivalent or better fit quality

### Model D: Physics-Space (Optional)
- **Flag**: `--constrain-signs physics`
- **Description**: Works in physics-space with k_J, η', ξ ≥ 0 constraints
- **Purpose**: Alternative constraint approach with direct physical interpretation
- **Expected**: Similar to Model B but with interpretable physical parameters

## Implementation Details

### CLI Usage

All variants use the same Stage 2 MCMC script with the `--constrain-signs` flag:

```bash
# Model A: Unconstrained baseline
python src/stage2_mcmc_numpyro.py \
  --stage1-results results/v15_stage1_production \
  --lightcurves data/lightcurves.csv \
  --out results/variant_A \
  --constrain-signs off

# Model B: Alpha-space constraint (c ≤ 0)
python src/stage2_mcmc_numpyro.py \
  --stage1-results results/v15_stage1_production \
  --lightcurves data/lightcurves.csv \
  --out results/variant_B \
  --constrain-signs alpha

# Model C: Orthogonalized basis
python src/stage2_mcmc_numpyro.py \
  --stage1-results results/v15_stage1_production \
  --lightcurves data/lightcurves.csv \
  --out results/variant_C \
  --constrain-signs ortho
```

### Automated Comparison

Use the comparison script to run all variants and generate a decision table:

```bash
# Quick test with subset (1000 samples per chain)
python scripts/compare_abc_variants.py \
  --nsamples 1000 \
  --nwarmup 500

# Full production run (2000 samples per chain)
python scripts/compare_abc_variants.py \
  --nsamples 2000 \
  --nwarmup 1000
```

## Model Comparison Metrics

The framework computes and compares:

### 1. Convergence Diagnostics
- **Divergences**: Should be 0 for all variants
- **R̂ (Gelman-Rubin)**: Should be ≈ 1.00
- **ESS (Effective Sample Size)**: Should be > 5000

### 2. Model Selection Criteria
- **WAIC** (Widely Applicable Information Criterion)
- **LOO** (Leave-One-Out Cross-Validation via PSIS)
- Higher ELPD values are better
- Differences < 2σ are considered equivalent

### 3. Fit Quality
- **RMS(μ)**: Root-mean-square residuals from Stage 3
- **Residual slope**: Trend of residuals vs redshift
- Differences < 0.01 mag are considered equivalent

### 4. Boundary Diagnostics
- **Model B/D**: Fraction of samples at constraint boundaries (< 5% is good)
- **Model A/C**: Sign distribution of coefficients

### 5. Basis Conditioning
- **Condition number**: κ(Φ) or κ(Q)
- Original basis: κ ≈ 2×10⁵
- Orthogonalized: κ < 50 (expected)

## Decision Framework

### Decision Rule

1. **If Model C passes all checks AND (WAIC ≥ best - 2σ) AND (RMS ≤ best + 0.01)**:
   - **✅ ADOPT MODEL C** as the default
   - Rationale: Fixes root cause, no performance penalty

2. **Else if Model B passes all checks AND outperforms C**:
   - **⚠️ ADOPT MODEL B** as pragmatic fix
   - Document that collinearity remains but is constrained

3. **Else if both B and C degrade fit materially**:
   - **📊 KEEP MODEL A** and document as scientific finding
   - This would suggest data genuinely prefers the ambiguous solution

### Success Criteria Table

| Metric | Condition | Expected A | Expected B | Expected C |
|--------|-----------|-----------|-----------|-----------|
| Monotonicity | Violations = 0 | ❌ FAIL | ✅ PASS | ✅ PASS |
| RMS(μ) | Δ ≤ 0.01 mag | 1.888 (baseline) | ≈ 1.888 | ≈ 1.888 |
| WAIC/LOO | Higher is better | Baseline | ? | ? |
| Convergence | R̂ ≈ 1.0, ESS > 5k | ✅ PASS | ? | ✅ PASS |
| Divergences | = 0 | ✅ 0 | ? | ✅ 0 |
| Boundary hits | < 5% | 0% | ? | 0% |
| Param corr | |r| < 0.9 | ❌ r ≈ -0.97 | ? | ✅ |
| Basis cond | κ < 100 | ❌ κ ≈ 2×10⁵ | ❌ | ✅ κ < 50 |

## Technical Implementation

### Code Changes

**Modified files:**
- `src/stage2_mcmc_numpyro.py`:
  - Added `--constrain-signs` CLI argument
  - Modified `numpyro_model_alpha_space()` to support all variants
  - Added WAIC/LOO computation with ArviZ
  - Added boundary diagnostics
  - Save variant metadata in output JSON

**New functions:**
- `_orthogonalize_basis(Φ)`: QR decomposition for Model C
- Variant-specific prior sampling logic in model

**New scripts:**
- `scripts/compare_abc_variants.py`: Automated A/B/C comparison
- `scripts/check_basis_correlation.py`: Diagnostic tool for collinearity

**New tests:**
- `tests/test_backtransform_identity.py`: Validates coefficient transformations

### Output Structure

Each variant produces:
```
results/abc_comparison_TIMESTAMP/
├── A_unconstrained/
│   ├── samples.json          # Contains WAIC/LOO metrics
│   ├── best_fit.json         # Contains variant metadata
│   ├── summary.json
│   └── *.npy
├── B_constrained/
│   └── ...
├── C_orthogonal/
│   └── ...
├── comparison_table.csv      # Summary comparison
└── comparison_table.json
```

## Diagnostic Tools

### Check Basis Correlation
```bash
python scripts/check_basis_correlation.py
```
Output shows correlation matrix and condition number for both original and orthogonalized bases.

### Manual Verification
```python
import json
import numpy as np

# Load variant results
with open('results/variant_C/samples.json') as f:
    samples = json.load(f)

print(f"WAIC: {samples['waic']:.2f} ± {samples['waic_se']:.2f}")
print(f"LOO:  {samples['loo']:.2f} ± {samples['loo_se']:.2f}")
print(f"Divergences: {samples['n_divergences']}")
print(f"Variant: {samples['constrain_signs_variant']}")
```

## Expected Outcomes

Based on the root cause analysis:

**Model C (Orthogonal) is expected to:**
- Reduce condition number from 2×10⁵ to < 50
- Eliminate sign ambiguity without hard constraints
- Produce narrower, more interpretable posteriors
- Maintain or slightly improve RMS
- Pass monotonicity check (likely even without constraints)
- Have 0 divergences and excellent convergence

**Model B (Constrained) is expected to:**
- Pass monotonicity check by construction
- Maintain baseline RMS (constraints shouldn't hurt fit if they match data)
- Potentially show some samples near boundaries if constraint is tight
- Still have high basis condition number (doesn't fix root cause)

**Model A (Unconstrained) baseline:**
- Known to fail monotonicity (1499/1499 violations)
- Good RMS (1.888 mag, 23.9% better than ΛCDM)
- 0 divergences, good convergence
- High parameter correlations (r ≈ -0.97)

## Next Steps

1. **Run quick test** with subset (~1200 SNe, 1000 samples)
2. **Review comparison table** and check decision framework
3. **If Model C wins**: Run full production with `--constrain-signs ortho`
4. **Generate final figures** with winning variant
5. **Update paper** with A/B/C comparison as evidence of robust model building

## References

- Cloud.txt recommendations (A/B/C testing proposal)
- MONOTONICITY_FINDINGS.md (original diagnostic)
- Basis correlation analysis: `results/v15_production/figures/basis_correlation.png`
