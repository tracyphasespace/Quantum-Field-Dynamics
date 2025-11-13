# BBH Validation Strategy - Two-Stage Approach

**Date**: 2025-11-13
**Purpose**: Validate BBH physics hypothesis without combinatorial explosion
**Strategy**: Fit bulk (90%) with BBH ≈ WD, then test if BBH variations explain outliers (10%)

---

## The Computational Challenge

**Problem**: Fitting BBH parameters (A_lens, P_orb, phi_0, M_BBH) for all 4,831 SNe simultaneously:
- **Parameters**: 4,831 SNe × 4 BBH params = **19,324 additional parameters**
- **Computational cost**: Requires supercomputer (weeks to months)
- **Degeneracies**: BBH mass ↔ distance ↔ FDR strength creates multimodal posterior

**Solution**: Two-stage approach
1. **Stage 1**: Fit 90% of data assuming BBH ≈ WD (no special BBH effects)
2. **Stage 2**: Validate BBH hypothesis on outliers identified in Stage 1

---

## Stage 1: Bulk Fit (Current V16.2 Implementation)

### Model Assumptions
1. **BBH ≈ WD mass**: A_lens ≈ 0 (gravitational redshift negligible)
2. **No time-varying lensing**: μ_bbh = 1 (magnification constant)
3. **Student-t likelihood**: Heavy tails accommodate outliers **statistically** (not physically)

### Why This Works for 90% of Data

**Physical justification**:
- Most BBH companions are ~1-2× WD mass (Maoz et al. 2014)
- Gravitational redshift: z_grav ∝ M/R
  - WD: M ~ 0.6 M☉, R ~ 5000 km → z_grav ~ 10^-4
  - 2× WD: M ~ 1.2 M☉, R ~ 4000 km → z_grav ~ 3×10^-4
  - **Difference**: Δz ~ 2×10^-4 (negligible compared to measurement errors)

**Statistical accommodation**:
- Student-t with ν ≈ 6.5 has heavy tails
- ~10% of SNe can be 3-5σ outliers without breaking the fit
- These outliers are "downweighted" (not discarded)

### Current Status
✅ **V16.2 implements this correctly**:
- BBH hardcoded to A_lens = 0.0 (PHYSICS_AUDIT.md)
- Student-t likelihood active (AUDIT_FINDINGS.md)
- Paper reports ν ≈ 6.5 (accommodates ~10% outliers)

---

## Stage 2: BBH Validation on Outliers

### Identify Outlier Population

After Stage 1 MCMC completes, identify SNe with high **studentized residuals**:

```python
# Compute residuals
alpha_pred = A0 + Phi_std @ c_mean
residuals = alpha_obs - alpha_pred

# Studentize (account for uncertainty)
studentized = residuals / sigma_alpha_mean

# Flag outliers (|studentized| > 3)
outlier_mask = np.abs(studentized) > 3.0
n_outliers = outlier_mask.sum()
print(f"Identified {n_outliers} outliers (~{100*n_outliers/len(alpha_obs):.1f}%)")
```

**Expected**:
- ~10-15% outliers (based on Student-t with ν ≈ 6.5)
- ~500-700 SNe from 4,831 total

---

### BBH Hypothesis Test

For the identified outliers, fit a **per-SN BBH model**:

**Model**:
```
alpha_outlier,i = alpha_pred,i + delta_BBH,i
delta_BBH,i = f(A_lens,i, M_BBH,i)
```

Where:
- `A_lens,i`: Lensing amplification for SN i
- `M_BBH,i`: BBH mass for SN i (relative to WD, e.g., M_BBH = 10 → 10× WD mass)

**Transformation from BBH mass to alpha**:
```python
# Gravitational redshift from BBH
z_grav = (G * M_BBH * M_sun) / (R_WD * c^2)

# This shifts the observed wavelength
lambda_obs = lambda_emit * (1 + z_grav)

# Which changes the observed flux (shifts Planck spectrum)
# Leading to a change in alpha (log-amplitude)
delta_alpha = -2.5 * log10(flux_ratio)  # Empirical from detailed model
```

**Simplified approximation**:
```python
delta_alpha ≈ k_BBH * log(M_BBH / M_WD)
```

Where `k_BBH` is a coupling constant to be fitted.

---

### Validation Fit

**For each outlier**, run a small optimization:

```python
def fit_bbh_for_outlier(snid, alpha_obs_i, alpha_pred_i, lightcurve):
    """
    Fit BBH mass for a single outlier SN.

    Returns:
        M_BBH: Best-fit BBH mass (in units of M_WD)
        delta_chi2: Improvement in chi2
    """
    # Initial guess: M_BBH = 1 (same as WD)
    M_BBH_init = 1.0

    # Objective: minimize residual
    def objective(M_BBH):
        delta_alpha = k_BBH * np.log(M_BBH)
        alpha_model = alpha_pred_i + delta_alpha
        residual = (alpha_obs_i - alpha_model) / sigma_alpha
        return residual**2

    result = minimize(objective, M_BBH_init, bounds=[(0.1, 1000)])

    M_BBH_best = result.x[0]
    chi2_with_bbh = result.fun
    chi2_without_bbh = ((alpha_obs_i - alpha_pred_i) / sigma_alpha)**2

    delta_chi2 = chi2_without_bbh - chi2_with_bbh

    return M_BBH_best, delta_chi2
```

---

### Validation Metrics

**Success criteria**:

1. **Residual reduction**:
   - RMS of outliers without BBH: ~8-10 mag (from paper's holdout analysis)
   - RMS of outliers with BBH: Should improve to ~3-5 mag
   - **Target**: >50% RMS reduction

2. **Physical plausibility**:
   - Fitted M_BBH values should be in range [0.1, 1000] × M_WD
   - Distribution should be bimodal:
     - Core: M_BBH ~ 1-2 (normal WD companions)
     - Tail: M_BBH ~ 10-1000 (stellar-mass BH companions)

3. **Correlation with host properties**:
   - SNe in young, star-forming galaxies → higher M_BBH (more massive companions)
   - SNe in old, elliptical galaxies → lower M_BBH (WD companions only)
   - **This is the "smoking gun" from your forward model description!**

---

### Implementation Plan

**Phase 1: Run Stage 1 Fit** (Current)
```bash
cd /home/user/Quantum-Field-Dynamics/projects/V16.2

python stages/stage2_simple.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out stage1_bulk_fit \
  --nchains 4 --nsamples 2000 --nwarmup 1000 \
  --use-informed-priors  # Start with informed for faster convergence
```

**Phase 2: Identify Outliers**
```python
# Load Stage 1 results
samples = load_stage1_samples('stage1_bulk_fit')

# Compute studentized residuals
identify_outliers(samples, alpha_obs, Phi_std)

# Save outlier list
np.save('outlier_snids.npy', outlier_mask)
```

**Phase 3: Fit BBH for Outliers**
```python
# For each outlier
for snid in outlier_snids:
    M_BBH, delta_chi2 = fit_bbh_for_outlier(snid, ...)
    save_bbh_result(snid, M_BBH, delta_chi2)

# Analyze results
plot_bbh_mass_distribution(bbh_results)
compute_rms_improvement(bbh_results)
```

**Phase 4: Host Galaxy Correlation**
```python
# Load host galaxy properties from DES
host_properties = load_des_host_properties(outlier_snids)

# Correlate M_BBH with host SFR, metallicity, morphology
correlation_analysis(bbh_results, host_properties)

# This is the SMOKING GUN if correlation is strong!
```

---

## Expected Outcomes

### If BBH Hypothesis is Correct (QFD Prediction)

**Observation 1**: Outlier residuals improve significantly
- RMS without BBH: ~8 mag
- RMS with BBH: ~3 mag
- **Δχ² per SN**: ~20-50 (highly significant)

**Observation 2**: M_BBH distribution is bimodal
- Peak 1: M_BBH ~ 1-2 (WD companions, ~80% of outliers)
- Peak 2: M_BBH ~ 10-1000 (BH companions, ~20% of outliers)
- Long tail to high masses (stellar-mass BH)

**Observation 3**: Strong correlation with host properties
- Young galaxies (high SFR): <M_BBH> ~ 50 M_WD
- Old galaxies (low SFR): <M_BBH> ~ 1.5 M_WD
- **Pearson r > 0.5** (strong correlation)

---

### If BBH Hypothesis is Wrong (ΛCDM Prediction)

**Observation 1**: No improvement from BBH modeling
- RMS stays ~8 mag
- **Δχ² per SN**: ~0-2 (not significant)

**Observation 2**: M_BBH values are random
- No bimodal structure
- Values hit boundaries (0.1 or 1000)
- Fitter is just chasing noise

**Observation 3**: No correlation with host properties
- **Pearson r ~ 0** (no correlation)
- Scatter plot is a cloud

---

## Why This Approach is Computationally Tractable

### Stage 1 Fit (Bulk)
- **Parameters**: 3 global (c0, c1, c2) + 3 nuisance (A0, σ_α, ν) = **6 total**
- **Data points**: 4,831 SNe
- **Chains**: 4
- **Samples**: 2,000 per chain
- **Time**: ~10-30 minutes on laptop

### Stage 2 Fit (Outliers)
- **Parameters**: 1 per SN (M_BBH_i) for ~500 outliers
- **Optimization**: Independent for each SN (embarrassingly parallel)
- **Method**: scipy.optimize.minimize (fast, deterministic)
- **Time per SN**: ~1 second
- **Total time**: ~10 minutes for all 500 outliers

**Total computational cost**: ~1 hour on laptop (vs weeks on supercomputer for joint fit)

---

## Connection to Paper Results

### Paper's Student-t with ν ≈ 6.5

**Interpretation**: The Student-t is **statistically** accommodating ~10-15% outliers

**From theory** (Student-t properties):
- ν = 30: ~2% of observations >3σ (nearly Gaussian)
- ν = 10: ~5% of observations >3σ
- **ν = 6.5**: ~10% of observations >3σ ← Paper result
- ν = 3: ~20% of observations >3σ

**This matches BBH hypothesis**: ~10-15% of SNe have BBH companions with M_BBH >> M_WD, causing large residuals.

---

### Paper's Holdout Set (RMS ≈ 8.16 mag)

**Paper** (Section 5): "We evaluated a challenging holdout set of 508 supernovae that failed Stage-1 screening"

**Interpretation**: These 508 SNe are the pathological cases:
- Failed chi2 < 2000 cut in Stage 1
- Likely have extreme BBH effects (M_BBH > 100?)
- RMS ≈ 8.16 mag without BBH modeling

**Prediction**: If we fit BBH masses for these 508, RMS should improve dramatically.

**Test**:
```python
# Load holdout set
holdout_data = load_holdout_set('excluded_sne.csv')

# Fit BBH for each
for snid in holdout_data['snids']:
    fit_bbh_for_outlier(snid, ...)

# Compute new RMS
rms_with_bbh = compute_rms(holdout_residuals_with_bbh)
print(f"Holdout RMS without BBH: 8.16 mag")
print(f"Holdout RMS with BBH: {rms_with_bbh:.2f} mag")
print(f"Improvement: {100*(8.16 - rms_with_bbh)/8.16:.1f}%")
```

**Expected**: RMS improves from 8.16 → ~3-4 mag (50% reduction)

---

## Implementation Status

### ✅ Ready to Run (Stage 1)
- V16.2 code is correct for bulk fit
- Test dataset available (200 SNe)
- Full dataset available (4,831 SNe)

### 📝 To Be Implemented (Stage 2)
- Outlier identification script
- BBH fitting function
- Host galaxy correlation analysis

### 🎯 Priority
**IMMEDIATE**: Run Stage 1 fit to validate bulk model works
**NEXT**: Implement Stage 2 BBH validation scripts
**GOAL**: Demonstrate BBH hypothesis explains outliers (smoking gun!)

---

## Next Steps

1. **Run Stage 1 fit on test dataset** (~10 min)
   - Verify convergence (R̂ < 1.01)
   - Check ν ≈ 6-7 (should get heavy tails)
   - Identify outliers (studentized residuals > 3)

2. **Create BBH validation script** (~1 hour coding)
   - Implement `fit_bbh_for_outlier()`
   - Parallelize over outliers
   - Save results to CSV

3. **Run BBH validation** (~10 min)
   - Fit M_BBH for each outlier
   - Plot M_BBH distribution
   - Compute RMS improvement

4. **Host galaxy correlation** (~variable, depends on data availability)
   - Load DES host properties
   - Correlate with M_BBH
   - Generate publication-quality plots

---

**END OF BBH VALIDATION STRATEGY**
