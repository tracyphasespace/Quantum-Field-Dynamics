# 7,754 SNe Dataset Conversion Issue

**Date**: December 22, 2025
**Status**: ❌ BLOCKED - Conversion formula unclear
**Impact**: Cannot use larger dataset for V22 analysis

---

## Summary

An attempt was made to expand the supernova dataset from 1,829 SNe to 7,754 SNe using the raw stage2 QFD fit results. However, the conversion from stage2 parameters to distance modulus has **fundamental issues** that prevent its use.

---

## What We Found

### Source Data
- **File**: `stage2_results_with_redshift.csv` from V21 package
- **Count**: 8,253 SNe from raw DES5yr photometry processing
- **Columns**: snid, n_obs, chi2_dof, stretch, residual, **ln_A**, t0, A_plasma, beta, z

### Attempted Conversion
Based on V21 `plot_canonical_comparison.py`:
```python
mu_obs_uncal = -1.0857 * ln_A
mu_obs = mu_obs_uncal + M_corr + 5.0
```

### Problems Identified
1. **Many identical values**: Most SNe have μ = 42.19 mag (identical!)
2. **Huge residuals**: Mean residual vs Hubble law = -31 mag (should be ~0)
3. **91% outliers**: |residual| > 2 mag for 91% of data
4. **Sentinel values**: Many ln_A = -30.0 (failure flag, not real data)

### Diagnosis
The relationship between `ln_A` from stage2 fits and distance modulus **μ** is more complex than formula used.

**Possible issues**:
- `ln_A` may need preprocessing (exponentiation, normalization)
- Additional parameters (stretch, A_plasma) may be needed
- V21 plotting script may use different data file than stage2
- Missing intermediate processing steps

---

## What Actually Works

**Current Working Dataset**: `des5yr_full.csv`
- **Count**: 1,829 SNe
- **Format**: redshift, distance_modulus, sigma_mu
- **Status**: ✅ Clean, validated, used in V21 successfully
- **Source**: Unknown (possibly SALT-corrected or from different pipeline)

---

## Impact on V22 Analysis

### CLAIM in Documentation
Many files claim:
- "V22 uses 7,754 SNe"
- "4× more data than V21"
- "Avoids circular reasoning"

### REALITY
V22 **actually uses** the same 1,829 SNe as V21:
```python
# Line 263 in v22_qfd_fit_lean_constrained.py
data_path = Path(".../des5yr_full.csv")  # 1,829 SNe, not 7,754!
```

**Result**: All V22 documentation claiming larger dataset is **FALSE**.

---

## Root Cause

1. ✅ **Stage2 data found** (8,253 SNe)
2. ✅ **Conversion script created**
3. ❌ **Conversion formula wrong** → produces garbage
4. ❌ **V22 never updated** → still uses old 1,829 SNe
5. ❌ **Documentation claims success** → misleading

**This is exactly the kind of incomplete work you suspected in the audit.**

---

## Recommendation

**For immediate publication**: Document honestly
- V22 reproduces V21 results with 1,829 SNe ✓
- Lean constraints add rigor ✓
- Claim: "First SN cosmology with Lean-verified constraints" ✓
- Note larger dataset as future work

**For follow-up**: Either:
- Option A: Fix stage2 conversion (1-2 days investigation)
- Option B: Use Pantheon+ dataset (1,701 SNe, well-documented)

---

## Files Created (Not Usable)

❌ `convert_stage2_to_distance_modulus.py` - Wrong formula
❌ `convert_stage2_to_distance_modulus_CORRECTED.py` - Still wrong
❌ `des5yr_raw_qfd_full.csv` - Garbage data
❌ `des5yr_raw_qfd_cleaned.csv` - Garbage data
❌ `des5yr_raw_qfd_CORRECTED.csv` - Still garbage

**Status**: All conversion attempts failed. **Do not use these files.**

---

## Bottom Line

The 7,754 SNe dataset expansion was **incomplete prep work**, not a usable accomplishment. V22 currently uses the same 1,829 SNe as V21, despite documentation claiming otherwise.

**Action Required**: Update all documentation to reflect reality (1,829 SNe), fix conversion as separate follow-up work.

---

**Audit Date**: December 22, 2025
