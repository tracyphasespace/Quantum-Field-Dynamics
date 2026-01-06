# V18 Components Missing from V21 - Restoration Guide

**Date**: December 22, 2025
**Purpose**: Guide for adding V18's MCMC pipeline to V21 to achieve good cosmology fits
**Target**: Reproduce V18's RMS=2.18 mag with 4,885 SNe

---

## Executive Summary

**V21 is incomplete** - it only has Stage1 fitting and plotting, missing the critical **Stage2 MCMC** that makes V18 work.

**V18 Pipeline** (Working - RMS=2.18 mag):
```
Stage1 (per-SN) → Stage2 MCMC (global params) → Stage3 (Hubble diagram)
```

**V21 Pipeline** (Incomplete - η=0, χ²/dof=176):
```
Stage1 (per-SN) → ??? → Plotting only
```

**To Fix**: Add V18's Stage2 and Stage3 to V21.

---

## What V21 Has ✅

### 1. Stage1 Fitting ✅
- **File**: `stage1_v20_fullscale.py` + `stage1_v20_fullscale_runner.py`
- **What it does**: Fits per-SN parameters (t0, ln_A, A_plasma, beta) for each supernova
- **Output**: 8,253 JSON files in `results/stage1_output/`
- **Status**: ✅ Working, memory-optimized

### 2. Data Loading ✅
- **File**: `v17_data.py`
- **What it does**: Loads lightcurve data
- **Status**: ✅ Working

### 3. Physics Models ✅
- **Files**: `v17_qfd_model.py`, `v17_lightcurve_model.py`
- **What they do**: QFD physics calculations (static universe, D=z×c/k_J)
- **Status**: ✅ Working

### 4. Plotting ✅
- **File**: `plot_canonical_comparison.py`
- **What it does**: Generates time dilation test and Hubble diagram
- **Status**: ✅ Working, but gets poor fit (η=0, χ²/dof=176) because no Stage2 MCMC

### 5. BBH Analysis ✅
- **File**: `analyze_bbh_candidates.py`
- **What it does**: Searches for BBH lensing signals
- **Status**: ✅ Working

---

## What V21 is MISSING ❌

### 1. Stage2 MCMC (CRITICAL) ❌

**Location in V18**: `/v18/pipeline/stages/stage2_mcmc_v18_emcee.py`

**What it does**:
- Takes Stage1 per-SN results (ln_A values)
- Fits **global QFD parameters** using emcee MCMC:
  - k_J_correction
  - eta_prime
  - xi
  - sigma_ln_A
- Outputs posterior samples and best-fit values

**Why it's critical**:
- Without Stage2 MCMC, you can't fit global cosmology parameters properly
- V21's plotting script just uses simplified model with η parameter, gets poor fit
- V18's Stage2 MCMC achieves RMS=2.18 mag (15.8% better than ΛCDM)

**V18 Results**:
```json
{
  "k_J_correction": 19.96 ± 0.06 km/s/Mpc,
  "eta_prime": -5.999 ± 0.002,
  "xi": -5.998 ± 0.003,
  "sigma_ln_A": 1.0000 ± 0.00008
}
```

**V18 Performance**:
- 4,885 SNe used (after quality cuts)
- RMS = 2.18 mag
- χ²/ν reasonable
- All parameters well-constrained

### 2. Stage3 Hubble Diagram (Needed for validation) ❌

**Location in V18**: `/v18/pipeline/stages/stage3_v18.py`

**What it does**:
- Takes Stage1 + Stage2 results
- Calculates distance moduli using best-fit global parameters
- Generates Hubble diagram with proper calibration
- Outputs: `hubble_data.csv` with columns:
  - snid, z, ln_A, mu_obs, mu_qfd, mu_lcdm, residual_qfd, residual_lcdm, chi2_per_obs

**Why it's needed**:
- Proper calibration of distance moduli
- Validation against V18 benchmark (RMS=2.18 mag)
- Outputs usable for V22 parameter fitting

### 3. emcee MCMC Library ❌

**Required**: `emcee` Python package for MCMC sampling

**Install**:
```bash
pip install emcee
```

**Used in**: `stage2_mcmc_v18_emcee.py`

---

## File Locations to Copy from V18

### Critical Files (Stage2 MCMC)

**Primary**:
```
v18/pipeline/stages/stage2_mcmc_v18_emcee.py  →  V21/stage2_mcmc_v18_emcee.py
```

**Supporting** (if needed):
```
v18/core/v17_qfd_model.py         →  Already in V21 ✓
v18/core/v17_lightcurve_model.py  →  Already in V21 ✓
v18/core/v17_data.py              →  Already in V21 ✓
v18/core/pipeline_io.py           →  May need to copy
```

### Stage3 Files (Hubble Diagram)

**Primary**:
```
v18/pipeline/stages/stage3_v18.py  →  V21/stage3_v18.py
```

### Dependencies

Check V18's requirements:
```
v18/requirements.txt  →  Compare with V21/requirements.txt
```

Likely additions needed:
- `emcee` (MCMC sampler)
- `corner` (for corner plots - optional but useful)

---

## Integration Steps

### Step 1: Copy Stage2 MCMC
```bash
# Copy Stage2 from V18
cp v18/pipeline/stages/stage2_mcmc_v18_emcee.py V21/

# Install emcee if not present
pip install emcee
```

### Step 2: Copy Stage3 Hubble
```bash
# Copy Stage3 from V18
cp v18/pipeline/stages/stage3_v18.py V21/
```

### Step 3: Run Full Pipeline
```bash
# Stage1 (already done in V21)
# Results in: results/stage1_output/*.json

# Stage2 MCMC (NEW - from V18)
python3 stage2_mcmc_v18_emcee.py \
  --stage1-results results/stage1_output \
  --lightcurves data/lightcurves_all_transients.csv \
  --out results/stage2_mcmc \
  --use-ln-a-space \
  --constrain-signs informed

# Stage3 Hubble Diagram (NEW - from V18)
python3 stage3_v18.py \
  --stage1-results results/stage1_output \
  --stage2-results results/stage2_mcmc \
  --out results/stage3_hubble \
  --quality-cut 2000
```

### Step 4: Validate Results

**Expected output** (should match V18):
```
Stage2 Best-Fit:
  k_J_correction ≈ 19.96 ± 0.06
  eta_prime ≈ -5.999 ± 0.002
  xi ≈ -5.998 ± 0.003

Stage3 Performance:
  N_SNe: ~4,885 (after quality cuts)
  RMS: ~2.18 mag
  QFD vs ΛCDM: ~15-16% improvement
```

**Compare**:
```bash
# V18 reference
cat v18/results/stage2_fullscale_5468sne/summary.json
cat v18/results/stage3_hubble/summary.json

# V21 new results (should match)
cat V21/results/stage2_mcmc/summary.json
cat V21/results/stage3_hubble/summary.json
```

---

## Expected Differences V18 vs V21

### Things that SHOULD be the same:
- ✅ Best-fit parameters (k_J_correction, eta_prime, xi) within uncertainties
- ✅ RMS ≈ 2.18 mag
- ✅ ~4,885 SNe used
- ✅ QFD improvement over ΛCDM

### Things that MAY differ slightly:
- ⚠️ Exact number of SNe (depends on quality cuts in Stage1)
- ⚠️ MCMC convergence details (random sampling)
- ⚠️ Numerical precision (different runs, different machines)

### Red flags (if these happen, something is wrong):
- ❌ RMS >> 2.5 mag (should be ~2.18)
- ❌ Parameters hit bounds (k_J=100, eta_prime=0, etc.)
- ❌ Very few SNe used (<1000)
- ❌ Non-convergence in MCMC

---

## Troubleshooting

### Issue 1: Import Errors
**Problem**: `ImportError: cannot import name 'X' from 'Y'`

**Solution**: Check that all V18 core modules are present:
```bash
ls -la V21/v17_*.py
# Should have: v17_qfd_model.py, v17_lightcurve_model.py, v17_data.py
```

Copy missing files from V18/core/

### Issue 2: emcee Not Found
**Problem**: `ModuleNotFoundError: No module named 'emcee'`

**Solution**:
```bash
pip install emcee
```

### Issue 3: Stage1 Results Format Mismatch
**Problem**: Stage2 can't read Stage1 output

**Solution**: Check V18 Stage1 output format vs V21 format. They should both be JSON with keys: `best_fit_params`, `success`, `final_neg_logL`, etc.

May need to regenerate Stage1 results with V18's stage1 code.

### Issue 4: Poor Fit Results
**Problem**: RMS >> 2.5 mag, parameters at bounds

**Likely causes**:
1. Using wrong data (SALT-corrected instead of raw)
2. Sign errors in ln_A conversion
3. Missing quality cuts
4. Wrong lightcurve matching

**Solution**: Compare data provenance:
- V18 uses: Raw DES5yr photometry
- V21 uses: Same raw data
- Check: `data/lightcurves_all_transients.csv` matches V18's input

---

## Success Criteria

**V21 + V18 components is successful if**:

1. ✅ Stage2 MCMC runs without errors
2. ✅ Parameters converge: k_J_correction ≈ 20, eta_prime ≈ -6, xi ≈ -6
3. ✅ Stage3 produces RMS ≈ 2.18 mag
4. ✅ ~4,885 SNe pass quality cuts
5. ✅ Results match V18 within uncertainties

**Then you can**:
- Use V21+V18 as basis for V22
- Apply Lean constraints to validated parameters
- Attempt multi-parameter fits for 15+ schema parameters
- Publish results with confidence

---

## Summary for New AI

**What to do**:
1. Copy `stage2_mcmc_v18_emcee.py` from V18 to V21
2. Copy `stage3_v18.py` from V18 to V21
3. Install `emcee`
4. Run Stage2 on V21's Stage1 results
5. Run Stage3 to validate
6. Compare results to V18 benchmark

**Expected outcome**: RMS ≈ 2.18 mag, matching V18 performance

**Then**: Use these working results for V22 Lean constraint validation

---

**Files Referenced**:
- V18 Stage2: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/pipeline/stages/stage2_mcmc_v18_emcee.py`
- V18 Stage3: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/pipeline/stages/stage3_v18.py`
- V18 Results: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/results/stage2_fullscale_5468sne/summary.json`
- V21 Base: `/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/projects/astrophysics/V21 Supernova Analysis package/`

**Status**: Ready for new AI to implement
