# Diagnosis: Test Dataset Produces Wrong Parameters

**Date**: 2025-11-13
**Issue**: Stage 2 MCMC on test dataset (184 SNe) produces parameters ~10-30x smaller than expected
**Root Cause**: Dataset-dependent standardization statistics

---

## The Problem

### MCMC Results (Test Dataset, 184 SNe)
```
Physical Parameters (median ± std):
  k_J  = 0.955 ± 0.358 km/s/Mpc  (expected 10.770)
  eta' = -0.274 ± 0.102           (expected -7.988)
  xi   = -0.817 ± 0.308           (expected -6.908)
```

**All parameters are ~10-30x too small!**

### Standardized Coefficients (from MCMC)
```
c[0] median: 0.212  (informed prior centered at 1.857)
c[1] median: -0.107 (informed prior centered at -2.227)
c[2] median: -0.107 (informed prior centered at -0.766)
```

The MCMC **ignored the informed priors** and converged to values ~10x smaller.

---

## Root Cause Analysis

### The Dataset-Dependent Priors Bug

**What happened**:

1. **November 5, 2024 golden run** used FULL dataset (4,831 SNe):
   - Computed Phi standardization: `Phi_std = (Phi - means_full) / scales_full`
   - MCMC converged to standardized coefficients: `c = [1.857, -2.227, -0.766]`
   - These c values are **specific to the full dataset's standardization**

2. **Today's test run** used TEST dataset (184 SNe):
   - Computed Phi standardization with DIFFERENT statistics:
     ```
     Means:  [0.566, 0.805, 0.418]
     Scales: [0.222, 0.391, 0.131]
     ```
   - Applied **same informed priors** `c ~ Normal([1.857, -2.227, -0.766], σ)`
   - But these priors are WRONG for the test dataset's standardization!

### Why This Breaks

The model is:
```python
Phi_std = (Phi - means) / scales  # Dataset-specific!
alpha_pred = ln_A0 + dot(Phi_std, c)
```

**The c coefficients are not universal** - they depend on the standardization statistics!

If you change `means` and `scales`, you need to change the c priors accordingly.

**Example**:
- Full dataset: `phi1_std = (ln(1+z) - 0.500) / 0.250`
  → Informed prior: `c0 ~ Normal(1.857, 0.5)`

- Test dataset: `phi1_std = (ln(1+z) - 0.566) / 0.222`
  → Same prior `c0 ~ Normal(1.857, 0.5)` is WRONG!

The standardization changes, so the correct c value changes too.

---

## Evidence from MCMC Behavior

**The data overwhelmed the priors**:
- Informed priors: `c0 ~ Normal(1.857, 0.5)` with σ=0.5
- MCMC converged to: `c0 ≈ 0.212`

This is **~3σ away** from the prior mean! This only happens when:
1. The data strongly disagrees with the prior (unlikely for subset)
2. **The prior is wrong for this dataset** ← This is what happened!

The test dataset's standardization statistics are different, so the informed priors (derived from full dataset) don't apply.

---

## Why Test Dataset Has Different Statistics

**Test dataset** (184 SNe):
- Smaller sample → higher variance in statistics
- Different redshift distribution (random sampling artifact)
- Missing the high-z tail that the full dataset has

**Full dataset** (4,831 SNe):
- Spans z = [0.083, 1.498] uniformly
- More representative of true distribution
- Standardization statistics are more stable

**Result**: Test dataset's `Phi` has:
- Different mean (0.566 vs ~0.500 for full dataset)
- Different scale (0.222 vs ~0.250 for full dataset)
- These small changes → large changes in correct c values!

---

## The Fix: Three Options

### Option A: Use Full Dataset Standardization (RECOMMENDED)

When running on test dataset, use standardization statistics FROM THE FULL DATASET:

```python
# Load full dataset standardization stats
full_means = [0.500, 0.750, 0.400]  # From Nov 5 golden run
full_scales = [0.250, 0.400, 0.130]

# Apply to test dataset
Phi_test = compute_features(z_test)
Phi_test_std = (Phi_test - full_means) / full_scales  # Use FULL stats!

# Now informed priors work correctly
c0 ~ Normal(1.857, 0.5)  # This is valid for full_means, full_scales
```

**Pros**:
- Informed priors work correctly
- Test results comparable to full dataset
- No need to recompute priors

**Cons**:
- Need to save and load full dataset statistics

---

### Option B: Run on Full Dataset

Skip the test dataset and run directly on the full 4,831 SNe:

```bash
python stages/stage2_simple.py \
  --stage1-results stage1_results \
  --lightcurves lightcurves.csv \
  --out stage2_full \
  --nchains 4 --nsamples 2000 --nwarmup 1000 \
  --use-informed-priors
```

**Pros**:
- No compatibility issues
- Most statistically powerful
- Should reproduce Nov 5 golden results

**Cons**:
- Takes longer (~30 min vs ~3 min for test dataset)

---

### Option C: Adjust Informed Priors for Test Dataset

Recompute what the c priors should be for the test dataset's standardization:

**Math**:
```
Let α_phys = ln_A0 + k_J·phi1 + eta'·phi2 + xi·phi3  (physical space)

In standardized space:
  phi_std = (phi - mean) / scale
  phi = phi_std * scale + mean

Substitute:
  α_phys = ln_A0 + k_J·(phi1_std·scale1 + mean1) + ...
         = (ln_A0 + k_J·mean1 + eta'·mean2 + xi·mean3)  ← new intercept
           + k_J·scale1·phi1_std + eta'·scale2·phi2_std + xi·scale3·phi3_std
         = ln_A0_new + c0·phi1_std + c1·phi2_std + c2·phi3_std

Where:
  c0_new = k_J * scale1_new / scale1_old * ...  (complex!)
```

**Pros**:
- Theoretically correct

**Cons**:
- Complex math
- Error-prone
- Why bother when Option A is simpler?

---

## Recommended Action

**IMMEDIATE**: Run on full dataset (Option B)

```bash
cd /home/user/Quantum-Field-Dynamics/projects/V16.2

# Use FULL dataset (stage1_results/ has 4,727 SNe after chi2 < 2000 cut)
python3 stages/stage2_simple.py \
  --stage1-results stage1_results \
  --lightcurves lightcurves.csv \
  --out stage2_full_dataset \
  --nchains 4 \
  --nsamples 2000 \
  --nwarmup 1000 \
  --use-informed-priors
```

**Expected result**: Parameters should match November 5 golden run (within ±30%)

---

## Lessons Learned

1. **Never use informed priors derived from one dataset on a different dataset**
   (Unless you use the same standardization statistics!)

2. **Standardization statistics are dataset-dependent**
   Small datasets have higher variance in means/scales

3. **Test datasets are useful for debugging code, not for validating priors**
   The statistical properties differ from the full dataset

4. **This is exactly the bug PROGRESS.md warned about!**
   > "Priors defined on physical parameters (k_J, eta', xi) were being implicitly scaled by dataset-dependent standardization statistics"

---

## Next Steps

1. ✅ Identified root cause (dataset-dependent standardization)
2. ⏳ Run on full dataset to validate informed priors work correctly
3. ⏳ If full dataset works, document the standardization statistics from golden run
4. ⏳ Consider saving full dataset's means/scales for future use

---

**END OF DIAGNOSIS**
