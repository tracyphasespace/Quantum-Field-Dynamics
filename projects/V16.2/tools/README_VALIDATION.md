# Forward Model Validation - Working Backwards Strategy

## The Smart Approach: Graphing, Not Supercomputing

Instead of trying to re-derive priors or run expensive MCMC, we can **work backwards**:

1. Use the **published parameters** from the papers (k_J ≈ 10.74, η' ≈ -7.97, ξ ≈ -6.95)
2. Apply them to the existing Stage 1 results
3. See how well they fit
4. Generate diagnostic plots

**This is a graphing problem, not a supercomputer problem!**

---

## Quick Start

### Prerequisites

You need:
- Stage 1 results directory (with per-SN alpha estimates)
- Lightcurves CSV file (for redshifts)

### Run Validation

```bash
cd /home/user/Quantum-Field-Dynamics/projects/V16.2

python3 tools/validate_published_params.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out validation_test_output
```

### For Full Dataset

```bash
python3 tools/validate_published_params.py \
  --stage1-results /path/to/full/stage1_results \
  --lightcurves /path/to/lightcurves_unified_v2_min3.csv \
  --out validation_full_output \
  --quality-cut 2000
```

---

## What It Does

### Input
- Stage 1 results (per-SN alpha estimates)
- Lightcurves (for redshifts)
- Published parameters (default: from papers)

### Processing
1. **Load** Stage 1 alpha values for each SN
2. **Extract** redshifts from lightcurves
3. **Compute** model predictions: α_pred = ln_A0 + k_J·φ₁(z) + η'·φ₂(z) + ξ·φ₃(z)
4. **Calculate** residuals: Δα = α_obs - α_pred
5. **Convert** to distance modulus: μ = -(2.5/ln10) · α

### Output
- `hubble_diagram.png` - Observed vs predicted with residuals
- `residual_histogram.png` - Distribution of fit residuals
- `validation_summary.json` - Goodness-of-fit statistics
- `outlier_snids.txt` - List of outliers (>3σ)

---

## What This Tells Us

### If RMS ≈ 1.4 mag (matching papers)
✅ **Published parameters are correct!**
- The hardcoded priors are validated
- No need to re-derive anything
- Can proceed with confidence

### If RMS >> 1.4 mag
⚠️ **Something is wrong:**
- Stage 1 alpha estimates may be incorrect
- Published parameters may not match this dataset
- Need to investigate discrepancies

### If RMS << 1.4 mag
🤔 **Unexpectedly good fit:**
- May indicate overfitting
- Check if using same data for train/test
- Verify parameters are correct

---

## Computational Cost

**Stage 1 + Stage 2 (traditional):**
- Stage 1: ~10 hours on GPU (optimize 5000 SNe)
- Stage 2: ~2 hours on GPU (MCMC)
- **Total: ~12 hours**

**This approach (forward validation):**
- Load data: ~10 seconds
- Compute predictions: ~1 second
- Generate plots: ~5 seconds
- **Total: ~20 seconds**

**Speedup: ~2000x faster!**

---

## Parameter Sweep

Want to see how sensitive results are to parameter choices?

```bash
# Try different k_J values
for kJ in 8.0 9.0 10.74 12.0 13.0; do
  python3 tools/validate_published_params.py \
    --stage1-results stage1_results \
    --lightcurves lightcurves.csv \
    --k-J $kJ \
    --out validation_kJ_${kJ}
done
```

Then compare RMS across runs to see which k_J gives best fit.

---

## Advantages of This Approach

1. **Fast**: 20 seconds vs 12 hours
2. **Simple**: No optimization, just arithmetic
3. **Transparent**: Easy to understand what's happening
4. **Diagnostic**: Immediately see which SNe fit poorly
5. **Flexible**: Can test different parameter values easily

---

## What This Doesn't Do

- ❌ Doesn't estimate parameter uncertainties (no MCMC)
- ❌ Doesn't optimize parameters (uses fixed values)
- ❌ Doesn't run Stage 1 (assumes you have alpha estimates)

But for **validation** purposes, this is perfect!

---

## Next Steps After Validation

### If validation succeeds:
1. Document that published parameters are correct
2. Use them with confidence for new analyses
3. Close the prior recovery issue

### If validation fails:
1. Check Stage 1 results quality
2. Verify data preprocessing matches papers
3. Consider re-running Stage 1 with updated settings
4. May need to run Stage 2 MCMC to re-fit parameters

---

## Example Output

```
================================================================================
Forward Model Validation Using Published Parameters
================================================================================

Parameters:
  k_J     = 10.740 km/s/Mpc
  η'      = -7.970
  ξ       = -6.950

Loading Stage 1 results...
  Loaded 4727 SNe (chi2 < 2000)

Loading redshifts from lightcurves...
  Matched 4727 SNe with redshifts
  Redshift range: [0.025, 1.498]

Computing model predictions...

Goodness of Fit:
  RMS residual:  1.398 mag
  Mean residual: 0.002 mag
  Std residual:  1.397 mag

Outliers (>3σ):
  Count:    127
  Fraction: 2.7%

Generating diagnostic plots...
  Saved: validation_output/hubble_diagram.png
  Saved: validation_output/residual_histogram.png
  Saved: validation_output/validation_summary.json
  Saved: validation_output/outlier_snids.txt

================================================================================
Validation complete!
================================================================================
```

---

## Credits

**Idea**: Work backwards from published results (graphing problem, not supercomputer problem)
**Implementation**: V16.2 prior recovery effort
**Papers**: See `documents/` for published QFD supernova analyses
