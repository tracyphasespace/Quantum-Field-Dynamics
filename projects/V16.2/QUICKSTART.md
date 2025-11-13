# V16.2 Quick Start - Forward Validation Approach

## The Problem

Original code to derive hardcoded priors was lost. Need to validate whether published parameters are correct.

## The Solution

**Work backwards!** Use published parameters and see how well they fit the data.

---

## Quick Validation (Test Dataset)

```bash
cd /home/user/Quantum-Field-Dynamics/projects/V16.2

python3 tools/validate_published_params.py \
  --stage1-results test_dataset/stage1_results \
  --lightcurves test_dataset/lightcurves_test.csv \
  --out validation_test
```

**Expected output:**
- Takes ~20 seconds
- Generates Hubble diagram and residual plots
- Reports RMS residual (should be ~1.4 mag if parameters are correct)

---

## Full Validation (All 5500 SNe)

If you have the full Stage 1 results and lightcurves:

```bash
python3 tools/validate_published_params.py \
  --stage1-results /path/to/full/stage1_results \
  --lightcurves /path/to/lightcurves_unified_v2_min3.csv \
  --out validation_full \
  --quality-cut 2000
```

---

## What to Look For

### ✅ Success (Parameters are correct)
```
Goodness of Fit:
  RMS residual:  1.398 mag     ← Close to published 1.4 mag
  Mean residual: 0.002 mag     ← Near zero
  Std residual:  1.397 mag

Outliers (>3σ):
  Fraction: 2.7%               ← Small outlier fraction
```

### ⚠️ Warning (Something may be wrong)
```
Goodness of Fit:
  RMS residual:  3.5 mag       ← Much worse than expected!
  Mean residual: 1.2 mag       ← Systematic bias
```

---

## Output Files

In the output directory you'll find:

- `hubble_diagram.png` - Visual check of model fit
- `residual_histogram.png` - Distribution of residuals
- `validation_summary.json` - Numerical goodness-of-fit metrics
- `outlier_snids.txt` - List of poorly-fitting SNe (BBH candidates?)

---

## Next Steps

### If validation succeeds (RMS ≈ 1.4 mag):
✅ Published parameters are correct!
✅ Hardcoded priors are validated!
✅ Can use V16 with confidence!

Close the recovery issue and document findings.

### If validation fails (RMS >> 1.4 mag):
⚠️ Need to investigate:
1. Check Stage 1 results quality
2. Verify data matches paper dataset
3. May need to re-run Stage 1 or Stage 2

---

## Why This Works

**Traditional approach** (expensive):
1. Run Stage 1: Optimize 5000 SNe (~10 hours GPU)
2. Run Stage 2: MCMC to fit k_J, η', ξ (~2 hours GPU)
3. Check if results match published values
4. **Total: ~12 hours**

**This approach** (smart):
1. Use published k_J, η', ξ directly
2. Compute predictions (simple arithmetic)
3. Calculate residuals and plot
4. **Total: ~20 seconds**

**2000x faster!**

---

## Dependencies

```bash
pip install numpy scipy matplotlib pandas jax
```

Or if already in QFD environment:
```bash
# Should already have everything needed
```

---

## Help

For more details:
- See `tools/README_VALIDATION.md` for full documentation
- See `RECOVERY.md` for the full recovery plan
- See `documents/` for published papers with parameter values

---

**Created**: 2025-11-13
**Purpose**: Validate published parameters via forward model evaluation
**Credit**: Working backwards strategy (graphing not supercomputing!)
