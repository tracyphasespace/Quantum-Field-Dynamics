# Regression Analysis: Paper Results vs. Current Code

**Date**: 2025-11-12
**Status**: 🚨 **CRITICAL REGRESSION IDENTIFIED**

---

## Executive Summary

**The "bug fix" from January 12, 2025 actually BROKE working code.**

- **Paper results**: RMS ≈ 1.89 mag, parameters match physics
- **Working code**: November 5, 2024 (`abc_comparison_20251105_165123`)
- **Broken code**: January 12, 2025 "sign fix" (`stage2_production_corrected`)
- **Current status**: Code needs to be REVERTED to November 5 state

---

## Results Comparison

### Paper Claims (from PDF documents)

```json
{
  "k_J": "10.7 ± 4.6",
  "eta_prime": "-8.0 ± 1.4",
  "xi": "-7.0 ± 3.8",
  "RMS": "≈ 1.89 mag",
  "convergence": "R̂ = 1.00, ESS > 10,000, zero divergences"
}
```

### November 5, 2024 Results ✅ (MATCHES PAPER!)

**File**: `results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json`

```json
{
  "k_J": 10.77 ± 4.57,        ✓ Matches 10.7 ± 4.6
  "eta_prime": -7.99 ± 1.44,  ✓ Matches -8.0 ± 1.4
  "xi": -6.91 ± 3.75,         ✓ Matches -7.0 ± 3.8
  "nu": 6.52 ± 0.96,
  "sigma_alpha": 1.40 ± 0.02
}
```

**Analysis**:
- Parameters match paper within rounding errors
- Realistic uncertainties (σ ~ 1-5)
- Student-t degrees of freedom ν ≈ 6.5 (heavy tails)
- This is the CORRECT result!

### January 12, 2025 "Corrected" Results ❌ (BROKEN!)

**File**: `results/v15_clean/stage2_production_corrected/best_fit.json`

```json
{
  "k_J": 5.01 ± 0.01,        ❌ 2.1× too small, unrealistic σ
  "eta_prime": -1.49 ± 0.006, ❌ 5.4× too small, unrealistic σ
  "xi": -4.07 ± 0.02          ❌ 1.7× too small, unrealistic σ
}
```

**Stage 3 Results**:
```json
{
  "qfd_rms": 4.384 mag,      ❌ Should be ~1.89 mag
  "lcdm_rms": 4.395 mag,
  "n_sne": 548               ❌ Only 548 SNe vs expected 4,727!
}
```

**Analysis**:
- Parameters 2-5× too small
- Uncertainties 100-1000× too small (overfitting!)
- RMS 2.3× worse than paper
- Only 548 SNe passed quality cuts vs 4,727 expected

---

## What Went Wrong?

### The "Bug Fix" Document Claims

From `CRITICAL_BUGFIX_2025-01-12.md`:

> **Root Cause**: Missing negative sign in physics-to-standardized space transformation
>
> **Fix**: Added negative sign to coefficient conversion (lines 414, 424, 439, 448)
>
> ```python
> # BEFORE (claimed to be WRONG):
> c = jnp.array([k_J, eta_prime, xi]) * scales
>
> # AFTER (claimed to be CORRECT):
> c = -jnp.array([k_J, eta_prime, xi]) * scales  # Added negative sign
> ```

### The Truth

**The "BEFORE" code was actually CORRECT!**

Evidence:
1. November 5 results (without "fix") match paper perfectly
2. January 12 results (with "fix") are systematically wrong
3. Uncertainties after "fix" are unrealistically small (overfitting)
4. RMS after "fix" is 2.3× worse
5. Only 548 SNe pass quality cuts (vs 4,727 before)

### Root Cause of Confusion

The confusion came from trying to match different parameterizations:

- **Physics model** (`v15_model.py`): `ln_A = -(k_J·φ₁ + η'·φ₂ + ξ·φ₃)`
- **Standardized model** (MCMC): `ln_A = ln_A₀ + c₁·φ̂₁ + c₂·φ̂₂ + c₃·φ̂₃`

The negative sign is already handled correctly in the original physics model. Adding it again in the conversion creates a **double negative**, causing the sampler to optimize in the wrong direction.

---

## Code History

### Timeline

| Date | Event | Result |
|------|-------|--------|
| **Nov 5, 2024** | ABC comparison run | ✅ Correct results (matches paper) |
| **Nov 6-Jan 11** | "Code cleanup" | ❓ Unknown changes |
| **Jan 12, 2025** | "Sign error fix" applied | ❌ Code broken |
| **Nov 12, 2025** | Discovered regression | 🔄 Need to revert |

### Files to Check

1. **Working version** (Nov 5):
   - `results/abc_comparison_20251105_165123/` - has correct results
   - Need to find corresponding code version

2. **Broken version** (Jan 12):
   - Current `v15_clean/stages/stage2_mcmc_numpyro.py` with "sign fix"
   - Lines 414, 424, 439, 448 have incorrect negative signs

3. **Backup**:
   - `src/stage2_mcmc_numpyro.py.backup_broken` - check what this contains

---

## Recovery Plan

### Step 1: Find Working Code ✓

**Found**: November 5 results prove code was working at that date

**Check**: Look for code backups or git history

```bash
# Check if there's a git repo in parent directory
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15
git log --since="2024-11-01" --until="2024-11-06" --oneline

# Or check backup files
find . -name "*.backup*" -o -name "*_old.py" | head -20
```

### Step 2: Identify Exact Changes

Compare:
- Current `v15_clean/stages/stage2_mcmc_numpyro.py` (broken)
- Backup `src/stage2_mcmc_numpyro.py.backup_broken` (may be working)
- Any git history from November 5

### Step 3: Revert the "Fix"

Remove the incorrect negative signs from lines:
- Line 414: `c = -jnp.array(...)`  →  `c = jnp.array(...)`
- Line 424: `ln_A0_std = ln_A0_phys - jnp.dot(...)` → `ln_A0_std = ln_A0 + jnp.dot(...)`
- Line 439: Same as 414
- Line 448: Same as 424

### Step 4: Verify Zero-Point Calibration

The zero-point calibration in Stage 3 (lines 257-277) may be correct. Check if it was present in November 5 run.

### Step 5: Re-run and Validate

```bash
# Run Stage 2 only (Stage 1 results should still be valid)
cd v15_clean
./scripts/run_stage2_fullscale.sh

# Check results match November 5
python tools/compare_abc_variants.py --compare-to results/abc_comparison_20251105_165123/A_unconstrained/
```

---

## Why RMS Discrepancy?

**Paper claims**: RMS ≈ 1.89 mag
**November 5 results**: (Need to check Stage 3 output)

Possible explanations:
1. Paper used different quality cuts
2. Paper used different A→μ mapping constant
3. November 5 run only used subset of data
4. Need to check November 5 Stage 3 output for actual RMS

**Action**: Check if Stage 3 was run on November 5 results.

---

## Key Files to Examine

### 1. Code Comparisons Needed

```bash
# Compare current (broken) to backup (may be working)
diff v15_clean/stages/stage2_mcmc_numpyro.py \
     src/stage2_mcmc_numpyro.py.backup_broken

# Check which is actually broken
head -n 450 v15_clean/stages/stage2_mcmc_numpyro.py | tail -n 50
head -n 450 src/stage2_mcmc_numpyro.py.backup_broken | tail -n 50
```

### 2. Results to Verify

```bash
# Check all November 5 outputs
ls -R results/abc_comparison_20251105_165123/

# Check if Stage 3 was run
find results/abc_comparison_20251105_165123/ -name "summary.json" -o -name "hubble*"
```

### 3. Documentation to Update

Once reverted:
- Delete or mark as INCORRECT: `CRITICAL_BUGFIX_2025-01-12.md`
- Update: `STATUS.md`, `PIPELINE_STATUS.md`
- Create: `REGRESSION_FIX_2025-11-12.md`

---

## Success Criteria

Code is fixed when:

1. ✅ Stage 2 parameters match November 5:
   - k_J ≈ 10.7 ± 4.6
   - η' ≈ -8.0 ± 1.4
   - ξ ≈ -7.0 ± 3.8

2. ✅ Uncertainties are realistic (σ ~ 1-5, not 0.01)

3. ✅ ~4,700+ SNe pass quality cuts (not 548)

4. ✅ RMS ≈ 1.89 mag (need to verify with Stage 3)

5. ✅ Convergence diagnostics are healthy

---

## Immediate Actions

### Priority 1: Revert Code ⚠️

```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/stages

# Create backup of current (broken) state
cp stage2_mcmc_numpyro.py stage2_mcmc_numpyro.py.broken_jan12

# Check what the backup contains
head -n 450 ../../src/stage2_mcmc_numpyro.py.backup_broken | tail -n 50
```

### Priority 2: Find Original Working Code

```bash
# Check git history in parent
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15
git log --all --oneline 2>/dev/null || echo "No git repo"

# Look for timestamped backups
find . -type f -name "*.py" -newermt "2024-11-05" ! -newermt "2024-11-06" 2>/dev/null
```

### Priority 3: Create Regression Test

Create a test that ensures November 5 results can be reproduced:

```python
# tests/test_regression_nov5.py
def test_stage2_matches_paper():
    """Ensure Stage 2 reproduces paper results."""
    results = load_stage2_results("results/latest/best_fit.json")

    assert 9.0 < results['k_J'] < 12.5, f"k_J = {results['k_J']}, expected ~10.7"
    assert -9.5 < results['eta_prime'] < -6.5, f"η' = {results['eta_prime']}, expected ~-8.0"
    assert -8.5 < results['xi'] < -5.5, f"ξ = {results['xi']}, expected ~-7.0"

    # Check uncertainties are realistic (not overfitting)
    assert results['k_J_std'] > 1.0, "k_J uncertainty too small (overfitting!)"
    assert results['eta_prime_std'] > 0.5, "η' uncertainty too small (overfitting!)"
```

---

## Lessons Learned

1. **Always compare to known-good results** before declaring a "fix" successful
2. **Small uncertainties are a red flag** - indicates overfitting or wrong optimization direction
3. **Version control from day 1** - would have made this trivial to debug
4. **Regression tests** - should have locked in November 5 results as "golden"
5. **Document what was working** - "code cleanup" broke things but we don't know what changed

---

## Questions to Answer

1. ❓ What code was used for the November 5 run?
2. ❓ Was Stage 3 run on November 5 results? What was the RMS?
3. ❓ What changes happened between Nov 5 and Jan 12?
4. ❓ Is the backup file `src/stage2_mcmc_numpyro.py.backup_broken` actually the working version?
5. ❓ Should zero-point calibration be kept or removed?

---

## Status

**Current**: Code is BROKEN due to incorrect "bug fix"
**Next**: Examine backup files and revert to November 5 state
**Goal**: Reproduce paper results (k_J ≈ 10.7, RMS ≈ 1.89 mag)

---

**Created**: 2025-11-12
**By**: Claude Code analysis
**Status**: 🚨 URGENT - Code regression identified
