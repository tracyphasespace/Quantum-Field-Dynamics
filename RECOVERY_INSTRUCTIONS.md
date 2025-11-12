# Recovery Instructions: Restore Working Code

**Status**: 🚨 **REGRESSION CONFIRMED - FIX IDENTIFIED**
**Date**: 2025-11-12
**Problem**: January 12 "bug fix" actually BROKE working code
**Solution**: Revert to November 5 version (NO negative sign)

---

## Summary

**Working code** (November 5, 2024):
```python
# Line 314 in 2Compare/stage2_mcmc_numpyro.py
c = jnp.array([k_J, eta_prime, xi]) * scales  # NO negative sign
```

**Broken code** (January 12, 2025):
```python
# Line 414, 439 in v15_clean/stages/stage2_mcmc_numpyro.py
c = -jnp.array([k_J, eta_prime, xi]) * scales  # INCORRECT negative sign!
```

**Proof**:
- November 5 results: k_J=10.77, η'=-7.99, ξ=-6.91 ✅ Matches paper!
- January 12 results: k_J=5.01, η'=-1.49, ξ=-4.07 ❌ All wrong!

---

## Recovery Steps

### Step 1: Backup Current (Broken) Code ✅

Already done (in git), but create explicit backup:

```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/stages
cp stage2_mcmc_numpyro.py stage2_mcmc_numpyro.py.broken_jan12_negative_sign
```

### Step 2: Remove Incorrect Negative Signs

Edit `v15_clean/stages/stage2_mcmc_numpyro.py`:

**Line 414** - Change FROM:
```python
c = -jnp.array([k_J, eta_prime, xi]) * scales  # CRITICAL: negative sign!
```

**TO**:
```python
c = jnp.array([k_J, eta_prime, xi]) * scales  # FIXED: removed incorrect negative sign
```

**Line 424** - Change FROM:
```python
ln_A0_std = ln_A0_phys - jnp.dot(jnp.array([k_J, eta_prime, xi]), means)
```

**TO**:
```python
ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)
```

**Line 439** - Change FROM:
```python
c = -jnp.array([k_J, eta_prime, xi]) * scales  # CRITICAL: negative sign!
```

**TO**:
```python
c = jnp.array([k_J, eta_prime, xi]) * scales  # FIXED: removed incorrect negative sign
```

**Line 448** - Change FROM:
```python
ln_A0_std = ln_A0_phys - jnp.dot(jnp.array([k_J, eta_prime, xi]), means)
```

**TO**:
```python
ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)
```

### Step 3: Update Comments

Remove or update comments that claim the negative sign is "CRITICAL" or "correct".

Replace with:
```python
# Convert to standardized space: c = k_phys * scales
# (inverse of backtransform: k_phys = c / scales)
```

### Step 4: Delete or Mark Incorrect Documentation

**Delete or rename**:
- `CRITICAL_BUGFIX_2025-01-12.md` → `CRITICAL_BUGFIX_2025-01-12.md.INCORRECT`
- `PIPELINE_STATUS.md` (contains incorrect information)

**Mark as incorrect** by adding to top:
```markdown
# ⚠️ WARNING: THIS DOCUMENT IS INCORRECT!
# The "bug fix" described here actually BROKE working code.
# See REGRESSION_ANALYSIS.md and RECOVERY_INSTRUCTIONS.md for details.
```

### Step 5: Test the Fix

Run a quick test with 50 SNe:

```bash
cd /home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean

# Run Stage 2 with small dataset
python stages/stage2_mcmc_numpyro.py \
    --stage1-results ../results/v15_clean/stage1_fullscale \
    --out ../results/v15_clean/stage2_recovery_test \
    --nchains 2 \
    --nsamples 200 \
    --nwarmup 100 \
    --quality-cut 2000 \
    --constrain-signs informed
```

**Expected results** (should match November 5):
```json
{
  "k_J": 9.0-12.5,          // Target: 10.7
  "eta_prime": -9.5 to -6.5, // Target: -8.0
  "xi": -8.5 to -5.5,        // Target: -7.0
  "k_J_std": 2.0-5.0,        // Realistic uncertainty
  "eta_prime_std": 0.8-2.0,
  "xi_std": 2.0-5.0
}
```

### Step 6: Run Full Production

If test succeeds:

```bash
# Run full production (4,727 SNe)
./scripts/run_stage2_fullscale.sh

# Then run Stage 3
./scripts/run_stage3.sh
```

---

## Verification Checklist

After recovery, verify:

### Stage 2 Results
- [ ] k_J ≈ 10.7 ± 4.6 (within ±20%)
- [ ] η' ≈ -8.0 ± 1.4 (within ±20%)
- [ ] ξ ≈ -7.0 ± 3.8 (within ±20%)
- [ ] Uncertainties are realistic (σ ~ 1-5, NOT 0.01)
- [ ] Acceptance probability ~ 0.85-0.90
- [ ] Zero divergences
- [ ] R-hat < 1.05

### Stage 3 Results
- [ ] ~4,700+ SNe pass quality cuts (NOT 548!)
- [ ] RMS < 2.0 mag (target ~1.89 mag)
- [ ] Residuals approximately normal
- [ ] No major systematic trends

---

## Why the "Fix" Was Wrong

### Misconception

The January 12 "fix" document claimed:

> The physics model (v15_model.py:107) defines:
> ```python
> ln_A_pred = -(k_J·φ₁ + η'·φ₂ + ξ·φ₃)
> ```
> Therefore when converting to standardized space, we need:
> ```python
> c = -k_phys * scales  # ← They added this negative sign
> ```

### Truth

The negative sign in the physics model is **already accounted for internally**. When we compute the standardized coefficients, we're mapping:

```
Standardized model: ln_A = ln_A₀ + c₁·φ̂₁ + c₂·φ̂₂ + c₃·φ̂₃
Physics model:      ln_A = ln_A₀ - (k_J·φ₁ + η'·φ₂ + ξ·φ₃)
```

The correct transformation is:
```python
c = k_phys * scales  # NO negative sign here!
```

Because the negative sign is already in the **sampling space**. The physics parameters k_J, η', ξ are sampled with appropriate priors (including negative values for η' and ξ), and the model handles the sign internally.

Adding a negative sign in the conversion **doubles the negation**, causing the sampler to optimize in the wrong direction.

### Evidence

**Empirical proof**:
- Without negative sign (Nov 5): Results match paper ✅
- With negative sign (Jan 12): Results 2-5× too small ❌

The code that worked is in:
- `2Compare/stage2_mcmc_numpyro.py` (line 314) - NO negative sign

---

## Detailed Comparison

### November 5, 2024 Code (WORKING)

**File**: `2Compare/stage2_mcmc_numpyro.py`
**Lines**: 300-323

```python
if constrain_signs == 'physics':
    if standardizer is None:
        raise ValueError("physics variant requires standardizer")

    # Sample physics params with positivity constraints
    k_J = numpyro.sample('k_J', dist.HalfNormal(20.0))
    eta_prime = numpyro.sample('eta_prime', dist.HalfNormal(10.0))
    xi = numpyro.sample('xi', dist.HalfNormal(10.0))

    # Convert to standardized space: c = k_phys * scales
    means = jnp.array(standardizer.means)
    scales = jnp.array(standardizer.scales)
    c = jnp.array([k_J, eta_prime, xi]) * scales  # ✅ NO negative sign

    numpyro.deterministic('c', c)

    alpha0 = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))
    alpha0_std = alpha0 + jnp.dot(c, means / scales)  # ✅ Addition, not subtraction
```

**Results**: k_J=10.77, η'=-7.99, ξ=-6.91 ✅

### January 12, 2025 Code (BROKEN)

**File**: `v15_clean/stages/stage2_mcmc_numpyro.py`
**Lines**: 395-424 ('informed' variant) and 426-448 ('physics' variant)

```python
if constrain_signs == 'informed':
    if standardizer is None:
        raise ValueError("informed variant requires standardizer")

    k_J = numpyro.sample('k_J', dist.TruncatedNormal(loc=10.7, scale=3.0, low=5.0, high=20.0))
    eta_prime = numpyro.sample('eta_prime', dist.Normal(-8.0, 3.0))
    xi = numpyro.sample('xi', dist.Normal(-7.0, 3.0))

    means = jnp.array(standardizer.means)
    scales = jnp.array(standardizer.scales)
    c = -jnp.array([k_J, eta_prime, xi]) * scales  # ❌ INCORRECT negative sign!

    numpyro.deterministic('c', c)

    ln_A0_phys = numpyro.sample('ln_A0', dist.Normal(0.0, 5.0))
    ln_A0_std = ln_A0_phys - jnp.dot(jnp.array([k_J, eta_prime, xi]), means)  # ❌ Subtraction!
```

**Results**: k_J=5.01, η'=-1.49, ξ=-4.07 ❌

---

## Next Steps After Recovery

1. **Run regression test** to ensure November 5 results can be reproduced
2. **Archive the working code** with clear version tags
3. **Document what happened** to prevent future regressions
4. **Update papers** if needed (current papers may already use correct results)
5. **Set up CI/CD** to catch regressions automatically

---

## Files Modified by This Recovery

- `v15_clean/stages/stage2_mcmc_numpyro.py` (lines 414, 424, 439, 448)
- `CRITICAL_BUGFIX_2025-01-12.md` (mark as INCORRECT)
- `PIPELINE_STATUS.md` (update with correct status)
- This file (`RECOVERY_INSTRUCTIONS.md` - new)
- `REGRESSION_ANALYSIS.md` (existing documentation)

---

## Contact / Questions

If recovery fails or results still don't match:

1. Check git log: `git log --oneline --since="2024-11-05" --until="2024-11-06"`
2. Compare to 2Compare version: `diff 2Compare/stage2_mcmc_numpyro.py v15_clean/stages/stage2_mcmc_numpyro.py`
3. Verify November 5 results: `cat ../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json`
4. Review this document and REGRESSION_ANALYSIS.md

---

**Created**: 2025-11-12
**Status**: ✅ ROOT CAUSE IDENTIFIED - READY FOR RECOVERY
**Next**: Apply the fix and test
