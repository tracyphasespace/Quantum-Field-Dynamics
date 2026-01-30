# Multi-Objective β-Scan Results (Option 1: Magnetic Moment)

**Date**: 2025-12-23
**Status**: ⚠️ **PARTIAL SUCCESS** - Degeneracy broken but β minimum shifted

---

## Executive Summary

The multi-objective optimization with magnetic moment constraint:
- ✓ **BREAKS the flat degeneracy** (96.9% variation vs <1% before)
- ⚠️ **ALL β values still converge** (11/11 = 100%)
- ✗ **Minimum at β = 3.200, NOT β = 3.058** (offset: 0.142)

**Conclusion**: Magnetic moment adds discriminating power but does NOT validate β = 3.058 as uniquely selected.

---

## Detailed Results

### Calibration (Critical First Step)

**Problem identified**: Original normalization factor (10.0) was 94.8× too small

**Calibration method**:
1. Used known electron solution at β = 3.058: (R=0.44, U=0.024, A=0.90)
2. Calculated raw magnetic moment: μ = 0.2 × 1.0 × 0.44 × 0.024 = 2.11×10⁻³
3. Required normalization: g_target / μ_raw = 2.00232 / 0.00211 = **948.0**

**File**: `validation_tests/calibrate_magnetic_moment.py`

### Multi-Objective Scan Results

**Configuration**:
- β range: [2.5, 3.5] with 11 points (Δβ = 0.1)
- Constraints: Mass + g-factor (equal weights)
- Tolerances: mass_residual < 1×10⁻³, g_residual < 0.1

**Results Table**:

| β Value | Total Objective | Mass Residual | g Residual | R      | U        |
|---------|----------------|---------------|------------|--------|----------|
| 2.500   | 4.97×10⁻¹³     | 6.69×10⁻⁷     | 2.23×10⁻⁷  | 0.4455 | 0.023704 |
| 2.600   | 4.55×10⁻¹³     | 6.42×10⁻⁷     | 2.07×10⁻⁷  | 0.4445 | 0.023760 |
| 2.700   | 4.76×10⁻¹³     | 6.52×10⁻⁷     | 2.25×10⁻⁷  | 0.4434 | 0.023815 |
| 2.800   | 4.84×10⁻¹³     | 6.55×10⁻⁷     | 2.35×10⁻⁷  | 0.4424 | 0.023870 |
| 2.900   | 4.83×10⁻¹³     | 6.53×10⁻⁷     | 2.37×10⁻⁷  | 0.4414 | 0.023923 |
| 3.000   | 4.90×10⁻¹³     | 6.57×10⁻⁷     | 2.40×10⁻⁷  | 0.4405 | 0.023976 |
| 3.100   | 4.03×10⁻¹³     | 5.77×10⁻⁷     | 2.63×10⁻⁷  | 0.4395 | 0.024029 |
| **3.200** | **2.52×10⁻¹³** | **3.26×10⁻⁷** | **3.82×10⁻⁷** | **0.4386** | **0.024081** |
| 3.300   | 4.75×10⁻¹³     | 6.44×10⁻⁷     | 2.47×10⁻⁷  | 0.4376 | 0.024132 |
| 3.400   | 4.73×10⁻¹³     | 6.40×10⁻⁷     | 2.51×10⁻⁷  | 0.4367 | 0.024183 |
| 3.500   | 4.71×10⁻¹³     | 6.37×10⁻⁷     | 2.55×10⁻⁷  | 0.4358 | 0.024233 |

**Minimum**: β = 3.200 with objective = 2.52×10⁻¹³

### Statistical Analysis

**Objective variation**:
- Range: [2.52×10⁻¹³, 4.97×10⁻¹³]
- Variation: 96.9% (factor of ~2)
- Compare to original β-scan: <1% variation (essentially flat)

**Parameter trends with β**:
- R: Decreases linearly (0.4455 → 0.4358)
- U: Increases linearly (0.02370 → 0.02423)
- Both constraints satisfied at all β

---

## Interpretation

### What Worked

1. **Magnetic moment DOES break degeneracy**:
   - Original scan (mass only): <1% variation, flat across β
   - Multi-objective scan: 96.9% variation, clear minimum
   - Different scalings (Mass ~ U²R³, μ ~ UR) do add information

2. **Formula functional form appears correct**:
   - μ = k × Q × R × U works after calibration
   - Both constraints can be satisfied simultaneously
   - Smooth variation with β

### What Didn't Work

1. **β minimum is SHIFTED**:
   - Expected: β = 3.058 (from fine structure constant)
   - Observed: β = 3.200
   - Offset: Δβ = 0.142 (~4.6%)

2. **All β values still work**:
   - 11/11 converged (100% success rate)
   - No failure mode demonstrated
   - Weak falsifiability remains

3. **Variation is moderate, not sharp**:
   - Factor of ~2 across full range
   - Not the "sharp minimum" needed for strong claim

---

## Possible Explanations

### 1. Geometric Factor Error
The coefficient k = 0.2 in μ = k × Q × R × U may be incorrect:
- Tracy estimated k ≈ 0.2 for uniform vorticity Hill vortex
- Actual QFD solution may have different circulation profile
- **Test**: Try k = 0.18 or k = 0.22 to see if β minimum shifts

### 2. Missing β-Dependence
The magnetic moment formula may need β-dependence:
- Current: μ = k × Q × R × U (β-independent)
- Possible: μ = k(β) × Q × R × U
- Or: μ involves density amplitude or other β-dependent terms

### 3. QFD Conventions Mismatch
Normalization factor (948) is empirical:
- Calibrated to match electron at β = 3.058
- May encode physics that should be explicit
- **Need**: Theoretical derivation of normalization

### 4. Model Limitation
The Hill vortex model may be too flexible:
- 3 parameters (R, U, amplitude) for 2 constraints (mass, μ)
- Still 1 DOF of freedom remaining
- **Need**: Third observable OR cross-lepton constraint

---

## Diagnostic Tests

### Test 1: Vary Geometric Factor k

```bash
# Test different k values
for k in 0.15 0.18 0.20 0.22 0.25; do
    # Modify geometric_factor in magnetic_moment_hill_vortex()
    # Re-calibrate normalization
    # Run β-scan
    # Check if minimum shifts toward 3.058
done
```

**Expected outcome**:
- If β minimum shifts with k → formula has correct form, wrong coefficient
- If β minimum stays at 3.2 → deeper model issue

### Test 2: Add β-Dependence

Test formula: μ = k × Q × R × U × f(β)

Try f(β) = 1/√β, β, √β, etc.

### Test 3: Cross-Lepton Constraint

Instead of optimizing each lepton independently, require:
- **Same β for all three leptons** (electron, muon, tau)
- Each gets its own (R, U, amplitude)
- Check if unique β emerges

---

## Comparison to Previous Tests

| Test | Method | β Success Rate | β Minimum | Variation |
|------|--------|---------------|-----------|-----------|
| **Original (loose tolerance)** | Mass only, tol=1×10⁻⁴ | 81% (17/21) | β=2.6 | <1% (flat) |
| **Production (tight tolerance)** | Mass only, tol=1×10⁻⁷ | Not run yet | TBD | TBD |
| **Option 2 (Fixed amplitude)** | Mass only, A fixed | ~100% | Flat | <2% (failed) |
| **Option 1 (Multi-objective)** | Mass + magnetic moment | 100% (11/11) | β=3.2 | 97% |

**Progression**:
- Option 2: Confirmed degeneracy moved to (R,U) space
- Option 1: Second observable breaks degeneracy BUT β minimum is wrong

---

## Manuscript Implications

### Current Claim (From BETA_SCAN_READY.md)

> "The fine structure constant α determines vacuum stiffness β = 3.058,
> which uniquely supports Hill vortex solutions at the three lepton masses."

### Evidence Status

**Against claim**:
- ✗ β = 3.058 is NOT the minimum (β = 3.2 is)
- ✗ All β values in [2.5, 3.5] support solutions
- ✗ Only factor-of-2 preference, not "unique selection"

**Supporting claim**:
- ✓ Magnetic moment does add discriminating power
- ✓ There IS a preferred β (moderate evidence)
- ✓ Model makes testable predictions

### Recommended Revisions

**Option A: Weaken claim to "compatibility"**

> "The vacuum stiffness β = 3.058 inferred from the fine structure constant
> is compatible with Hill vortex solutions at the observed lepton masses and
> magnetic moments, with β in the range [2.5, 3.5] showing factor-of-2
> variation in fit quality."

**Option B: Investigate and fix formula**

1. Derive theoretical normalization (don't calibrate empirically)
2. Test different geometric factors k
3. Consider β-dependent terms in magnetic moment
4. Re-run with corrected formula

**Option C: Add third constraint**

- Charge radius
- Cross-lepton consistency (same β for all)
- Form factors or other observables

---

## Next Steps (Recommended)

### Immediate (Debug magnetic moment formula)

1. **Theoretical derivation**: Ask Tracy to derive normalization from first principles
   - Why is normalization ≈ 948?
   - Should it involve β, ħ, c, or other constants?
   - Is geometric factor k = 0.2 correct for QFD Hill vortex?

2. **Sensitivity analysis**: Test k ∈ [0.15, 0.25] to see if β minimum shifts

3. **Check for β-dependence**: Does μ formula need β terms?

### If formula is confirmed correct

Run **cross-lepton multi-objective scan**:
- Optimize (R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, **β**)
- Constraints: All three masses + magnetic moments
- **One β shared across all leptons**
- Check if unique β emerges

### If still no unique β

**Manuscript strategy**:
- Downgrade claim from "uniquely determines" to "constrains to β ≈ 3 ± 0.5"
- Frame as "compatibility test" rather than "prediction"
- Emphasize model's ability to match TWO observables (mass + μ) with consistent parameters
- Consider lower-tier journal or major revision

---

## Files Created

1. `validation_tests/calibrate_magnetic_moment.py` - Normalization calibration
2. `validation_tests/test_multi_objective_beta_scan.py` - Multi-objective solver (updated with calibrated normalization)
3. `validation_tests/results/multi_objective_beta_scan.json` - Scan results
4. `MULTI_OBJECTIVE_RESULTS.md` - This document

---

## Bottom Line

**Question**: Does adding magnetic moment restore β = 3.058 as uniquely selected?

**Answer**: **NO**, but it's closer than before:
- Flat degeneracy → 96.9% variation ✓
- 100% β success rate (still no failures) ✗
- Minimum at β = 3.2, not β = 3.058 ✗

**Critical decision**:
- If magnetic moment formula can be corrected → Re-run and hope for β = 3.058
- If formula is correct → Major manuscript revision needed

**Recommendation**: Ask Tracy to review magnetic moment derivation before proceeding.
