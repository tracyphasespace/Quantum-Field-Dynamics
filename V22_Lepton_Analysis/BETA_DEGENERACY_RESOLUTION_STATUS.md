# β-Degeneracy Resolution - Complete Status Report

**Date**: 2025-12-23
**Status**: ⚠️ **PARTIAL RESOLUTION** - Degeneracy identified and partially broken

---

## Problem Statement

**Original issue** (from reviewer feedback):
> "The β-scan shows weak falsifiability: 81% of β values converged with flat residuals.
> Without a failure mode, the optimizer appears too flexible."

**Core question**: Why does the optimizer find solutions at almost every β value?

---

## Diagnostic Journey (Complete Timeline)

### Phase 1: Root Cause Analysis ✓ COMPLETE

**Tests implemented**: `test_beta_degeneracy_diagnostic.py`

#### Test A: Echo Test (Verify β enters calculation)
```
Result: E_stab/β = 0.0672 (constant, CV < 1%)
Verdict: ✓ β IS entering calculation correctly
```

#### Test B: Frozen-Parameter Test
```
Method: Optimize at β=3.058, evaluate at other β without re-optimizing
Result: Residual varies by 8.5 MILLION %
Verdict: ✓ β DOES matter physically (not a plumbing bug)
```

#### Test C: Restricted Refit (Only amplitude varies)
```
Method: Fix R and U, optimize only amplitude
Result: amplitude × √β = 1.5708 (constant, CV = 0.00%)
Verdict: ✓ SCALING DEGENERACY CONFIRMED
```

**Diagnosis**: Not a bug. Real mathematical degeneracy where amplitude ∝ 1/√β compensates for β changes in E_stab.

**Root cause**:
```
E_stab ~ β × amplitude²
If amplitude → amplitude/√β, then:
E_stab ~ β × (amplitude/√β)² = amplitude² (β cancels!)
```

---

### Phase 2: Option 2 - Fixed Amplitude Test ✗ FAILED

**Hypothesis**: If amplitude ∝ 1/√β is the degeneracy, fixing amplitude should restore β identifiability.

**Implementation**: `test_fixed_amplitude_beta_scan.py`

**Test**: Fixed amplitude at [0.25, 0.5, 0.75, 0.9], optimized only (R, U)

**Results**:
```
Amplitude = 0.25: <2% residual variation (FLAT)
Amplitude = 0.50: <2% residual variation (FLAT)
Amplitude = 0.75: <2% residual variation (FLAT)
Amplitude = 0.90: <2% residual variation (FLAT)
```

**Conclusion**: Degeneracy has moved to (R, U) space. With 2 free parameters (R, U) and 1 constraint (mass), continuous solution manifold still exists.

**Implication**: Need second observable - no other option will work.

---

### Phase 3: Option 1 - Multi-Objective with Magnetic Moment ⚠️ PARTIAL SUCCESS

**Hypothesis**: Different scaling of mass (E ~ U²R³) vs magnetic moment (μ ~ UR) should break (R,U) degeneracy.

#### Step 3a: Formula Implementation

**Tracy's magnetic moment formula** (from Hill spherical vortex):
```python
μ = k × Q × R × U
where:
  k ≈ 0.2  (geometric factor for uniform vorticity)
  Q = 1.0  (fundamental charge)
  R = vortex radius
  U = circulation velocity
```

**Initial implementation**: Failed (all β failed convergence)

**Problem**: Normalization factor was 94.8× too small

#### Step 3b: Calibration ✓

**Method**: Use known electron solution at β = 3.058 to empirically determine normalization.

**Calibration** (`calibrate_magnetic_moment.py`):
```
Known solution: R = 0.44, U = 0.024
Raw μ = 0.2 × 1.0 × 0.44 × 0.024 = 2.11×10⁻³
Target g = 2.00231930436256
Required normalization = g / μ = 948.0
```

**Update**: Changed normalization from 10.0 → 948.0

#### Step 3c: Multi-Objective Scan Results

**Configuration**:
- β range: [2.5, 3.5], 11 points
- Constraints: Mass + g-factor (equal weights)
- Convergence: mass_residual < 10⁻³, g_residual < 0.1

**Results**:

| β     | Objective  | Mass Res | g Res  | Converged |
|-------|-----------|----------|--------|-----------|
| 2.500 | 4.97×10⁻¹³ | 6.69×10⁻⁷ | 2.23×10⁻⁷ | ✓ |
| 2.600 | 4.55×10⁻¹³ | 6.42×10⁻⁷ | 2.07×10⁻⁷ | ✓ |
| 2.700 | 4.76×10⁻¹³ | 6.52×10⁻⁷ | 2.25×10⁻⁷ | ✓ |
| 2.800 | 4.84×10⁻¹³ | 6.55×10⁻⁷ | 2.35×10⁻⁷ | ✓ |
| 2.900 | 4.83×10⁻¹³ | 6.53×10⁻⁷ | 2.37×10⁻⁷ | ✓ |
| 3.000 | 4.90×10⁻¹³ | 6.57×10⁻⁷ | 2.40×10⁻⁷ | ✓ |
| 3.100 | 4.03×10⁻¹³ | 5.77×10⁻⁷ | 2.63×10⁻⁷ | ✓ |
| **3.200** | **2.52×10⁻¹³** | **3.26×10⁻⁷** | **3.82×10⁻⁷** | **✓** |
| 3.300 | 4.75×10⁻¹³ | 6.44×10⁻⁷ | 2.47×10⁻⁷ | ✓ |
| 3.400 | 4.73×10⁻¹³ | 6.40×10⁻⁷ | 2.51×10⁻⁷ | ✓ |
| 3.500 | 4.71×10⁻¹³ | 6.37×10⁻⁷ | 2.55×10⁻⁷ | ✓ |

**Success metrics**:
- ✓ **Degeneracy broken**: 96.9% objective variation (vs <1% before)
- ✓ **Both constraints satisfied**: All residuals well within tolerance
- ✓ **Smooth variation**: Objective function shows clear minimum

**Failure metrics**:
- ✗ **Wrong minimum**: β = 3.200, not β = 3.058 (offset: 0.142)
- ✗ **All β work**: 11/11 converged (100% success rate)
- ✗ **Weak selectivity**: Only factor-of-2 variation

---

## Current Understanding

### What We Know

1. **The degeneracy is real and understood**:
   - Original: amplitude ∝ 1/√β scaling
   - After fixing amplitude: (R, U) manifold degeneracy
   - Mathematical origin: 3 DOF, 1 constraint → 2D solution space

2. **Second observable does help**:
   - Flat degeneracy → 96.9% variation
   - Different scalings (Mass ~ U²R³, μ ~ UR) add information
   - Optimizer finds consistent solutions

3. **But β = 3.058 is NOT validated**:
   - Minimum at β = 3.200 (4.6% offset)
   - All β values in range [2.5, 3.5] work
   - No failure mode demonstrated

### What We Don't Know

1. **Is the magnetic moment formula correct?**:
   - Functional form μ = k × Q × R × U from Tracy
   - Geometric factor k = 0.2 from uniform vorticity assumption
   - Normalization factor 948 is empirical (not derived)
   - May need β-dependence or other terms

2. **Why is the minimum shifted?**:
   - Expected β = 3.058 (from fine structure constant)
   - Observed β = 3.200
   - Possible causes:
     - Wrong geometric factor k
     - Missing β-dependent terms
     - Model limitation (Hill vortex too simple)

3. **Is the model fundamentally too flexible?**:
   - 100% success rate (no failures)
   - Only moderate variation (factor of 2)
   - May need third constraint or different approach

---

## Files Created (Complete List)

### Diagnostic Scripts
1. **test_beta_degeneracy_diagnostic.py**
   - Three tests (Echo, Frozen-parameter, Restricted refit)
   - Confirms scaling degeneracy
   - Results: `results/beta_degeneracy_diagnostic.json`

2. **test_fixed_amplitude_beta_scan.py** (Option 2)
   - Fixed amplitude at [0.25, 0.5, 0.75, 0.9]
   - Optimizes only (R, U)
   - Results: Shows degeneracy moved to (R,U) space

3. **calibrate_magnetic_moment.py**
   - Empirical calibration of normalization factor
   - Uses known electron solution at β = 3.058
   - Found: normalization = 948.0 (not 10.0)

4. **test_multi_objective_beta_scan.py** (Option 1)
   - Multi-objective optimization: mass + g-factor
   - Magnetic moment: μ = k × Q × R × U
   - Calibrated normalization
   - Results: `results/multi_objective_beta_scan.json`

5. **plot_multi_objective_results.py**
   - Visualization of multi-objective scan
   - 6-panel figure showing objectives, residuals, parameters
   - Output: `results/multi_objective_beta_scan.png`

### Documentation
6. **REVIEWER_FEEDBACK_ACTION_PLAN.md**
   - Complete action plan from reviewer feedback
   - Diagnostic roadmap

7. **BETA_SCAN_RESULTS_CRITICAL.md**
   - Analysis of initial β-scan (tolerance 10⁻⁴)
   - 81% success rate, weak falsifiability

8. **BETA_SCAN_READY.md**
   - Status before production tolerance scan
   - Outlines Outcome A (narrow window) vs Outcome B (still wide)

9. **MULTI_OBJECTIVE_RESULTS.md**
   - Detailed analysis of Option 1 results
   - Calibration procedure
   - Interpretation and next steps

10. **BETA_DEGENERACY_RESOLUTION_STATUS.md** (this file)
    - Complete timeline of diagnostic journey
    - Consolidated status report

---

## Summary of Progress

### Achievements ✓

1. **Identified root cause** of β-flatness (scaling degeneracy)
2. **Ruled out bugs** (β enters calculation correctly)
3. **Tested and failed Option 2** (fixed amplitude) with clear diagnosis
4. **Implemented Option 1** (multi-objective with magnetic moment)
5. **Calibrated magnetic moment** formula empirically
6. **Broke flat degeneracy** (96.9% variation achieved)
7. **Satisfied both constraints** simultaneously at all β

### Remaining Issues ✗

1. **β minimum shifted** from 3.058 to 3.200
2. **All β values work** (100% success rate)
3. **Weak selectivity** (only factor-of-2 variation)
4. **Magnetic moment normalization unclear** (empirical, not theoretical)
5. **Manuscript claim not validated** (β = 3.058 not uniquely selected)

---

## Next Steps (Decision Tree)

### Path A: Fix Magnetic Moment Formula (RECOMMENDED)

**If formula can be corrected theoretically**:

1. **Ask Tracy**:
   - Derive normalization factor from first principles
   - Verify geometric factor k = 0.2 for QFD Hill vortex
   - Check for missing β-dependent terms
   - Examine density profile (may not be uniform vorticity)

2. **Test sensitivity**:
   - Vary k ∈ [0.15, 0.25] and re-run scan
   - Check if β minimum shifts toward 3.058
   - Try β-dependent formula: μ = k(β) × Q × R × U

3. **If successful**:
   - Re-run multi-objective scan with corrected formula
   - Check if minimum at β ≈ 3.058
   - Check if variation increases (sharper minimum)
   - Proceed with manuscript

### Path B: Add Third Constraint

**If magnetic moment formula is correct as-is**:

1. **Cross-lepton multi-objective**:
   - Optimize (R_e, U_e, A_e, R_μ, U_μ, A_μ, R_τ, U_τ, A_τ, **β**)
   - Constraints: 6 observables (3 masses + 3 g-factors)
   - **One shared β** for all leptons
   - Check if unique β emerges

2. **Alternative third observable**:
   - Charge radius (R_rms)
   - Form factors
   - Anomalous moment (a = (g-2)/2) precision

### Path C: Manuscript Revision

**If neither Path A nor B restores β = 3.058**:

1. **Weaken claims**:
   - "uniquely determines" → "constrains to β ≈ 3 ± 0.5"
   - "prediction" → "compatibility test"
   - "evidence for" → "consistent with"

2. **Add sections**:
   - "Model limitations and parameter degeneracy"
   - "Need for additional observables"
   - "Future work: Cross-lepton constraints"

3. **Consider lower-tier journal** or major revision

---

## Critical Questions for Tracy

### Theoretical Questions

1. **Magnetic moment normalization**:
   - Why is the normalization factor ≈ 948?
   - Should it involve fundamental constants (ħ, c, e, m_e)?
   - Can you derive it from first principles?

2. **Geometric factor**:
   - Is k = 0.2 correct for the QFD Hill vortex?
   - Does the density perturbation profile affect k?
   - Should k depend on β or other parameters?

3. **Formula completeness**:
   - Does μ = k × Q × R × U need additional terms?
   - Should there be β-dependence?
   - Should amplitude enter the magnetic moment?

### Strategic Questions

4. **Manuscript direction**:
   - Given β minimum at 3.200 (not 3.058), what do you recommend?
   - Should we pursue Path A (fix formula) or Path B (add constraint)?
   - Are you comfortable weakening the claim to "compatibility"?

5. **Model interpretation**:
   - Is factor-of-2 variation across β ∈ [2.5, 3.5] acceptable?
   - Should we expect sharper selection (factor of 10+)?
   - What level of β-selectivity would validate the Golden Loop?

---

## Bottom Line

**Question**: Can we restore β = 3.058 as uniquely selected by the lepton spectrum?

**Current Answer**: **PARTIAL** - We've made progress but haven't reached the goal:

| Metric | Before | After Option 1 | Target |
|--------|--------|---------------|--------|
| **Objective variation** | <1% | 96.9% | >1000% |
| **β minimum** | 2.6 | 3.200 | 3.058 |
| **β success rate** | 81% | 100% | <30% |
| **Selectivity** | None | Moderate | Sharp |

**Status**:
- ✓ Diagnosed degeneracy mechanism
- ✓ Implemented second observable
- ✓ Broke flat degeneracy
- ⚠️ Wrong β minimum
- ✗ Still weak falsifiability

**Decision point**: Tracy needs to review magnetic moment derivation to determine if Path A (formula correction) is viable, or if we need Path B (third constraint) or Path C (manuscript revision).

**Immediate next action**: Present findings to Tracy and get guidance on magnetic moment formula before proceeding further.
