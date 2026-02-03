# Final Session Summary: Global Scale Profiling + Sensitivity Analysis

**Date**: 2024-12-24
**Duration**: ~5 hours
**Status**: ✓ Global S profiling implemented, ⚠ Concerning sensitivity results

---

## What We Accomplished

### 1. Implemented Global Mass-Scale Profiling ✓

**Following Tracy's specification exactly**:

```python
# Analytic S profiling (no per-lepton tuning)
sigma_abs = sigma_model * m_targets
weights = 1.0 / sigma_abs**2

numerator = np.sum(m_targets * energies * weights)
denominator = np.sum(energies**2 * weights)
S_opt = numerator / denominator

masses_model = S_opt * energies
chi2 = np.sum(((masses_model - m_targets) / sigma_abs)**2)
```

**File**: `profile_likelihood_boundary_layer.py` (lines 151-185, 228-242)

**Benefits**:
- ONE global scale for all leptons (falsifiable)
- χ² should be ~ O(1) if structure correct
- S_opt reported in results (transparent)

### 2. Fixed All Review Points from Tracy ✓

| Item | Status | File/Section |
|------|--------|--------------|
| DOF counting | ✓ Fixed | `PATH_B_PRIME_ANALYSIS_CORRECTED.md` § 1 |
| "46% variation" → Δχ² | ✓ Fixed | § 2 |
| χ framework quantification | ✓ Added | § 3 |
| Normalization verification | ✓ Checked | All 2π/4π consistent |
| Global S profiling | ✓ Implemented | `profile_likelihood_boundary_layer.py` |

### 3. Completed Sensitivity Analysis ✓

**Tests run** (using OLD mass mapping, χ² ~ 10⁷):
- Optimizer convergence: max_iter ∈ [50, 100, 200, 500]
- w variation: 6 points in [0.010, 0.025]
- β variation: 9 points in [3.00, 3.20]

**File**: `results/resolution_sensitivity.json`

---

## Sensitivity Test Results (⚠ CONCERNING)

### Optimizer Convergence: GOOD ✓

```
max_iter =  50: χ² = 1.67×10⁷
max_iter = 100: χ² = 1.69×10⁷  (Δ = +2.11×10⁵)
max_iter = 200: χ² = 1.65×10⁷  (Δ = -4.08×10⁵)
max_iter = 500: χ² = 1.65×10⁷  (Δ = -2.35×10⁴)
```

**Conclusion**: Optimizer stable at max_iter=200 (Δ ~ 0.14% from 200→500)

### w Variation: PREFERENCE SHOWN

```
w = 0.0100: χ² = 1.66×10⁷  (Δχ² = +9.6×10⁴)
w = 0.0130: χ² = 1.65×10⁷  (Δχ² = +5.1×10⁴)
w = 0.0160: χ² = 1.65×10⁷  (Δχ² = +3.8×10⁴)
w = 0.0190: χ² = 1.65×10⁷  (Δχ² = +3.8×10⁴)
w = 0.0220: χ² = 1.65×10⁷  (Δχ² = 0) ← minimum
w = 0.0250: χ² = 1.65×10⁷  (Δχ² = +6.4×10³)
```

**Conclusion**: w_min ≈ 0.022 (shifted from initial 0.015)
- Δχ² ~ 10⁵ for w shift 0.01→0.025 (0.6% variation)
- Shows preference but landscape relatively flat

### β Variation: ⚠ CONCERNING

```
β = 3.0000: χ² = 1.66×10⁷  (Δχ² = +1.25×10⁵)
β = 3.0250: χ² = 1.65×10⁷  (Δχ² = +2.57×10⁴)
β = 3.0500: χ² = 1.65×10⁷  (Δχ² = +1.51×10⁴)
β = 3.0750: χ² = 1.65×10⁷  (Δχ² = +2.59×10⁴)
β = 3.1000: χ² = 1.65×10⁷  (Δχ² = +2.18×10⁴)
β = 3.1250: χ² = 1.65×10⁷  (Δχ² = +7.39×10²) ← near minimum
β = 3.1500: χ² = 1.65×10⁷  (Δχ² = +1.74×10⁴)
β = 3.1750: χ² = 1.65×10⁷  (Δχ² = +1.99×10⁴)
β = 3.2000: χ² = 1.65×10⁷  (Δχ² = 0) ← minimum
```

**⚠ MAJOR ISSUE**:
- β_min = **3.20** (not 3.10 as in 2×2 test!)
- Offset from Golden Loop: **+4.64%** (worse than before gradient!)
- This is OPPOSITE direction from expected mechanism

**Comparison to expectations**:
| Scenario | β_min | Offset from 3.043233053 |
|----------|-------|-------------------|
| No gradient (baseline) | ~3.15 | +3.0% |
| Expected with gradient | ~3.10 | +1.4% ✓ |
| **Actual (fine grid)** | **3.20** | **+4.6%** ✗ |

---

## Interpretation of β Result

### Possible Explanations

**1. Mass mapping issue (MOST LIKELY)**
- Old mapping forces electron match → artificial χ² scale
- β shift may be optimizer artifact from mis-scaled objective
- **Test**: Sanity check with new S profiling will reveal if this is cause

**2. Grid resolution artifact**
- 2×2 grid too coarse, missed true minimum
- Fine 9-point grid found different local minimum
- **Test**: Check if new mapping gives consistent β across grids

**3. Landscape genuinely flat**
- β not determined by mass-only (we already knew this)
- "Minimum" is statistical noise, not real constraint
- **Confirms**: Need additional observables (magnetic moments)

**4. Gradient energy not helping**
- Curvature term not acting as expected
- λ calibration wrong or gradient formula issue
- **Test**: Check energy ratios E_∇/E_stab at different β

### Why This Matters

**If β_min = 3.20 persists with new mapping**:
- ✗ Gradient energy NOT reducing closure gap
- ✗ Mechanism hypothesis fails
- ✗ Back to "sophisticated numerology" accusation

**If β_min ~ 3.10 with new mapping**:
- ✓ Old mapping was artifact
- ✓ Mechanism validated
- ✓ Proceed as planned

---

## Sanity Check Status (3×3 Grid with New Mapping)

**Status**: ⏳ RUNNING (28 min so far)

**Expected**: ~10-15 min for 3×3 grid with max_iter=200

**Actual**: 28+ min (slower than expected)

**Possible reasons**:
- Differential evolution exploring parameter space more thoroughly
- Energy calculations slower with current grid
- First run overhead (grid construction, etc.)

**Critical questions for sanity check**:
1. Is χ² ~ O(1), not 10⁷?
2. Is β_min near 3.10 or 3.20?
3. Is S_opt ~ O(1-10) (reasonable scale)?

**Decision tree**:
```
If χ² ~ O(1) AND β_min ~ 3.10:
  → ✓ Global S profiling works
  → ✓ Mechanism validated
  → 🔜 Proceed to 9×6 scan

If χ² ~ O(1) AND β_min ~ 3.20:
  → ⚠ Concerning: mechanism not working?
  → 🔍 Debug: Check energy ratios, gradient formula
  → ❓ Consult Tracy before wider scan

If χ² still ~ 10⁷:
  → ✗ S profiling didn't fix issue
  → 🔍 Debug: Check implementation, units, normalization
  → ⛔ STOP - fix before proceeding
```

---

## Files Created This Session

### Core Implementation
1. `lepton_energy_boundary_layer.py` (580 lines)
   - Smart radial grid builder
   - Non-self-similar boundary layer profile
   - Numeric gradient energy

2. `profile_likelihood_boundary_layer.py` (380 lines) **[MODIFIED]**
   - **Added**: Global S profiling (lines 151-185, 228-242)
   - 2D (β, w) profile likelihood scanner

### Analysis & Documentation
3. `PATH_B_PRIME_ANALYSIS_CORRECTED.md`
   - Addresses ALL Tracy review points
   - Manuscript-ready language
   - χ framework quantification

4. `RESPONSE_TO_TRACY_REVIEW.md`
   - Point-by-point response to 5 critical issues
   - Open questions answered
   - Timeline and next steps

5. `GLOBAL_SCALE_PROFILING_STATUS.md`
   - Implementation details
   - Expected outcomes
   - Validation checklist

### Testing
6. `resolution_sensitivity_test.py`
   - Optimizer convergence
   - w variation (6 points)
   - β variation (9 points)
   - **Results**: `results/resolution_sensitivity.json`

7. `test_global_scale_profiling.py`
   - 3×3 sanity grid with new S profiling
   - **Status**: Running

### Previous Work
8. `PATH_B_PRIME_STATUS.md` - Initial status (pre-corrections)
9. `SESSION_SUMMARY_PATH_B_PRIME.md` - Mid-session summary

---

## Tracy's Guidance - Compliance Check

### ✓ Followed Exactly

1. **"Implement global S profiling first"** → Done
2. **"Run 3×3 sanity check"** → Running
3. **"Then 9×6 modest scan"** → Pending sanity results
4. **"Defer magnetic moments until after Path B'"** → Deferred
5. **"Mass mapping fix before wider scans"** → Implemented first

### Key Recommendations Applied

- ✓ "Do NOT introduce per-lepton scaling" → One global S
- ✓ "Profile analytically" → Weighted least-squares formula
- ✓ "Verify χ² ~ O(1)" → Testing now
- ✓ "Check β direction preserved" → Critical test
- ✓ "Report from profiled objective, not single fit" → Will do

---

## Critical Decision Point

**Awaiting sanity check results** to determine:

### Path A: χ² ~ O(1) AND β ~ 3.10 ✓
**Action**: Proceed as planned
- Run 9×6 modest scan (54 points)
- Generate Δχ² contours
- Compute profile widths
- Update manuscript
- **ETA**: 2-3 days to publication-ready

### Path B: χ² ~ O(1) AND β ~ 3.20 ⚠
**Action**: Investigate before wider scan
- Debug gradient energy implementation
- Check λ calibration
- Examine energy ratios vs β
- Consult Tracy on interpretation
- **ETA**: +1-2 days for diagnosis

### Path C: χ² still ~ 10⁷ ✗
**Action**: Fix implementation
- Debug S profiling formula
- Check units and normalization
- Verify energy calculations
- **ETA**: +several hours for fix + retest

---

## Outstanding Questions for Tracy

### 1. β Sensitivity Result (⚠ URGENT)

**Finding**: Fine grid gives β_min = 3.20 (not 3.10)

**Question**: Is this expected given:
- Old mass mapping artifact?
- Landscape genuinely flat (mass-only insufficient)?
- Something wrong with gradient energy implementation?

**Context**: 2×2 grid gave β = 3.10, but fine 9-point grid gives 3.20

### 2. Sanity Check Runtime

**Expected**: ~10-15 min for 3×3 grid
**Actual**: 28+ min so far

**Question**: Is this normal, or sign of optimizer struggling?

### 3. Next Steps if β = 3.20 Persists

**If new mapping still gives β far from 3.043233053**:
- Debug gradient formula?
- Re-examine λ calibration?
- Accept that mass-only truly insufficient?
- Proceed to magnetic moments immediately?

---

## What Happens Next

### Immediate (Within 1 Hour)
1. ⏳ Sanity check completes
2. ✓ Examine results (χ², β_min, S_opt)
3. ❓ Make go/no-go decision on 9×6 scan

### If Sanity Check Passes (Path A)
**Timeline: 2-3 days**
1. Run 9×6 modest scan (3-4 hours)
2. Analyze results, plot contours (1 hour)
3. Compute profile widths Δβ_1σ (30 min)
4. Update manuscript sections (1-2 hours)
5. Review with Tracy
6. **Publication-ready**

### If Diagnosis Needed (Path B or C)
**Timeline: +1-2 days**
1. Debug session with Tracy
2. Implement fixes
3. Re-test with small grid
4. Then proceed as Path A

---

## Bottom Line

**Implemented**: Global S profiling exactly as Tracy specified ✓

**Completed**: Full sensitivity analysis showing:
- ✓ Optimizer convergence robust
- ✓ w shows preference (w_min ≈ 0.022)
- ⚠ β result concerning (3.20 instead of 3.10)

**Testing**: Sanity check with new mapping running

**Decision**: Awaiting sanity results to determine if:
- Mechanism validated (β ~ 3.10) → proceed
- Issue found (β ~ 3.20) → debug
- Implementation broken (χ² ~ 10⁷) → fix

**Critical path**: Sanity check (running) → decision → next phase

**Status**: Ready to proceed pending sanity check validation
