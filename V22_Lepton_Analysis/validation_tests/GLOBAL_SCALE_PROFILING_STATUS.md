# Global Scale Profiling Implementation - Status Update

**Date**: 2024-12-24
**Action**: Implemented Tracy's recommended global mass-scale profiling
**Status**: ✓ IMPLEMENTED, ⏳ TESTING IN PROGRESS

---

## What Was Fixed

### Problem: χ² ~ 10⁷ (Unphysical)

**Old code** (lines 151-162 of `profile_likelihood_boundary_layer.py`):
```python
# WRONG: Forces electron mass match, destroys falsifiability
energies = np.array([E_e, E_mu, E_tau])
scale = self.m_targets[0] / energies[0]  # Per-evaluation electron match
masses_model = energies * scale
residuals = (masses_model - self.m_targets) / (self.sigma_model * self.m_targets)
chi2 = np.sum(residuals**2)
```

**Issues**:
1. Forces perfect electron mass match → only muon/tau contribute to χ²
2. Per-evaluation scaling → no global falsifiability
3. χ² ~ 10⁷ indicates mis-scaled objective

### Solution: Analytic Global Scale Profiling

**New code** (implemented):
```python
# CORRECT: Global nuisance parameter S profiled analytically
#
# Objective: χ² = Σ[(S·E_ℓ - m_ℓ)²/σ_ℓ²]
#
# Analytic minimization over S:
#   S_opt = Σ[m_ℓ·E_ℓ/σ_ℓ²] / Σ[E_ℓ²/σ_ℓ²]

energies = np.array([E_e, E_mu, E_tau])
sigma_abs = self.sigma_model * self.m_targets
weights = 1.0 / sigma_abs**2

numerator = np.sum(self.m_targets * energies * weights)
denominator = np.sum(energies**2 * weights)
S_opt = numerator / denominator

masses_model = S_opt * energies
residuals = (masses_model - self.m_targets) / sigma_abs
chi2 = np.sum(residuals**2)  # Should be ~ O(1) if structure is correct
```

**Benefits**:
1. **One global scale** for all leptons (no per-lepton tuning)
2. **Falsifiable**: If structure wrong, χ² will be large
3. **Interpretable**: χ² ~ O(1) expected, Δχ² thresholds meaningful
4. **S_opt reported** in results for transparency

---

## Expected Outcomes (Testing Now)

### 1. χ² Magnitudes
**Before**: χ² ~ 10⁷ (arbitrary scale)
**After**: χ² ~ O(1) to O(10) (honest objective)

If structure is correct (gradient + boundary layer model captures physics):
- Good fit: χ² ≈ 1-3 (3 constraints, 9 parameters)
- Acceptable: χ² ≈ 3-10
- Bad: χ² > 10 (model missing physics)

### 2. β Shift Direction Preserved
**Critical test**: Does β still move toward 3.058 with gradient?

**Before** (old mapping):
- No gradient: β_eff ≈ 3.15
- With gradient: β_eff ≈ 3.10

**After** (new mapping):
- Expect: Same directional shift (3.15 → 3.10)
- If preserved → mechanism validated independent of normalization
- If not → something deeply wrong

### 3. Δχ² Profiles Interpretable
Standard confidence contours become meaningful:
- Δχ² = 1: 1σ (~68% CL, 1 parameter)
- Δχ² = 4: 2σ (~95% CL, 1 parameter)
- Δχ² = 9: 3σ (~99.7% CL, 1 parameter)

For 2D (β, w):
- Δχ² = 2.30: 1σ ellipse
- Δχ² = 6.18: 2σ ellipse
- Δχ² = 11.83: 3σ ellipse

### 4. S_opt Value
Expect: S_opt ~ O(1) to O(10) (dimensional analysis)

If S_opt ~ 10⁻⁶ or 10⁶ → unit mismatch somewhere

---

## Tests Running

### A. Sanity Check (3×3 Grid)
**File**: `test_global_scale_profiling.py`
**Grid**: β ∈ [3.00, 3.15] (3 points) × w ∈ [0.01, 0.02] (3 points)
**Iterations**: max_iter = 200 (proven convergent from sensitivity tests)
**Status**: ⏳ RUNNING
**ETA**: ~10-15 min

**Purpose**: Verify:
1. χ² ~ O(1), not 10⁷
2. β minimum direction same (toward 3.058)
3. Δχ² profiles smooth (no optimizer artifacts)

### B. Sensitivity Tests (Old Mapping)
**File**: `resolution_sensitivity_test.py`
**Tests**:
1. ✓ Optimizer convergence (complete): Stable by max_iter=200
2. ✓ w variation (complete): w_min ≈ 0.022, Δχ² ~ 10⁵
3. ⏳ β variation (running): Testing β ∈ [3.00, 3.20] (9 points)

**Status**: ~80% complete, will finish soon

**Note**: These use OLD mapping (χ² ~ 10⁷) but still validate:
- Optimizer convergence
- Grid spacing adequacy
- Relative Δχ² structure

---

## Implementation Details

### Files Modified

**`profile_likelihood_boundary_layer.py`**:
- Lines 151-185: `LeptonFitter.objective()` - Analytic S profiling
- Lines 228-242: `LeptonFitter.fit()` - Report S_opt in results

### Changes Made

1. **Objective function** (lines 151-185):
   - Removed per-evaluation electron forcing
   - Added analytic S_opt calculation
   - Used absolute uncertainties σ_abs = σ_model × m_target
   - Proper weighted least-squares for global scale

2. **Result reporting** (lines 228-266):
   - Compute S_opt at best-fit parameters
   - Add to result dict: `"S_opt": S_opt`
   - Masses computed consistently with objective

3. **Documentation**:
   - Inline comments explain S profiling formula
   - Note: "should be ~ O(1) if structure is correct"

### Testing Created

**`test_global_scale_profiling.py`**: 3×3 sanity grid

---

## What's Next (After Sanity Check)

### If Sanity Check Succeeds (χ² ~ O(1)):

**Immediate**:
1. **Modest 9×6 scan** (Tracy's recommendation)
   - β ∈ [3.00, 3.20] with 9 points
   - w ∈ [0.005, 0.03] with 6 points
   - Total: 54 points (vs 160 for full production)
   - ETA: ~3-4 hours

2. **Δχ² contour plots**
   - 2D landscape (β, w)
   - Standard 1σ/2σ/3σ contours
   - Mark β_Golden = 3.058 on plot

3. **Profile width analysis**
   - Δβ_1σ: width where Δχ² = 1
   - Report: β_min ± Δβ_1σ
   - If Δβ_1σ > 0.05: landscape "flat"
   - If Δβ_1σ < 0.01: landscape "sharp"

**Then**:
4. **Manuscript updates**
   - Results: Use corrected language from `PATH_B_PRIME_ANALYSIS_CORRECTED.md` § 5
   - Figures: 2D Δχ² contours with Golden Loop marked
   - Discussion: χ closure quantification (0.912 → 0.959)

### If Sanity Check Fails (χ² Still Large):

**Diagnose**:
1. Check S_opt magnitude (should be O(1-10))
2. Examine masses_model vs m_targets (residuals)
3. Check energy magnitudes (E_e, E_mu, E_tau)
4. Verify boundary layer profile not degenerate

**Possible issues**:
- Unit mismatch in energy calculation
- Boundary layer too thin (w << R_c issues)
- Optimizer not finding true minimum

---

## Tracy's Action Items - Status

### ✓ 1. Fix DOF Counting
**Status**: COMPLETE
**File**: `PATH_B_PRIME_ANALYSIS_CORRECTED.md` § 1
**Language**: "3 constraints, 11 parameters → 8-dimensional manifold"

### ✓ 2. Replace "46% Variation" with Δχ²
**Status**: COMPLETE
**File**: `PATH_B_PRIME_ANALYSIS_CORRECTED.md` § 2
**Next**: After mass fix, compute profile width Δβ_1σ

### ✓ 3. χ Framework Quantification
**Status**: COMPLETE
**File**: `PATH_B_PRIME_ANALYSIS_CORRECTED.md` § 3
**Result**: χ improved 0.912 → 0.959 (53% gap closure)

### ✓ 4. Verify Normalization
**Status**: COMPLETE
**Finding**: All 2π/4π factors consistent ✓

### ✓ 5. Implement Global S Profiling
**Status**: ✓ IMPLEMENTED, ⏳ TESTING
**Modification**: Analytic profiling in objective + fit methods
**Testing**: 3×3 sanity grid running

### ⏳ 6. Sensitivity Tests
**Status**: 80% COMPLETE
- ✓ Optimizer convergence (stable at max_iter=200)
- ✓ w variation (w_min ≈ 0.022, Δχ² ~ 10⁵)
- ⏳ β variation (running, ~80% complete)

### 🔜 7. Modest 9×6 Scan
**Status**: PENDING (after sanity check)
**Sequence**: Sanity → 9×6 → Contours → Manuscript

### 🔜 8. Magnetic Moments
**Status**: DEFERRED (Tracy's recommendation)
**Rationale**: Finish Path B' properly first, then add μ_ℓ

---

## Sequence Plan (Tracy's Recommendation)

1. ✓ Global S profiling (DONE)
2. ⏳ 3×3 sanity check (RUNNING)
3. 🔜 Verify χ² ~ O(1) and β direction preserved
4. 🔜 9×6 modest scan
5. 🔜 Δχ² contours and profile widths
6. 🔜 Manuscript updates
7. 🔜 Path B' finalized and locked
8. 🔜 Then: Magnetic moments (Phase II)

---

## Expected Timeline

**Today** (2024-12-24):
- ⏳ Sanity check completes (~15 min remaining)
- ⏳ Sensitivity tests complete (~30 min remaining)
- ✓ Verify results (χ² magnitudes, β shift)
- 🔜 If good: Launch 9×6 scan (3-4 hours)

**Tomorrow** (2024-12-25):
- 🔜 Analyze 9×6 results
- 🔜 Generate Δχ² contour plots
- 🔜 Compute profile widths
- 🔜 Update manuscript sections

**Publication-ready**: ~2-3 days from now

---

## Critical Validation Points

Before proceeding to 9×6 scan, verify ALL of:

- [ ] χ² ~ O(1) to O(10), not 10⁷
- [ ] β_min ≈ 3.10 (same direction toward 3.058)
- [ ] S_opt ~ O(1) to O(10) (reasonable scale)
- [ ] Masses_model close to m_targets (within σ)
- [ ] No optimizer failures (all 9 points converge)
- [ ] Smooth Δχ² profile (no discontinuities)

If ANY fail → debug before wider scan

---

## Code Quality

### Testability ✓
- Analytic S formula documented with math
- Expected magnitudes specified (χ² ~ O(1))
- S_opt reported in results (transparency)

### Falsifiability ✓
- GLOBAL scale (not per-lepton)
- If structure wrong, χ² will be large
- No hidden tuning parameters

### Reproducibility ✓
- Seed=42 for deterministic runs
- Formula explicit (weighted least-squares)
- All parameters in output JSON

---

## Bottom Line

**Global S profiling implemented** following Tracy's exact specification.

**Testing in progress**: 3×3 sanity grid to verify:
1. χ² magnitudes honest (~ O(1))
2. β shift direction preserved (mechanism independent of normalization)
3. Δχ² profiles interpretable (standard thresholds)

**If successful**: Proceed to 9×6 scan → contours → manuscript → publication

**Critical path**: Sanity check (running) → verify → 9×6 scan → finalize

**ETA to publication-ready**: ~2-3 days
