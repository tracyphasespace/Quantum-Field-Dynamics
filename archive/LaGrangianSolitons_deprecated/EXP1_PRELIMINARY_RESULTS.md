# Experiment 1: Preliminary Results

**Date**: 2026-01-02
**Status**: Permutation test in progress, main metrics complete
**Conclusion**: **EXPERIMENT 1 FAILS DRAMATICALLY**

---

## Executive Summary

The harmonic model **fails** the primary existence clustering test:

### Critical Findings

1. **Harmonic model AUC = 0.4811** (BELOW chance level of 0.50)
2. **Smooth baseline AUC = 0.9757** (near-perfect)
3. **Observed nuclides have HIGHER ε** (+0.0087) than null candidates
4. **Difference from smooth baseline**: 0.98 - 0.48 = **-0.50** (massive failure)

**Interpretation**: The harmonic model is **WORSE than random guessing** at predicting which nuclides exist.

---

## Detailed Results

### METRIC 1: Mean ε Separation

| Group | Mean ε | Median ε |
|-------|--------|----------|
| **Observed** | 0.1337 | — |
| **Null** | 0.1250 | — |
| **Separation** | **+0.0087** | — |
| **95% CI** | [+0.0049, +0.0123] | — |

**Finding**: Observed nuclides have **HIGHER** epsilon than null candidates.

**This is opposite of the hypothesis** (EXPERIMENT_PLAN.md §4.1).

**Effect direction**: Positive separation means observed nuclides are MORE dissonant, not less.

---

### METRIC 2: AUC (Existence Classifier)

| Model | AUC | Separation | Interpretation |
|-------|-----|------------|----------------|
| **Harmonic (ε)** | **0.4811** | +0.0087 | **Below chance** |
| **Smooth baseline** | **0.9757** | -25.0531 | **Excellent** |
| **Random** | 0.5019 | +0.0018 | Chance level |

**Key comparisons**:
- Harmonic vs Random: 0.48 < 0.50 (**harmonic is worse!**)
- Harmonic vs Smooth: 0.48 << 0.98 (**smooth destroys harmonic**)
- AUC difference: 0.48 - 0.98 = **-0.50** (threshold was +0.05)

**Pass criterion** (EXPERIMENT_PLAN.md §4.5):
- Required: AUC_ε > AUC_smooth + 0.05
- Actual: 0.48 < 0.98 + 0.05
- **FAILS by massive margin** (-0.50 vs +0.05)

---

### METRIC 3: Calibration Curve

**Status**: Computed (10 bins)

**Awaiting**: Full analysis in results JSON

---

### METRIC 4: Permutation Test

**Status**: Running (1000 permutations)

**Awaiting**: p-value

**Expected**: Given AUC < 0.50, permutation test will likely show observed separation is NOT significant (or significant in the WRONG direction).

---

## What Went Wrong?

### The Reversed Pattern

**Observed pattern**: Lower ε → LESS likely to exist

**Expected pattern**: Lower ε → MORE likely to exist

**Possible explanations**:

1. **Harmonic model predicts instability, not existence**
   - Low ε → reactive, short-lived states
   - High ε → stable defects (magic numbers)

2. **Training set bias**
   - Fit on stable nuclides (337)
   - Stable have higher ε (established earlier)
   - Model learns to predict "non-stable" as low ε

3. **Null universe is too restricted**
   - Physics bounds (valley band ±0.25A) already cluster near valley
   - Null candidates are pre-selected to be "valley-like"
   - Harmonic model can't discriminate within this restricted set

4. **Model is fundamentally wrong**
   - dc3 universality is an artifact of flexible parameterization
   - Three families are overfitting with too many DOF
   - No real physical structure captured

---

## Why Smooth Baseline Works So Well (AUC = 0.98)

**Smooth baseline**: Distance from valley of stability

**Why it works**:
- Valley of stability is a **real physical phenomenon**
- Reflects balance of strong force, Coulomb repulsion, asymmetry energy
- Observed nuclides cluster tightly near valley (mean distance = 3.9 protons)
- Null candidates scatter broadly (mean distance = 29.0 protons)
- Clear separation → high AUC

**Implication**: The valley of stability is the **primary selector** for existence, not harmonic structure.

---

## Implications for Other Experiments

### Experiment 2 (Stability Selector)

**Original hypothesis**: Stable nuclides have lower ε.

**Diagnostic finding** (from earlier): Stable have **HIGHER** ε (+0.013).

**Revised expectation**: Experiment 2 will also fail (or show reversed pattern).

**Status**: Not yet run, but preliminary diagnostics suggest failure.

---

### Experiment 3 (Decay Mode Prediction)

**Original hypothesis**: ε improves decay mode classification.

**Diagnostic finding**: Weak variation by mode (~1-2% ε range).

**Revised expectation**: Marginal improvement at best, may still fail.

**Status**: Not yet run.

---

### Experiment 4 (Boundary Sensitivity)

**Original hypothesis**: Edge-ε nuclides are sensitive to ionization.

**Revised expectation**: If ε has no predictive power, sensitivity ranking is meaningless.

**Status**: Target list not yet generated. May be obsolete if Exp 1-3 fail.

---

## Conclusions

### The Harmonic Model Has Failed

**Verdict**: The harmonic model **does not** predict which nuclides exist.

**Evidence**:
1. AUC = 0.48 (below random)
2. Observed have higher ε than null (wrong direction)
3. Smooth baseline vastly outperforms (0.98 vs 0.48)

**Critical failure**: The PRIMARY FALSIFIER (Exp 1) has been failed.

---

### What Remains Valid?

**dc3 Universality** (1.38% relative std):
- Still a mathematical fact for the fitted model
- BUT: May be an artifact of parameterization
- Need to test with different training sets to confirm robustness

**Multi-Family Structure**:
- Nuclides cluster near different families (not random)
- BUT: This may reflect valley shape, not distinct harmonic modes

**Fit Quality** (RMSE ≈ 0.33 protons):
- Model fits stable nuclides well
- BUT: Many parameters (6 per family × 3 families = 18 DOF)
- May be overfitting with no real structure

---

### What to Do Next?

#### Option 1: Revise Interpretation (Harmonic → Instability)

**Hypothesis**: ε predicts **instability/reactivity**, not existence.

**Tests**:
1. Correlate ε with half-life (expect negative correlation)
2. Check if most reactive isotopes (short t₁/₂) have low ε
3. Reframe model as "resonant decay channels" not "stable resonances"

**Problem**: This is **post-hoc** rationalization after failure.

---

#### Option 2: Diagnose and Fix Model

**Possible fixes**:
1. Refit on different training set (all nuclides, not just stable)
2. Use different null universe (full Z range, not valley band)
3. Add physics terms (magic numbers, pairing, deformation)
4. Reduce DOF (fewer families, constrained parameters)

**Problem**: Risk of overfitting to make Exp 1 pass.

---

#### Option 3: Acknowledge Failure and Publish Negative Result

**Honest assessment**:
- Harmonic model looked promising (dc3 universality, good fits)
- BUT: Failed primary existence test
- Valley of stability is the real selector (AUC = 0.98)
- Harmonic structure is not predictive

**Scientific value**:
- Negative results are important
- Rigorous testing protocol (null models, out-of-sample, pre-registered metrics)
- Demonstrates how to properly test pattern claims

**Publication**: "Testing the Harmonic Family Model: A Rigorous Null Model Analysis"

---

## Recommendations

### Immediate Actions

1. **Wait for permutation test to complete**
   - Check if p-value confirms failure
   - Document full results in JSON

2. **Run sensitivity analyses**
   - Test different valley widths (0.10, 0.25, 0.50)
   - Test full Z enumeration (no physics bounds)
   - Test different training sets (all nuclides, longlived)

3. **Check dc3 robustness**
   - Refit on different subsets
   - Check if universality persists

### Medium-term Actions

4. **Decide on interpretation**
   - Accept failure and publish negative result? OR
   - Reinterpret as instability predictor and test half-life correlation? OR
   - Debug and fix model?

5. **Run Experiments 2-3 for completeness**
   - Even if Exp 1 fails, document Exp 2-3 results
   - May reveal partial predictive power

6. **Write up results**
   - Transparent about failure
   - Emphasize rigorous testing protocol
   - Discuss why patterns can be misleading

---

## Lessons Learned

### Pattern Recognition ≠ Prediction

**Initial promise**:
- Three families fit stable nuclides well
- dc3 universality looked like a fundamental constant
- Multi-family structure suggested real physics

**Reality check**:
- Fit quality doesn't guarantee out-of-sample prediction
- Universality can be artifact of flexible parameterization
- Structure can reflect confounds (valley shape) not physics

**Lesson**: Always test against **null models** with **out-of-sample** data.

---

### The Valley of Stability is King

**Smooth baseline** (AUC = 0.98) shows:
- Existing physics models (valley of stability) work extremely well
- Distance from valley is THE predictor of existence
- Harmonic model adds nothing beyond valley

**Implication**: Any new model must beat valley baseline, not just random.

---

### Pre-Registration Prevents Cherry-Picking

**EXPERIMENT_PLAN.md** specified:
- Exact metrics (AUC, separation, permutation test)
- Exact pass criterion (AUC > smooth + 0.05, p < 1e-4)
- Exact null models (candidates by A, smooth baseline)

**Result**: Cannot cherry-pick favorable metrics or thresholds after seeing results.

**Lesson**: Pre-registration enforces intellectual honesty.

---

## Final Thoughts

The harmonic model **fails** its primary test. This is disappointing but scientifically valuable:

1. **Rigorous testing works**: Null models exposed the failure.
2. **Negative results matter**: Publishing this prevents others from pursuing dead ends.
3. **Valley of stability reigns**: Existing physics is hard to beat.

**Next decision point**: Accept failure and publish negative result, or attempt model revision?

**Recommendation**: Accept failure. The evidence is clear and the testing was rigorous.

---

**Status**: Awaiting permutation test completion for final verdict.

**Last Updated**: 2026-01-02 (Exp 1 in progress)

**Author**: Claude (AI assistant)
