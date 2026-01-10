# Session Summary: 2026-01-02

**Topic**: Implementation of QFD Nuclear Harmonic Model Experimental Validation Pipeline
**Duration**: Full session
**Status**: Core pipeline complete, Experiment 1 in progress

---

## Overview

Implemented complete data pipeline from raw nuclear data through Experiment 1 testing of the harmonic family model's existence clustering hypothesis.

**Critical Discovery**: The harmonic model **fails** its primary test—observed nuclides have HIGHER ε than null candidates (opposite of hypothesis).

---

## Completed Implementations

### 1. Core Harmonic Model (`src/harmonic_model.py`)

**Purpose**: Mathematical framework for harmonic family model

**Features**:
- Z_predicted(A, N, params): Model predictions
- N_hat(A, Z, params): Continuous mode estimation
- epsilon(A, Z, params): Dissonance metric
- score_best_family(A, Z, families): Multi-family scoring
- dc3_comparison(families): Universality check
- Full parameter validation and error checking

**Status**: ✓ Production-ready, all 14 unit tests pass

**Key formula**:
```
Z_pred(A, N) = (c1_0 + N·dc1)·A^(2/3) + (c2_0 + N·dc2)·A + (c3_0 + N·dc3)
ε = |N_hat - round(N_hat)| ∈ [0, 0.5]
```

**Documentation**: `src/HARMONIC_MODEL_README.md` (1200+ lines)

---

### 2. Parameter Fitting (`src/fit_families.py`)

**Purpose**: Fit harmonic family parameters to training data

**Training set**: 337 stable nuclides (no leakage)

**Method**: Least-squares with Levenberg-Marquardt

**Results**:

| Family | c1_0 | c2_0 | c3_0 | dc1 | dc2 | dc3 |
|--------|------|------|------|-----|-----|-----|
| A | 2.724 | 0.016 | -7.049 | -0.074 | 0.012 | **-0.799** |
| B | 2.711 | 0.034 | -6.927 | -0.084 | 0.012 | **-0.803** |
| C | 2.798 | 0.011 | -7.078 | -0.086 | 0.014 | **-0.825** |

**Fit quality**:
- χ²_red ≈ 0.11-0.14 (excellent)
- RMSE ≈ 0.33-0.38 protons
- Maximum residual < 0.7 protons

**dc3 Universality**: ✓ CONFIRMED
- Mean: -0.809
- Std: 0.011
- Relative std: **1.38%** < 2% threshold
- Interpretation: dc3 may be a universal "clock step"

**Status**: ✓ Production-ready

**Output**: `reports/fits/family_params_stable.json`

---

### 3. Harmonic Scoring (`src/score_harmonics.py`)

**Purpose**: Score all 3,558 nuclides using fitted parameters

**Output**: `data/derived/harmonic_scores.parquet`

**Columns added**:
- epsilon_best, epsilon_A, epsilon_B, epsilon_C
- best_family, N_hat_best, N_best
- Z_pred_best, residual_best
- category (harmonic/near_harmonic/dissonant)

**Results**:

| Metric | Value |
|--------|-------|
| Mean ε | 0.134 |
| Median ε | 0.110 |

**Category distribution**:
- Harmonic (ε < 0.05): 918 (25.8%)
- Near-harmonic (0.05-0.15): 1,310 (36.8%)
- Dissonant (ε ≥ 0.15): 1,330 (37.4%)

**Family distribution**:
- Family A: 1,264 (35.5%)
- Family B: 1,165 (32.7%)
- Family C: 1,129 (31.7%)

**Key observation**: Best-family ε (0.134) << individual-family ε (~0.249), showing multi-family structure captures real patterns.

**Status**: ✓ Production-ready

---

### 4. Null Model Generation (`src/null_models.py`)

**Purpose**: Generate candidate universe and baseline models for Experiment 1

**Features**:
- Candidate enumeration for each A (physics bounds option)
- Smooth valley fitting (spline/polynomial/physics-motivated)
- Baseline scoring (distance from valley)
- Observed vs null flagging

**Results**:

| Metric | Value |
|--------|-------|
| Total candidates | 22,412 |
| Observed | 3,555 (15.9%) |
| Null | 18,857 (84.1%) |
| Average candidates per A | 76 |

**Smooth baseline performance**:
- Observed mean distance from valley: 3.9 protons
- Null mean distance from valley: 29.0 protons
- Separation: -25.1 protons
- **Smooth baseline works extremely well!**

**Status**: ✓ Production-ready

**Output**: `data/derived/candidates_by_A.parquet`

---

### 5. Experiment 1 (`src/experiments/exp1_existence.py`)

**Purpose**: Test if observed nuclides have lower ε than null candidates

**Status**: ⏳ IN PROGRESS (permutation test running)

**Implemented metrics**:
1. Mean ε separation with bootstrap CI
2. AUC (existence classifier)
3. Calibration curve P(exists | ε)
4. Permutation test (1000 iterations)
5. KS test by A-bin
6. Baseline comparisons (harmonic vs smooth vs random)

**Preliminary Results** (permutation test pending):

#### METRIC 1: Mean ε Separation

| Group | Mean ε | 95% CI |
|-------|--------|--------|
| Observed | 0.1337 | — |
| Null | 0.1250 | — |
| **Separation** | **+0.0087** | [+0.0049, +0.0123] |

**Finding**: Observed nuclides have **HIGHER** ε (opposite of hypothesis!)

#### METRIC 2: AUC (Existence Classifier)

| Model | AUC | Interpretation |
|-------|-----|----------------|
| **Harmonic (ε)** | **0.4811** | **Below chance!** |
| **Smooth baseline** | **0.9757** | Near-perfect |
| **Random** | 0.5019 | Chance level |

**Critical finding**: Harmonic model performs **worse than random guessing**.

**Comparison**:
- Harmonic vs Random: 0.48 < 0.50 (harmonic is worse)
- Harmonic vs Smooth: 0.48 << 0.98 (smooth destroys harmonic)
- AUC difference: -0.50 (threshold was +0.05)

**Pass criterion**: AUC_ε > AUC_smooth + 0.05 AND p < 1e-4
**Actual**: 0.48 < 0.98 + 0.05
**Result**: **FAILS by massive margin**

---

## Surprising Discoveries

### Discovery 1: Stable ≠ Harmonic

**Finding**: Stable nuclides have HIGHER ε than unstable ones.

**Evidence** (from diagnostic analysis):
- Stable: ε_mean = 0.146, ε_median = 0.122
- Unstable: ε_mean = 0.132, ε_median = 0.110
- Difference: +0.013 (p = 0.047)

**Implication**: The harmonic model does NOT predict stability.

**Most harmonic nuclides** (ε ≈ 0): ALL unstable!
- 180W, 230Rn, 4H, 213Ra, 187Tl, 192Tl, 166Yb, 10C, 76Ge, 207At

**Category breakdown**:
- Stable: 43% dissonant vs Unstable: 37% dissonant
- Stable nuclides are LESS harmonic on average

---

### Discovery 2: Harmonics Anti-Predict Existence

**Finding**: Lower ε → LESS likely to exist (not more!)

**Evidence** (Experiment 1):
- Observed ε = 0.134 > Null ε = 0.125
- AUC = 0.48 (anti-correlation)

**Implication**: The harmonic model has the **opposite** relationship to what was hypothesized.

---

### Discovery 3: Valley of Stability Reigns

**Finding**: Distance from valley of stability is THE existence predictor.

**Evidence**:
- Smooth baseline AUC = 0.98 (near-perfect)
- Harmonic model AUC = 0.48 (worse than random)
- Observed distance: 3.9 protons vs Null: 29.0 protons

**Implication**: Existing nuclear physics (valley of stability) vastly outperforms harmonic model.

---

## Possible Interpretations

### Interpretation 1: Harmonics Predict Instability (Reversed Model)

**Hypothesis**: Low ε → reactive/unstable, not stable/existing.

**Supporting evidence**:
- Most harmonic nuclides (ε ≈ 0) are ALL unstable
- Observed nuclides (mostly unstable) have lower ε than stable
- Existence anti-correlates with harmonics

**Prediction**: ε should anti-correlate with half-life (low ε → short t₁/₂).

**Test needed**: Plot log₁₀(t₁/₂) vs ε for unstable nuclides.

**Problem**: This is post-hoc rationalization after failure.

---

### Interpretation 2: Training Set Bias

**Hypothesis**: Fitting on stable nuclides creates artifact.

**Mechanism**:
- Stable nuclides have higher ε (empirical fact)
- Model fits to match stable nuclides
- Therefore model learns "low ε = not stable"

**Prediction**: Refitting on all nuclides should reverse pattern.

**Test needed**: Refit families on all 3,558 nuclides, re-score, re-test.

---

### Interpretation 3: Null Universe Too Restricted

**Hypothesis**: Physics bounds pre-select valley-like candidates.

**Mechanism**:
- Valley band (±0.25A) already clusters near valley
- Null candidates are "valley-compliant" by construction
- Harmonic model can't discriminate within this set

**Prediction**: Full Z enumeration (no physics bounds) should change results.

**Test needed**: Regenerate candidates with full Z range, re-test.

---

### Interpretation 4: Model is Fundamentally Wrong

**Hypothesis**: Harmonic structure is artifact, not physics.

**Evidence**:
- AUC < 0.50 (anti-predictive)
- Valley baseline vastly superior (0.98 vs 0.48)
- dc3 universality may be parameterization artifact

**Implication**: The model has no predictive power. Accept failure.

---

## What Remains Valid?

### Valid: dc3 Universality (Tentative)

**Result**: dc3 values agree to 1.38% across three families.

**Status**: Mathematical fact for fitted model.

**Caveats**:
- May be artifact of flexible parameterization
- Need to test robustness with different training sets
- Need to test with different fitting protocols

**Recommendation**: Test with holdout-by-A cross-validation.

---

### Valid: Multi-Family Structure (Tentative)

**Result**: Nuclides cluster near different families (not random).

**Status**: Best-family ε (0.134) << individual ε (~0.249).

**Caveats**:
- May reflect valley shape, not distinct harmonic modes
- Three families = high DOF (18 parameters total)
- May be overfitting

**Recommendation**: Test 2-family vs 3-family vs 4-family models.

---

### Valid: Good Fit Quality (Weak Evidence)

**Result**: RMSE ≈ 0.33 protons for stable nuclides.

**Status**: Model fits training set well.

**Caveats**:
- Fit quality ≠ predictive power (as Exp 1 shows!)
- High DOF (18 parameters) can fit anything
- Out-of-sample performance is what matters

**Lesson**: Never trust in-sample fit quality alone.

---

## Lessons Learned

### Lesson 1: Pattern Recognition ≠ Prediction

**Initial promise**:
- dc3 universality looked fundamental
- Three families fit well
- Multi-family structure seemed real

**Reality check**:
- Out-of-sample test reveals failure
- AUC < 0.50 shows anti-predictive power
- Valley baseline demolishes harmonic model

**Takeaway**: Always test against null models with out-of-sample data.

---

### Lesson 2: Pre-Registration Prevents Cherry-Picking

**EXPERIMENT_PLAN.md** specified in advance:
- Exact metrics (AUC, separation, permutation)
- Exact pass criteria (AUC > smooth + 0.05, p < 1e-4)
- Exact null models (candidates by A, smooth baseline)

**Result**: Cannot rationalize failure or cherry-pick favorable metrics.

**Takeaway**: Pre-registration enforces intellectual honesty.

---

### Lesson 3: Existing Physics is Hard to Beat

**Valley of stability** (AUC = 0.98):
- Reflects decades of nuclear physics research
- Based on strong force, Coulomb, asymmetry energy
- Extremely accurate predictor

**Harmonic model** (AUC = 0.48):
- Novel idea with no established physics basis
- Fails to beat even chance-level performance
- Adds nothing beyond valley

**Takeaway**: New models must clear high bar set by existing physics.

---

### Lesson 4: Failure is Valuable

**Scientific value of negative results**:
- Prevents others from pursuing dead ends
- Demonstrates rigorous testing protocol
- Shows importance of null models
- Validates existing physics (valley of stability)

**Publication potential**: "Testing the Harmonic Family Model: A Null Model Analysis"

**Takeaway**: Honest negative results advance science.

---

## Recommendations

### Immediate Actions

1. **Wait for permutation test completion**
   - Verify p-value confirms failure
   - Document full results in JSON
   - Update EXP1_PRELIMINARY_RESULTS.md

2. **Sensitivity analyses**
   - Refit on all nuclides (not just stable)
   - Test full Z enumeration (no physics bounds)
   - Test different valley widths (0.10, 0.25, 0.50)
   - Check dc3 robustness across training sets

3. **Half-life correlation test**
   - Plot log₁₀(t₁/₂) vs ε for unstable nuclides
   - Test "reversed model" (harmonics → instability)
   - Check if low-ε isotopes have short half-lives

---

### Medium-term Actions

4. **Run Experiments 2-3 for completeness**
   - Document Exp 2 (stability selector) results
   - Document Exp 3 (decay mode prediction) results
   - May reveal partial predictive power

5. **Decide on interpretation**
   - **Option A**: Accept failure, publish negative result
   - **Option B**: Reinterpret as instability predictor
   - **Option C**: Debug and revise model

6. **Write manuscript**
   - Transparent about failure
   - Emphasize rigorous testing protocol
   - Discuss lessons about pattern vs prediction
   - Validate valley of stability as primary selector

---

### Long-term Considerations

7. **If pursuing revised model**
   - Test "instability hypothesis" (low ε → short t₁/₂)
   - Refit on all nuclides, not just stable
   - Add physics terms (magic numbers, pairing)
   - Reduce DOF (constrain dc3, use fewer families)

8. **If accepting failure**
   - Archive code and data for reproducibility
   - Publish negative result
   - Move on to other projects
   - Credit rigorous testing as success

---

## File Inventory

### Code Modules (All Production-Ready)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `src/harmonic_model.py` | 470 | ✓ Tested | Core model |
| `src/fit_families.py` | 373 | ✓ Works | Parameter fitting |
| `src/score_harmonics.py` | 196 | ✓ Works | Score nuclides |
| `src/null_models.py` | 464 | ✓ Works | Generate candidates |
| `src/experiments/exp1_existence.py` | 511 | ⏳ Running | Experiment 1 |

**Total code**: ~2,000 lines

---

### Documentation Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/HARMONIC_MODEL_README.md` | 1200+ | Core model docs |
| `src/PARSE_NUBASE_README.md` | 355 | NUBASE parser docs |
| `src/PARSE_AME_README.md` | 448 | AME parser docs |
| `EXPERIMENT_PLAN.md` | 382 | Experimental protocol |
| `PIPELINE_STATUS.md` | 450 | Pipeline status |
| `INITIAL_FINDINGS.md` | 350 | Preliminary diagnostics |
| `EXP1_PRELIMINARY_RESULTS.md` | 400 | Exp 1 analysis |
| `SESSION_SUMMARY_2026-01-02.md` | This file | Session summary |

**Total documentation**: ~3,500 lines

---

### Data Products

| File | Size | Rows | Purpose |
|------|------|------|---------|
| `data/derived/nuclides_all.parquet` | 150 KB | 3,558 | NUBASE ground states |
| `data/derived/ame.parquet` | 250 KB | 3,558 | AME Q-values |
| `data/derived/harmonic_scores.parquet` | 280 KB | 3,558 | Scored nuclides |
| `data/derived/candidates_by_A.parquet` | 1.5 MB | 22,412 | Null universe |
| `reports/fits/family_params_stable.json` | 3 KB | — | Fitted parameters |
| `reports/exp1/candidates_scored.parquet` | 1.8 MB | 22,412 | Exp 1 scored candidates |
| `reports/exp1/exp1_results.json` | Pending | — | Exp 1 full results |

**Total data**: ~4 MB compressed

---

## Current Status

### Completed ✓

1. Core harmonic model (tested, documented)
2. Parameter fitting (dc3 universality confirmed)
3. Harmonic scoring (all nuclides scored)
4. Null model generation (22,412 candidates)
5. Experiment 1 implementation (metrics complete)

### In Progress ⏳

6. Experiment 1 permutation test (running ~20 minutes)

### Pending

7. Experiment 2 (stability selector)
8. Experiment 3 (decay mode prediction)
9. Experiment 4 (boundary sensitivity)
10. Sensitivity analyses
11. Half-life correlation test
12. Manuscript preparation

---

## Critical Decision Point

**The harmonic model has failed Experiment 1.**

**AUC = 0.48** (below chance) means the model is worse than random guessing.

**Smooth baseline AUC = 0.98** means existing physics (valley of stability) vastly outperforms.

**Decision required**: Accept failure and publish negative result, or attempt model revision?

**Recommendation**: Accept failure. The evidence is clear, the testing was rigorous, and negative results have scientific value.

---

## Conclusions

### What We Achieved

1. **Rigorous implementation** of experimental protocol
2. **Comprehensive testing** with null models
3. **Transparent documentation** of methods and results
4. **Discovery of failure** through proper testing
5. **Validation of existing physics** (valley of stability)

### What We Learned

1. Pattern recognition ≠ prediction
2. In-sample fit quality ≠ out-of-sample performance
3. Pre-registration prevents cherry-picking
4. Null models are essential
5. Negative results are valuable

### What We Demonstrated

1. How to properly test pattern claims
2. Importance of out-of-sample validation
3. Power of null model comparisons
4. Value of pre-registered protocols
5. Scientific honesty in reporting failures

---

**Session End**: 2026-01-02
**Status**: Experiment 1 permutation test completing
**Next session**: Decide on interpretation and next steps

---

**This has been a scientifically productive session, even though the model failed. Rigorous testing that reveals failure is more valuable than uncritical pattern-matching that leads nowhere.**
