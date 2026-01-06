# Initial Findings from Harmonic Scoring

**Date**: 2026-01-02
**Status**: Preliminary diagnostic results (full experiments pending)
**Warning**: These are **initial observations**, not controlled experiments

---

## Executive Summary

### Surprising Result: Stable Nuclides Have HIGHER Epsilon

**Finding**: Stable nuclides show **higher** average dissonance (ε) than unstable nuclides.

- **Stable**: ε_mean = 0.146, ε_median = 0.122
- **Unstable**: ε_mean = 0.132, ε_median = 0.110
- **Difference**: +0.013 (stable higher, p = 0.047)

**This is opposite of the original hypothesis** (EXPERIMENT_PLAN.md §5.1) which predicted stable nuclides would have *lower* ε.

---

## Detailed Results

### Epsilon by Stability

| Group | N | Mean ε | Median ε | Std ε |
|-------|---|--------|----------|-------|
| Stable | 337 | 0.1457 | 0.1222 | 0.1126 |
| Unstable | 3,221 | 0.1324 | 0.1100 | 0.1031 |
| **Difference** | | **+0.0133** | **+0.0122** | |

**Statistical test**:
- Kolmogorov-Smirnov D = 0.078
- p-value = 0.047 < 0.05 (significant)
- **Distributions are significantly different, but in unexpected direction**

### Category Distribution

| Category | Stable | Unstable | Difference |
|----------|--------|----------|------------|
| Harmonic (ε < 0.05) | 27.3% | 25.6% | +1.7% |
| Near-harmonic (0.05-0.15) | 29.4% | 37.6% | -8.2% |
| Dissonant (ε ≥ 0.15) | 43.3% | 36.8% | **+6.5%** |

**Interpretation**: Stable nuclides are MORE dissonant on average!

---

## Most Harmonic Nuclides (ε ≈ 0)

**Top 10 lowest ε**:

| Nuclide | A | Z | ε | Family | Stable? |
|---------|---|---|---|--------|---------|
| 180W | 180 | 74 | 0.0000 | B | **No** |
| 230Rn | 230 | 86 | 0.0000 | A | **No** |
| 4H | 4 | 1 | 0.0000 | C | **No** |
| 213Ra | 213 | 88 | 0.0001 | A | **No** |
| 187Tl | 187 | 81 | 0.0001 | A | **No** |
| 192Tl | 192 | 81 | 0.0002 | B | **No** |
| 166Yb | 166 | 70 | 0.0002 | C | **No** |
| 10C | 10 | 6 | 0.0002 | B | **No** |
| 76Ge | 76 | 32 | 0.0003 | C | **No** |
| 207At | 207 | 85 | 0.0004 | A | **No** |

**Key observation**: ALL top-10 most harmonic nuclides are UNSTABLE!

---

## Most Dissonant Nuclides (ε ≈ 0.5)

**Top 10 highest ε**:

| Nuclide | A | Z | ε | Family | Stable? |
|---------|---|---|---|--------|---------|
| 35P | 35 | 15 | 0.4845 | B | **No** |
| 26Na | 26 | 11 | 0.4728 | C | **No** |
| 23O | 23 | 8 | 0.4715 | A | **No** |
| 28Na | 28 | 11 | 0.4665 | A | **No** |
| 61Co | 61 | 27 | 0.4570 | B | **No** |
| 59Fe | 59 | 26 | 0.4520 | C | **No** |
| 66Ni | 66 | 28 | 0.4502 | A | **No** |
| 41Cl | 41 | 17 | 0.4477 | B | **No** |
| 11Li | 11 | 3 | 0.4424 | B | **No** |
| 31Si | 31 | 14 | 0.4420 | A | **No** |

**Key observation**: All top-10 most dissonant are also UNSTABLE (no stable in either extreme).

---

## Epsilon by Decay Mode

| Mode | N | Mean ε | Median ε | Harmonic % | Dissonant % |
|------|---|--------|----------|------------|-------------|
| Alpha | 574 | 0.1266 | 0.1064 | 25.1% | 34.1% |
| Beta⁻ | 1,393 | 0.1374 | 0.1116 | 25.3% | 38.4% |
| Beta⁺ | 983 | 0.1293 | 0.1104 | 24.9% | 36.2% |
| EC | 111 | 0.1382 | 0.1168 | 31.5% | 37.8% |
| Proton | 63 | 0.1306 | 0.1108 | 28.6% | 33.3% |
| Neutron | 19 | 0.1164 | 0.0570 | 42.1% | 26.3% |
| Fission | 64 | 0.1223 | 0.1086 | 31.2% | 37.5% |

**Observations**:
- **Neutron emitters** have lowest mean ε (0.116) and highest harmonic fraction (42%)
- **Beta⁻ emitters** have highest mean ε (0.137)
- Differences are modest (~1-2% variation)

---

## Possible Interpretations

### Interpretation 1: Harmonics Predict Instability (Not Stability)

**Hypothesis**: Low ε → short-lived, high reactivity, rapid decay.

**Supporting evidence**:
- Most harmonic nuclides are ALL unstable
- Stable nuclides have HIGHER ε on average
- Magic numbers (stability islands) may be *anti-resonant* (off-harmonic)

**Prediction**: ε should anti-correlate with half-life (low ε → short t₁/₂).

**Test**: Plot log₁₀(t₁/₂) vs ε for unstable nuclides.

---

### Interpretation 2: Training Set Bias

**Hypothesis**: Fitting on stable nuclides forced model to match them, making unstable nuclides appear "more harmonic" by contrast.

**Supporting evidence**:
- Model was fit to minimize residuals for stable nuclides
- Unstable nuclides are out-of-sample relative to fit objective

**Counterargument**:
- If true, stable nuclides should have ε ≈ 0 (perfect fit)
- But stable ε_mean = 0.146 (NOT close to 0!)
- This suggests model is NOT just overfitting training set

**Test**: Refit on unstable nuclides, check if pattern reverses.

---

### Interpretation 3: A-Dependence Confound

**Hypothesis**: Stable and unstable nuclides cluster at different A ranges.

**Supporting evidence**:
- Light elements (A < 20) are mostly unstable
- Heavy elements (A > 200) are ALL unstable
- Mid-range (A ≈ 40-200) has both stable and unstable

**Prediction**: ε vs A should show different distributions for stable vs unstable.

**Test**: Match stable/unstable by A-bin and recompute effect size.

---

### Interpretation 4: Shell Effects vs Harmonics

**Hypothesis**: Stable nuclides are determined by shell closures (magic numbers), which are *orthogonal* to harmonic structure.

**Supporting evidence**:
- Magic numbers (Z = 2, 8, 20, 28, 50, 82; N = 2, 8, 20, 28, 50, 82, 126) are empirical
- Harmonic model has no explicit magic number terms
- Stable nuclides at magic numbers may be "defects" in harmonic lattice

**Prediction**: Nuclides at magic numbers should have HIGHER ε.

**Test**: Check ε distribution for doubly-magic vs non-magic nuclides.

---

## Implications for Experiments

### Experiment 1 (Existence Clustering)

**Original hypothesis**: Observed nuclides have lower ε than null candidates.

**Revised expectation**:
- May still pass if unstable nuclides dominate (3,221 unstable vs 337 stable)
- But effect size may be smaller than expected

**Critical test**: Compare ε_observed (all nuclides) vs ε_candidates (null universe).

---

### Experiment 2 (Stability Selector)

**Original hypothesis**: Stable nuclides have lower ε.

**Diagnostic result**: **OPPOSITE PATTERN** (stable have higher ε).

**Revised hypotheses to test**:
1. **Reversed selector**: Stable have HIGHER ε (current result)
2. **No effect**: After A-matching, difference disappears (confound)
3. **U-shaped**: Both very low and very high ε are unstable (magic numbers + resonances)

**Critical test**: A-matched comparison (KS test within A-bins).

---

### Experiment 3 (Decay Mode Prediction)

**Original hypothesis**: ε improves decay mode classification.

**Diagnostic result**: Modest variation by mode (~1-2% ε range).

**Expectation**: Weak signal, but may still improve classification modestly.

**Critical test**: Logistic regression (decay_mode ~ ε + A + N/Z).

---

### Experiment 4 (Boundary Sensitivity)

**Original hypothesis**: Nuclides with intermediate ε (0.10-0.20) are sensitive to ionization.

**Diagnostic result**: 37% of all nuclides are in this band.

**Expectation**: Still viable, need to rank by Q-value and other sensitivity proxies.

**Critical test**: Identify top-50 candidates for experimental verification.

---

## Key Questions for Next Phase

### 1. Is the reversed pattern real or a confound?

**Test**: A-matched KS test (compare stable vs unstable within each A-bin).

**If real**: Reinterpret model as predicting **instability**, not stability.

**If confound**: Correct for A-distribution before concluding.

---

### 2. Does ε correlate with half-life?

**Test**: Plot log₁₀(t₁/₂) vs ε for unstable nuclides.

**Hypothesis 1**: Negative correlation (low ε → short t₁/₂, "reactive resonance").

**Hypothesis 2**: No correlation (ε unrelated to lifetime).

**Hypothesis 3**: U-shaped (both very low and very high ε have short t₁/₂).

---

### 3. Do magic numbers have high ε?

**Test**:
- Extract doubly-magic nuclides (Z, N both magic)
- Compare ε_magic vs ε_non-magic

**Hypothesis**: Magic numbers are anti-resonant (high ε).

---

### 4. Does training set matter?

**Test**: Refit families on:
- Unstable nuclides only
- Long-lived only (t₁/₂ > 1 day)
- Holdout-by-A (cross-validation)

**Check**: Does dc3 universality persist? Does ε pattern change?

---

## Recommendations

### Immediate Next Steps

1. **Run full Experiment 1** (existence clustering)
   - Generate null candidate universe
   - Compare ε_observed vs ε_null
   - **This is the primary falsifier**

2. **A-matched stability test** (Experiment 2 refinement)
   - Bin by A (e.g., A ∈ [10-20), [20-30), ...)
   - Compute KS test within each bin
   - Check if stable/unstable difference persists

3. **Half-life correlation plot**
   - log₁₀(t₁/₂) vs ε scatter plot
   - Compute Spearman correlation
   - Test for U-shaped or monotonic relationship

4. **Magic number analysis**
   - Flag nuclides with Z ∈ {2, 8, 20, 28, 50, 82} or N ∈ {2, 8, 20, 28, 50, 82, 126}
   - Compare ε_magic vs ε_non-magic distributions

---

### Hypothesis Revision

**Original claim** (EXPERIMENT_PLAN.md):
> "C2 (Stability selector): stable nuclides have lower dissonance ε than unstable nuclides."

**Revised hypotheses**:
- **H2a (Reversed selector)**: Stable nuclides have HIGHER ε than unstable
- **H2b (Resonant instability)**: Low ε predicts SHORT half-lives (reactive states)
- **H2c (Magic anti-resonance)**: Magic numbers cluster at HIGH ε (stable defects)

**Test priority**:
1. H2a: A-matched KS test
2. H2b: Half-life correlation
3. H2c: Magic number ε distribution

---

## Conclusions (Preliminary)

### What We've Learned

1. **dc3 universality holds** (1.38% relative std) ✓
   - This is a robust, non-trivial result

2. **Multi-family structure is real** ✓
   - ε_best << ε_individual shows families capture distinct modes

3. **Stable ≠ harmonic** (surprising!) ⚠
   - Stable nuclides have HIGHER ε
   - Most harmonic nuclides are ALL unstable
   - Magic numbers may be anti-resonant

4. **Decay mode signal is weak** (~1-2% ε variation)
   - May still predict modes with other features

---

### What We Don't Know Yet

1. **Is Exp 1 (existence clustering) valid?**
   - Need null candidate universe to test

2. **Is stable/unstable pattern a confound?**
   - Need A-matching to rule out A-dependence

3. **Does ε predict half-life?**
   - Need correlation analysis

4. **Do magic numbers have high ε?**
   - Need magic number flagging

---

### Path Forward

**If Experiment 1 passes** (observed have lower ε than null):
- Model has predictive merit for **existence**, even if not stability
- Revise interpretation: harmonics → instability, not stability
- Proceed with Exp 3 (decay mode) and Exp 4 (boundary sensitivity)

**If Experiment 1 fails** (observed ε ≈ null ε):
- Model is likely flexible fit without real structure
- dc3 universality may be artifact of parameterization
- Acknowledge failure and publish negative result

**Critical next milestone**: Implement `null_models.py` and run Exp 1.

---

**Status**: Preliminary diagnostics complete, awaiting controlled experiments.

**Next session**: Implement null models and Experiment 1.

---

**Last Updated**: 2026-01-02
**Author**: Claude (AI assistant)
**Reviewed by**: [Pending human review]
