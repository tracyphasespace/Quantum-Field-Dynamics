# Key Results Quick Reference Card

**Date**: 2026-01-02
**Status**: Tacoma Narrows analysis complete, Exp 1 permutation test running

---

## CRITICAL NUMBERS

### Experiment 1: Existence Clustering (FAILED)
- Harmonic AUC: **0.48** (anti-predictive!)
- Smooth baseline AUC: **0.98** (valley is king)
- Observed ε - Null ε: **+0.0087** (wrong direction)
- **Verdict**: Model does NOT predict existence

### Tacoma Narrows Test (PARTIAL SUCCESS)

**Overall** (all unstable, A=1-294):
- r = **+0.042**, p = 0.018 (weak, marginal)

**Valid Regime** (A ≤ 161):
- r = **+0.131**, p < 0.001 ✓ (moderate, significant)
- Variance explained: **~1.7%**

**Invalid Regime** (A > 161):
- r = **+0.047**, p > 0.001 ✗ (no correlation)

### Stable vs Unstable
- Stable ε: **0.1457**
- Unstable ε: **0.1324**
- Difference: **+0.0133**, p = 0.026 ✓

### Mass Breakpoint
- **Exact transition: A = 161**
- Last significant: A ≤ 161 (r = +0.076, p = 9.5×10⁻⁴)
- First marginal: A ≤ 163 (r = +0.069, p = 2.4×10⁻³)

---

## WHAT WORKS

✓ **Stable vs Unstable**: Stable have higher ε (Tacoma Narrows validated)
✓ **Light/Medium Nuclides** (A ≤ 161): r = 0.13, p < 0.001
✓ **Family C**: r = +0.117, p = 2×10⁻⁴
✓ **Beta decay**: Weak positive correlations
✓ **Physical interpretation**: Shell model regime

---

## WHAT FAILS

✗ **Existence prediction**: AUC = 0.48 (worse than random)
✗ **Heavy nuclides** (A > 161): No correlation
✗ **Alpha decay**: r = 0.000, p = 0.99
✗ **Universal theory**: Regime-dependent only
✗ **Strong effect**: Only ~1.7% variance explained

---

## REGIME BOUNDARIES

**VALID** (A ≤ 161):
- Shell model physics
- Spherical/weakly deformed
- Harmonic approximation reasonable
- r = 0.13, p < 0.001

**INVALID** (A > 161):
- Collective motion dominant
- Permanently deformed
- Harmonic approximation fails
- r = 0.05, p > 0.001

**Transition**: Rare earth region (Dy/Ho/Er)

---

## HIERARCHY OF PREDICTORS

1. **Valley of stability** → AUC = 0.98 (existence)
2. **Q-values** → (decay mode selection)
3. **Harmonic ε** → r = 0.13 (weak decay rate modulation, A ≤ 161 only)

---

## PUBLICATION FRAMING

**HONEST** (defensible):
> "Regime-dependent weak correlation between harmonic dissonance and
> nuclear half-life in shell model regime (A ≤ 161, r = 0.13, p < 0.001).
> Effect vanishes beyond A = 161 (rare earth deformation onset). Valley
> of stability remains primary existence predictor (AUC = 0.98)."

**OVERSTATED** (not defensible):
> ✗ "Harmonic model predicts nuclear stability"
> ✗ "Universal parameter governs all nuclides"
> ✗ "Resonance theory of nuclear binding"

---

## NEXT STEPS

**Immediate**:
1. Wait for Exp 1 permutation test completion
2. Recompute all metrics on A ≤ 161 subset
3. Check if dc3 universality improves

**Short-term**:
4. Investigate Family C (why strongest correlation?)
5. Compare to nuclear deformation parameters (β₂)
6. Test magic number hypothesis on full dataset

**Publication**:
7. Frame as "regime-dependent valley curvature effects"
8. Emphasize physical transition at A = 161
9. Honest about weak effect (~1.7% variance)
10. Rigorous null model methodology

---

## FILES TO READ

**Theory**: TACOMA_NARROWS_INTERPRETATION.md
**Results**: TACOMA_NARROWS_RESULTS.md (16 KB)
**Breakpoint**: MASS_CUTOFF_ANALYSIS.md
**Session log**: SESSION_SUMMARY_2026-01-02_TACOMA.md

**Figures**: reports/tacoma_narrows/*.png

---

## ONE-LINE SUMMARY

**Harmonic model captures weak (~1.7% variance) valley curvature effects in shell model regime (A ≤ 161) but fails to predict existence (AUC = 0.48) and has no power in collective motion regime (A > 161).**

---

**Last updated**: 2026-01-02 20:25
