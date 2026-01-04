# Harmonic Mode Transitions vs Experimental Half-Lives

## Summary

**Date:** 2026-01-02

**Total isotopes analyzed:** 47

- Alpha decay: 24 isotopes
- Beta- decay: 15 isotopes
- Beta+ decay: 8 isotopes

## Key Findings

### 1. Selection Rule Validation

**Allowed transitions (|ΔN|≤1):** 36/47 (76.6%)

- Mean log(half-life): 8.36
- Mean Q-value: 3.26 MeV

**Forbidden transitions (|ΔN|>1):** 11/47 (23.4%)

- Mean log(half-life): 9.10
- Mean Q-value: 3.63 MeV

**Result:** Forbidden transitions are 5.5× slower

### 2. Decay Mode Predictions

**Beta- decay (prediction: ΔN < 0):**
- Correct: 15/15 (100.0%)

**Beta+ decay (prediction: ΔN > 0):**
- Correct: 8/8 (100.0%)

**Alpha decay:**
- Same mode (ΔN=0): 1/24 (4.2%)
- Mean ΔN: 1.21 ± 0.51

### 3. Regression Model (Alpha Decay)

**Baseline (Geiger-Nuttall):**
```
log(t) = -20.45 + 65.77/√Q
RMSE = 4.073
```

**With harmonic correction:**
```
log(t) = -24.14 + 67.05/√Q + 2.56*|ΔN|
RMSE = 3.869
Improvement: 5.0%
```

## Detailed Data

See `harmonic_halflife_results.csv` for complete dataset.

## Interpretation

The harmonic mode quantum number N acts as a **selection rule** for nuclear decay:

- **Allowed transitions** (|ΔN| ≤ 1): Fast, high probability
- **Forbidden transitions** (|ΔN| > 1): Slow, low probability

This is analogous to atomic spectroscopy where electric dipole transitions require Δl = ±1 (selection rule from angular momentum conservation).

In the nuclear case, the harmonic mode N represents the **resonance pattern** of the nucleon field. Large changes in N require significant rearrangement of the nuclear wave function, suppressing the transition rate.
