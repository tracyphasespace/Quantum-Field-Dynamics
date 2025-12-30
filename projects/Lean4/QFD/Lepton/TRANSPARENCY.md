# QFD Lepton Soliton Model - Transparency Summary

**Last Updated**: 2025-12-29

## Purpose

This document clarifies what is experimentally input, what is fitted, and what is derived in the QFD lepton formalization. It provides an honest assessment of the model's predictive power and remaining gaps.

---

## Inputs vs Outputs

| Quantity | Source | Status |
|----------|--------|--------|
| Fine structure constant α | Experimental (1/137.035999) | **Input** |
| Nuclear coefficients c₁, c₂ | Fitted to 5,842 nuclides (NuMass.csv) | **Input** (from separate fit) |
| Vacuum stiffness β = 3.058 | Derived in FineStructure.lean from (α, c₁, c₂) | **Derived** (depends on nuclear fit) |
| Gradient coupling ξ | Fitted to lepton mass spectrum (Stage 2 MCMC) | **Fitted** (≈1.0 after corrections) |
| Time coupling τ | Fitted to lepton mass spectrum (Stage 2 MCMC) | **Fitted** |
| Circulation α_circ | Calibrated from muon g-2 anomaly | **Calibrated** (goal: derive from geometry) |
| Compton radius R | R = ℏ/(mc) for each lepton | **Input** (from experimental mass) |

---

## Model Summary

**Physical Picture**: Hill spherical vortex with Compton-scale radius R = ℏ/(mc)

**Energy Functional**:
```
E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV
```

**Current Status**:
- Energy evaluated on Hill profile (no self-consistent solver yet)
- Parameters (β, ξ, τ) set as described above
- Radius R fixed by experimental mass for each lepton

---

## What is Calibrated vs Checked

### Spin (L ≈ ℏ/2)
- **Status**: Consistency check
- **Method**: Given (β, ξ, τ, R), model reproduces L ≈ ℏ/2 at U ≈ 0.876c
- **Assessment**: Demonstrates internal consistency, not an independent prediction

### Anomalous Magnetic Moment (g-2)
- **Electron**: V₄ = -ξ/β ≈ -0.327 matches Schwinger coefficient
- **Muon**: Requires tuning α_circ to match experimental anomaly
- **Status**: Consistency check, not parameter-free prediction
- **Reason**: ξ comes from fitting the same energy functional

### Tau Regime
- **Status**: Incomplete
- **Gap**: Requires higher-order terms (V₆) not yet formalized
- **Assessment**: Outside current model scope

---

## Strengths

1. **Cross-Sector Consistency**: Fine structure constant α and vacuum stiffness β link electromagnetic, nuclear, and lepton sectors through a common geometric framework

2. **Generation Structure**: Once parameters are fixed, the model reproduces spin and magnetic anomaly magnitudes across lepton generations

3. **Mathematical Rigor**: Degeneracy resolution and uniqueness theorems formalized in Lean 4 with complete proofs

---

## Current Limitations

1. **Parameter Fitting**: Three fitted parameters (β, ξ, τ) to match three lepton masses means no hold-out validation for mass sector

2. **Circulation Coupling**: α_circ still calibrated from muon data; needs independent derivation for true predictive power

3. **g-2 Claims**: Should be framed as consistency checks until ξ and α_circ have independent constraints

4. **No Self-Consistent Solution**: Energy evaluated on assumed Hill profile, not solved from field equations

---

## Recommended Next Steps

1. **Documentation Clarity**:
   - Distinguish input constants from fitted parameters in all documentation
   - Frame g-2 results as "consistent with experiment when ξ ≈ 1" rather than "predicts"
   - Remove claims of "no free parameters" where parameters are fitted

2. **Validation**:
   - Provide scripts/notebooks for Compton-scale correction showing ξ ≈ 1
   - Document Stage 2 MCMC procedure for reproducibility
   - Show correlation structure of fitted parameters

3. **Independent Constraints**:
   - Derive α_circ from geometric principles (if possible)
   - Find independent observable constraining ξ
   - With independent constraints, g-2 becomes true prediction

4. **Honest Assessment**:
   - Clearly state that current g-2 agreement uses calibrated α_circ
   - Acknowledge three parameters fitted to three masses
   - Frame as "promising geometric framework" rather than "complete theory"

---

## Formal Verification Status

The mathematical theorems in VortexStability.lean and AnomalousMoment.lean are **correctly proven**:
- Degeneracy resolution: Complete proof with 0 sorries
- Uniqueness of radius: Complete proof
- Spin quantization: Demonstrated for given parameters

**What the proofs show**:
- The two-parameter (β, ξ) model has unique solutions
- The model structure is internally consistent
- Mathematical relationships are rigorous

**What the proofs do NOT show**:
- That the chosen parameter values are fundamental
- That the model is the unique explanation for lepton properties
- That predictions are parameter-free

---

## Summary

The QFD lepton formalization demonstrates that a geometric vortex model with Hill profile can:
- Reproduce observed lepton masses (by parameter fitting)
- Match spin ℏ/2 (consistency check)
- Match g-2 anomalies (with calibrated α_circ)

The mathematical framework is rigorously proven. The physical interpretation requires:
- Independent derivation of fitted parameters for predictive claims
- Honest acknowledgment of what is input vs output
- Conservative language in all documentation

This is a promising geometric framework that merits further development, not a complete predictive theory.
