# QFD Harmonic Model - Validation Matrix

**Purpose**: Honest assessment for external replication
**Date**: 2026-01-09
**Status**: Mixed results - some validations pass, one fails

---

## Executive Summary

The harmonic model makes TWO types of predictions:
1. **Threshold predictions** (WHERE stability ends) → ✅ All pass
2. **Existence predictions** (WHICH nuclides exist) → ❌ Fails

This is not a failure of QFD - it's a scope limitation. The model predicts BOUNDARIES, not INTERIORS.

---

## Validation Results

### PASSING: Threshold Predictions (Engines A-D)

| Test | Prediction | Validation | Result |
|------|------------|------------|--------|
| **Engine A: Neutron Drip** | (c₂/c₁)·A^(1/3) > 1.701 | 20/20 highest-ratio nuclei at drip | ✅ **100%** |
| **Engine B: Fission** | Elongation ζ > 2.0 triggers fission | All 517 actinides have ζ < 1.4 | ✅ **Consistent** |
| **Engine C: Cluster Decay** | N²_parent ≈ N²_daughter + N²_cluster | 10/10 emit magic clusters (N=1,2) | ✅ **100%** |
| **Engine D: Proton Drip** | (c₂/c₁)·A^(1/3) > 0.539 | 103/107 preserve harmonic mode | ✅ **96.3%** |
| **Fission Asymmetry** | Integer partition of excited N_eff | 6/6 symmetry predictions correct | ✅ **100%** |

**Interpretation**: The geometric threshold model correctly predicts WHERE nuclear stability ends.

### FAILING: Existence Classification (exp1)

| Test | Prediction | Validation | Result |
|------|------------|------------|--------|
| **exp1_existence** | Harmonic ε predicts which (A,Z) exist | AUC = 0.481 (worse than random 0.50) | ❌ **FAIL** |

**Baseline comparison**:
- Harmonic model: AUC = 0.481 ❌
- Smooth valley: AUC = 0.976 ✅
- Random: AUC = 0.500

**Interpretation**: The harmonic model does NOT predict which specific nuclides exist. The simple valley of stability (Z ≈ 0.4A) does this much better.

---

## Why exp1 Fails (Honest Assessment)

### 1. Different Questions

**Threshold prediction** (Engines A-D):
> "At what point does a nucleus become unstable?"
> Answer: When geometric ratios exceed critical values.

**Existence prediction** (exp1):
> "Given (A,Z), does this specific nuclide exist?"
> Answer: Depends on valley of stability, not harmonic mode.

### 2. The Model's Actual Scope

The harmonic model describes:
- Family geometry (surface tension, volume pressure)
- Harmonic resonance modes (N = -3 to +10)
- Stability BOUNDARIES (drip lines, fission limits)

It does NOT describe:
- Why Z ≈ 0.4A is the stability valley
- Which specific (A,Z) combinations are bound
- Interior structure of the valley

### 3. GIGO Warning

The 18-parameter model was FIT to nuclear data:
- 3 families × 6 coefficients = 18 parameters
- Fitted to 3,558 nuclides
- Risk: Overfitting to training data

The threshold predictions (Engines A-D) are MORE robust because:
- They test EXTRAPOLATION (boundaries, not interiors)
- They use derived ratios (c₂/c₁), not raw fits
- Critical values (1.701, 0.539, 2.0) emerge from data

---

## What This Means for QFD

### Claims We CAN Make ✅

1. "The harmonic model correctly predicts nuclear stability BOUNDARIES"
2. "Drip lines emerge from geometric tension ratios"
3. "Fission asymmetry follows from integer partition constraints"
4. "Cluster decay conserves harmonic energy (Pythagorean)"

### Claims We CANNOT Make ❌

1. ~~"The harmonic model predicts which nuclides exist"~~
2. ~~"Harmonic ε is a universal stability predictor"~~
3. ~~"The 18 parameters explain all nuclear structure"~~

---

## Replication Instructions

### To Reproduce PASSING Results (Engines A-D)

```bash
cd projects/nuclear-physics/harmonic_halflife_predictor
python scripts/neutron_drip_scanner.py    # Engine A
python scripts/fission_neck_scan.py       # Engine B
python scripts/cluster_decay_scanner.py   # Engine C
python scripts/validate_proton_engine.py  # Engine D
```

### To Reproduce FAILING Results (exp1)

```bash
cd projects/particle-physics/LaGrangianSolitons
python src/experiments/exp1_existence.py \
  --candidates data/derived/candidates_by_A.parquet \
  --params reports/fits/family_params_stable.json \
  --out reports/exp1_replication
```

Expected output: `✗ EXPERIMENT 1 FAILS`

---

## Scientific Integrity Statement

We report ALL results, including failures. The exp1 failure is:
- Documented in this repository
- Reproducible by external scientists
- Explained with honest assessment

Hiding negative results would be scientific misconduct. We prefer honest limitations over false claims.

---

## Connection to Golden Loop

The Golden Loop β = 3.043 derives vacuum stiffness from α:
- e^β/β = (α⁻¹ × c₁)/π²
- c₂ = 1/β = 0.329 (bare volume coefficient)

This is INDEPENDENT of the harmonic model fitting.

The harmonic model's c₂/c₁ ratios (0.12-0.26) are EFFECTIVE values after:
- Shell corrections
- Family variations
- Coulomb screening

Both are consistent - bare vs dressed parameters.

---

## Recommendations for Reviewers

1. **Replicate Engines A-D first** - These are the strong results
2. **Replicate exp1** - Confirm it fails (it should)
3. **Check our interpretation** - Is the threshold/existence distinction valid?
4. **Test Golden Loop independently** - Does β = 3.043 predict fissility α⁻¹/β ≈ 45?

We welcome critical review. Science advances through honest disagreement.

---

**Document Version**: 1.0
**Author**: QFD Project
**License**: CC-BY-4.0
