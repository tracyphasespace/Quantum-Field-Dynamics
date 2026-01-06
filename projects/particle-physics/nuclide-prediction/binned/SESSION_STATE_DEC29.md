# Session State: Three-Component Nuclear Charge Model

**Date**: 2025-12-29
**Status**: 🟡 IN PROGRESS - c₁=0 issue needs investigation
**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclide-prediction/binned/`

---

## Current Results

### Three-Track Model Performance
- **Hard Assignment**: RMSE = 1.4824 Z, R² = 0.9969 ✅
- **Soft Weighting**: RMSE = 1.9053 Z, R² = 0.9949 ✅
- **Improvement**: 2.6× better than single baseline (3.83 → 1.48 Z)

### Component Parameters (threshold = 2.5)

| Track | c₁ | c₂ | Count | Formula |
|-------|-----|-----|-------|---------|
| Charge-Rich | 1.07474 | 0.24883 | 1,473 | Q = 1.075·A^(2/3) + 0.249·A |
| Charge-Nominal | 0.52055 | 0.31918 | 2,663 | Q = 0.521·A^(2/3) + 0.319·A |
| **Charge-Poor** | **0.00000** | **0.38450** | 1,706 | **Q = 0.385·A** |

---

## 🔴 OPEN QUESTION: Why is c₁ = 0 for Charge-Poor?

### Hypothesis 1: Constraint Boundary
**Code**: `three_track_ccl.py` line ~151
```python
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF],
                       bounds=([0, 0], [2, 1]))  # ← c₁ ≥ 0 enforced
```

**Issue**: If optimal c₁ for charge-poor is **negative**, `curve_fit` clamps it to 0 (lower bound).

**Test**: Remove bounds and see if c₁ goes negative:
```python
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF])
                       # No bounds - allow negative c₁
```

### Hypothesis 2: Physical Reality
**Interpretation**: Charge-poor nuclei genuinely have no surface term?

**Physics**:
- Surface term (A^(2/3)) represents boundary curvature effects
- Charge-poor = neutron excess = r-process nucleosynthesis
- Rapid neutron capture prevents surface equilibration
- Result: Only bulk volume term (A) survives

**Counter-evidence**: This seems unlikely - all nuclei should have surface tension.

### Hypothesis 3: Wrong Functional Form
**Alternative models** for charge-poor:
```
Model A: Q = c₀ + c₁·A^(2/3) + c₂·A     (add constant offset)
Model B: Q = c₁·A^β                      (power law, β ≠ 2/3)
Model C: Q = c₁·A + c₂·A^(4/3)          (volume + higher order)
```

**Test**: Fit charge-poor data to alternative forms and compare RMSE.

---

## Action Items

### 1. Remove Bounds Constraint ⚠️ HIGH PRIORITY
**File**: `three_track_ccl.py`
**Change**: Remove `bounds=([0,0], [2,1])` from `curve_fit`
**Expected**: c₁ for charge-poor may become negative
**Question**: What does negative c₁ mean physically?

### 2. Compare with qfd_hydrogen_project Implementation
**Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/workflows/`
**Goal**: Check if their 3-bin model has same issue
**Files to review**:
- Scripts with "3bin", "three_track", "mixture" in name
- Documentation explaining c₁=0 or negative c₁

### 3. Test Alternative Functional Forms
**For charge-poor track only**:
- Fit Q = c₀ + c₁·A^(2/3) + c₂·A
- Fit Q = c₁·A^β (optimize β)
- Compare RMSE with current Q = 0.385·A

### 4. Physical Interpretation Workshop
**Questions**:
- In QFD soliton picture, what does c₁ represent?
- Can surface term be negative or zero?
- Does r-process nucleosynthesis justify c₁→0?
- Connection to V4 circulation hypothesis?

---

## Files Created This Session

### Implementation
- `three_track_ccl.py` - Main implementation with threshold tuning
- `gaussian_mixture_ccl.py` - EM algorithm (needs fixing, RMSE=3.83 Z)

### Results
- `three_track_model.json` - Best parameters (threshold=2.5, c₁=0 for charge-poor)
- `global_model.json` - Gaussian mixture (suboptimal)

### Visualizations
- `three_track_analysis.png` - Three baselines + threshold performance
- `three_component_fit.png` - Gaussian mixture components
- `convergence.png` - EM convergence

### Documentation
- `README.md` - Quick start guide
- `THREE_TRACK_RESULTS.md` - Full analysis
- `MODEL_COMPARISON.md` - Single vs Three-Track vs Gaussian Mixture
- `SESSION_STATE_DEC29.md` - This file

---

## Cross-References

### Current Project
- **Single baseline validation**: `../validate_ccl_predictions.py`
- **Results**: `../NUBASE_VALIDATION_RESULTS.md` (RMSE=3.82 Z, 88.53% accuracy)
- **Paper**: `/mnt/c/Users/TracyMc/Downloads/Three_Bins_Two_Parameters_for_Quantum_Fitting_of_Nuclei.md`
- **Lean formalization**: `../projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean`

### Other Project (TO REVIEW)
- **Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/workflows/`
- **Purpose**: Compare their 3-bin implementation
- **Questions**: Do they have c₁=0? How do they handle it?

---

## Next Steps

1. **Immediate**: Review qfd_hydrogen_project workflows for 3-bin model comparison
2. **Test**: Remove bounds constraint and check if c₁ goes negative
3. **Analyze**: Plot charge-poor data separately to visualize A^(2/3) vs A contribution
4. **Document**: Explain c₁=0 physically or fix it algorithmically
5. **Compare**: Paper achieves 1.107 Z, we have 1.482 Z - is c₁=0 the gap?

---

## Performance Gap Analysis

| Model | RMSE | Gap to Paper |
|-------|------|--------------|
| Paper Target | 1.107 Z | Baseline |
| Our Three-Track | 1.482 Z | +0.375 Z (34% worse) |
| Single Baseline | 3.828 Z | +2.721 Z |

**Hypothesis**: If we fix the charge-poor c₁ issue, we might close the gap to paper's 1.107 Z.

**Test Plan**:
1. Remove bounds → check if c₁ < 0 improves RMSE
2. Try alternative functional forms → find best fit for charge-poor
3. Compare with qfd_hydrogen_project → see their solution
4. Re-tune threshold with corrected model → optimize globally

---

## Code Locations

### Key Function (needs investigation)
**File**: `three_track_ccl.py`
**Lines**: 145-165 (fit method)

```python
# CURRENT CODE (enforces c₁ ≥ 0):
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF],
                       bounds=([0, 0], [2, 1]))

# PROPOSED TEST (allow negative c₁):
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF])
                       # OR: bounds=([-1, 0], [2, 1])
```

---

## Questions for User

1. **Physics**: In QFD soliton theory, can the surface term coefficient be zero or negative?
2. **Implementation**: Should we enforce c₁ ≥ 0, or allow negative values?
3. **Comparison**: What does the qfd_hydrogen_project 3-bin model show for charge-poor c₁?
4. **Target**: Is 1.482 Z acceptable, or must we reach 1.107 Z to match paper?

---

## Resumption Checklist

When returning to this work:

- [ ] Review this document (SESSION_STATE_DEC29.md)
- [ ] Check qfd_hydrogen_project workflows (next action)
- [ ] Understand c₁=0 issue (is it constraint artifact or physics?)
- [ ] Test removing bounds constraint
- [ ] Compare alternative functional forms for charge-poor
- [ ] Document final resolution
- [ ] Update Lean formalization if needed

---

**Status**: 🟡 PAUSED - Awaiting investigation of c₁=0 issue
**Next Action**: Review `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/workflows/`
**Goal**: Understand if c₁=0 is expected or needs fixing
