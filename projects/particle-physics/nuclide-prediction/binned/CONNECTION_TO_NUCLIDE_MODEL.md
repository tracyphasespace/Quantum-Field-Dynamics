# Connection: Charge Predictions ↔ Binding Energy Model

**Date**: 2025-12-29

---

## Two Complementary QFD Nuclear Projects

### Project 1: Charge Prediction (Our Work Today)
**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclide-prediction/`

**Question**: Given mass number A, what charge Z minimizes ChargeStress?

**Model**:
```
Q(A) = c₁·A^(2/3) + c₂·A  (or with c₀ offset)
```

**Parameters**:
- c₁ ≈ 0.5 (surface term)
- c₂ ≈ 0.3 (volume term)
- Per-track variations

**Performance**: RMSE = 1.46 Z (three-track model)

**Applications**:
- Nuclear stability valley
- Beta decay mode prediction
- Charge regime classification

### Project 2: Binding Energy Model (NuclideModel)
**Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/NuclideModel/`

**Question**: Given (A, Z), what is the binding energy E_bind?

**Model**:
```
E_bind = f(V2, V4, c_sym, ...)
```
where V2, V4 are QFD circulation terms

**Parameters** (Trial 32):
```json
{
  "c_v2_base": 2.201711,
  "c_v4_base": 5.282364,
  "c_sym": 25.0,
  "alpha_e_scale": 1.007419,
  "beta_e_scale": 0.504312,
  ...
}
```

**Performance**:
- Light nuclei (A < 60): < 1% error ✅
- Heavy nuclei (A > 120): ~8% underbinding ⚠️

**Applications**:
- Nuclear mass predictions
- Binding energy calculations
- Stability assessment

---

## How They Connect

### Conceptual Link

**Charge Model** (geometric):
- **WHERE** nuclei live in (A, Z) space
- Defines the stability valley
- ChargeStress = |Z - Q_backbone(A)|

**Binding Energy Model** (energetic):
- **HOW STABLE** each (A, Z) configuration is
- Actual binding energies
- Determines which nuclei exist

### Physical Interpretation

```
ChargeStress (our work) ≈ Geometric configuration cost
                          ↓
                   Manifests as
                          ↓
                  Energy penalty in binding
                          ↓
          Captured by c_sym term in NuclideModel
```

**Evidence**: NuclideModel's symmetry energy
```python
V_sym = c_sym × (N-Z)² / A^(1/3)
```

This is the ENERGETIC cost of deviating from optimal N/Z ratio!

Our charge model finds the optimal Z for given A (minimum ChargeStress).
Their model computes the energy penalty for ANY (A, Z).

### Mathematical Connection

**Our Q(A) formula**:
```
Q(A) = c₁·A^(2/3) + c₂·A
```

**Their binding energy** has similar terms:
- Surface energy: ∝ A^(2/3) (matches our c₁ term!)
- Volume energy: ∝ A (matches our c₂ term!)

The **SAME GEOMETRY** (soliton field structure) appears in both!

### Parameter Correspondence

| Charge Model | Energy Model | Physical Meaning |
|--------------|--------------|------------------|
| c₁ (surface) | Surface term in E_bind | Boundary curvature cost |
| c₂ (volume) | Volume term in E_bind | Bulk packing density |
| ChargeStress | c_sym × (N-Z)²/A^(1/3) | Charge asymmetry penalty |

---

## Regional Parameter Sets

### Charge Model Findings

**Three charge regimes**:
1. **Charge-Rich**: c₁ = +1.075, c₂ = +0.249
2. **Charge-Nominal**: c₁ = +0.521, c₂ = +0.319
3. **Charge-Poor**: c₁ = -0.147, c₂ = +0.411

### NuclideModel Findings

**Mass-dependent performance**:
1. **Light** (A < 60): < 1% error (parameters work great)
2. **Medium** (60 ≤ A < 120): 2-9% error (degrading)
3. **Heavy** (A ≥ 120): ~8% underbinding (systematic)

**Recommendation from FINDINGS.md**: Regional parameter sets

### Unified Hypothesis

**BOTH models need regional parameters!**

**Proposal**: The THREE CHARGE REGIMES correspond to THREE MASS REGIONS
- Charge-poor ≈ Light nuclei (high surface/volume)
- Charge-nominal ≈ Medium nuclei (balanced)
- Charge-rich ≈ Heavy nuclei? (or different mapping)

**Test**: Check if NuclideModel's regional parameters align with our charge tracks

---

## Cross-Validation Opportunity

### Test 1: Stability Predictions

**Charge Model** says: Z = Q(A) ± threshold → stable

**Energy Model** says: E_bind(A, Z) maximized → stable

**Hypothesis**: These should agree!

**Test**:
1. For each A, find Z_charge from our model
2. For that A, compute E_bind(A, Z) for all Z using NuclideModel
3. Check if max(E_bind) occurs at Z ≈ Z_charge

**Expected**: Strong correlation (both models capture same physics)

### Test 2: ChargeStress ↔ Symmetry Energy

**Our model**: ChargeStress = |Z - Q_backbone(A)|

**Their model**: E_sym = c_sym × (N-Z)²/A^(1/3) = c_sym × (A-2Z)²/A^(1/3)

**Transform**:
```
ChargeStress = |Z - Q(A)|
             = |Z - (c₁·A^(2/3) + c₂·A)|

For Z ≈ Q(A):
  N - Z ≈ A - 2Q(A)
  (N-Z)² ≈ (A - 2·(c₁·A^(2/3) + c₂·A))²
```

**Hypothesis**: ChargeStress should correlate with sqrt(E_sym)

**Test**: Plot ChargeStress vs E_sym for all stable isotopes

### Test 3: Regional Parameter Derivation

**Goal**: Derive charge model (c₁, c₂) from energy model (V2, V4)

**Approach**:
1. For each mass region, extract optimal (c_v2, c_v4) from NuclideModel
2. Hypothesis: c₁ = f(c_v2), c₂ = g(c_v4)
3. Fit functions f, g
4. Test if derived (c₁, c₂) match our empirical values

**Expected outcome**: Unified parameter set across both models

---

## Integration Pathway

### Stage 1: Empirical Correlation ✅ (Done)
- Charge model: 1.46 Z RMSE
- Energy model: < 1% for light nuclei
- Both models work independently

### Stage 2: Cross-Validation (Next)
1. **Run NuclideModel** on all 5,842 isotopes
2. **Compare stability predictions** with our charge model
3. **Correlate ChargeStress with E_sym**
4. Document agreement/disagreement

### Stage 3: Unified Parameterization
1. **Regional parameters**: Align charge tracks with mass regions
2. **Derive relationships**: c₁(c_v2), c₂(c_v4)
3. **Single parameter set**: Reduce total free parameters
4. Theoretical basis from V4 circulation

### Stage 4: Lean Formalization
1. **Formalize both models** in Lean
2. **Prove relationships** between ChargeStress and E_sym
3. **Unified constraints**: Both models satisfy same bounds
4. Cross-realm theorem linking charge and energy

---

## Key Insights from FINDINGS.md

### 1. Heavy Isotope Underbinding

**Their finding**: All A > 120 underbound by 7-9%

**Implication for us**:
- Heavy nuclei need MORE binding
- Our charge-rich track (high A?) may need adjustment
- Regional parameters confirmed necessary

### 2. c_v2_mass ≈ 0

**Their finding**: No exponential mass compounding needed

**Implication for us**:
- Our c₁, c₂ can be constants (not A-dependent functions)
- But regional piecewise constants may work
- Aligns with our three-track approach

### 3. Charge Asymmetry Energy

**Their c_sym = 25.0 MeV** successfully captures (N-Z)² effects

**Connection to our ChargeStress**:
```
ChargeStress ∝ |Z - Q(A)|
             ∝ |deviation from optimal Z|
             ∝ sqrt((N-Z)² term in E_sym)
```

**Test**: Is c_sym ≈ c₁² or c₂²? Dimensional analysis needed.

### 4. Virial Convergence

**Their finding**: Virial convergence independent of mass

**Implication**:
- Field solver quality not the issue for heavy nuclei
- Missing physics (surface, pairing), not numerical problems
- Same may apply to our charge model

---

## Unified QFD Nuclear Framework

### Complete Picture

```
           QFD Soliton Field
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
   Charge Model       Energy Model
   (Geometric)        (Energetic)
        ↓                   ↓
   Q(A) = c₁·A^(2/3)   E_bind(A,Z,V2,V4)
         + c₂·A              ↓
        ↓              Three mass regions
   Three charge       (Light/Medium/Heavy)
   regimes                  ↓
   (Rich/Nominal/Poor) Performance varies
        ↓                   ↓
   RMSE = 1.46 Z     Light: <1% error
                     Heavy: ~8% error
        ↓                   ↓
        └─────────┬─────────┘
                  ↓
         Cross-Validation
         Unified Parameters
         Lean Formalization
```

### Unified Research Goals

1. **Understand charge-energy connection**
   - ChargeStress → symmetry energy
   - Prove theoretical relationship

2. **Regional parameterization**
   - Both models need it
   - Can they share parameters?

3. **Heavy isotope physics**
   - Both struggle with A > 120
   - Missing surface? Pairing? Deformation?

4. **Lean integration**
   - Formalize both models
   - Prove equivalences
   - Unified theorem base

---

## Action Items

### Immediate
1. ✅ Document connection (this file)
2. Read NuclideModel source code
3. Extract V2, V4 parameters from Trial 32
4. Compare with our c₁, c₂

### Short-term
1. Run NuclideModel on all 5,842 isotopes
2. Correlate ChargeStress with E_sym
3. Test stability prediction agreement
4. Plot regional correspondence

### Long-term
1. Derive c₁(c_v2), c₂(c_v4) relationships
2. Unified regional parameters
3. Cross-formalize in Lean
4. Single parameter set for both models

---

## Summary

**Two projects, one physics**:
- **Charge model**: WHERE nuclei live (geometry)
- **Energy model**: HOW STABLE they are (energetics)

**Same underlying structure**:
- Both use A^(2/3) (surface) and A (volume) terms
- Both need regional parameters
- Both struggle with heavy isotopes

**Opportunity**:
- Cross-validate predictions
- Unify parameter sets
- Reduce total free parameters
- Strengthen theoretical foundation

**Next step**: Run NuclideModel's Trial 32 on full dataset and compare with our charge predictions

---

**Status**: Connection documented, integration pathway defined
**Both models**: Production-ready independently
**Integration**: Potential for significant parameter reduction
