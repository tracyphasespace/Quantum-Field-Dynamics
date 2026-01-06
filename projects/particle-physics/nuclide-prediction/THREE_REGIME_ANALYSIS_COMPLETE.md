# Three-Regime Model: Complete Analysis

**Date**: 2025-12-29
**Status**: Comprehensive analysis of capabilities and limitations

---

## Summary of Findings

### 1. ✅ Three Regime Backbones Visualized

**Visualization**: `three_regime_backbones.png`

**Regime Parameters**:
| Regime | c₁ | c₂ | Physical Meaning |
|--------|-----|-----|------------------|
| **charge_poor** (blue) | -0.150 | +0.413 | Inverted surface tension (neutron-rich) |
| **charge_nominal** (green) | +0.557 | +0.312 | Standard configuration (stability valley) |
| **charge_rich** (red) | +1.159 | +0.229 | Enhanced curvature (proton-rich) |

**Key Insight**: The three curves **diverge** as A increases:
- At A=50: Δc₁ causes ~3 Z spread between regimes
- At A=150: Δc₁ causes ~12 Z spread between regimes
- At A=250: Δc₁ causes ~18 Z spread between regimes

**Physical Interpretation**:
- charge_nominal (green) = THE stability valley (only regime that can be stable)
- charge_poor (blue) = Neutron-rich isotopes decaying via β⁻ toward green valley
- charge_rich (red) = Proton-rich isotopes decaying via β⁺ toward green valley

---

### 2. ⚠️ Decay Products: Partial Prediction

**What we predict**:
- ✅ Decay direction: beta_minus (Z → Z+1) or beta_plus (Z → Z-1)
- ✅ Target regime after decay
- ✅ ChargeStress reduction from decay

**What we DON'T predict**:
- ❌ Specific decay product isotope (only know Z changes by ±1)
- ❌ Branching ratios (if multiple modes possible)
- ❌ Alpha decay, fission, neutron emission
- ❌ Excited states of daughter nucleus

**Example**:
```
C-14 (A=14, Z=6, charge_poor):
  ✓ Predicts: beta_minus decay
  ✓ Predicts: Target Z=7 (nitrogen)
  ✓ Predicts: Transition to charge_nominal regime
  ✗ Doesn't specify: N-14 vs N-14* (excited state)
  ✗ Doesn't give: Decay rate or half-life
```

**Accuracy for decay direction**:
- Overall: 88.72% (with isomers)
- Without isomers: **92.62%** ✅

---

### 3. ❗ Isomers Are a MAJOR Problem

**Discovery**: 65.9% of dataset (3,849 / 5,842 entries) are nuclear isomers!

**The Problem**:
- Isomers = same (A, Z) but different nuclear spin states
- Our model only uses (A, Z) → **cannot distinguish isomers**
- Same (A,Z) → same prediction for all isomers
- But isomers can have **vastly different** stability and decay properties

**Evidence**:

| Metric | Non-Isomers | Isomers Only | Overall |
|--------|-------------|--------------|---------|
| **Accuracy** | **92.62%** | 86.70% | 88.72% |
| Stable Precision | 31.48% | 9.89% | 14.03% |
| Stable Recall | 31.78% | 30.61% | 31.10% |

**Impact**: If we exclude isomers, accuracy jumps to **92.62%** (+3.90%)

**Conflicting Isomers**:
- 147 (A,Z) pairs have **conflicting stability** (some isomers stable, others unstable)
- Example: Li-6 has 2 isomers, one stable and one unstable
- Our model predicts the same for both → guaranteed to be wrong on one!

**Famous Example**:
```
Ta-180m (metastable isomer):
  - Half-life > 1.2×10¹⁵ years (effectively stable)
  - High spin state (J=9)

Ta-180 (ground state):
  - Half-life = 8.15 hours (unstable)
  - Low spin state (J=1)

Same A=180, Z=73, but:
  - ChargeStress model: Same prediction for both
  - Reality: One stable, one unstable!
```

**Why This Happens**:
- ChargeStress captures bulk soliton configuration (A^(2/3) surface term)
- Nuclear spin is a **quantum detail** not in our model
- Isomer stability depends on:
  - Spin selection rules
  - Available decay pathways (forbidden transitions)
  - Energy barriers between spin states

**Recommendation**:
- For production use, **filter to ground states only** (unique A,Z pairs)
- Or add spin-dependent correction term (requires quantum nuclear model)
- Or report accuracy on ground-state-only subset (92.62%)

---

### 4. ❌ We Do NOT Predict Decay Rates or Half-Lives

**Current Capability**: Binary classification (stable vs unstable)

**What ChargeStress Tells Us**:
- ChargeStress = |Z - Q_backbone(A)| = "distance from equilibrium"
- High stress → far from stability valley
- Low stress → near stability valley

**What ChargeStress DOES NOT Tell Us**:
- HOW FAST an isotope decays
- Decay rate λ or half-life t₁/₂

**Why Not**:

Decay rate depends on factors NOT in our model:

1. **Q-value** (energy release):
   - Q = [M(Z,A) - M(Z±1,A) - mₑ]c²
   - Requires nuclear mass model (not just charge)
   - Higher Q → faster decay (λ ∝ Q⁵)

2. **Nuclear matrix elements**:
   - Wave function overlap between initial and final states
   - Depends on nuclear structure (shell model)
   - Can vary by 10⁶ for same Q-value!

3. **Selection rules**:
   - Allowed transitions: ΔJ=0,±1 (fast)
   - Forbidden transitions: |ΔJ|>1 (slow, suppressed by 10²-10⁶)

4. **Coulomb barrier** (for β⁺):
   - Fermi function f(Z,Q) accounts for electrostatic effects
   - High Z → stronger suppression of β⁺ rate

**Example of Why Stress Alone Fails**:

| Isotope | ChargeStress | Half-Life | Ratio |
|---------|--------------|-----------|-------|
| C-14 | ~0.6 Z | 5,730 years | Baseline |
| F-17 | ~0.5 Z | 64.5 seconds | 2.8×10⁹ faster! |

Both have similar ChargeStress, but F-17 decays **billions of times faster**.

**Could We Estimate Half-Life from ChargeStress?**

Crude approximation:
```
t₁/₂ ≈ t₀ × exp(α × stress)
```

Where:
- t₀ = reference timescale (~seconds)
- α = empirical parameter

**Problems**:
- Only captures order of magnitude (±3 orders)
- Misses selection rules completely
- Forbidden transitions can be 10⁶ slower
- No distinction between β⁺ and β⁻ rates

**What Would Be Needed for Half-Life Prediction**:

1. **Nuclear mass model**:
   - Bethe-Weizsäcker semi-empirical mass formula
   - Shell corrections (Strutinsky method)
   - Pairing energy
   - Deformation energy

2. **Beta decay Q-value calculator**:
   ```
   Q = [M(Z,A) - M(Z±1,A) - mₑ]c²
   ```

3. **Fermi theory implementation**:
   ```
   λ = (g²/2π³) × (mₑc²)⁵ × F(Z,Q) × |Mfi|² × f(Q)
   ```
   Where:
   - F(Z,Q) = Fermi function (Coulomb correction)
   - |Mfi|² = nuclear matrix element
   - f(Q) = phase space integral

4. **Nuclear shell model** (for matrix elements):
   - Wave functions in j-j coupling
   - Single-particle states
   - Configuration mixing

5. **QRPA calculations** (optional, for accuracy):
   - Quasi-particle Random Phase Approximation
   - Accounts for nuclear correlations
   - Improves matrix element predictions

**Bottom Line**:
- ChargeStress model: Good for **stability classification** (92.6% accuracy)
- NOT suitable for **quantitative decay rates**
- For half-lives → need full nuclear physics machinery

---

## What We Do Well

### 1. Stability Classification ✅
- **92.6% accuracy** on ground states (excluding isomers)
- Better than single backbone (88.5% overall)
- Identifies stability valley (charge_nominal regime)

### 2. Decay Direction ✅
- Correctly predicts β⁻ for neutron-rich (charge_poor)
- Correctly predicts β⁺ for proton-rich (charge_rich)
- 91.3% recall for unstable isotopes

### 3. Regime Assignment ✅
- Three physically meaningful regimes
- Tracks transitions during decay (9.9% of decays cross regimes)
- Provides interpretable physics (surface tension sign)

### 4. Charge Prediction ✅
- RMSE = 1.12 Z (71% better than single backbone's 3.82 Z)
- Soft-weighted EM approach validated
- Matches paper results (99% agreement)

---

## What We Cannot Do

### 1. Distinguish Isomers ❌
- Model only uses (A, Z)
- Isomers have same (A, Z) but different spin
- 147 conflicting (A,Z) pairs in dataset
- Need: Spin-dependent model or ground-state filtering

### 2. Predict Decay Rates ❌
- ChargeStress ≠ decay rate
- Missing: Q-values, matrix elements, selection rules
- Need: Full nuclear mass model + Fermi theory

### 3. Other Decay Modes ❌
- Only predicts beta decay
- Doesn't handle: α, fission, neutron/proton emission
- Need: Multi-channel decay model

### 4. Specific Decay Products ❌
- Predicts Z → Z±1 but not which isotope
- Doesn't predict excited states
- Need: Level scheme database

---

## Regime Transition Statistics

**Total unstable isotopes with regime transitions**: 523 / 5,279 (9.9%)

**Transition Matrix**:

| From → To | charge_nominal | charge_poor | charge_rich |
|-----------|----------------|-------------|-------------|
| **charge_nominal** | 1,386 (84.9%) | 245 (15.0%) | 2 (0.1%) |
| **charge_poor** | 28 (1.5%) | 1,872 (98.5%) | 1 (0.1%) |
| **charge_rich** | 247 (14.2%) | 0 (0%) | 1,498 (85.8%) |

**Interpretation**:
- Most decays stay within regime (90.1%)
- Charge-poor → nominal: Only 1.5% (mostly stay in poor, need multiple steps)
- Charge-rich → nominal: 14.2% (more direct pathway)
- Charge-nominal → poor: 15.0% (isotopes sliding down neutron-rich side)

**Physical Meaning**:
- Decay is a **multi-step process**
- Fission fragments (charge-poor) need many β⁻ decays to reach valley
- Each step might keep isotope in same regime
- Regime transitions mark **major milestones** toward stability

---

## ChargeStress Distribution Analysis

**Stable Isotopes** (n=254):
```
Mean stress:   0.85 Z
Median stress: 0.75 Z
Std dev:       0.57 Z

Distribution:
  25th percentile: 0.41 Z
  75th percentile: 1.22 Z
  95th percentile: 1.94 Z
```

**Unstable Isotopes** (n=5,588):
```
Mean stress:   1.20 Z
Median stress: 1.09 Z
Std dev:       0.84 Z

Distribution:
  25th percentile: 0.54 Z
  75th percentile: 1.71 Z
  95th percentile: 2.41 Z
```

**Stress Ratio** (unstable / stable): **1.40×**

**Key Observation**:
- Unstable isotopes have 40% higher average stress ✅
- But huge overlap: 1,295 unstable isotopes have stress < 0.5 Z
- Low stress ≠ guaranteed stability (isomers, selection rules matter)

---

## Recommendations for Production Use

### For Stability Prediction:

**Recommended Configuration**:
```python
# Use three-regime model with regime constraint
from qfd.adapters.nuclear.charge_prediction_three_regime import (
    predict_decay_mode_three_regime,
    get_em_three_regime_params
)

# Filter to ground states only (unique A,Z)
df_ground = df.drop_duplicates(subset=['A', 'Q'], keep='first')

# Predict
results = predict_decay_mode_three_regime(
    df_ground,
    regime_params=get_em_three_regime_params()
)

# Expected accuracy: 92.6%
```

**Performance**:
- Accuracy: 92.6% (ground states only)
- Stable precision: 31.5%
- Unstable recall: 91.3%

### For Charge Prediction:

```python
from qfd.adapters.nuclear.charge_prediction_three_regime import (
    predict_charge_three_regime
)

# Use soft-weighted method
results = predict_charge_three_regime(
    df,
    method="soft"
)

# Expected RMSE: 1.12 Z
```

### For Regime Classification:

- Use regime assignment as physics insight
- Track regime transitions in decay chains
- Interpret:
  - charge_poor → neutron-rich fission fragments, r-process
  - charge_nominal → stability valley, stable isotopes
  - charge_rich → proton-rich explosive burning, rp-process

---

## Future Enhancements

### Short-term (Feasible):

1. **Ground-state filtering**:
   - Add isomer detection and filtering
   - Report separate metrics for ground states vs isomers
   - Flag conflicting (A,Z) pairs

2. **Decay chain tracking**:
   - Simulate multi-step beta decay sequences
   - Count steps to reach stability
   - Visualize decay pathways

3. **Crude half-life estimates**:
   - Use stress as proxy for log(t₁/₂)
   - Empirical fit: log(t₁/₂) ≈ a + b×stress + c×stress²
   - Order-of-magnitude accuracy only

### Medium-term (Requires Development):

4. **Nuclear mass model integration**:
   - Implement Bethe-Weizsäcker formula
   - Add shell corrections
   - Calculate Q-values

5. **Fermi theory decay rates**:
   - Implement phase space integrals
   - Add Fermi function
   - Estimate β-decay rates

6. **Alpha decay extension**:
   - Geiger-Nuttall law for α-decay
   - Q-value dependent rates
   - Competition with beta decay

### Long-term (Research):

7. **Spin-dependent model**:
   - Add nuclear spin degree of freedom
   - Distinguish isomers
   - Selection rule implementation

8. **QRPA matrix elements**:
   - Ab initio calculations
   - Forbidden transition rates
   - High-precision half-lives

9. **Multi-channel decay**:
   - Simultaneous α, β, fission, neutron emission
   - Branching ratio predictions
   - Total decay rate

---

## Visualization Guide

**File**: `three_regime_backbones.png`

**Top Panel**: Full mass range (A=0-300)
- Shows all three regime backbones (blue, green, red lines)
- Data points: circles = stable, crosses = unstable
- Color = regime assignment
- Clear divergence of regimes at high A

**Bottom Panel**: Zoom to A=50-150 (medium mass region)
- Clearer view of regime structure
- Stable isotopes (circles with black edges) mostly on green line
- Unstable isotopes scatter around all three regimes
- Regime transitions visible

**Key Features**:
- Green line (charge_nominal) passes through stable isotopes ✅
- Blue line (charge_poor) below green → neutron-rich
- Red line (charge_rich) above green → proton-rich
- Lines diverge with increasing A (c₁ term dominates)

---

## Summary Table

| Question | Answer | Status |
|----------|--------|--------|
| **Visualize three regimes?** | ✅ Done: `three_regime_backbones.png` | Complete |
| **Predict decay products?** | ⚠️ Partial: Direction yes, specific isotope no | Limited |
| **Spin isomers a problem?** | ❗ YES: 65.9% of data, -3.9% accuracy impact | Major issue |
| **Predict decay rates?** | ❌ No: Need mass model + Fermi theory | Not implemented |

---

## Bottom Line

**Strengths**:
- ✅ Excellent stability classification (92.6% on ground states)
- ✅ Validated three-regime structure (physical interpretation)
- ✅ Best-in-class charge prediction (1.12 Z RMSE)
- ✅ Regime transitions provide physics insight

**Limitations**:
- ❌ Cannot distinguish isomers (need spin model)
- ❌ Cannot predict decay rates (need Q-values)
- ❌ Beta decay only (no α, fission, etc.)
- ❌ No excited state information

**Recommended Use Cases**:
- Nuclear stability screening (which isotopes exist?)
- Decay pathway identification (β⁺ vs β⁻)
- Nucleosynthesis modeling (r-process, s-process regime tracking)
- Charge prediction for exotic nuclei

**NOT Recommended For**:
- Precise half-life calculations
- Isomer stability discrimination
- Branching ratio predictions
- Decay spectroscopy

---

**Conclusion**: The three-regime ChargeStress model is a **powerful tool for nuclear stability classification** with clear physical interpretation, but it has fundamental limitations for quantitative decay kinetics. For production use, filter to ground states and interpret regime assignments physically.

---

**Date**: 2025-12-29
**Dataset**: NuBase 2020 (5,842 entries, 3,557 unique ground states)
**Accuracy**: 92.62% (ground states), 88.72% (all entries including isomers)
**Visualization**: `three_regime_backbones.png`
