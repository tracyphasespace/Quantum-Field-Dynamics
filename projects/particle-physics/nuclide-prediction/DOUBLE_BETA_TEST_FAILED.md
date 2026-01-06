# Double-Beta Decay Prediction: Test Failed

**Date**: 2025-12-30
**Prediction**: Magic bulk masses (A=124, 130, 136, 138) should show enhanced 2νββ decay rates
**Result**: **PREDICTION FAILED** - Magic isotopes show suppression, not enhancement

---

## The Prediction

From universal pairing structure analysis, we predicted:

> "Double-beta decay (ΔZ=2) should be enhanced at magic bulk masses because direct paired topology transition bypasses the energetically forbidden unpaired intermediate state."

**Mechanism proposed**:
- Sequential β⁻β⁻: (paired) → (unpaired, Q<0) → (paired) [FORBIDDEN]
- Direct 2νββ: (paired) → (paired) preserving bivector closure [ALLOWED with enhancement]

**Expected**: Magic bulk masses A=124, 130, 136, 138 show 2-5x faster 2νββ rates.

---

## Test Results

### Available 2νββ Data

| Isotope | Z | A | Magic? | Q (keV) | t½ (years) | Quality |
|---------|---|---|--------|---------|------------|---------|
| Ca-48 | 20 | 48 | | 4272 | 4.4×10¹⁹ | good |
| Ge-76 | 32 | 76 | | 2039 | 1.8×10²¹ | excellent |
| Se-82 | 34 | 82 | | 2995 | 9.2×10¹⁹ | good |
| Zr-96 | 40 | 96 | | 3350 | 2.3×10¹⁹ | good |
| Mo-100 | 42 | 100 | | 3034 | 7.1×10¹⁸ | excellent |
| Cd-116 | 48 | 116 | | 2814 | 2.8×10¹⁹ | good |
| Te-128 | 52 | 128 | | 867 | 2.5×10²⁴ | good |
| **Te-130** | **52** | **130** | **✓** | **2527** | **7.0×10²⁰** | **excellent** |
| **Xe-136** | **54** | **136** | **✓** | **2458** | **2.2×10²¹** | **excellent** |
| Nd-150 | 60 | 150 | | 3367 | 7.0×10¹⁸ | good |

Only **2 magic isotopes** available for testing: Te-130 (A=130) and Xe-136 (A=136).

### Q-Value Corrected Analysis

Double-beta decay rates scale approximately as t½ ~ Q⁻¹¹ (phase space factor). Must correct for Q-value differences before comparing.

**Normalization**: Scale all half-lives to equivalent Q=3000 keV using Mo-100 as reference.

**Results**:
```
Magic isotopes (n=2):
  Te-130: Normalized t½ = 1.06×10²⁰ years
  Xe-136: Normalized t½ = 2.46×10²⁰ years
  Mean: 1.76×10²⁰ years

Non-magic isotopes (n=8):
  Mean normalized t½: 2.99×10²⁰ years
  Median normalized t½: 2.56×10¹⁹ years
```

### Residuals from Q-Value Prediction

Comparing measured vs expected (from Q⁻¹¹ scaling):

**Te-130 (A=130, MAGIC)**:
- Expected from Q-value: 1.06×10²⁰ years
- Measured: 7.0×10²⁰ years
- Residual: +0.82 log units
- **Result**: 6.6x SLOWER than Q-value predicts

**Xe-136 (A=136, MAGIC)**:
- Expected from Q-value: 2.46×10²⁰ years
- Measured: 2.2×10²¹ years
- Residual: +0.95 log units
- **Result**: 8.9x SLOWER than Q-value predicts

**Fast decayers (non-magic) for comparison**:
- Mo-100 (A=100): residual = 0.00 (reference)
- Nd-150 (A=150): residual = +0.49 log units

---

## Verdict

**✗ PREDICTION FAILED**

Both magic bulk masses show:
1. **Suppressed rates** (slower than Q-value predicts)
2. **No enhancement** relative to non-magic isotopes
3. **Opposite behavior** from prediction (slower, not faster)

The magic bulk masses are actually **6-9x slower** than expected from Q-value alone, while some non-magic isotopes (Mo-100, Nd-150) match Q-value predictions closely.

---

## Why Did It Fail?

### Consistency with Previous Findings

This failure is **consistent** with the thermodynamics/kinetics separation discovered earlier:

**What We Found Before** (Session Dec 30):
- ChargeStress (curve position) predicts decay **thermodynamics** (WHAT decays, WHICH direction)
- ChargeStress does **NOT** predict decay **kinetics** (HOW FAST)
- Correlation: r = 0.12 (essentially zero)

**Examples**:
- Co-60: Low ChargeStress (0.26) → Fast decay (5.3 years)
- Rb-87: High ChargeStress (1.08) → Slow decay (49 billion years)
- U-238: Very high ChargeStress (3.65) → Very slow decay (4.5 billion years)

**Interpretation**: Internal soliton structure (topology, vortex winding) determines activation barriers, not geometric stress alone.

### Application to Double-Beta Decay

**Pairing structure predicts**:
- ✓ Which decay modes are **allowed** (thermodynamics)
- ✓ Beta direction (above/below valley)
- ✓ ΔZ = 2 transitions preserve paired topology

**Pairing structure does NOT predict**:
- ✗ Decay **rates** (kinetics)
- ✗ Half-lives
- ✗ Matrix element magnitudes

**The Error**: We assumed that topological favorability (paired → paired) would translate to **faster rates**. But just like ChargeStress doesn't predict half-life, pairing geometry doesn't predict matrix elements.

### What Controls 2νββ Rates?

From nuclear physics:
```
t½(2νββ) = [G(Q,Z)]⁻¹ × |M_GT - M_F|⁻²
```

Where:
- G(Q,Z): Phase space factor (Q-value dependence)
- M_GT, M_F: Gamow-Teller and Fermi matrix elements

**Matrix elements** depend on:
1. Nuclear wavefunctions (shell model states)
2. Overlap integrals between initial/final states
3. Nucleon-nucleon correlations
4. Spin-isospin structure

**Magic bulk mass** (A=130, 136) affects:
- Soliton packing geometry (number of stable charge states)
- Pairing topology (paired bivector structure)

**Does NOT necessarily affect**:
- Matrix element magnitudes
- Wavefunction overlaps
- Transition probabilities

---

## Implications for QFD Theory

### What This Teaches Us

1. **Pairing is real** (validated by ΔZ=2 spacing universality, 79%)
2. **Pairing affects thermodynamics** (which states are stable, ✓ confirmed)
3. **Pairing does NOT affect kinetics** (decay rates, ✗ double-beta failed)

This is **fundamental separation** in QFD:
```
Geometry (ChargeStress, pairing) → Thermodynamics
Internal structure (vortices, topology) → Kinetics
```

Both are needed for complete description.

### What We Still Don't Have

**Missing from current QFD formalism**:
- Soliton internal structure solutions (vortex configurations)
- Topology change barriers (activation energies)
- Matrix elements from bivector dynamics
- Connection between geometry and transition amplitudes

**To predict double-beta rates**, we would need:
- Full QFD field equation solutions for nuclear solitons
- Calculate overlap between initial/final bivector configurations
- Derive Gamow-Teller operator equivalent in QFD
- Compute transition amplitude from first principles

**We are not there yet**.

---

## Revised Understanding

### What Magic Bulk Masses Mean

**Magic bulk masses (A=124, 130, 136, 138)** are:
- Geometric resonances in soliton packing
- Bulk sizes that accommodate **multiple stable charge states**
- Evidence for ΔA ≈ 6 periodicity in packing structure

**They do NOT mean**:
- Enhanced decay rates
- Special stability against specific processes
- Lower activation barriers

**Analogy**: Magic bulk masses are like resonant cavity sizes that support multiple electromagnetic modes. The cavity size determines **which modes exist**, not **how strongly they couple** to external fields.

### What Pairing Means

**Bivector charge pairing (ΔZ=2 dominance)**:
- Geometric preference for paired bivector structures
- Affects which soliton configurations are stable
- Universal across all mass regions (79% of multi-stable cases)

**Pairing predicts**:
- ✓ Even-even isotopes more stable than odd-odd (confirmed empirically)
- ✓ ΔZ=2 spacing in multi-stable bulk masses (confirmed, 79%)
- ✓ Even bulk mass preference (confirmed, 18:1)

**Pairing does NOT predict**:
- ✗ Decay rates or half-lives
- ✗ Matrix element enhancements
- ✗ Double-beta decay enhancement (tested, failed)

---

## What Should We Predict Instead?

### Falsifiable QFD Predictions (Kinetics-Free)

Since QFD geometry predicts **thermodynamics** but not **kinetics**, we should focus on:

**1. Stability Predictions** (no rate needed):
- Which isotopes are stable vs unstable
- Beta decay direction (β⁺ vs β⁻)
- Fission vs beta competition zones
- **Already validated: 95.4% beta direction accuracy**

**2. Magic Bulk Mass Existence**:
- Predict A=124, 130, 136, 138 support multiple stable Z
- **Confirmed in existing data**
- Extend to superheavy: Predict A=302 for E120

**3. Charge Density Scaling**:
- Predict Z/A ~ 0.557·A⁻¹/³ + 0.312
- **Validated: <3% error across Z=20-90**

**4. Even-Odd Effects** (no rates):
- Even-even more stable than odd-odd (thermodynamics)
- **Empirically confirmed**

**5. Pairing Structure**:
- ΔZ=2 spacing universal in multi-stable bulk masses
- **Validated: 79% across all regions**

### What We Should NOT Predict (Yet)

Until we solve QFD field equations for internal structure:
- ✗ Half-lives
- ✗ Decay rates
- ✗ Cross-sections
- ✗ Matrix elements
- ✗ Branching ratios

These require **kinetics**, which needs internal soliton structure beyond geometry.

---

## Lessons Learned

### Scientific Process Working Correctly

1. Made prediction from QFD theory (double-beta enhancement)
2. Tested against data (Te-130, Xe-136)
3. **Prediction failed** (6-9x opposite behavior)
4. Understood why (geometry ≠ kinetics)
5. **Revised theory** (pairing is thermodynamic only)

This is **good science**. Failed predictions teach us limits of current formalism.

### What We Learned

**QFD geometric framework is powerful for**:
- Stability landscapes (curves, valleys)
- Thermodynamic preferences (which states favored)
- Scaling laws (surface-bulk, Z/A vs A)
- Structural patterns (pairing, magic masses)

**QFD geometric framework is NOT sufficient for**:
- Decay rates (need internal structure)
- Transition amplitudes (need bivector dynamics)
- Quantum matrix elements (need field solutions)

**Next frontier**: Solve QFD field equations to get internal structure → kinetics.

---

## Updated Testable Predictions

### Remove from Prediction List

~~1. Double-beta decay enhancement at magic bulk masses~~ (FAILED)
~~2. Factor 2-5x faster 2νββ rates for A=124,130,136,138~~ (FAILED)

### Keep These Predictions

**Thermodynamic predictions** (kinetics-free):

1. **Superheavy island at A=302, Z=120**
   - Magic bulk mass + even Z + low ChargeStress
   - Testable when E120 synthesized

2. **ΔA ≈ 6 spacing continues** (or predictably shifts)
   - Search for magic masses in Z<20 (light)
   - Search for magic masses in Z>90 (heavy)

3. **Even-even stability preference**
   - Even-Z, even-N more stable than odd-odd
   - Already confirmed but quantifiable

4. **Charge density evolution**
   - Z/A = 0.557·A⁻¹/³ + 0.312 holds for ALL stable isotopes
   - Extend validation to Z=1-100

5. **Multi-stability pattern**
   - Even bulk masses support multiple stable Z
   - ΔZ=2 spacing in multi-stable cases
   - Already confirmed (79%) but can extend

### Novel Prediction (Kinetics-Free)

**Isobar stability pattern**:

For fixed bulk mass A (isobars), predict which Z values are stable based on:
- Distance to charge_nominal curve (ChargeStress)
- Pairing (even Z preferred)
- Magic bulk mass flexibility (A=124,130,136,138 allow wider Z range)

**Test**: For each A=100 to A=150:
- Measure how many stable isobars exist
- Check if A=124,130,136,138 have MORE stable isobars
- Predict which Z values stable for each A

This is **pure thermodynamics** (no rates) and distinguishes QFD from shell model (which predicts magic Z/N, not magic A).

---

## Summary

**Prediction tested**: Magic bulk masses show enhanced 2νββ rates
**Result**: **FAILED** (6-9x slower than expected)

**Why it failed**: Pairing affects thermodynamics (which states stable) but NOT kinetics (decay rates)

**Consistency**: Matches earlier finding that ChargeStress ≠ half-life (r=0.12)

**Lesson**: QFD geometric framework predicts stability landscapes (thermodynamics) but needs internal structure solutions for kinetics (rates, matrix elements)

**Revised predictions**: Focus on stability patterns (which isotopes exist, not how fast they decay)

**This is good science**: Failed prediction → better understanding of theory limits → refined predictions.

---

**Date**: 2025-12-30
**Status**: Prediction tested and failed; theory understanding refined
**Next**: Search literature for prior discussion of "magic mass numbers" A=124,130,136,138
