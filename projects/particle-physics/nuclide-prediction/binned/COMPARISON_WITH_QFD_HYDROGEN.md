# Comparison: Our Three-Track Model vs qfd_hydrogen_project

**Date**: 2025-12-29
**Purpose**: Understand different 3-bin approaches and resolve c₁=0 issue

---

## qfd_hydrogen_project Implementation

**Location**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/workflows/3Bins_of_Nuclides.py`

### Methodology

**1. Backbone Fit**:
```python
# Fit to STABLE nuclides only
popt_backbone, _ = curve_fit(backbone, A[stable_mask], Q[stable_mask])
c1, c2 = popt_backbone
```
Result: Q_backbone = c₁·A^(2/3) + c₂·A fitted to 254 stable isotopes

**2. Classification**:
```python
deltaQ = Q - Q_pred_backbone

stable_bin = df[stable_mask]                # On backbone
over_bin   = df[(~stable_mask) & (deltaQ > 0)]  # Q > backbone
starve_bin = df[(~stable_mask) & (deltaQ < 0)]  # Q < backbone
```

Three bins:
- **Stable**: The 254 stable isotopes (on backbone by definition)
- **Overcharged**: Unstable with Q > backbone (too much charge)
- **Starved**: Unstable with Q < backbone (too little charge)

**3. Correction Model**:
```python
def model(AQ, k1, k2):
    A, Q = AQ
    neutron_excess = A - 2*Q
    return backbone(A, c1, c2) + k1*neutron_excess + k2*neutron_excess**2
```

**NOT three separate baselines!** One backbone + bin-specific corrections based on neutron excess.

### Key Differences from Our Approach

| Aspect | qfd_hydrogen | Our three-track |
|--------|--------------|-----------------|
| **Backbone source** | Stable only (254) | All isotopes (5,842) or Phase 1 reference |
| **Model type** | One backbone + corrections | Three separate baselines |
| **Correction** | k₁·(A-2Q) + k₂·(A-2Q)² | Separate (c₁, c₂) per track |
| **Parameters** | 2 (backbone) + 2×3 (corrections) = 8 | 2×3 (three baselines) = 6 |
| **Classification** | Sign of deviation | Magnitude threshold |
| **Physical basis** | Neutron excess | Charge regime |

### Why They Don't Have c₁=0 Issue

**Answer**: They don't fit separate c₁, c₂ for each bin! They use:
```
Q = single_backbone(A) + correction(neutron_excess)
```

So c₁, c₂ are the same for all bins, only k₁, k₂ differ.

---

## Our Three-Track Model

**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclide-prediction/binned/three_track_ccl.py`

### Methodology

**1. Reference Backbone**:
```python
# Use Phase 1 validated parameters
C1_REF = 0.496296
C2_REF = 0.323671
```

**2. Classification**:
```python
def classify_track(Z, A, threshold=2.5):
    Q_ref = backbone(A, C1_REF, C2_REF)
    deviation = Z - Q_ref

    if deviation > threshold: return 'charge_rich'
    elif deviation < -threshold: return 'charge_poor'
    else: return 'charge_nominal'
```

**3. Separate Regressions**:
```python
# Fit Q = c1_k·A^(2/3) + c2_k·A for each track k
for track in ['charge_rich', 'charge_nominal', 'charge_poor']:
    popt, _ = curve_fit(backbone, A_track, Q_track,
                        bounds=([0, 0], [2, 1]))  # ← ISSUE: c₁ ≥ 0 enforced
    c1_k, c2_k = popt
```

### The c₁=0 Problem

**Observation**: Charge-poor track has c₁ = 0.00000

**Hypothesis 1: Boundary Constraint**
```python
bounds=([0, 0], [2, 1])  # Forces c₁ ≥ 0
```

If the optimal c₁ for charge-poor is negative, `curve_fit` clamps it to 0.

**Test**: Remove bounds and check if c₁ goes negative.

**Hypothesis 2: Physical Reality**
Charge-poor nuclei genuinely have no surface term? This seems unlikely - all nuclei should have surface effects.

**Hypothesis 3: Model Mismatch**
The functional form Q = c₁·A^(2/3) + c₂·A may not be appropriate for charge-poor nuclei. They might need:
- Power law: Q = c·A^β where β ≠ 2/3
- Offset: Q = c₀ + c₁·A^(2/3) + c₂·A
- Higher order: Q = c₁·A^(2/3) + c₂·A + c₃·A^(4/3)

---

## Paper's Published Model

**Source**: `Three_Bins_Two_Parameters_for_Quantum_Fitting_of_Nuclei.md`

### Methodology (Inferred)

**Description**: "Gaussian Mixture of Regressions on the simple charge-mass baseline, Q(A)=c₁A²/³+c₂A"

**Method**: Expectation-Maximization (EM) algorithm

**Result**: Three components with separate (c₁, c₂, σ, π) each, achieving RMSE = 1.107 Z

### Alignment with Our Approach

The paper's approach is **closest to our three-track model**:
- Three separate baselines (not one backbone + corrections)
- Each component has own (c₁, c₂)
- Soft weighting via posterior probabilities

**Difference**:
- Paper uses unsupervised EM (data-driven clustering)
- We use physics-based classification (deviation from reference)

---

## Analysis: Why c₁=0 for Charge-Poor?

### Data Investigation Needed

Let me analyze the charge-poor data directly:

**Questions**:
1. What does A vs Q scatter look like for charge-poor track?
2. Does it visually follow A^(2/3) + A scaling?
3. What happens if we allow negative c₁?
4. Does pure linear (Q = c₂·A) fit better?

### Physical Interpretation Attempts

**Attempt 1: r-Process Explanation**
- Charge-poor = neutron-rich = r-process nucleosynthesis
- Rapid neutron capture prevents surface equilibration
- Only bulk volume term survives
- **Counter**: Still should have some surface curvature

**Attempt 2: Soliton Field Picture**
- In QFD, nuclei are charge density distributions
- Surface term (A^(2/3)) = boundary curvature
- Volume term (A) = bulk packing
- **Question**: Can soliton have zero surface curvature?

**Attempt 3: Classification Artifact**
- Maybe charge-poor classification captures nuclei far from equilibrium
- These nuclei are so unstable that normal scaling breaks down
- Need different functional form entirely

---

## Proposed Tests

### Test 1: Remove Bounds Constraint ⚠️ HIGH PRIORITY

**File**: `three_track_ccl.py` line ~151

**Current**:
```python
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF],
                       bounds=([0, 0], [2, 1]))
```

**Test A: No bounds**:
```python
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF])
```

**Test B: Allow negative c₁**:
```python
popt, pcov = curve_fit(backbone, A_track, Q_track,
                       p0=[C1_REF, C2_REF],
                       bounds=([-1, 0], [2, 1]))
```

**Expected**: c₁ for charge-poor may become negative

**Impact on RMSE**: Will it improve from 1.48 Z toward 1.11 Z target?

### Test 2: Alternative Functional Forms

**For charge-poor track only**, try:

**Model A: Pure power law**
```python
def power_law(A, c, beta):
    return c * A**beta

# Fit for charge-poor
popt, _ = curve_fit(power_law, A_poor, Q_poor, p0=[0.4, 1.0])
```

**Model B: With offset**
```python
def backbone_offset(A, c0, c1, c2):
    return c0 + c1 * A**(2/3) + c2 * A

# Fit for charge-poor
popt, _ = curve_fit(backbone_offset, A_poor, Q_poor, p0=[0, C1_REF, C2_REF])
```

**Model C: Neutron excess correction (like qfd_hydrogen)**
```python
def neutron_correction(AQ, k1, k2):
    A, Q = AQ
    neutron_excess = A - 2*Q
    return backbone(A, C1_REF, C2_REF) + k1*neutron_excess + k2*neutron_excess**2
```

### Test 3: Visualize Charge-Poor Data

**Goal**: Understand the A vs Q relationship visually

**Code**:
```python
import matplotlib.pyplot as plt

# Filter charge-poor nuclei
charge_poor = df[df['track'] == 'charge_poor']
A_poor = charge_poor['A'].values
Q_poor = charge_poor['Q'].values

# Plot raw data
plt.scatter(A_poor, Q_poor, s=5, alpha=0.6, label='Data')

# Overlay different models
A_range = np.linspace(A_poor.min(), A_poor.max(), 500)

# Current (c₁=0): Q = 0.385·A
plt.plot(A_range, 0.385 * A_range, 'r-', label='Linear (c₁=0)')

# If c₁ were negative: Q = -0.1·A^(2/3) + 0.5·A
plt.plot(A_range, -0.1 * A_range**(2/3) + 0.5 * A_range, 'b--', label='c₁<0 example')

# Pure A^(2/3): Q = 1.0·A^(2/3)
plt.plot(A_range, 1.0 * A_range**(2/3), 'g:', label='Surface only')

plt.xlabel('Mass A')
plt.ylabel('Charge Z')
plt.legend()
plt.title('Charge-Poor Track: Model Comparison')
plt.show()
```

---

## Recommendations

### Immediate Action (Priority 1)

1. **Test removing bounds** in `three_track_ccl.py`
   - Run with no constraints
   - Check if c₁ goes negative and RMSE improves
   - Document physical interpretation

2. **Visualize charge-poor data**
   - Create scatter plot of A vs Q for charge-poor
   - Overlay current fit (linear)
   - Overlay alternative models
   - Identify best visual fit

3. **Compare with paper**
   - Paper achieves 1.107 Z, we have 1.482 Z (gap = 0.375 Z)
   - Is c₁=0 the cause of this gap?
   - Or is it EM vs hard classification difference?

### Secondary (Priority 2)

4. **Try qfd_hydrogen approach**
   - Use single backbone (from stable only)
   - Add neutron excess corrections per bin
   - Compare RMSE with our three-baseline approach

5. **Hybrid model**
   - Charge-rich & charge-nominal: Use c₁·A^(2/3) + c₂·A
   - Charge-poor: Use alternative form (power law, offset, etc.)
   - Mixed functional forms may better capture physics

### Future Work (Priority 3)

6. **Cross-realm derivation**
   - Connect c₁, c₂ to V4, β, λ² from QFD theory
   - Test if charge-poor c₁=0 has theoretical basis
   - Reduce free parameters via fundamental constants

7. **Lean formalization**
   - Formalize three-track model in Lean
   - Prove constraints on each track separately
   - Handle c₁=0 case explicitly

---

## Summary

**Key Finding**: qfd_hydrogen_project uses a fundamentally different approach:
- One backbone (from stable) + bin-specific neutron excess corrections
- NOT three separate baselines like our model and the paper

**Our c₁=0 Issue**: Likely caused by bounds constraint in `curve_fit`
- Test by removing bounds
- May need negative c₁ or alternative functional form for charge-poor
- Could explain 0.375 Z gap to paper target (1.48 vs 1.11 Z)

**Next Step**: Run Test 1 (remove bounds) and see if RMSE improves

---

## Action Plan

```bash
# 1. Create test version without bounds
cd /home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclide-prediction/binned

# 2. Edit three_track_ccl.py:
#    Remove bounds=([0,0], [2,1]) from curve_fit (line ~151)

# 3. Re-run and compare:
python three_track_ccl.py > results_no_bounds.txt

# 4. Check if:
#    - c₁ for charge-poor becomes negative
#    - RMSE improves toward 1.11 Z target
#    - Physical interpretation makes sense
```

**Expected Outcome**: c₁ likely negative for charge-poor, RMSE may improve

**Risk**: Negative c₁ may lack physical interpretation in current theory
