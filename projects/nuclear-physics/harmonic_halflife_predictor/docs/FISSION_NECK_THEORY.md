# Engine B: The Geometric Origin of Spontaneous Fission

**Hypothesis**: Spontaneous Fission is the "Rayleigh-Plateau" instability of a vacuum soliton.

**Date:** 2026-01-03  
**Status:** Testing  
**Significance:** First geometric prediction of fission from surface tension

---

## The Physical Model

### 1. The "Peanut" Shape
At A > 161, we proved the soliton bifurcates (Two-Center model).
- Single sphere → Elongated ellipsoid → Two separated centers
- Deformation parameter β describes the elongation
- Neck connects the two lobes

### 2. Deformation Parameter (β)
We estimate this from the QFD Harmonic fit (c2/c1 ratio):
- Higher c2/c1 → More volume energy → More deformation
- Family B (surface-dominated): low β
- Family C (volume-dominated): high β

### 3. Elongation Factor (ζ)
The aspect ratio of the soliton is derived from β:

```
ζ = (1 + β) / (1 - β/2)
```

**Physical meaning:**
- Numerator: Length extension along symmetry axis
- Denominator: Radius contraction of the "neck"
- Volume conservation: R_long × R_neck² = constant

**Example:**
- β = 0 (sphere): ζ = 1.00 (no elongation)
- β = 0.3: ζ = 1.51 (moderately deformed)
- β = 0.6: ζ = 2.29 (highly deformed, thin neck)

---

## The Rayleigh-Plateau Instability

### Classic Fluid Dynamics

A liquid cylinder becomes unstable when:
```
L > π × D_neck
```

The surface tension tries to minimize surface area → breaks into spherical droplets.

**Famous example:** Water stream from faucet breaks into drops.

### Nuclear "Liquid Drop"

The nucleus behaves similarly:
- **Surface tension:** Strong force binding (∝ c1)
- **Volume pressure:** Fermi energy (∝ c2)
- **Instability:** When elongation > critical → neck snaps → fission

**Prediction:**
```
log(T_1/2^SF) ∝ -k × ζ

Where k is the "snap resistance" coefficient
```

When ζ exceeds critical elongation (ζ_crit ≈ 2.0-2.5):
- Neck becomes too thin
- Surface tension can't restore shape
- Fission half-life → near zero (immediate splitting)

---

## Comparison to Traditional Fission Theory

| Aspect | Traditional (Bohr-Wheeler) | Geometric Soliton |
|--------|---------------------------|-------------------|
| **Mechanism** | Quantum tunneling through barrier | Classical Rayleigh-Plateau instability |
| **Barrier** | Calculated from liquid drop + shell | Determined by surface tension (c1) |
| **Deformation** | Empirical β from collective model | Derived from c2/c1 ratio |
| **Prediction** | Barrier penetration integral | Elongation factor ζ |
| **Parameters** | ~10-15 (barrier height, inertia, etc.) | 3 (c1, c2, β) |

Both approaches may be equivalent - quantum tunneling is the microscopic picture, geometric instability is the macroscopic picture.

---

## Mathematical Formulation

### Elongation Factor

For prolate (football-shaped) nucleus:
```python
R_long = R0 × (1 + β)      # Long axis radius
R_neck = R0 × (1 - β/2)    # Neck radius (approximate)

ζ = R_long / R_neck = (1 + β) / (1 - β/2)
```

### Beta Estimation from Harmonic Model

```python
# From geometric quantization
c1_eff = c1_0 + N × dc1
c2_eff = c2_0 + N × dc2

# Deformation proxy (higher c2/c1 → more elongated)
β_est = k × (c2_eff / c1_eff)

Where k ≈ 0.5-1.0 (calibration factor)
```

### Fission Half-Life Prediction

Empirical relationship:
```
log10(T_1/2) = A - B × ζ + C × Z²/A

Where:
  A = baseline stability
  B = geometric snap coefficient
  C = Coulomb correction
```

**Hypothesis:** B should be universal (same for all actinides).

---

## Critical Elongation

### Prediction: ζ_crit ≈ 2.0-2.5

Based on Rayleigh-Plateau instability for cylinders:
```
Critical aspect ratio ≈ π ≈ 3.14
```

But nucleus is not a perfect cylinder - it's a "peanut" with two bulges.

**Modified criterion:**
```
ζ_crit = 2.0 - 2.5 (empirical, to be determined from data)
```

Nuclei with ζ > ζ_crit should fission **immediately** (t_1/2 < 1 μs).

---

## Experimental Signatures

### If the Model is Correct:

1. **Correlation:** log(T_1/2) vs ζ should be linear with negative slope

2. **Family dependence:**
   - Family B (low c2/c1): Less deformed → Longer SF half-lives
   - Family C (high c2/c1): More deformed → Shorter SF half-lives

3. **Sharp cutoff:** No nuclei should exist with ζ > 2.5 (impossible geometry)

4. **Magic numbers resist:** Shell closures should reduce effective β
   - Example: Pb-208 (doubly magic) resists deformation even if ζ_predicted is high

---

## Connection to Other Engines

### Unified Geometric Decay Framework

All three exotic decay modes are geometric instabilities:

| Engine | Mode | Geometric Cause | Critical Parameter |
|--------|------|-----------------|-------------------|
| **C (Three-Peanut)** | Cluster decay | N² conservation breaks | \|Δ(N²)\| > 3 |
| **A (Skin Failure)** | Neutron drip | Pressure > Tension | (c2/c1) × A^(1/3) > 1.7 |
| **B (Neck Snap)** | Fission | Elongation > Critical | ζ > 2.0-2.5 |

All three use the **same 18 parameters** (3 families × 6 coefficients).

---

## Testing Strategy

### Phase 1: Correlation Test
- Collect actinide data (Z ≥ 90)
- Calculate β_est from harmonic model
- Compute ζ = (1+β)/(1-β/2)
- Plot log(T_1/2) vs ζ
- Measure correlation coefficient r

**Success criterion:** r < -0.3 (moderate negative correlation)

### Phase 2: Critical Elongation
- Identify highest ζ_survivors (nuclei that don't fission immediately)
- Estimate ζ_crit from data
- Predict which superheavy elements should be fission-stable

### Phase 3: Predictive Validation
- Use model to predict SF half-lives for unmeasured isotopes
- Guide experimental searches for "islands of stability"
- Test on superheavy elements (Z > 110)

---

## Expected Results

### Strong Correlation Scenario (r < -0.5)

**Interpretation:** Fission is dominated by geometric instability
- Surface tension model is correct
- Can predict SF from pure geometry
- Liquid drop = soliton surface dynamics

### Weak Correlation Scenario (|r| < 0.3)

**Interpretation:** Shell effects dominate over geometry
- Magic numbers stabilize against elongation
- Need quantum corrections to classical picture
- Hybrid model required

### No Correlation Scenario (r ≈ 0)

**Interpretation:** Fission is pure quantum tunneling
- Geometry is not the primary driver
- Back to Bohr-Wheeler theory
- Harmonic model doesn't apply to fission

---

## Implications if Validated

### Scientific Impact

1. **Unified decay theory:** All decay modes from geometry
2. **Island of stability:** Predict which superheavy elements resist fission
3. **Nuclear reactors:** Geometric design principles for fuel stability
4. **Astrophysics:** Fission barrier in r-process nucleosynthesis

### Philosophical Impact

If fission is Rayleigh-Plateau instability:
- Strong force = surface tension of vacuum
- Nuclear "liquid drop" is literal, not metaphorical
- QFD soliton model captures classical dynamics
- Quantum mechanics emerges from geometric quantization

**This would be as revolutionary as proving the atom is a standing wave.**

---

## References

- Rayleigh-Plateau instability: Rayleigh, Proc. London Math. Soc. (1878)
- Nuclear fission theory: Bohr & Wheeler, Phys. Rev. (1939)
- Liquid drop model: Weizsäcker (1935)
- Geometric quantization: This work (harmonic_halflife_predictor)

---

**Next step:** Run `scripts/fission_neck_scan.py` to test correlation.

