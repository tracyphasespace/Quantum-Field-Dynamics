# Testing V₄_nuc = β Hypothesis

**Date**: 2025-12-30
**Goal**: Test if quartic soliton stiffness V₄_nuc equals vacuum bulk modulus β
**Status**: THEORETICAL INVESTIGATION

---

## Physical Context

### The Soliton Energy Functional

In QFD, nucleons are solitons with energy:
```
E[ρ] = ∫ (-μ²ρ + λρ² + κρ³ + V₄_nuc·ρ⁴) dV
```

**Terms**:
- **μ²ρ**: Quadratic (attractive)
- **λρ²**: Harmonic (restoring)
- **κρ³**: Cubic (asymmetry)
- **V₄_nuc·ρ⁴**: **Quartic (prevents over-compression)** ← TARGET

### The Vacuum Bulk Modulus

**β = 3.058231** (from Golden Loop)

**Physical meaning**: Resistance to vacuum compression
- Larger β → stiffer vacuum → harder to compress
- This SHOULD directly set the quartic coefficient!

---

## The Hypothesis

**V₄_nuc = β** (or simple multiple)

**Physical reasoning**:
1. β governs vacuum resistance to compression
2. Quartic term prevents soliton over-compression
3. Same physics → same parameter!

**Alternative forms**:
- V₄_nuc = β (direct)
- V₄_nuc = 4πβ (geometric factor)
- V₄_nuc = (1/4π)β (inverse geometric)
- V₄_nuc = β² (squared)

---

## What We Know

### From StabilityCriterion.lean

**Potential**:
```lean
def V (mu lam kappa beta : ℝ) (x : ℝ) : ℝ :=
  -mu^2 * x + lam * x^2 + kappa * x^3 + beta * x^4
```

**Stability requirement**: β > 0 (proved in Lean)

**Bounds**: No empirical value given (generic parameter)

### From Schema

**NuclearParams includes**:
- V4: Energy (well depth) ✅ ALREADY DERIVED = λ/(2β²)
- No separate "V4_nuc" or "beta_quartic" parameter

**Interpretation**: V₄_nuc might not be a free parameter at all!

---

## Dimensional Analysis

### Natural Units (ℏ = c = 1)

**Energy functional per volume**:
```
[E/V] = [energy density] = [mass]⁴
```

**Term analysis**:
```
μ²ρ:      [mass]² × [mass] = [mass]³ ✗ (needs dimension adjustment)
λρ²:      [mass] × [mass]² = [mass]³ ✗
V₄_nuc·ρ⁴: [?] × [mass]⁴ = [mass]³ ✗
```

**Wait, dimensions don't match!**

**Corrected**: If ρ has dimensions [density] = [mass]/[volume]:
```
In 3D: [ρ] = [mass]/[length]³ = [mass]⁴ (natural units)
```

**Then**:
```
V₄_nuc·ρ⁴: [?] × [mass]¹⁶ = [mass]⁴ (energy density)
→ [V₄_nuc] = [mass]⁻¹²
```

**But β is dimensionless!** ❌

### Dimensionless Formulation

**If ρ is dimensionless** (scaled density):
```
ρ = ρ_physical / ρ_scale
```

**Then**:
```
V₄_nuc·ρ⁴: dimensionless × dimensionless = dimensionless ✓
```

**In this case**: V₄_nuc = β makes sense! ✓

---

## Phenomenological Constraints

### From Nuclear Density

**Nuclear saturation density**: ρ₀ ≈ 0.16 fm⁻³

**Energy minimization**: dE/dρ = 0 at ρ = ρ₀
```
-μ² + 2λρ₀ + 3κρ₀² + 4V₄_nuc·ρ₀³ = 0
```

**If we know** μ, λ, κ, ρ₀, we can solve for V₄_nuc:
```
V₄_nuc = (μ² - 2λρ₀ - 3κρ₀²) / (4ρ₀³)
```

**Problem**: We don't have empirical values for μ, κ in dimensionless form!

### From Binding Energy

**Binding per nucleon**: B/A ≈ 8 MeV

**Energy functional value**:
```
E[ρ₀] = -μ²ρ₀ + λρ₀² + κρ₀³ + V₄_nuc·ρ₀⁴
```

**At equilibrium**:
```
E[ρ₀] = -(B/A) × A ≈ -8A MeV
```

**This gives constraint but needs other parameters**

---

## Testing V₄_nuc = β

### Hypothesis 1: V₄_nuc = β (direct)

**Value**: V₄_nuc = 3.058231 (dimensionless)

**Check**: Does this prevent over-compression?

**Quartic term at ρ = 1** (scaled):
```
V₄_nuc·ρ⁴ = 3.058 × 1⁴ = 3.058
```

**Compared to quadratic**:
```
λρ² = λ × 1² = λ
```

**If λ ~ 938 MeV** (vacuum stiffness):
```
Ratio: 3.058 / 938 ≈ 0.0033 (quartic much smaller)
```

**Conclusion**: At ρ ~ 1, quartic is small. But at ρ >> 1, quartic dominates ✓

**Status**: ⚠️ PLAUSIBLE but needs full functional minimization

### Hypothesis 2: V₄_nuc = 4πβ

**Value**: V₄_nuc = 4π × 3.058 = 38.4

**Physical meaning**: Geometric surface factor

**Quartic term at ρ = 1**:
```
V₄_nuc·ρ⁴ = 38.4 × 1 = 38.4
```

**Much stronger stabilization**

**Status**: ⚠️ Could work, needs empirical check

### Hypothesis 3: V₄_nuc = β²

**Value**: V₄_nuc = (3.058)² = 9.351

**Physical meaning**: Squared stiffness

**Quartic term at ρ = 1**:
```
V₄_nuc·ρ⁴ = 9.351
```

**Status**: ⚠️ Intermediate between β and 4πβ

### Hypothesis 4: V₄_nuc = β/(4π)

**Value**: V₄_nuc = 3.058 / (4π) = 0.244

**Physical meaning**: Inverse geometric factor

**Quartic term at ρ = 1**:
```
V₄_nuc·ρ⁴ = 0.244
```

**Very weak stabilization**

**Status**: ❌ Probably too small

---

## Comparison with Other Parameters

### Parameters involving β

We've derived:
- c₂ = 1/β = 0.327
- V₄ (well depth) = λ/(2β²) = 50 MeV
- α_n = (8/7)β = 3.495
- β_n = (9/7)β = 3.932
- γ_e = (9/5)β = 5.505

**Pattern**: Most are β times simple fraction

**Prediction**: V₄_nuc = k × β where k is simple (likely 1, 4π, or fraction)

---

## The 4π Connection

### Why 4π?

**User mentioned**: V₄_nuc "likely related to 4π"

**Geometric reasons**:
1. **Sphere surface area**: 4πr² (geometry of soliton)
2. **Solid angle**: 4π sr (full sphere)
3. **Coulomb constant**: ke = 1/(4πε₀)
4. **Volume integral**: ∫ dΩ = 4π

**Nucleon as sphere**:
- Radius r₀ ~ 1 fm
- Volume ~ (4π/3)r₀³
- Surface ~ 4πr₀²

**If V₄_nuc relates to surface energy**:
```
V₄_nuc ~ (surface factor) × β ~ 4π × β
```

**Value**: V₄_nuc = 4πβ = 4π × 3.058 = **38.35**

---

## Testing Approach

### What We Need

Since we lack direct empirical value for V₄_nuc, we need to:

1. **Check internal consistency**:
   - Does V₄_nuc = β (or k×β) give stable solitons?
   - Do energy functional minima match nuclear properties?

2. **Numerical simulation**:
   - Solve E[ρ] minimization with V₄_nuc = β
   - Check if ρ₀ ≈ 0.16 fm⁻³
   - Check if B/A ≈ 8 MeV

3. **Compare with other derivations**:
   - We have V₄ (well depth) = λ/(2β²)
   - We have other β-dependent parameters
   - Does V₄_nuc = β fit the pattern?

### Preliminary Assessment

**Most likely**: V₄_nuc = β or 4πβ

**Reasoning**:
1. β is the fundamental stiffness parameter
2. Quartic term should inherit this stiffness
3. Factor could be 1 (direct) or 4π (geometric)
4. Other multiples (β², √β, etc.) seem less natural

**Next steps**:
1. Formalize in Lean with V₄_nuc = β assumption
2. Prove stability criterion holds
3. Check if this matches nuclear phenomenology
4. Test alternative V₄_nuc = 4πβ

---

## Preliminary Conclusion

**Hypothesis**: V₄_nuc = β (quartic soliton stiffness = vacuum bulk modulus)

**Status**:
- ✅ Physically motivated (same compression physics)
- ✅ Dimensionally consistent (if ρ dimensionless)
- ⚠️ Needs numerical validation (stability check)
- ⏳ No direct empirical value to compare

**Confidence**: MODERATE (70%)

**Alternative**: V₄_nuc = 4πβ (30%)

**Action**: Proceed with Lean formalization assuming V₄_nuc = β, document as hypothesis, validate through soliton stability proofs

---

**Generated**: 2025-12-30
**Status**: Theoretical investigation complete
**Hypothesis**: V₄_nuc = β (or 4πβ)
**Next**: Lean formalization + stability proofs

🔬 **V₄_NUC = β HYPOTHESIS FORMULATED** 🔬
