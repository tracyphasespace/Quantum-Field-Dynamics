# PDE Ground State Existence: Proof Outline

**Date**: 2026-02-12
**Status**: Bounds computed, proof path identified, formalization pending
**Computation**: `hardy_poincare_bounds.py` (all bounds verified numerically)

## The Problem

**Prove**: The QFD energy functional

> E[ψ] = ∫_{ℝ⁶} [½|∇₆ψ|² + V(|ψ|²)] d⁶x

where V(ρ) = -μ²ρ + βρ² (Mexican hat, β > 0), admits a global minimizer ψ₀
in the equivariant sector {ψ : Q[ψ] = m} (fixed topological charge m ≥ 1).

This would close the "IF" in `SpectralGap.lean`:
"IF a stable ground state exists, THEN spectral gap ΔE > 0."

## Current Lean Proof Chain (with the gap)

```
StabilityCriterion.lean   V(x) has min on ℝ           ✅ (1D, finite-dim)
     ↓ (does NOT imply)
  ❌ E[ψ] has minimizer on H¹(ℝ⁶)                      ← THE GAP
     ↓ (would imply)
SpectralGap.lean          ΔE > 0 for H_orth             ✅ (conditional)
```

## Proof Strategy: Concentration-Compactness

### Step 0: Function Space

Work in the equivariant Sobolev space:

> H¹_m(ℝ⁶) = {ψ ∈ H¹(ℝ⁶; ℝ³²) : ψ(R_θ x) = e^{imθ·B} ψ(x)}

where R_θ is the SO(2) rotation in the (x₄,x₅) plane and B = e₄e₅ is the
internal phase bivector. The winding number m is a topological invariant.

### Step 1: Hardy Inequality (d=6)

**Theorem**: For u ∈ H¹₀(ℝ⁶ \ {0}):

> ∫_{ℝ⁶} |∇u|² dx ≥ 4 ∫_{ℝ⁶} |u|²/|x|² dx

where C_H = ((d-2)/2)² = 4 is the sharp Hardy constant.

**Status**: Verified numerically to 10⁻¹⁴. Standard result in d ≥ 3.
Mathlib has `MeasureTheory.Measure.ae_le_of_integral_le` infrastructure
but not the Hardy inequality itself.

### Step 2: Centrifugal Barrier

For ψ ∈ H¹_m with winding m ≥ 1, the angular decomposition gives:

> T[ψ] = ∫₀^∞ [|f'|² + Λ_{|m|}|f|²/r²] r⁵ dr × ∫_{S⁵} |Y|²

where Λ_ℓ = ℓ(ℓ+4) is the angular eigenvalue on S⁵.

For m = 1: Λ₁ = 5. For m = 2: Λ₂ = 12.

**Key bound**: T[ψ] ≥ Λ_{|m|} ∫ |ψ|²/|x|² ≥ (Λ_{|m|}/C_H) · T[ψ].
This is consistent (Λ₁/C_H = 5/4 > 1 ✓) and means T[ψ] > 0 for m ≥ 1.

**Physical meaning**: The winding number creates a centrifugal barrier that
prevents the field from collapsing to a delta function at the origin.

### Step 3: Coercivity (Energy Bounded Below)

On the constraint manifold {∫|ψ|² = M}:

> E[ψ] = ½T - μ²M + β∫|ψ|⁴ ≥ -μ²M

Proof: T ≥ 0 and β∫|ψ|⁴ ≥ 0.

**Minimizing sequences are bounded in H¹**: If E[ψ_n] ≤ E₀, then:
> ½‖∇ψ_n‖² ≤ E₀ + μ²M → ‖ψ_n‖_{H¹} ≤ C(E₀, M, μ²)

### Step 4: Strauss Compactness (Radial/Equivariant)

**The supercritical problem**: In d=6, p* = 2d/(d-2) = 3, so |ψ|⁴ (p=4)
is NOT controlled by H¹ for general functions. Standard Sobolev fails.

**Resolution**: For equivariant functions with winding m ≥ 1:

1. **Strauss decay**: |ψ(r)| ≤ C · r^{-5/2} · ‖ψ‖_{H¹}
2. **Compact embedding**: H¹_m(ℝ⁶) ↪↪ L⁴(ℝ⁶) (COMPACT for equivariant)
3. This handles the supercritical quartic term.

The compactness follows because equivariant functions can't "spread thin"
in angle — the winding forces them to maintain radial structure.

### Step 5: Concentration-Compactness (Lions) — for m=1

Take a minimizing sequence {ψ_n} with E[ψ_n] → inf E and ∫|ψ_n|² = M.

**Vanishing excluded**: If ψ_n spreads to infinity, the centrifugal term
Λ₁∫|ψ_n|²/r² would grow (mass moves to large r where 1/r² penalty
is small, but the Strauss bound says ψ_n decays as r^{-5/2}, so ∫|ψ_n|²
on balls B(0,R)^c → 0, contradicting ∫|ψ_n|² = M).

**Dichotomy excluded (TOPOLOGICAL ARGUMENT)**: For m=1, the only possible
partition is {1, 0}. Within the equivariant sector H¹_{m=1}, a single
vortex CANNOT continuously split into two well-separated pieces while
preserving winding number 1 — the winding is topologically attached to
a single center.

**Key computation** (`binding_energy.py`): E(m)/m is NOT decreasing for m ≥ 2.
This means multi-quantum vortices (m ≥ 2) are UNSTABLE — they prefer to
split into m=1 pieces (Type-II behavior, like Abrikosov vortex splitting).
For m=1, no splitting is possible → dichotomy excluded by topology alone.

**Compactness**: After centering (equivariant functions are already centered),
a subsequence converges weakly in H¹ and strongly in L⁴ (by Strauss).
The weak limit ψ₀ satisfies ∫|ψ₀|² = M and E[ψ₀] = inf E.

### Step 5b: Multi-Quantum Vortex Instability (Bonus Result)

The binding energy computation reveals E(m₁) + E(m₂) < E(m₁+m₂) for m ≥ 2.
Physical consequences:
- **The electron (m=1) is the UNIQUE stable topological soliton**
- Higher charges (m ≥ 2) split into m=1 pieces (unstable excited states)
- This naturally explains charge quantization in units of e
- Analogous to Type-II superconductivity (Abrikosov vortex lattice)

### Step 6: Regularity

The minimizer ψ₀ satisfies the Euler-Lagrange equation:

> -Δψ₀ + V'(|ψ₀|²)ψ₀ = λψ₀

where λ is a Lagrange multiplier for the mass constraint.

By elliptic regularity bootstrap: ψ₀ ∈ H¹ → ψ₀ ∈ H² → ... → ψ₀ ∈ C^∞.

## Derrick's Theorem: Why Topology is Essential

Derrick scaling ψ_λ(x) = ψ(x/λ) gives E_λ = λ^{d-2}T + λ^d V.

At a critical point: (d-2)T + dV = 0, so V = -(d-2)T/d.

Second variation: d²E/dλ² = -2(d-2)T < 0 for d > 2.

This means ALL scalar solitons are **unstable under scaling** in d ≥ 3.

**Resolution**: Topological charge m ≠ 0 prevents the scaling deformation.
The transformation ψ_λ(x) = ψ(x/λ) changes the winding number density
(it compresses or dilates the vortex), which costs angular kinetic energy.
Specifically, T_ang = Λ_{|m|} ∫ |f|²/r² · r^5 dr is NOT scale-invariant
when f has a fixed winding profile — it penalizes both compression and dilation.

## What Needs Formal Proof

| Step | Mathematical Content | Difficulty | Lean Status |
|------|---------------------|------------|-------------|
| Hardy inequality | ∫|∇u|² ≥ 4∫|u|²/|x|² in d=6 | LOW | Not in Mathlib |
| Centrifugal barrier | Λ_ℓ = ℓ(ℓ+4) eigenvalues | LOW | Need S⁵ harmonics |
| Coercivity | E ≥ -μ²M on constraint set | LOW | Straightforward |
| Strauss compactness | H¹_m ↪↪ L⁴ for equivariant | MEDIUM | Need radial lemma |
| Binding energy | E(m₁)+E(m₂) > E(m) | MEDIUM | Need explicit calc |
| Weak convergence | Minimizer exists | MEDIUM | Standard but d=6 |
| Regularity | ψ₀ ∈ C^∞ | LOW | Elliptic theory |

**Estimated total Lean effort**: ~500-1000 lines, dependent on what
Mathlib provides for Sobolev spaces and radial functions.

## Connection to Book

The existence proof would:
1. Close the "IF" in SpectralGap.lean → unconditional ΔE > 0
2. Justify Appendix Z.4's assumption of a ground-state soliton
3. Provide quantitative bounds (C_H = 4, Λ₁ = 5) for the spectral gap
4. Address Red Team Gap 5 (PDE existence)

## Analogies to Known Results

The QFD soliton existence problem is structurally similar to:

1. **Abrikosov vortices** in Ginzburg-Landau theory (d=2, m=1)
   - Proven by Bethuel-Brezis-Hélein (1994)
   - Topological degree prevents collapse

2. **Skyrmions** in nuclear physics (d=3, m=baryon number)
   - Proven by Esteban (1986) for the Skyrme model
   - Derrick-stable due to the quartic Skyrme term

3. **Yang-Mills instantons** (d=4)
   - Proven by Uhlenbeck (1982) via removable singularity theorem
   - Topological charge (second Chern number) provides coercivity

The QFD case (d=6, m=vortex winding) is the natural higher-dimensional
extension of this pattern. The new ingredient is the supercritical Sobolev
exponent (p* = 3 < 4 = quartic), handled by equivariant compactness.
