# Red Team Roadmap: Four Structural Gaps

**Date**: 2026-02-12
**Source**: External red-team analysis of QFD v8.9
**Purpose**: Map each identified gap to (a) what's already documented, (b) what the computation path looks like, (c) estimated difficulty

---

## Gap 1: κ̃ → H₀ Dimensional Bridge

**Red-team claim**: The conversion from dimensionless κ̃ ≈ 85.6 to K_J in km/s/Mpc lacks a first-principles derivation.

**What the book says**: §9.3.1 and line 5234 — "The identification κ̃ ≈ K_J [km/s/Mpc] is a numerical coincidence whose dimensional bridge awaits derivation."

**What exists**:
- κ̃ = ξ_QFD × β^(3/2) ≈ 85.58 (dimensionless, from α alone)
- K_J ≈ 85.6 km/s/Mpc numerically matches
- The K_J–M degeneracy (§12.10.1a) means the *shape* of μ(z) is testable but the absolute scale is absorbed into calibration

**What's missing**: A derivation of the vacuum's natural length scale L₀ in meters that connects κ̃ to K_J without using an empirical anchor (proton Compton wavelength or Planck length).

**Computation path**:
1. The 6C action S = ∫ dτ d⁶x̃ L_6C uses dimensionless coordinates x̃
2. Physical units enter via the scale factors E₀/L₀³ (Appendix B.2)
3. Need: derive L₀ from the soliton's *self-consistent* size in the β-stiff vacuum
4. If L₀ = f(β) × ℏ/(m_e c), then K_J = κ̃ × c / L₀ gives the bridge

**Difficulty**: HIGH. This is essentially the hierarchy problem — deriving why L₀ ≈ 10⁻¹⁵ m rather than the Planck scale. May require deriving m_e from L_6C directly.

**Status in codebase**: K_J now COMPUTED in `shared_constants.py` (was stale hardcoded). The √β coincidence (χ_QFD/χ_ΛCDM ≈ 1.74 ≈ √β) from the CMB solver is documented but unexplained.

---

## Gap 2: 6C → 4D Inelastic Breakdowns

**Red-team claim**: The spectral gap proof shows the gap exists but says nothing about what happens when collision energy exceeds ΔE.

**What the book says**: Ch.13 §13.4 identifies three failure scenarios: high-energy inelastic collisions, pair creation/annihilation, extreme ψ-field gradients. The "Transition Logic" is narrative.

**What exists**:
- `SpectralGap.lean`: Proves ΔE > 0 for the Hessian's orthogonal subspace
- Appendix Z.4.B: Formal spectral gap theorem (L[ψ] on H_orth has positive spectrum)
- The spectral decomposition H = H_sym ⊕ H_orth is rigorous

**What's missing**: The state-mixing matrix elements ⟨ψ_4D | H_6C | ψ_6D⟩ when E_collision > ΔE. Without these, the theory describes cold vacuum well but the hot limit (pair production, deep inelastic scattering) remains phenomenological.

**Computation path**:
1. Parameterize excited states of L[ψ] above the gap: η_n with eigenvalues λ_n > ΔE
2. Compute transition amplitudes: T_{4D→6D} = ⟨η_n | V_coupling | ψ_4D⟩
3. The coupling V is the cross-term between H_sym and H_orth in the full Hamiltonian
4. Sum over final states to get total inelastic cross-section σ_inel(E)

**Difficulty**: VERY HIGH. This is the QFD analogue of computing DIS structure functions. The full 6D field configuration space is 64-component at each point.

**Status in codebase**: `SpectralGap.lean` proves the gap exists. No computation of transition rates across the gap.

---

## Gap 3: σ = β³/(4π²) — Constitutive Postulate → Derivation

**Red-team claim**: The shear modulus σ used in the Padé saturation for tau g-2 is a constitutive postulate, not derived from L_6C.

**What the book says**: Appendix V line 13591 — "the dimensional analysis identifies σ = β³/(4π²) as the simplest scaling consistent with the Cl(3,3) active phase space, but it does not prove that no other combination is possible."

**What exists**:
- σ = β³/(4π²) ≈ 0.714 defined in `appendix_g_solver.py` line 296
- Full Padé saturation: V_circ/(1 + γₛx + δₛx²) implemented and validated
- Parameter status table (V line 13603): σ = "Constitutive postulate"
- Hessian framework: Z.4 (spectral gap), Z.8 (soliton spectrum), Z.3 (orbital stability)
- Scale-Dependent Hessian in Lean: `Scale_Dependent_Hessian.lean` (V₄ sign flip, not σ)
- Rigor spectrum in edits25: σ explicitly categorized as "Constitutive postulate"
- BOOK_PROOF_GAP_MANIFEST.md: σ listed as MISSING from Lean

**What's missing**: The eigenvalue calculation connecting L_6C → Hessian → shear mode → σ.

**Computation path** (the specific mathematical steps):

### Step 1: Hill vortex ground state in Cl(3,3) representation

The Hill vortex stream function is:
```
ψ(r,θ) = (3U/2a²)(a² - r²) r sin²θ    for r ≤ a
```
where a = vortex radius, U = circulation speed. The velocity field:
```
v_r = (1/r²sinθ) ∂ψ/∂θ,  v_θ = -(1/r sinθ) ∂ψ/∂r
```
In the Cl(3,3) multivector representation, this maps to:
```
ψ_0(X) = ρ(r) R(θ,φ) B   where B = e₄e₅ (internal bivector)
```
The density profile ρ(r) = ρ₀(1 - r²/a²) is parabolic (Hill).

### Step 2: Energy functional and second variation

The total energy is:
```
E[ψ] = ∫ [½|∇₆ψ|² + V(⟨ψ†ψ⟩₀)] d⁶x
```
where V(ρ) = -μ²ρ + λρ² (Mexican hat, Appendix B). Expand ψ = ψ₀ + εη:
```
E[ψ₀ + εη] = E[ψ₀] + ε² ∫ [½|∇₆η|² + V''(ρ₀)|η|²] d⁶x + O(ε³)
```
The Hessian operator is:
```
L[ψ₀] η = -∇₆² η + V''(ρ₀) η
```

### Step 3: Multipolar decomposition of perturbations

Decompose η into angular harmonics on S⁵ (the unit sphere in 6D):
```
η(r,Ω) = Σ_{ℓ,m} f_ℓ(r) Y_ℓm(Ω)
```
The Hessian separates:
```
L_ℓ f_ℓ = [-d²/dr² - (5/r)(d/dr) + ℓ(ℓ+4)/r² + V''(ρ₀)] f_ℓ
```
where ℓ(ℓ+4) is the 6D angular momentum eigenvalue.

**Key identification**:
- ℓ = 0 mode → isotropic compression → V₄ (bulk modulus β)
- ℓ = 2 mode → shape distortion at fixed volume → V₆ (shear modulus σ)
- ℓ = 4 mode → higher-order torsion → V₈ (torsional stiffness δₛ)

### Step 4: The σ eigenvalue

For the ℓ=2 mode, the effective potential in the Hessian is:
```
V_eff(r) = ℓ(ℓ+4)/r² + V''(ρ₀(r))
       = 12/r² + V''(ρ₀(1-r²/a²))
```
The lowest eigenvalue of L₂ determines the shear mode stiffness. If:
```
λ₂ = β³/(4π²) = σ
```
then the constitutive postulate becomes a theorem.

### Step 5: Why β³/(4π²) is plausible

The dimensional argument in the book:
- β³: Shear is a rank-3 deformation (volume-dependent), so scales as (bulk modulus)³
- 4π²: The solid-angle factor 2×(2π²) = 4π² is the surface area of two S³'s (the Cl(3,3) has two 3-spheres in its geometry)

The stronger argument (not in the book yet):
- The Hill vortex has V''(ρ₀) = λ - 3λρ₀²/ρ_c² where λ = β is the quartic coupling
- For the parabolic profile, ∫ V''(ρ₀) r⁴ dr ∝ β³ (three powers from the cubic profile)
- The angular integration over S⁵ gives 2π³ ≈ 4π² × (π/4) ≈ 4π²

This is the computation that needs to be done explicitly.

**Difficulty**: MEDIUM-HIGH. The angular decomposition is standard (hyperspherical harmonics). The radial eigenvalue is a Sturm-Liouville problem that can be solved numerically. The hard part is setting up the Cl(3,3) multivector → scalar Hessian reduction correctly.

**Status in codebase**: Numerical scaffolding built and producing results.

### Numerical Results (2026-02-12)

**v1-v3 solvers** (RETRACTED — solver bug):
- Used `eigh_tridiagonal` on a non-symmetric matrix (the (d-1)/r first-derivative
  term makes upper ≠ lower diagonals). This is INCORRECT.
- The "2.83% agreement" of the (ℓ_s=3, ℓ_t=1) mode was an artifact.
- The asymmetric matrix has COMPLEX eigenvalues (imag parts ~22000);
  `eigh_tridiagonal` silently forced real outputs.

**v4b solver** (self-adjoint form, CORRECT):
- Uses substitution u(r) = r^((d-1)/2) f(r) to obtain symmetric eigenvalue problem
- Verified against general eigensolver (numpy.linalg.eigvals) and weight-symmetrized
  form — all three agree to 6 digits
- **Correct eigenvalues**: compression λ₀ = -9.17, shear (3,1) λ₀ = -6.28
- **Correct ratio**: λ_shear/λ_compression = **0.685** (192% from target 0.235)
- No (ℓ_s, ℓ_t) mode comes close to β²/(4π²) in the scalar Hessian

**v3 profile findings** (still valid):
- The ratio IS profile-independent (all 6 profiles give identical results)
- This confirms the angular sector determines the ratio, not the radial profile
- But the ratio itself is 0.685, not 0.228

**v5-v8 solvers** (full Cl(3,3) Geometric Algebra approach):
- v5: GA-native potential Hessian — massively degenerate, grade-blind
- v6: Full 4-index kinetic tensor K_{ABij} — PROVEN: K_sym = g_{ij} G_{AB}
  (second-order Hessian has NO grade coupling in flat space)
  K_anti ≠ 0 but is a total derivative → doesn't enter Hessian
- v7: SO(3)×SO(3) rotor decomposition — internal Casimir max = 9,
  far below L²_eff = 31.35 needed for target ratio
- v8 (DEFINITIVE): Left-mult matrices L_i = G⁻¹M_i satisfy:
  - L_i² = g_{ii} I (error 0.00e+00)
  - L_iL_j + L_jL_i = 0 for i≠j (error 0.00e+00)
  - Full 64-channel system DECOUPLES completely:
    • 2-channel longitudinal → ratio = 0.815 (247% off)
    • 30 scalar transverse → ratio = 0.630 (168% off)
    • 32 timelike (repulsive, no bound states)

**Key conclusion**: The flat-space Cl(3,3) Hessian eigenvalue spectrum does NOT
reproduce σ = β³/(4π²). This is now PROVEN for the full 64-component
multivector space, not just the scalar reduction. The Clifford algebra
guarantees complete decoupling in flat space. The constitutive postulate
remains a postulate. Possible escape routes:
(a) Curved-space effects (soliton self-gravity → spin-orbit coupling)
(b) Beltrami/rotating-frame constraint (Coriolis coupling via L_{45})
(c) Non-perturbative (scattering amplitudes, not small oscillations)
(d) Topological (winding number / index theorem)
(e) Genuinely constitutive, constrained only by experiment (Belle II τ g-2)

---

## Gap 4: Gravitational Constant Hierarchy

**Red-team claim**: ξ_QFD = k_geom² × (5/6) gives the *form* of the gravitational coupling but the 10⁴⁰ hierarchy suppression is phenomenological.

**What the book says**: Three routes to ξ_QFD ≈ 16 exist (§12.7, line 5237), but deriving G_Newton from β alone requires the proton-to-Planck-length ratio.

**What exists**:
- ξ_QFD = K_GEOM² × (5/6) ≈ 16.2 in `shared_constants.py`
- The dimensional projection 6D→4D gives the 5/6 factor (active/total dimensions)
- k_geom = k_Hill × (π/α)^(1/5) is derived (Z.12)

**What's missing**: The bridge from ξ_QFD (dimensionless) to G (m³/kg/s²). This requires knowing the vacuum density ρ_vac in SI units, which circles back to Gap 1 (the natural length scale).

**Computation path**: If Gap 1 is solved (L₀ derived), then G = c⁴/(ξ_QFD × ρ_vac × L₀⁴) follows immediately. The two gaps are coupled.

**Difficulty**: VERY HIGH (coupled to Gap 1).

---

---

## Gap 5: PDE Ground State Existence (Analytic Gap)

**Red-team claim**: The spectral gap proof (SpectralGap.lean) assumes the ground-state
soliton exists. StabilityCriterion.lean proves the 1D potential has a global minimum,
but the infinite-dimensional energy functional E[ψ] on H¹(ℝ⁶; Cl(3,3)) has not been
shown to admit a minimizer. This is the QFD analogue of the Yang-Mills mass gap problem.

**What exists (Lean, 0 sorries)**:
- `StabilityCriterion.lean`: V(x) = -μ²x + λx² + κx³ + βx⁴ has global min (1D, EVT)
- `SpectralGap.lean`: IF ψ₀ exists with quantized topology THEN ΔE > 0
- `VacuumEigenvalue.lean`: β unique eigenvalue of Golden Loop
- `Soliton/Quantization.lean`: Vortex charge is quantized (hard-wall → discrete spectrum)

**What exists (numerical, `projects/field-theory/pde-existence/`)**:
- Hardy constant C_H = 4 in d=6 (verified numerically)
- Angular eigenvalues Λ_ℓ = ℓ(ℓ+4) on S⁵
- Sobolev critical exponent p* = 3 (supercritical |ψ|⁴)
- Strauss decay |u(r)| ≤ C·r^{-5/2}·‖u‖_{H¹} (radial compactness)
- Derrick scaling: d²E/dλ² = -8T < 0 (scalar unstable, needs topology)
- Centrifugal barrier Λ₁ = 5 for winding m=1 (prevents collapse)
- Coercivity: E ≥ -μ²M on {∫|ψ|² = M}
- Concentration-compactness checklist: all 4 conditions verified

**What's missing**:
- STEP A: Binding energy inequality E(m₁)+E(m₂) > E(m) (excludes dichotomy)
- STEP B: Weak lower semicontinuity of E[ψ] (Strauss + supercritical quartic)
- STEP C: Regularity of minimizer (elliptic bootstrap, standard)
- STEP D: Lean formalization (Mathlib lacks concentration-compactness)

**Difficulty**: HIGH for rigorous proof, MEDIUM for the mathematical argument.
The path is clear (Hardy + topology + concentration-compactness → existence),
but the supercritical exponent (p* = 3, quartic = 4) in d=6 requires the
equivariant/radial restriction for compact embedding.

**Key insight**: Topology provides the coercivity that Derrick's theorem denies
to pure scalars. The winding number m ≥ 1 creates a centrifugal barrier
(Λ_min = 5) analogous to angular momentum in the hydrogen atom.

---

## Priority Assessment

| Gap | Impact if Closed | Difficulty | Testability | Priority |
|-----|-----------------|------------|-------------|----------|
| 5 (PDE existence) | Closes the "IF" in spectral gap | HIGH | Mathematical proof | **1st** |
| 3 (σ from Hessian) | ~~Promotes tau sector~~ **CLOSED** — constitutive postulate | N/A | Belle II τ g-2 | Done |
| 1 (κ̃ → H₀) | Unifies cosmology with particle physics | HIGH | Already tested (K_J) | 2nd |
| 2 (Inelastic 6D) | Extends framework to hot/dense regimes | VERY HIGH | LHC, heavy-ion | 3rd |
| 4 (G hierarchy) | Solves the deepest problem in physics | VERY HIGH | Already tested (G) | 4th |

**Recommendation**: Gap 5 (PDE existence) is now the highest priority. It has the
clearest mathematical path (functional analysis, not physics), and closing it would
make SpectralGap.lean's conclusion unconditional.

---

## Computation Reference

- `projects/particle-physics/sigma-eigenvalue/` — Gap 3 (σ, CLOSED: v1-v9)
- `projects/field-theory/pde-existence/` — Gap 5 (PDE existence, IN PROGRESS)
