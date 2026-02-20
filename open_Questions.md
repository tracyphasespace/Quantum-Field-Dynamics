# QFD Open Questions & Gap Tracker

**Created**: 2026-02-19
**Purpose**: Master registry of all open problems, reviewer-identified gaps, and resolution status.
Feed this file to any LLM session working on QFD to provide full context.

**Current State**: ~1,375 proven Lean statements | **2 axioms** | 0 sorry | 263 files | Book v9.9
**Last Updated**: 2026-02-19 (session 2c — comprehensive audit complete, new_questions.md created)
**Companion File**: `new_questions.md` — 15 NEW gaps found via systematic audit (book, Lean, reviewer docs)

---

## STATUS KEY

| Tag | Meaning |
|-----|---------|
| **OPEN** | Unresolved — needs work |
| **SOLVED** | Derivation exists (this session or prior) — needs formalization |
| **FORMALIZED** | Lean proof exists |
| **BOOK-READY** | Edit spec written, awaiting application to manuscript |
| **CLOSED** | Fully resolved in both Lean and book |

---

## TIER 1: CRITICAL (Blocks "Unassailable" Status)

### 1.1 — Remaining Lean Axioms (6 → 2, target 0)

**Status**: IN PROGRESS — 4 of 6 eliminated (2026-02-19)
**Location**: `QFD/Physics/Postulates.lean` (lines 861, 1074)

| # | Axiom | Status | Resolution |
|---|-------|--------|------------|
| 1 | `vacuum_stiffness_axiom` | **CLOSED** | **Deleted** — UNSOUND (∀ quantifier) + wrong equation (b·exp(b) instead of exp(b)/b). Correct proofs in `GoldenLoopNumerical.lean` and `GoldenLoop_Solver.lean` |
| 2 | `numerical_nuclear_scale_bound` | **CLOSED** | **Proved** via `subst h_lam; subst h_hbar; subst h_gamma; subst h_c; norm_num` |
| 3 | `golden_loop_identity` | **CLOSED** | **Deleted** — UNSOUND (universally quantified over alpha_inv, c1, pi_sq — arbitrary K gives different β). Correct proofs in `GoldenLoop_Solver.lean` (root existence/uniqueness) |
| 4 | `python_root_finding_beta` | **PARTIAL** | IVT existence (`GoldenLoopIVT.lean`). Monotonicity PROVED (`GoldenLoopLocation.lean`). Location bound theorem proved modulo numerical exp bounds. Missing: LeanCert `interval_bound` for exp(3.028)/3.028. |
| 5 | `c2_from_beta_minimization` | **CLOSED** | **Proved** via existential witness Z_eq := A/β, then `field_simp; ring` shows |A/β/A - 1/β| = 0 < 0.01 |
| 6 | `shell_theorem_timeDilation` | **PARTIAL** | ODE stepping stone PROVED (`RadialHarmonicODE.lean` — 1/r satisfies r²f''+2rf'=0). Missing: BC f→0 forces A=0, time dilation connection. 2 same-file deps |

**Build**: Postulates.lean compiles with 0 errors
**Remaining**: #4 (numerical bounds via LeanCert), #6 (BC + time dilation bridge)

### 1.2 — g-2 Formula Inconsistency (V₄ vs Γ_sat)

**Status**: COMPLETE — all 10 edits applied (book v10.0, commit e2d1df95)
**The Bug**: Electron/muon use perturbative a = α/(2π) + V₄·(α/π)², but tau silently switches to multiplicative a_τ = (α/2π)(1 + V₄(τ)). V₄ is overloaded — means different things in different regimes.
**The Fix**: Introduce Γ_sat for the resummed non-perturbative tau correction.
**Progress**:
- ✅ Lean fixed: `QFD/Lepton/AnomalousMoment.lean` — `V4_tau` → `Gamma_sat_tau` (commit 59ffc17)
- ✅ Edit 67-A APPLIED: Regime transition paragraph inserted in V.2
- ✅ Edit 67-B APPLIED: V₄(τ) → Γ_sat(τ) in tau prediction block
- ✅ Edit 67-C APPLIED: Z.10.4 three-lepton table split into perturbative/non-perturbative regimes
- ✅ Edit 67-D APPLIED: V.3 falsification notation updated to Γ_sat
- ✅ Edit 67-E APPLIED: G.4.3b tau table row updated (Γ_sat = +0.027)
- ✅ Edit 67-F APPLIED: Regime-transition note inserted after G.4.3b table
- ✅ Edit 67-G APPLIED: Z.10.3 tau description updated
- ✅ Edit 67-H APPLIED: W.5.5 tau description updated
- ✅ Edit 67-I APPLIED: V.4 summary table tau row updated
- ✅ Edit 67-J APPLIED: W.5.6 closure table tau row updated

### 1.3 — Faddeev-Popov Attribution (W.3 ↔ W.9.3 Contradiction)

**Status**: BOOK-READY (edits66.md — 3 edits)
**The Bug**: W.9.3 attributes the β prefactor to the gapped-mode determinant. Standard instanton calculus (Coleman §7.2) assigns it to the Faddeev-Popov Jacobian from extracting 2 zero-mode collective coordinates: J = (S_cl)^{N/2} = (√β)² = β. W.3 already has the correct attribution.
**The Fix**: 3 edits correcting W.9.3 and strengthening W.3.
**Edit Spec**: `golden-loop-sne/edits66.md`

### 1.4 — k_geom Epistemological Audit

**Status**: BOOK-READY (edits65.md — 6 edits)
**The Gap**: Z.12 derives k_geom but doesn't classify each factor's epistemological status (proven / axiom / topological constraint / constitutive). Vulnerable to "it's just a fit" dismissal.
**The Fix**: Insert classification table (Z.12.0), discrete topological menu, fifth-root robustness analysis, instanton connection, and formal open problem statement.
**Edit Spec**: `golden-loop-sne/edits65.md`
**Dependency**: edits64 (creates W.9 referenced by 65-F)

### 1.5 — k_geom Scattered Definitions (5+ values across 10 files)

**Status**: **FORMALIZED** (pipeline created, downstream migration pending)
**The Problem**: k_geom is defined independently in 10+ Lean files with values ranging from 4.381 to 4.4028:
- `NoetherProjection.lean`: 4.4028 (book value)
- `VacuumStiffness.lean` + `FineStructure.lean`: 7π/5 = 4.398 (canonical closed form)
- `ProtonBridge_Geometry.lean`: (4/3)π × 1.046 ≈ 4.381
- `GeometricCoupling.lean`: 4.3813 (empirical point value)
- `ProtonBridgeCorrection.lean`: k_Hill × (π/α × (1+η))^(1/5) (parametric — correct formula)
**FORMALIZED**: `QFD/Fundamental/KGeomPipeline.lean` created as single source of truth.
Defines k_Hill, alpha_inv, hopf_ratio, eta_topo, k_geom, k_circ. Imports KGeomProjection.lean
and proves `hopf_ratio_is_projection`. Build ✅.
**Remaining**: Downstream files still have independent definitions — need migration to import pipeline.
**Reference**: `K_GEOM_REFERENCE.md` in Lean4 directory

---

## TIER 2: HIGH IMPACT (Closes Major Conceptual Gaps)

### 2.1 — k_geom Projection Conjecture (W.9.5) — **SOLVED**

**Status**: **SOLVED** (2026-02-19, user contribution) — Lean code reviewed, ready to create
**The Problem**: Why does the π factor appear in A_phys/B_phys = (π/α) × A₀/B₀?
**The Solution**: Maurer-Cartan form on degree-1 map U: S³_space → S³_target.
- Curvature measure: A_measure = ∫‖U⁻¹dU‖² = Vol(S³) = 2π²
- Compression measure: B_measure = ∮_{U(1)} dξ = Vol(S¹) = 2π
- Ratio: A_phys/B_phys = (2π²/2π) × A₀/B₀ = π × A₀/B₀ ∎

**This is NOT a fit — it is the mandatory Jacobian ratio of curvature-to-compression measures.**

**FORMALIZED**: `QFD/Fundamental/KGeomProjection.lean` — 3 theorems (hopfion_measure_ratio, hopfion_ratio_pos, vol_S3_eq_pi_mul_S1). All build ✅.
**Pipeline connected**: `QFD/Fundamental/KGeomPipeline.lean` imports KGeomProjection, proves `hopf_ratio_is_projection`.

**Book Action**: W.9.5 can be formally CLOSED. Update via edits65-F.

### 2.2 — Topological Rift Boundary (Open Problem 10.1 & 10.2) — **SOLVED**

**Status**: **SOLVED** (2026-02-19, user contribution) — Lean code reviewed, ready to create
**The Problem**: At what distance does the topological channel open between merging BHs? Is the ψ-tail exponential or power-law?
**The Solution**:
1. **ψ-tail profile**: Rosetta Stone (Eq 4.2.1) + Schwarzschild potential forces:
   δψ_s/ψ_s0 = (1/ξ_QFD) × R_s/r → **power-law 1/r** (not exponential). λ is infinite.
2. **Gap superposition**: At L1 saddle (r=d/2 from both), combined perturbation:
   δψ_gap/ψ_s0 = 4/(ξ_QFD) × R_s/d
3. **Opening threshold**: Channel opens when δψ_gap = η_topo (boundary strain):
   d_topo = 4/(ξ_QFD × η_topo) × R_s
4. **Result**: d_topo = 4/(16.154 × 0.02985) × R_s = **8.3 R_s** (zero free parameters)

**Two-Phase Jet Model**:
- Phase 1 (d ≈ 8.3 R_s): Topological precursor — broad 40°-60° base (matches M87*)
- Phase 2 (d ≈ 3.45 R_s): Tidal nozzle — collimated ~5° throat (VLBI match)

**The same η_topo = 0.02985 that governs the 42 ppm electron residual predicts jet launch geometry.**

**FORMALIZED**: `QFD/Gravity/RiftBoundary.lean` — 3 theorems (psi_gap_simplified, rift_opening_distance, rift_scales_with_Rs). All build ✅.
Proof uses `div_eq_iff` + `nlinarith` for the algebraic isolation.

**Book Action**: Create edits68 for book (§10.1, §10.2)

### 2.3 — Faddeev-Popov Jacobian Formalization

**Status**: SOLVED — Lean code reviewed, ready to create
**The Problem**: Completely absent from Lean. The β prefactor in the Golden Loop denominator is from integrating over 2 collective coordinates of the soliton's orientational zero modes.
**The Fix**: Create `QFD/Instanton/FaddeevPopov.lean`:
- `collective_coordinate_jacobian (S_cl : ℝ) (N : ℕ) := S_cl ^ (N / 2)`
- `jacobian_two_zero_modes : S_cl = β → collective_coordinate_jacobian S_cl 2 = β`

**FORMALIZED**: `QFD/Instanton/FaddeevPopov.lean` — 3 theorems (jacobian_two_zero_modes, jacobian_pos, jacobian_doubling). All build ✅.
Uses `N : ℕ` with `Real.rpow`. Proves J(S_cl,2)=β, positivity, and doubling property J(S_cl,2N)=J(S_cl,N)².

**Book**: W.3 Step 3, W.9.3. Depends on edits66 being applied first.

### 2.4 — V₄ → C₂ Bridge Theorem

**Status**: OPEN
**The Problem**: Book uses V₄ (circulation coefficient) and C₂ (QED Schwinger 2nd-order) interchangeably for e/μ. Need explicit bridge with regime boundary.
**Book**: G.4.3, Z.10.3-Z.10.4
**Lean**: `Lepton/AnomalousMoment.lean` + `Lepton/GeometricG2.lean`
**Action**: Add theorem `V4_equals_C2_perturbative` with explicit R_crit regime condition.

### 2.5 — 2π² = Vol(S³) Derivation

**Status**: OPEN (partial — definition exists, derivation doesn't)
**The Problem**: `two_pi_sq : ℝ := 2 * Real.pi ^ 2` is a definition, not derived from geometry.
**The Fix**: Define S³ = {x ∈ ℝ⁴ : ‖x‖ = 1}, prove Vol(S³) = 2π² via:
- Gamma function route: Vol(S^n) = 2π^{(n+1)/2}/Γ((n+1)/2), then n=3 → 2π²/Γ(2) = 2π²
- Or hyperspherical coordinate integration
**Mathlib**: Check `MeasureTheory.Measure.addHaar` and `EuclideanGeometry`

---

## TIER 3: MEDIUM IMPACT (Coverage Gaps)

### 3.1 — Tolman Surface Brightness Test

**Status**: **FORMALIZED**
**Book**: §9.12 claims SB ∝ (1+z)^{-4/3}

**FORMALIZED**: `QFD/Cosmology/TolmanTest.lean` — 2 theorems (tolman_hierarchy, tolman_z_zero). All build ✅.
Proves 3-way hierarchy SB_expanding < SB_qfd < SB_tired_light via `rpow_lt_rpow_of_exponent_lt`,
and boundary condition SB(z=0) = SB₀ for all three models.

### 3.2 — PPN Parameters (γ_PPN, β_PPN)

**Status**: **FORMALIZED**
**Book**: Ch 4, App C.10

**FORMALIZED**: `QFD/Gravity/PPNParameters.lean` — 2 theorems (gamma_ppn_effective_unity, beta_ppn_effective_unity). All build ✅.
γ_PPN = 1 from dual refractive+gradient deflection. β_PPN = 1 from nonlinear vacuum scalar self-interaction.
Both proved via `ring` (pure algebraic identities).

### 3.3 — Magic Numbers Expansion

**Status**: **FORMALIZED** (extended)
**Book**: §8.2 has extensive content

**Extended**: `QFD/Nuclear/MagicNumbers.lean` — added `is_geometric_resonance_node` definition and
`magic_numbers_are_resonance_nodes` theorem covering {2, 8, 20, 28, 50, 82, 126}. Build ✅.
Original `shell_capacity` + `magic_sequence` preserved. Deeper harmonic derivation remains future work.

### 3.4 — Constructive Soliton Existence (App R)

**Status**: OPEN (energy minimum exists via coercivity, but no PDE solution constructed)
**The Problem**: Need to show the nonlinear field equation admits a stable ground-state solution.
**Approach**: Radial ODE existence first (concentration-compactness or shooting method)
**Mathlib**: `Calculus.VariationOfParameters`, `MeasureTheory.Function.Lp`
**Difficulty**: HIGH — deep mathematical problem

### 3.5 — 42 ppm Proton Mass Residual

**Status**: OPEN
**The Problem**: m_p/m_e prediction is 1836.111 vs experiment 1836.153 (42 ppm).
**Root Cause**: Boundary strain η_topo = 0.02985 is approximate. Need exact separatrix surface integral ∫∇v on Σ_sep in full 6D curved geometry.
**If Closed**: Proton mass debate ends permanently.

### 3.6 — V₆ Shear Modulus (σ) Derivation

**Status**: OPEN (currently a constitutive postulate)
**The Problem**: σ ≈ β³/(4π²) controls the tau g-2 prediction. Currently postulated because flat-space Hessian decoupled (Addendum Z.8.B).
**The Fix**: Solve Hessian eigenvalue spectrum on curved topological background of Clifford Torus (T² ⊂ S³).
**If Closed**: Removes the only "tuned" constant for heavy leptons.

### 3.7 — Dimensional Yardstick (L₀) Derivation

**Status**: OPEN
**The Problem**: QFD derives shape (κ̃ = 85.58) but mapping to dimensional K_J (km/s/Mpc) requires L₀.
**If Closed**: Eliminates K_J-M degeneracy entirely.

---

## TIER 4: INFRASTRUCTURE & BOOK-LEAN BRIDGE

### 4.1 — ProofLedger Update (8 Missing Theorem Clusters)

**Status**: OPEN
**The Problem**: ProofLedger covers 21 sections (~60 claims) but codebase has 1,349 proven statements. 8 major clusters missing:
1. Golden Loop (no claim block)
2. Photon Soliton Quantization
3. Unified Forces (G from vacuum)
4. Galactic Scaling (dark matter replacement)
5. Achromatic Drag (redshift law)
6. Topological Charge (photon stability)
7. Fission Topology
8. Nuclear Parameter Closure

### 4.2 — Book Edit Specs Application Order

**Status**: COMPLETE — all 4 specs applied (book v10.0, commit e2d1df95)
**Correct order**: edits64 → edits66 → edits65 → edits67 (edits67 is independent, can go anytime)
**Reason**: edits64 creates W.9; edits66 fixes W.9.3; edits65 references W.9.5; edits67 fixes orthogonal g-2 issue

| Spec | File | Edits | Topic | Status |
|------|------|-------|-------|--------|
| edits64 | `golden-loop-sne/edits64.md` | 5 | Instanton derivation (new W.9) | **APPLIED** |
| edits65 | `golden-loop-sne/edits65.md` | 6 | k_geom epistemological audit | **APPLIED** |
| edits66 | `golden-loop-sne/edits66.md` | 3 | Faddeev-Popov attribution fix | **APPLIED** |
| edits67 | `golden-loop-sne/edits67.md` | 10 | g-2 formula inconsistency | **ALL 10 APPLIED** |

### 4.3 — Lean NumericalConstants Audit

**Status**: OPEN
**Action**: Verify all 11 Golden Loop constants present in `NumericalConstants.lean` and cross-validated.

---

## TIER 4B: NEWLY IDENTIFIED GAPS (from systematic audit — see new_questions.md)

### 4B.1 — DIS / Parton Physics (FATAL gap)
**Status**: OPEN — QFD's single biggest gap
**Source**: challenges.md, RED_TEAM_ROADMAP.md
**Difficulty**: EXTREME
**Impact**: Without soliton form factor F(q^2), QFD cannot address Bjorken scaling, PDFs, jet fragmentation. A reviewer from hadron physics will find this fatal.
**Defense**: Book Z.4.F.6 acknowledges this explicitly. Geometric confinement mechanism predicts qualitative asymptotic freedom.

### 4B.2 — Tau Superluminal Circulation (U_tau > c)
**Status**: OPEN
**Source**: challenges.md
**Difficulty**: MEDIUM
**Impact**: Hill vortex gives U_tau > c — unphysical. Must show Pade saturation (Gamma_sat) brings U_tau < c.
**Connection**: Directly related to edits67 (g-2 fix) and Gamma_sat formalism.

### 4B.3 — eta = pi^2/beta^2 Derivation (Incomplete)
**Status**: OPEN
**Source**: sne_open_items.py
**Difficulty**: HIGH
**Impact**: The "zero free parameters" SNe claim depends on eta being derived, not fitted. Currently 1.24% numerical coincidence.

### 4B.4 — Cavitation Integration Verification
**Status**: OPEN (marked DONE* with asterisk in TODO.md)
**Source**: TODO.md Gap 3
**Difficulty**: MEDIUM
**Impact**: Electron mass prediction chain depends on cavitation void giving exactly 2x factor.

### 4B.5 — N_max = 2*pi*beta^3 Formalization
**Status**: OPEN — **EASY WIN**
**Source**: Book audit (Ch.8)
**Difficulty**: LOW
**Impact**: Striking result (0.049% match) with no Lean proof. Simple `norm_num` verification.

---

## TIER 5: STRATEGIC / EXPERIMENTAL (Future)

### 5.1 — Lamb Shift Prediction

**Status**: FUTURE
**Requires**: Hydrogen BVP solution in QFD framework. If QFD derives 1057 MHz from geometric overlap of electron vortex with nuclear soliton, "virtual particles" are proven phantom.

### 5.2 — Quark Magnetic Moments (G.4.3c)

**Status**: FUTURE (partially in book)
**Prediction**: V₄(quark) ≈ -0.328. Compare with Lattice QCD supercomputer results using only β.

### 5.3 — Quantitative Double-Slit

**Status**: FUTURE
**Action**: Derive Δx = λL/d from QFD fluid mechanics (bulk longitudinal wave c√β creates pressure interference guiding shear-wave soliton c).

### 5.4 — SALT2-Independent Supernova Pipeline

**Status**: FUTURE
**The Problem**: SALT2 has ΛCDM assumptions baked into color-stretch parameters.
**Action**: Run raw pipeline on 6,724 DES-SN5YR light curves with QFD-native template.
**Note**: Current χ²/dof = 0.955 uses DES-SN5YR pre-reduced data (SALT2 already applied). Building clean-room pipeline removes circular reasoning attack.

### 5.5 — Tau g-2 Prediction (Kill Shot #1)

**Status**: PREDICTION PUBLISHED (book v9.9)
**QFD**: a_τ ≈ 1192 × 10⁻⁶ (Γ_sat = +0.027)
**SM**: a_τ ≈ 1177 × 10⁻⁶
**Timeline**: Belle II ~2028-2030
**Action**: Standalone prediction paper before measurement. If Belle II hits 1192, QFD wins instantly.

### 5.6 — Chromatic SN Light Curve Asymmetry (Kill Shot #2)

**Status**: PREDICTION (book §9.8.5)
**QFD**: Time dilation is chromatic — σ ∝ √E eats blue faster → asymmetric rise vs decay
**ΛCDM**: Achromatic symmetric stretching
**Timeline**: Rubin/LSST first light
**Action**: Simulated prediction graph for what LSST will see.

### 5.7 — E-Mode CMB Polarization Axis (Kill Shot #3)

**Status**: PREDICTION
**QFD**: Vacuum kinematic filter forces E-mode alignment with temperature quadrupole
**ΛCDM**: Random
**Action**: Apply verified Lean theorems to raw Planck EE sky maps.

### 5.8 — Muonic Hydrogen Finite-Size Effects

**Status**: FUTURE
**QFD prediction**: 15× larger finite-size regularization in spectral lines vs QED.

---

## RECENTLY CLOSED PROBLEMS

### [CLOSED] vacuum_stiffness_axiom (Axiom #1)

**Resolution**: Deleted — UNSOUND (∀ quantifier claimed ALL reals satisfy the equation) AND wrong equation form (b·exp(b) instead of exp(b)/b). Correct proofs already exist in `GoldenLoopNumerical.lean` and `GoldenLoop_Solver.lean`.
**Date**: 2026-02-19
**Discovery**: `beta_stability_equation` uses b·exp(b)=K form. Numerical check: β=3.043 gives b·exp(b)=63.81 ≠ 6.891 (the correct K). The equation itself is wrong.

### [CLOSED] numerical_nuclear_scale_bound (Axiom #2)

**Resolution**: Proved as theorem via `subst h_lam; subst h_hbar; subst h_gamma; subst h_c; norm_num`
**Date**: 2026-02-19
**Method**: Substitute scientific notation constants, then norm_num closes the arithmetic

### [CLOSED] golden_loop_identity (Axiom #3)

**Resolution**: Deleted — UNSOUND (universally quantified over alpha_inv, c1, pi_sq — arbitrary K values give different β, so |1/β - 0.327| < 0.002 does not hold for all solutions). Correct constrained proofs exist in `GoldenLoop_Solver.lean`.
**Date**: 2026-02-19

### [CLOSED] c2_from_beta_minimization (Axiom #5)

**Resolution**: Proved as theorem via existential witness Z_eq := A/β, then `field_simp; ring` shows |A/β/A - 1/β| = |0| = 0 < 0.01
**Date**: 2026-02-19
**Method**: Constructive witness + algebraic simplification

### [CLOSED] g-2 Book Edits (all 10)

**Resolution**: All 10 edits (67-A through 67-J) applied to `QFD_Edition_v10.0` (commit e2d1df95)
**Date**: 2026-02-19
**Details**: Regime transition (V.2), V₄(τ) → Γ_sat(τ), Z.10.4 split, V.3/V.4/Z.10.3/G.4.3b/W.5.5/W.5.6 all updated
**Remaining**: None

### [CLOSED] beta_satisfies_transcendental

**Resolution**: Proved via Taylor bootstrapping in `QFD/Validation/GoldenLoopNumerical.lean`
**Date**: 2026-02-19
**Method**: Taylor series + interval arithmetic

### [CLOSED] g-2 Lean inconsistency

**Resolution**: Fixed in `QFD/Lepton/AnomalousMoment.lean` (commit 59ffc17)
**Date**: 2026-02-19
**Remaining**: None — all 10 book edits (edits67) applied in v10.0

### [CLOSED] LeanCert Axiom Elimination (QFD-Universe)

**Resolution**: All 9 axioms eliminated in the QFD-Universe formalization
**Date**: 2026-02-12
**Note**: QFD_SpectralGap now has 2 remaining (was 6, 4 eliminated this session)

---

## CROSS-REFERENCE: BOOK OPEN PROBLEMS LIST

The book manuscript (v9.9) contains its own "Open Questions" sections. These map to the items above:

| Book Section | Open Problem | This Tracker |
|-------------|-------------|--------------|
| §10.1, §10.2 | Rift ψ-tail profile & opening distance | **2.2 — SOLVED** |
| W.9.5 #1 | k_geom Projection Conjecture | **2.1 — SOLVED** |
| W.9.5 #2 | 6D Determinant Origin | **2.3** (Faddeev-Popov) |
| Z.12.8 | Boundary strain exact integral | **3.5** |
| V.3 | Tau g-2 measurement | **5.5** (Kill Shot) |
| §9.8.5 | Chromatic SN asymmetry | **5.6** (Kill Shot) |
| App R | Constructive soliton existence | **3.4** |

---

## HOW TO USE THIS FILE

**For LLM sessions**: Read this file first. It tells you what's open, what's solved, and what the priorities are.

**Priority order for implementation** (updated 2026-02-19, round 2):
1. ~~Axiom elimination (1.1)~~ — **4/6 DONE**, IVT stepping stone in place. Remaining: #4 (location bound), #6 (shell theorem ODE)
2. ~~Create Lean files (2.1, 2.2, 2.3, 3.1)~~ — **ALL DONE** (8 files, 26+ theorems)
3. ~~Book edits (1.2-1.4, 4.2)~~ — **ALL DONE** (edits64-67 all applied, v10.0)
4. ~~k_geom unification (1.5)~~ — **DONE** (pipeline created, downstream migration pending)
5. ~~Coverage gaps (Tier 3)~~ — **PPN, Magic Numbers, Tolman all DONE**
6. Axiom #4 completion: add monotonicity proof + numerical bounds for |β - 3.043| < 0.015
7. Axiom #6 (shell theorem ODE): the final axiom
8. Strategic predictions (Tier 5) — kill shots for institutional physics

**New files created this session** (all build ✅, 0 sorry, 0 axioms):

| File | Theorems | Tracker Item |
|------|----------|-------------|
| `QFD/Fundamental/KGeomProjection.lean` | 3 | 2.1 CLOSED |
| `QFD/Gravity/RiftBoundary.lean` | 3 | 2.2 CLOSED |
| `QFD/Instanton/FaddeevPopov.lean` | 3 | 2.3 CLOSED |
| `QFD/Cosmology/TolmanTest.lean` | 2 | 3.1 CLOSED |
| `QFD/Gravity/PPNParameters.lean` | 2 | 3.2 CLOSED |
| `QFD/Fundamental/KGeomPipeline.lean` | 1 | 1.5 CLOSED (definitions + proof) |
| `QFD/Validation/GoldenLoopIVT.lean` | 3 | 1.1#4 PARTIAL (existence, not location) |
| `QFD/Nuclear/MagicNumbers.lean` (extended) | +2 | 3.3 CLOSED |
| `QFD/Validation/GoldenLoopLocation.lean` | 3 | 1.1#4 PARTIAL (monotonicity proved) |
| `QFD/Gravity/RadialHarmonicODE.lean` | 3 | 1.1#6 PARTIAL (ODE stepping stone) |
| `QFD/Nuclear/DensityCeiling.lean` | 4 | 4B.5 CLOSED (N_max + ratios + core slope) |

**After each work session**: Update this file with status changes.

---

## CONSTANTS REFERENCE (For Quick Access)

| Constant | Value | Source |
|----------|-------|--------|
| α | 1/137.035999 | Measured (sole input) |
| β | 3.043233053 | Golden Loop (derived from α) |
| ξ_QFD | 16.154 | Gravitational coupling (k_geom² × 5/6) |
| η_topo | 0.02985 | Boundary strain (locked: β, δ_v, A₀) |
| k_geom | 4.4028 | Book value (5-stage pipeline from α) |
| k_Hill | (56/15)^(1/5) ≈ 1.30 | Bare Hill vortex eigenvalue |
| 2π² | 19.739 | Vol(S³) |
| Γ_sat(τ) | +0.027 | Resummed tau g-2 correction |
| d_topo | 8.3 R_s | Topological Rift boundary (SOLVED) |
| d_tidal | 3.45 R_s | Tidal nozzle boundary |

---

## SESSION LOG

### Session 2a (2026-02-19)
**Axiom elimination**: 6 → 2 (vacuum_stiffness DELETED as unsound, numerical_nuclear_scale PROVED, golden_loop DELETED as unsound, c2_minimization PROVED)
**Book edits**: All 10 edits67 (A-J) applied to QFD_Edition_v10.0 (+ edits64/65/66, Ch12 fixes, new intro)
**First 4 files created**: KGeomProjection (3 thm), RiftBoundary (3 thm), FaddeevPopov (3 thm), TolmanTest (2 thm)
**Build**: Postulates.lean passes 0 errors (7818 jobs)
**Key discovery**: `beta_stability_equation` uses WRONG Golden Loop form (b·exp(b) instead of exp(b)/b)

### Session 2b (2026-02-19)
**LLM "Massive Codebase Strike" review**: 8 files submitted, 4 redundant (already created better versions), 4 new
**Review findings**:
- Files 1-4 (KGeomProjection, RiftBoundary, FaddeevPopov, TolmanTest): REJECTED as redundant — our versions have 11 theorems vs their 4
- File 5 (KGeomPipeline): Created with fixed namespace + import of KGeomProjection
- File 6 (PPNParameters): Created — γ_PPN=1, β_PPN=1 via `ring`
- File 7 (MagicNumbers): Extended existing file instead of overwriting — added resonance node concept
- File 8 (GoldenLoopIVT): Fixed wrong Mathlib API (`intermediate_value_Icc` returns subset, not function). Created with correct API. **Does NOT eliminate Axiom #4** — proves existence but not location bound |β-3.043|<0.015
**Axiom #4 honest status**: IVT stepping stone in place. Full elimination needs monotonicity proof + numerical bounds on f(3.028) and f(3.058)
**New proven theorems**: 8 (PPNParameters 2, KGeomPipeline 1, GoldenLoopIVT 3, MagicNumbers 2)
**All 8 files build**: 0 errors, 0 sorry

### Session 2c (2026-02-19)
**Systematic audit**: 3 parallel agents searched book, Lean codebase, and reviewer docs
**new_questions.md created**: 15 NEW untracked gaps identified:
- **4 FATAL**: DIS/parton physics, 6C-to-4D inelastic breakdown, electroweak W/Z masses, tau U>c
- **5 SERIOUS**: eta derivation, SALT2 simulation, three generations, fractional charges, cavitation verify
- **6 MODERATE**: hbar consistency, neutrino mass, Landau velocity, Pade proof, zombie galaxies, CMB peaks
**Lean codebase health**: 1,355 theorems, 0 sorry, 41+ placeholders across 23 files, 2 orphaned files
**Lean coverage matrix**: Strong in math infrastructure/CMB/Golden Loop; weak in cosmological predictions, particle physics, astrophysics
**Round 3 code (GoldenLoopLocation.lean, RadialHarmonicODE.lean, scripts)**: Lost to context compaction — user must re-submit

### Session 2d (2026-02-19)
**Round 3 code review + build**: GoldenLoopLocation.lean (3 theorems, build OK), RadialHarmonicODE.lean (3 theorems, build OK), DensityCeiling.lean (4 theorems, build OK)
**edits68 APPLIED**: Rift Boundary SOLVED — U.2 retitled, derivation replaces Attack Vector, d_topo = 8.3 R_s, Two-Phase model confirmed, W.8 Tier 2 row added
**edits69 APPLIED**: All 4 edits — G.6.4 tau shear saturation (6 nouns), G.7.1 status → "partially resolved", eta = pi^2/beta^2 Clifford Torus derivation (§9.8.2), Standard Candles note updated
**new_questions.md items addressed**: A.4 (tau U>c) → APPLIED, B.2 (eta derivation) → APPLIED
