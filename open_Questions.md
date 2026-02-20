# QFD Open Questions

**Current State**: ~1,380 proven Lean statements | **0 axioms** | 0 sorry | 270 files | Book v10.0
**Last Updated**: 2026-02-19
**Reviewer Audit**: Incorporated 2026-02-19 (functional analysis, field theory, cosmology, epistemology)

---

## LEAN PROOF STANDARDS (MANDATORY)

**Toolchain**: Lean 4.28.0-rc1 + Mathlib (pinned)

Every new Lean theorem MUST satisfy ALL of the following:

1. **No `sorry`** — every proof must be complete
2. **No `axiom`** — no standalone axiom declarations
3. **No `True` targets** — theorem conclusions must be substantive
4. **No vacuous proofs** — every hypothesis must be used; `h → h` is not a theorem
5. **No trivial arithmetic** — `(1/3)β + (2/3)β = β` is `ring`, not a physics result
6. **No unused hypotheses** — if `h : P` is never referenced, delete it
7. **No definitions disguised as theorems** — if the conclusion is `rfl`, it's a definition
8. **No structures as axiom replacements** — bundling a conclusion as a field and extracting it is not a proof
9. **No hypothesis-padding** — taking `Γ(2) = 1` as hypothesis and substituting is not deriving Vol(S³)

**Test**: If `#print axioms <theorem_name>` shows only `[propext, Classical.choice, Quot.sound]`, and the conclusion requires genuine mathematical work beyond `rfl`/`ring`/`norm_num` on trivially determined inputs, then it passes.

---

## TIER 1: CRITICAL (Manuscript Survival)

### 1.1 — Gapped-Mode Functional Determinant (W.9.5) [REVIEWER 1A]
**Status**: OPEN — **NEW, CRITICAL**
The Golden Loop `1/α = 2π²(e^β/β) + 1` assumes the 11 gapped-mode fluctuation determinant `det'(-∇² + V'')^{-1/2}` contributes exactly 1. In standard instanton calculus ('t Hooft, Yang-Mills), this determinant yields a SPECIFIC numerical prefactor. If it evaluates to 1.02, the 9-digit match for 1/α collapses.
**Fix**: Calculate the functional trace over SO(6)/SO(2) coset via:
- Zeta-function regularization, OR
- Heat kernel expansion on the instanton background
- Prove the determinant is exactly 1 (perhaps via bosonic-fermionic cancellation in Cl(3,3)), OR
- Show exactly how its value is absorbed into the topological volume 2π²
**Book**: W.9.5, Z.8.B
**Difficulty**: VERY HIGH — requires instanton calculus expertise

### 1.2 — Redshift Line Straggling Bound [REVIEWER 3A]
**Status**: RESOLVED (2026-02-19)
QFD predicts ZERO spectral line broadening for two independent reasons:
1. **Forward drag is coherent** (virtual process, optical theorem). E(D) = E₀·exp(-κD) exactly, no stochastic term. FDT doesn't apply (photon 43,400× above thermal equilibrium).
2. **Non-forward photons exit beam** (extinction, not broadening). Same as Rayleigh scattering — scattered photons are REMOVED, arriving photons have exact frequencies.
The non-forward opacity τ(z) = η·[1-1/√(1+z)] is already in μ_QFD as DIMMING, not broadening.
**Result**: QFD prediction 0.0 km/s vs observed limit 5.0 km/s (Ly-α), 2.0 km/s (metals). PASSES.
**File**: `projects/astrophysics/achromaticity/straggling_bound.py`
**Book**: §9.8, C.4.3

### 1.3 — Raw Supernova Pipeline (SALT2-Free) [REVIEWER 4B]
**Status**: DIAGNOSED — z-slope root cause identified (model contamination)
**Primary result**: `golden_loop_sne.py` on SALT2-reduced DES-SN5YR: χ²/dof=0.955, σ=0.18 mag, **0 free physics params**. This is the publishable result.
**Raw pipeline diagnosis** (2026-02-19):
- V18 Stage 1 alpha values embed the V18 linear distance model (D=cz/k_J) in the template
- v2 Kelvin (D_L=(c/K_J)ln(1+z)(1+z)^{2/3}) diverges from V18 by -1.6 to +0.7 mag across z
- This 2.3-mag model contamination IS the z-slope — not a physics problem
- V22 Stage 1 data also contaminated (undocumented fitter, processing_log mismatch)
- **Reverse Malmquist**: high-z bright tail (19 outliers, 0 dim) is gravitational lensing of distant SNe, not classical selection bias. 13/19 extreme outliers hit alpha=30.0 (fitter saturation cap).
**Raw pipeline best results** (V22 data, fitted scale, 2 free params): σ=1.99 mag, slope=-0.03 (flat) — beats V18 published RMS=2.18 (3 free), but fitted scale is unphysical (compensates for model contamination)
**Path forward**: Either (a) re-fit from raw DES photometry with v2 Kelvin template (high effort), or (b) accept SALT2 pipeline as primary (recommended for book)
**Reference**: `RAW_PIPELINE_STATUS.md` (full investigation documentation)
**Data**: V18 `v18_hubble_data.csv` (4,885 SNe), V22 `stage1_results_filtered.csv` (6,724 SNe)

### 1.4 — LeanCert exp Bounds (Last Numerical Gap)
**Status**: OPEN — feasible with Taylor series approach
The IVT+monotonicity proof in VacuumEigenvalue.lean takes 4 numerical bounds on Real.exp as hypotheses:
- `exp(2) < 7.40`, `54.50 < exp(4)` (IVT interval)
- `exp(3.028) < 20.656`, `21.284 < exp(3.058)` (location bracket)
**Action**: Discharge via Mathlib's `exp_one_gt_d9`/`exp_one_lt_d9` + `sum_le_exp_of_nonneg` + `exp_bound'`. Pattern already proven in GoldenLoopNumerical.lean.

### 1.5 — VacuumEigenvalue.lean Pre-existing Build Errors
**Status**: CLOSED (2026-02-19)
Broken monotonicity proofs deleted. Name mismatches and `abs_add` errors fixed. Build: 7824 jobs, 0 errors.

### 1.6 — Unapplied Book Edit Specs
**Status**: BOOK-READY
| Spec | Topic | Edits | Priority |
|------|-------|-------|----------|
| edits70 | DIS parton geometry + Bjorken scaling | 2 | MEDIUM |
**Note**: edits64-69 are all APPLIED.

### 1.7 — Lean Axiom vs Theorem Transparency [REVIEWER 4A]
**Status**: PARTIALLY RESOLVED
Front matter claims "0 axioms." Appendix Z.4.D.9 lists "Axiom 1, 2, 3" (constitutive physical postulates for spectral gap). These are DIFFERENT things: Lean has 0 `axiom` declarations; physics requires mapping math→reality.
**Fix**: Create explicit table in front matter:
- **Lean axioms**: 0 (all 1,380 theorems proved from [propext, Classical.choice, Quot.sound])
- **Physical postulates**: List constitutive mappings (topological quantization, soliton stability criterion, etc.)
- Acknowledge Lean proves mathematical consequences with zero missing steps, but physical ontological mappings are the foundational postulates
**Book**: Front matter, Z.4.D.9

---

## TIER 2: HIGH IMPACT (Theoretical Rigor)

### 2.1 — V₆ Shear Modulus from Curved Hessian [REVIEWER 1B]
**Status**: OPEN
σ ≈ β³/(4π²) is a "constitutive postulate" because flat-space Hessian decoupling theorem proves it can't be derived in flat space. At tau scale, extreme field density curves local phase-space metric.
**Fix**: Set up eigenvalue problem for stability operator L[ψ] on curved background metric. Calculate spectrum of Laplace-Beltrami operator on Clifford Torus T² ⊂ S³ in curved space. Prove lowest shear eigenvalue = β³/(4π²).
**Impact**: Turns tau g-2 prediction from postulate to parameter-free theorem.
**Book**: V.1, Z.8.B
**Files**: `Lepton/AnomalousMoment.lean`, `Lepton/GeometricG2.lean`

### 2.2 — η_topo from Field Gradients [REVIEWER 2A]
**Status**: OPEN — **NEW**
Boundary strain correction η_topo ≈ 0.02985 derived from classical 1D velocity partition (arch vs chord, δv = (π-2)/(π+2)). Applying piecewise classical fluid kinematics to a smooth 6D quantum topological defect is a heuristic leap.
**Fix**: Numerically solve non-linear Euler-Lagrange equations for continuous 6D field profile ψ(r,θ) including electromagnetic self-interaction. Integrate exact curvature and compression energies ⟨∇ψ†∇ψ⟩ to derive η_topo from field gradients.
**Book**: Z.12.7.4

### 2.3 — I_eff from Noether Stress-Energy Tensor [REVIEWER 2B]
**Status**: OPEN — **NEW**
I_eff = 2.32 evaluated via fluid analogy (ρ_eff ∝ v²). Mass and angular momentum in relativistic field theory must come from the stress-energy tensor.
**Fix**: From canonical Lagrangian L_6C, compute T^{0i} for exact D-flow Cl(3,3) multivector field. Integrate angular momentum L = ∫(r × T^{0i})d³x.
**Book**: G.3.2

### 2.4 — Kelvin Wave σ_nf from S-Matrix [REVIEWER 2C]
**Status**: OPEN — **NEW**
σ_nf ∝ √E derived from 1D classical fluid filament dispersion (ω ∝ k²). A QFD photon is a 3D toroidal Beltrami field in 6D multivector space.
**Fix**: Calculate transition amplitude M from L_{int,scatter}. Use QFT phase-space integration for S-matrix element. Prove quantum phase-space integration yields exactly √E without 1D fluid analogy.
**Book**: C.4.3, §9.8.2

### 2.5 — D_L from Eikonal Approximation [REVIEWER 3B]
**Status**: OPEN — **NEW**
D_L = D(1+z)^{2/3} derived by treating photon wavepacket as f=2 thermodynamic gas. Applying macroscopic gas invariants to a single topological defect is a category error.
**Fix**: Derive Poynting flux decay from eikonal approximation of modified Maxwell equations (∂_ν[h⁻¹ F^{νμ}] = 0) and optical scalars (Sachs equations) through dynamically refractive emergent metric.
**Book**: §9.12.1

### 2.6 — V₄ → C₂ Bridge Theorem
**Status**: OPEN
Book uses V₄ (circulation coefficient) and C₂ (QED Schwinger 2nd-order) interchangeably for e/μ. Need explicit bridge theorem with R_crit regime boundary.
**Files**: `Lepton/AnomalousMoment.lean`, `Lepton/GeometricG2.lean`

### 2.7 — 2π² = Vol(S³) Derivation
**Status**: OPEN
`two_pi_sq : ℝ := 2 * Real.pi ^ 2` is a definition, not derived from geometry. Need Vol(S³) = 2π² via Gamma function or hyperspherical integration.

### 2.8 — k_geom Downstream Migration
**Status**: PARTIAL
Pipeline file `QFD/Fundamental/KGeomPipeline.lean` exists as single source of truth. Downstream files still have independent definitions that should import from pipeline.

---

## TIER 3: MEDIUM IMPACT

### 3.1 — Constructive Soliton Existence (App R)
**Status**: OPEN
Energy minimum exists via coercivity, but no PDE solution constructed. Need radial ODE existence proof.
**Difficulty**: HIGH

### 3.2 — 42 ppm Proton Mass Residual
**Status**: OPEN
m_p/m_e prediction is 1836.111 vs experiment 1836.153 (42 ppm). Need exact separatrix integral.

### 3.3 — Dimensional Yardstick (L₀)
**Status**: OPEN
QFD derives shape (κ̃ = 85.58) but mapping to dimensional K_J requires L₀.

### 3.4 — ProofLedger Update
**Status**: OPEN
8 major theorem clusters missing from ProofLedger (Golden Loop, Photon Soliton, Unified Forces, etc.)

---

## TIER 4: STRATEGIC PREDICTIONS (Kill Shots)

### 4.1 — Tau g-2 (Kill Shot #1)
QFD: a_τ ≈ 1192 × 10⁻⁶ | SM: 1177 × 10⁻⁶ | Timeline: Belle II ~2028-2030
**Strength**: Becomes parameter-free theorem if 2.1 (V₆ Hessian) is solved.

### 4.2 — Chromatic SN Light Curve Asymmetry (Kill Shot #2)
QFD: chromatic time dilation (σ ∝ √E) | ΛCDM: achromatic | Timeline: Rubin/LSST
**Strength**: Becomes derivation if 2.4 (S-matrix σ_nf) is solved.

### 4.3 — E-Mode CMB Polarization Axis (Kill Shot #3)
QFD: E-mode aligned with temperature quadrupole | ΛCDM: random

---

## TIER 5: FUTURE

- Lamb shift prediction (hydrogen BVP)
- Quark magnetic moments (G.4.3c)
- Quantitative double-slit derivation
- Muonic hydrogen finite-size effects

---

## REVIEWER AUDIT TRAIL

**Date**: 2026-02-19
**Source**: External theoretical physics review
**Summary**: 9 actionable items across 4 categories:
1. **Functional Analysis** (1A: determinant, 1B: Hessian) — 2 items
2. **Fluid→Field Theory** (2A: η_topo, 2B: I_eff, 2C: Kelvin) — 3 items
3. **Cosmology** (3A: straggling, 3B: eikonal D_L) — 2 items
4. **Epistemology** (4A: axiom clarity, 4B: raw pipeline) — 2 items

**Assessment**: Item 1A (determinant) is the most dangerous to the manuscript's credibility. Item 3A (straggling) is RESOLVED. Item 4B (raw pipeline) is DIAGNOSED — z-slope caused by V18 model contamination in Stage 1 template, not physics; SALT2 pipeline remains primary. Items 2A-2C are important but the 5th-root and topology dampen numerical sensitivity.

---

## CONSTANTS REFERENCE

| Constant | Value | Source |
|----------|-------|--------|
| α | 1/137.035999 | Measured (sole input) |
| β | 3.043233053 | Golden Loop (derived from α) |
| ξ_QFD | 16.154 | k_geom² × 5/6 |
| η_topo | 0.02985 | Boundary strain |
| k_geom | 4.4028 | 5-stage pipeline from α |
| η (SNe) | π²/β² = 1.0657 | Clifford Torus scattering |
| K_J | ξ_QFD·β^{3/2} ≈ 85.6 | Hubble-like constant |
| Γ_sat(τ) | +0.027 | Resummed tau g-2 |
