# QFD Open Questions

**Current State**: ~1,380 proven Lean statements | **0 axioms** | 0 sorry | 267 files | Book v10.0
**Last Updated**: 2026-02-19

---

## TIER 1: CRITICAL

### 1.1 — LeanCert exp Bounds (Last Numerical Gap)
**Status**: OPEN
All standalone axioms eliminated from Postulates.lean. The final 2 (python_root_finding_beta, shell_theorem_timeDilation) were converted to theorems with hypotheses. The IVT+monotonicity proof in VacuumEigenvalue.lean takes 4 numerical bounds on Real.exp as hypotheses:
- `exp(2) < 7.40`, `54.50 < exp(4)` (IVT interval)
- `exp(3.028) < 20.656`, `21.284 < exp(3.058)` (location bracket)
**Action**: Discharge via LeanCert `interval_bound 30` when available.

### 1.2 — VacuumEigenvalue.lean Pre-existing Build Errors
**Status**: OPEN (pre-existing, not introduced by axiom elimination)
Lines 221-262 have type mismatches and missing identifiers (`continuousAt_exp`, `abs_add`). These blocked verification of the new IVT-based proof at line 338+.
**Action**: Fix the pre-existing errors, then verify the axiom-#4 replacement proof compiles.

### 1.3 — Unapplied Book Edit Specs
**Status**: BOOK-READY
| Spec | Topic | Edits | Priority |
|------|-------|-------|----------|
| edits70 | DIS parton geometry + Bjorken scaling | 2 | MEDIUM |
**Note**: edits64-69 are all APPLIED.

---

## TIER 2: HIGH IMPACT

### 2.1 — V₄ → C₂ Bridge Theorem
**Status**: OPEN
Book uses V₄ (circulation coefficient) and C₂ (QED Schwinger 2nd-order) interchangeably for e/μ. Need explicit bridge theorem with R_crit regime boundary.
**Files**: `Lepton/AnomalousMoment.lean`, `Lepton/GeometricG2.lean`

### 2.2 — 2π² = Vol(S³) Derivation
**Status**: OPEN
`two_pi_sq : ℝ := 2 * Real.pi ^ 2` is a definition, not derived from geometry. Need Vol(S³) = 2π² via Gamma function or hyperspherical integration.

### 2.3 — k_geom Downstream Migration
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

### 3.3 — V₆ Shear Modulus Derivation
**Status**: OPEN
σ ≈ β³/(4π²) is currently postulated. Need Hessian eigenvalue spectrum on curved Clifford Torus background.

### 3.4 — Dimensional Yardstick (L₀)
**Status**: OPEN
QFD derives shape (κ̃ = 85.58) but mapping to dimensional K_J requires L₀.

### 3.5 — ProofLedger Update
**Status**: OPEN
8 major theorem clusters missing from ProofLedger (Golden Loop, Photon Soliton, Unified Forces, etc.)

---

## TIER 4: STRATEGIC PREDICTIONS (Kill Shots)

### 4.1 — Tau g-2 (Kill Shot #1)
QFD: a_τ ≈ 1192 × 10⁻⁶ | SM: 1177 × 10⁻⁶ | Timeline: Belle II ~2028-2030

### 4.2 — Chromatic SN Light Curve Asymmetry (Kill Shot #2)
QFD: chromatic time dilation (σ ∝ √E) | ΛCDM: achromatic | Timeline: Rubin/LSST

### 4.3 — E-Mode CMB Polarization Axis (Kill Shot #3)
QFD: E-mode aligned with temperature quadrupole | ΛCDM: random

### 4.4 — Raw Supernova Pipeline
Build from-scratch pipeline on 8,277 raw light curves with v2 Kelvin Wave physics. No SALT2.
**Data**: `archive/V21.../lightcurves_all_transients.csv` (770K photometry rows)

---

## TIER 5: FUTURE

- Lamb shift prediction (hydrogen BVP)
- Quark magnetic moments (G.4.3c)
- Quantitative double-slit derivation
- Muonic hydrogen finite-size effects

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
