# QFD Quick Reference Guide
**Version**: 1.0
**Date**: December 23, 2025
**Purpose**: Essential context for AI assistants and collaborators

> **Read this file FIRST at the start of every work session**

---

## Critical Conventions (NEVER CHANGE THESE)

### 1. Clifford Algebra Signature
```
Cl(3,3) with signature: (+, +, +, -, -, -)
         Phase Space:   (x, y, z, px, py, pz)
         NOT:           (t, x, y, z, ...) ← WRONG ordering
```

**Why this matters**:
- Switching to (---+++) will break ALL the math
- This happened in 2024-11 and cost a month of work
- The (+++) phase space convention enables proper 4D Minkowski emergence

**Lean Implementation**: `QFD/GA/Cl33.lean`, signature33 function
**Book Reference**: Appendix Z.2

### 2. Time Convention
```
Time τ > 0: ALWAYS POSITIVE SCALAR (like temperature)
- NOT a dimension you can traverse
- NOT imaginary
- NOT reversible
- Emergent from internal bivector rotation rate
```

**Book Reference**: Aha Moment #16 ("Time is a Local Temperature")
**Lean Reference**: `QFD/Schema/Couplings.lean`, Time as Unitless scalar

### 3. "Imaginary" i is BANNED
```
WRONG: ψ = a + bi (complex number with imaginary i)
RIGHT: ψ = scalar + bivector·B where B² = -1 (geometric object)
```

**Physical meaning**: B is a real rotation plane, not an abstract symbol
**Book Reference**: Aha Moment #20 ("Real 'i'")
**Lean Reference**: `QFD/GA/Cl33.lean`, bivector products

### 4. Core Compression Law (CCL) Parameters
```lean
Z_ideal = c₁·A^(2/3) + c₂·A

c₁ ∈ (0, 1.5)      // Surface tension (open interval)
c₂ ∈ [0.2, 0.5]    // Volume packing (closed interval)

Fitted values (Phase 1, 2550 nuclides, AME2020):
c₁ = 0.496296
c₂ = 0.323671
R² = 0.9832
```

**Status**:
- ✅ Bounds PROVEN in Lean (CoreCompressionLaw.lean:26-42)
- ✅ Fitted values satisfy bounds (theorem phase1_satisfies_constraints)
- ⚠️ Functional form is FITTED, not derived from first principles

**Book Reference**: Chapter 8, Appendix N
**Lean Reference**: `QFD/Nuclear/CoreCompressionLaw.lean`
**Python Reference**: `V22_Nuclear_Analysis/scripts/v22_ccl_fit_lean_constrained.py`

---

## Project Structure Overview

### Repository Layout
```
QFD_SpectralGap/
├── projects/Lean4/QFD/              ← 213 theorems, 0 sorries (core modules)
│   ├── ProofLedger.lean             ← MASTER INDEX (read this!)
│   ├── CLAIMS_INDEX.txt             ← All theorem names
│   ├── CONCERN_CATEGORIES.md        ← Critical assumptions
│   ├── Nuclear/                     ← CCL, TimeCliff, decay
│   ├── Cosmology/                   ← VacuumRefraction, ScatteringBias
│   ├── Lepton/                      ← MassSpectrum, GeometricAnomaly
│   ├── GA/Cl33.lean                 ← Clifford algebra (0 axioms!)
│   └── Schema/                      ← Parameter constraints
├── V22_Nuclear_Analysis/            ← Nuclear paper (publication-ready)
├── V22_Supernova_Analysis/          ← Cosmology paper
├── V22_Lepton_Analysis/             ← Lepton paper
├── schema/v0/                       ← JSON schemas, solvers
└── QFD Book Dec 21 2025.txt         ← 213K words, 40+ Aha Moments
```

### Book Structure
- **Chapters 1-9**: ~200 pages, conceptual explanations, minimal math
- **Appendices A-Z**: ~550 pages, rigorous derivations, heavy math
- **40+ Aha Moments**: Summary table (lines 56-195 in book file)

### Key Appendices (Most Referenced)
| Appendix | Topic | Lean Formalization |
|----------|-------|-------------------|
| **A** | Adjoint Stability & Energy Positivity | AdjointStability_Complete.lean |
| **Z.2** | Cl(3,3) Structure | GA/Cl33.lean |
| **Z.4** | Spacetime Emergence (Centralizer) | SpacetimeEmergence_Complete.lean |
| **Z.15** | [Bivector classes] | BivectorClasses_Complete.lean |
| **N** | Nuclear Physics (CCL, Time Cliff) | Nuclear/*.lean |
| **O** | Core Compression Law Details | Nuclear/CoreCompression.lean |
| **Q** | Charge Quantization | Soliton/Quantization.lean |

---

## Top 20 Claims ↔ Lean Proofs Map

### Nuclear Physics
1. **"CCL parameter bounds are proven from soliton stability"**
   - Lean: `QFD.Nuclear.CoreCompressionLaw.stability_requires_bounds` (line 104)
   - File: `QFD/Nuclear/CoreCompressionLaw.lean:104`
   - Status: ✅ PROVEN (0 sorries)

2. **"Energy is minimized at the CCL backbone"**
   - Lean: `QFD.Nuclear.CoreCompression.energy_minimized_at_backbone` (line 75)
   - File: `QFD/Nuclear/CoreCompression.lean:75`
   - Status: ✅ PROVEN

3. **"Beta decay reduces charge stress toward backbone"**
   - Lean: `QFD.Nuclear.CoreCompression.beta_decay_reduces_stress` (line 132)
   - File: `QFD/Nuclear/CoreCompression.lean:132`
   - Status: ✅ PROVEN

4. **"Phase 1 CCL fit satisfies theoretical constraints"**
   - Lean: `QFD.Nuclear.CoreCompressionLaw.phase1_satisfies_constraints` (line 165)
   - File: `QFD/Nuclear/CoreCompressionLaw.lean:165`
   - Status: ✅ PROVEN

### Foundations
5. **"Energy is positive definite from QFD adjoint"**
   - Lean: `QFD.AdjointStability_Complete.energy_is_positive_definite` (line 157)
   - File: `QFD/AdjointStability_Complete.lean:157`
   - Status: ✅ PROVEN
   - Book: Appendix A.2.2

6. **"Cl(3,3) generators square to signature"**
   - Lean: `QFD.GA.generator_squares_to_signature` (line 130)
   - File: `QFD/GA/Cl33.lean:130`
   - Status: ✅ PROVEN (was axiom, now theorem!)

7. **"Generators anticommute for i ≠ j"**
   - Lean: `QFD.GA.generators_anticommute` (line 163)
   - File: `QFD/GA/Cl33.lean:163`
   - Status: ✅ PROVEN from Mathlib anchors

8. **"Emergent spacetime has Minkowski signature"**
   - Lean: `QFD.SpacetimeEmergence_Complete.emergent_signature_is_minkowski` (line 245)
   - File: `QFD/SpacetimeEmergence_Complete.lean:245`
   - Status: ⚠️ PARTIAL (proves generators in centralizer, not full equivalence)
   - Book: Appendix Z.4.A

### Charge & Solitons
9. **"Charge quantization factor is -40"**
   - Lean: `QFD.Soliton.GaussianMoments.ricker_moment_value` (line 123)
   - File: `QFD/Soliton/GaussianMoments.lean:123`
   - Status: ✅ PROVEN from Gamma functions

10. **"Ricker profile bounded: S(x) ≤ 1"**
    - Lean: `QFD.Soliton.RickerAnalysis.S_le_one` (line 42)
    - File: `QFD/Soliton/RickerAnalysis.lean:42`
    - Status: ✅ PROVEN

11. **"Soliton amplitude satisfies hard wall constraint"**
    - Lean: `QFD.Soliton.RickerAnalysis.soliton_always_admissible_aux` (line 319)
    - File: `QFD/Soliton/RickerAnalysis.lean:319`
    - Status: ✅ PROVEN with amplitude bound

### Schema & Consistency
12. **"Parameter space is nonempty (has valid solutions)"**
    - Lean: `QFD.Nuclear.CoreCompressionLaw.ccl_parameter_space_nonempty` (line 52)
    - File: `QFD/Nuclear/CoreCompressionLaw.lean:52`
    - Status: ✅ PROVEN (constructive proof)

13. **"Parameter space is bounded (optimization converges)"**
    - Lean: `QFD.Nuclear.CoreCompressionLaw.ccl_parameter_space_bounded` (line 64)
    - File: `QFD/Nuclear/CoreCompressionLaw.lean:64`
    - Status: ✅ PROVEN

14. **"Theory is falsifiable (exists invalid parameters)"**
    - Lean: `QFD.Nuclear.CoreCompressionLaw.theory_is_falsifiable` (line 189)
    - File: `QFD/Nuclear/CoreCompressionLaw.lean:189`
    - Status: ✅ PROVEN (c₂ = 0.1 violates theory)

### Cosmology
15. **"Vacuum refraction index relates to field density"**
    - Lean: `QFD.Cosmology.VacuumRefraction.refraction_from_density`
    - File: `QFD/Cosmology/VacuumRefraction.lean`
    - Status: ✅ Definition + basic properties proven

16. **"Scattering bias causes redshift without expansion"**
    - Lean: `QFD.Cosmology.ScatteringBias.redshift_from_scattering`
    - File: `QFD/Cosmology/ScatteringBias.lean`
    - Status: ⚠️ Framework defined, empirical validation in Python

### Leptons
17. **"Lepton mass spectrum from geometric isomers"**
    - Lean: `QFD.Lepton.MassSpectrum.*`
    - File: `QFD/Lepton/MassSpectrum.lean`
    - Status: ⚠️ Structure defined, mass calculation in progress

18. **"Anomalous magnetic moment from geometric correction"**
    - Lean: `QFD.Lepton.GeometricAnomaly.*`
    - File: `QFD/Lepton/GeometricAnomaly.lean`
    - Status: ⚠️ Framework, numerical match in Python

### Classical Limits
19. **"QFD matches Schwarzschild metric to first order"**
    - Lean: `QFD.Gravity.SchwarzschildLink.qfd_matches_schwarzschild_first_order`
    - File: `QFD/Gravity/SchwarzschildLink.lean`
    - Status: ✅ PROVEN for weak field limit

20. **"Shell theorem emerges from QFD field equations"**
    - Lean: `QFD.Electron.AxisAlignment.shell_theorem_emergence`
    - File: `QFD/Electron/AxisAlignment.lean`
    - Status: ⚠️ Sketch proof, full derivation in progress

---

## What's PROVEN vs. What's PHENOMENOLOGICAL

### ✅ PROVEN in Lean (Mathematical Certainty)

**Foundation**:
- Cl(3,3) Clifford algebra structure (0 axioms, all theorems!)
- Generator squaring and anticommutation
- Energy positivity from adjoint construction
- Charge quantization factor (-40 from Gaussian integrals)

**Nuclear Physics**:
- CCL parameter bounds from soliton stability
- Energy minimization at backbone
- Beta decay stress reduction
- Phase 1 fit satisfies constraints
- Theory is falsifiable

**Soliton Analysis**:
- Ricker profile bounds
- Hard wall constraints
- Gaussian moment integrals

**Schema**:
- Parameter space nonempty, bounded, consistent
- All constraints mutually compatible

### ⚠️ PHENOMENOLOGICAL (Empirical Fits with Rigorous Constraints)

**Nuclear Physics**:
- CCL functional form Z = c₁·A^(2/3) + c₂·A (fitted to data, not derived)
- Decay modulus µ(A) mass dependence (observed, not proven)
- Harmonic → linear envelope transition (empirical observation)

**Cosmology**:
- Scattering bias parameters (fitted to SNe Ia data)
- CMB temperature (steady-state assumption, not proven)
- Vacuum refraction coefficients (fitted to lensing)

**Leptons**:
- Mass ratios m_µ/m_e, m_τ/m_e (geometric model, numerical match)
- g-2 anomaly correction (calculated, matches experiment)

**Astrophysics**:
- Black hole ejection mechanism (proposed, not derived)
- Rift dynamics (qualitative model)
- Galaxy formation timescales (observational match)

### ❌ UNPROVEN (Speculative/In Progress)

**Major Claims Without Lean Proofs**:
- "All forces are gradients of one field" (philosophical framework)
- "Black holes recycle matter via Rifts" (qualitative mechanism)
- "CMB is present-day equilibrium, not Big Bang relic" (interpretation)
- "Neutrino flavors from geometric superposition" (model proposed)
- "Superconductivity from enforced entanglement" (speculative)

---

## Common AI Mistakes to AVOID

### 1. Signature Flipping ⚠️ CRITICAL
**WRONG**: "Let's use Cl(3,3) with (---+++) to match standard conventions"
**RIGHT**: "QFD uses (+++) for phase space, (---) for momentum"
**Why**: Changing this breaks 213 Lean theorems and all Python code
**Detection**: Check `signature33` in Cl33.lean = [1, 1, 1, -1, -1, -1]

### 2. Time Sign Confusion
**WRONG**: "Time can be negative in QFD"
**RIGHT**: "Time τ is always positive (like temperature)"
**Why**: Negative time creates causality violations
**Detection**: All `Time` types in Schema have `val > 0` constraint

### 3. Claiming CCL is "Derived"
**WRONG**: "QFD derives the CCL from first principles"
**RIGHT**: "QFD fits the CCL with Lean-proven parameter bounds"
**Why**: Functional form A^(2/3) + A is empirical
**Correct Claim**: "Parameters proven to satisfy soliton stability constraints"

### 4. Confusing "Proven" vs. "Fitted"
**WRONG**: "The decay modulus µ(A) is proven to decrease with mass"
**RIGHT**: "The decay modulus µ(A) is observed to decrease (Light: 1.99, Heavy: 1.17)"
**Why**: No Lean theorem for µ(A) scaling exists yet
**Path to Proof**: Need WKB barrier penetration + envelope analysis theorems

### 5. Using "Imaginary i" Language
**WRONG**: "ψ = a + bi where i² = -1"
**RIGHT**: "ψ = scalar + bivector, where bivector² = -1"
**Why**: QFD replaces complex numbers with geometric algebra
**Book Reference**: Aha Moment #20

### 6. Overstating Centralizer Proof
**WRONG**: "Lean proves Cent(B) ≅ Cl(3,1)"
**RIGHT**: "Lean proves Cl(3,1) generators lie in Cent(B)"
**Why**: Containment proven, not isomorphism
**See**: CONCERN_CATEGORIES.md, PHASE_CENTRALIZER

### 7. Mixing "Binding Energy" Language
**WRONG**: "Nuclei are bound by negative binding energy"
**RIGHT**: "Nuclei are stable configurations minimizing elastic strain"
**Why**: QFD doesn't use binding energy concept
**See**: V22_Nuclear_Analysis/PHASE1_V22_COMPARISON.md:62-88

### 8. Adding Parameters Without Justification
**WRONG**: "Let's add a parameter c₃ for shell corrections"
**RIGHT**: "Current CCL uses 2 parameters; shell terms are future work"
**Why**: Every parameter must have Lean-proven bounds
**Process**: Define in Lean → Prove bounds → Add to Schema → Implement in Python

---

## Complete Parameter List (17 Free + 5 Fixed)

### Grand Solver Parameters

**Source**: `QFD/Schema/Couplings.lean` (line 72: `count_parameters = 17`)

**Total**: 17 free parameters to solve + 5 fixed constants

---

### 📦 **NUCLEAR PARAMETERS (7)**

| # | Symbol | Name | Type | Bounds (Lean-proven) | Fitted/Status |
|---|--------|------|------|---------------------|---------------|
| 1 | **c₁** | Surface term | Unitless | (0.5, 1.5) | **0.496** ✅ Fitted |
| 2 | **c₂** | Volume term | Unitless | (0.0, 0.1) | **0.324** ✅ Fitted |
| 3 | **V₄** | Potential depth | Energy (eV) | (10⁶, 10⁹) | ~10⁷ ⚠️ To solve |
| 4 | **k_c2** | Mass scale | Mass (eV) | (10⁵, 10⁷) | ~10⁶ ⚠️ To solve |
| 5 | **α_n** | Nuclear fine structure | Unitless | (1.0, 10.0) | **~3.5** ⚠️ Genesis |
| 6 | **β_n** | Asymmetry coupling | Unitless | (1.0, 10.0) | **~3.9** ⚠️ Genesis |
| 7 | **γ_e** | Geometric shielding | Unitless | (1.0, 10.0) | **~5.5** ⚠️ Genesis |

**Notes**:
- c₁, c₂: Core Compression Law, Lean-proven bounds, fitted to AME2020 (R² = 98.3%)
- V₄: Nuclear potential well depth (Time Cliff model)
- α_n, β_n, γ_e: "Genesis Constants" from hydrogen spectrum fit (~2.5, ~3.9, ~5.5)
- Constraint: |α_n - 3.5| < 1.0, |β_n - 3.9| < 1.0, |γ_e - 5.5| < 2.0

**Lean File**: `QFD/Schema/Couplings.lean:33-40`, `QFD/Schema/Constraints.lean:31-62`

---

### 🌌 **COSMOLOGY PARAMETERS (5)**

| # | Symbol | Name | Type | Bounds (Lean-proven) | Fitted/Status |
|---|--------|------|------|---------------------|---------------|
| 8 | **k_J** | Hubble equivalent | km/s/Mpc | (50, 100) | ~70 ⚠️ To solve |
| 9 | **η'** | Conformal time scale | Unitless | [0.0, 0.1) | ⚠️ To solve |
| 10 | **A_plasma** | SNe plasma dispersion | Unitless | [0.0, 1.0) | ⚠️ To solve |
| 11 | **ρ_vac** | Vacuum energy density | kg/m³ | (0, 10⁻²⁶) | ⚠️ To solve |
| 12 | **w_dark** | Dark energy EOS | Unitless | (-2.0, 0.0) | ⚠️ To solve |

**Notes**:
- k_J: NOT the Hubble constant (QFD has no expansion), but apparent rate from scattering
- A_plasma: Flux-dependent scattering coefficient for supernovae dimming
- ρ_vac: Vacuum field density (steady-state, not dark energy)
- w_dark: Equation of state parameter (placeholder for future vacuum dynamics)

**Lean File**: `QFD/Schema/Couplings.lean:43-49`, `QFD/Schema/Constraints.lean:64-83`

---

### ⚛️ **PARTICLE PARAMETERS (5)**

| # | Symbol | Name | Type | Bounds (Lean-proven) | Fitted/Status |
|---|--------|------|------|---------------------|---------------|
| 13 | **g_c** | Geometric charge coupling | Unitless | [0.9, 1.0] | ~0.95 ⚠️ To solve |
| 14 | **V₂** | Weak potential scale | Energy (eV) | (0, 10¹²) | ⚠️ To solve |
| 15 | **λ_R** | Ricker wavelet width | Unitless | (0.1, 10.0) | ⚠️ To solve |
| 16 | **μ_e** | Electron mass seed | Mass (eV) | (5×10⁵, 6×10⁵) | **511 keV** ✅ Known |
| 17 | **μ_ν** | Neutrino mass seed | Mass (eV) | (10⁻³, 1.0) | ~0.01 ⚠️ To solve |

**Notes**:
- g_c: Charge coupling strength (near unity = maximum geometric locking)
- V₂: Weak interaction potential (cf. V₄ for nuclear)
- λ_R: Ricker wavelet shape parameter (soliton width)
- μ_e: Electron mass (known, serves as calibration)
- μ_ν: Neutrino mass scale (constrained by oscillation experiments)

**Lean File**: `QFD/Schema/Couplings.lean:51-56`, `QFD/Schema/Constraints.lean:85-105`

---

### 🔒 **FIXED CONSTANTS (5)** - Not Free Parameters

| Symbol | Name | Value (SI) | Status |
|--------|------|------------|--------|
| **c** | Speed of light (vacuum) | 299,792,458 m/s | Fixed |
| **G** | Gravitational constant | 6.674×10⁻¹¹ m³/kg/s² | Fixed |
| **ℏ** | Reduced Planck constant | 1.055×10⁻³⁴ J·s | Fixed |
| **e** | Elementary charge | 1.602×10⁻¹⁹ C | Fixed |
| **k_B** | Boltzmann constant | 1.381×10⁻²³ J/K | Fixed |

**Note**: In QFD philosophy, these are ultimately derivable from ψ-field dynamics, but treated as fixed inputs for now.

**Lean File**: `QFD/Schema/Couplings.lean:25-30`

---

## Parameter Status Summary

### ✅ **FITTED & VALIDATED** (2 parameters)
- c₁, c₂: Core Compression Law from AME2020 (2,550 nuclides, R² = 98.3%)

### ⚠️ **PARTIALLY CONSTRAINED** (4 parameters)
- α_n, β_n, γ_e: Genesis constants from hydrogen fit (~3.5, ~3.9, ~5.5)
- μ_e: Electron mass (known from experiment = 511 keV)

### 🔴 **TO BE SOLVED** (11 parameters)
- V₄, k_c2: Nuclear potential & mass scale
- k_J, η', A_plasma, ρ_vac, w_dark: Cosmology sector (5 params)
- g_c, V₂, λ_R, μ_ν: Particle sector (4 params)

**Grand Solver Goal**: Simultaneously fit all 17 parameters to:
1. Nuclear binding energies (AME2020: 5,800 isotopes)
2. Supernova distances (DES-SN5YR: 1,635 SNe Ia)
3. CMB power spectrum (Planck 2018)
4. Lepton masses (e, μ, τ, ν)
5. Decay rates and half-lives

---

## Cross-Domain Consistency (Lean-Proven Constraints)

### Nuclear ↔ Particle Consistency
```lean
nuclear_particle_consistent:
  3.0 < (γ_e · g_c) < 8.0
```
Geometric shielding must be compatible with charge coupling.

### Cosmology ↔ Particle Consistency
```lean
cosmo_particle_consistent:
  μ_e / ρ_vac > 0  -- Placeholder for vacuum-mass relation
```

**Lean File**: `QFD/Schema/Constraints.lean:117-131`

---

## Parameter Sensitivity (From Schema)

**HIGH SENSITIVITY** (>10% effect on observables):
- V₄ (nuclear well depth)
- k_J (Hubble-like parameter)
- g_c (geometric charge)
- μ_e (electron mass calibration)

**MEDIUM SENSITIVITY** (1-10% effect):
- α_n, c₁ (nuclear couplings)
- η' (conformal time scale)

**LOW SENSITIVITY** (<1% effect):
- λ_R (wavelet width)
- w_dark (dark EOS)

**Lean File**: `QFD/Schema/Constraints.lean:196-213`

---

## Charge Quantization (Not a Free Parameter)

```
Q_vortex = 40·v₀·σ⁶        // From Ricker integral = -40
```

The factor **40** is mathematically derived (Lean-proven in GaussianMoments.lean), not fitted.

---

## How Parameters Are Used

### Nuclear Binding Energy
```
E_bind = f(c₁, c₂, V₄, k_c2, α_n, β_n, γ_e | A, Z)
```

### Cosmological Observables
```
SNe distance modulus = f(k_J, η', A_plasma | z)
CMB power spectrum  = f(k_J, ρ_vac, w_dark | ℓ)
```

### Particle Masses
```
m_lepton = f(g_c, V₂, λ_R, μ_e, μ_ν | geometry)
```

### Cross-Domain Tests
- Nuclear + Particle: Hydrogen spectrum → α_n, β_n, γ_e
- Cosmology + Nuclear: Supernova nucleosynthesis
- All domains: Unified field gradient predicts all forces

---

## File Navigation Shortcuts

### Most Important Files (Read These First)
1. **`QFD/ProofLedger.lean`** - Master claim→theorem map (5 min read)
2. **`QFD/CONCERN_CATEGORIES.md`** - Critical assumptions (3 min read)
3. **`QFD/AXIOM_ELIMINATION_STATUS.md`** - What's proven vs. axiomatized (5 min)
4. **`QFD/Nuclear/CoreCompressionLaw.lean`** - Nuclear parameter proofs (10 min)
5. **`V22_Nuclear_Analysis/PHASE1_V22_COMPARISON.md`** - Best documented analysis

### Quick Lookups
- **All theorem names**: `QFD/CLAIMS_INDEX.txt` (grep this file)
- **Lean↔Python map**: `QFD/LEAN_PYTHON_CROSSREF.md`
- **Signature definition**: `QFD/GA/Cl33.lean:40-55`
- **Parameter bounds**: `QFD/Nuclear/CoreCompressionLaw.lean:26-42`
- **Energy positivity**: `QFD/AdjointStability_Complete.lean:157`

### Paper-Specific
- **V22 Nuclear**: `V22_Nuclear_Analysis/PHASE1_V22_COMPARISON.md`
- **Decay Corridor**: `/mnt/c/Users/TracyMc/Downloads/Decay_Corridor (7) (1).pdf`
- **Book**: `/mnt/c/Users/TracyMc/Downloads/7.2 QFD Book Dec 21 2025.txt`

---

## Validation Commands (Run Before Submitting Papers)

### Check Lean Proofs Build
```bash
cd /home/tracy/development/QFD_SpectralGap/projects/Lean4
lake build QFD
# Should show: Build completed successfully (no errors)
```

### Check for Sorries
```bash
rg "sorry" QFD --glob "*.lean"
# Should return: empty (0 sorries in core modules)
```

### Check Signature Convention
```bash
rg "signature33" QFD/GA/Cl33.lean -A 2
# Should show: [1, 1, 1, -1, -1, -1]
```

### Check CCL Parameters Match
```bash
# Lean bounds
rg "C1_MIN|C1_MAX|C2_MIN|C2_MAX" QFD/Nuclear/CoreCompressionLaw.lean

# Python bounds (should match)
rg "C1_MIN|C1_MAX|C2_MIN|C2_MAX" V22_Nuclear_Analysis/scripts/
```

### Verify No Axioms in Core Modules
```bash
rg "^axiom " QFD/GA/Cl33.lean QFD/EmergentAlgebra.lean QFD/Nuclear/CoreCompressionLaw.lean
# Should return: empty (all are theorems now)
```

---

## Session Workflow Template

### Starting a New Session
1. **Read this file** (QFD_QUICK_REFERENCE.md)
2. **Ask user**: "What are we working on today?"
3. **Get context**: "Which files should I read?"
4. **Clarify scope**: "Are we proving, implementing, or documenting?"
5. **Check conventions**: Verify signature, time sign, parameter bounds

### During Session
- **Before suggesting changes**: Check if it contradicts a convention
- **Before claiming "proven"**: Grep ProofLedger.lean for theorem
- **Before adding parameters**: Ask about Lean bounds
- **When uncertain**: Say "I don't see this in my context, can you point me to the file?"

### Ending Session
- **Document decisions**: What conventions were established?
- **List changes**: Which files were modified?
- **Note for next time**: What should be remembered?
- **Suggest validation**: Which checks should the user run?

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial creation from project analysis |

---

## Quick Answers to Common Questions

**Q: Is QFD proven?**
A: The math is proven (Lean theorems). The physics is testable (makes predictions). Not all physics claims have Lean proofs yet.

**Q: What's the status of axioms?**
A: 5/5 target axioms eliminated. Core modules have 0 sorries, 0 axioms (except 1 helper for general Gaussian moments).

**Q: Can I add a new parameter?**
A: Only if you: (1) Define it in Lean, (2) Prove its bounds, (3) Add to Schema, (4) Implement in Python, (5) Document in ProofLedger.

**Q: Where's the decay modulus µ(A) proven?**
A: It's not (yet). It's an empirical observation. To prove it, need WKB + envelope analysis theorems.

**Q: Why Cl(3,3) and not Cl(1,3)?**
A: Phase space (x,y,z,px,py,pz) is 6D. Spacetime emerges as 4D Minkowski from the centralizer. See Appendix Z.4.

**Q: Is the CCL derived or fitted?**
A: Fitted functional form, proven parameter bounds. Future work: Derive functional form from soliton field equations.

**Q: How do I cite Lean proofs in a paper?**
A: See ProofLedger.lean for theorem names. Format: "`theorem_name` (File.lean:line) proves [claim]"

**Q: What if a paper claim contradicts a Lean proof?**
A: Lean is correct (it's machine-verified). Revise the paper claim or find the gap in Lean coverage.

---

**END OF QUICK REFERENCE**

*For detailed proofs, read ProofLedger.lean*
*For concerns/assumptions, read CONCERN_CATEGORIES.md*
*For the full theory, read the 213K-word book*

This reference is your map. The Lean proofs are your foundation. The book is your vision.
