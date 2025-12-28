# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Documentation

**Read these guides before starting work**:

1. **AI_WORKFLOW.md** - Mandatory workflow for all AI assistants (build verification, error handling)
2. **MATHLIB_SEARCH_GUIDE.md** - How to find Mathlib theorems, handle type system issues (NEW!)
3. **COMPLETE_GUIDE.md** - Full system architecture, proof patterns, theorem index
4. **PROTECTED_FILES.md** - Core infrastructure files (DO NOT MODIFY)

**Quick references**:
- **WORK_QUEUE.md** - Prioritized task list
- **BUILD_STATUS.md** - Current build health, sorry count
- **CLAIMS_INDEX.txt** - All 482 proven theorems (grep-able)

## Repository Overview

This is a **Lean 4 formalization** of Quantum Field Dynamics (QFD) theorems, proving spacetime emergence, cosmology, nuclear physics, and particle physics using the Clifford algebra Cl(3,3). The project contains **482 proven statements** (364 theorems + 118 lemmas) across **180 Lean files** - **109 NEW FILES created on 2025-12-27!**

**Critical Context**: This is a formal proof repository where correctness is paramount. Every change must be verified by the Lean compiler via `lake build`.

## Essential Build Commands

### ⚠️ CRITICAL BUILD WARNING - PREVENT OOM CRASHES ⚠️

**NEVER RUN MULTIPLE `lake build` COMMANDS IN PARALLEL OR SIMULTANEOUSLY**

Running multiple builds at the same time will cause **each build to compile Mathlib independently**, resulting in:
- ❌ **Out-Of-Memory (OOM) system crashes** - your session WILL die
- ❌ **Corrupted build artifacts** - broken state, hard to recover
- ❌ **Lost session context** - you'll lose sync with your work
- ❌ **Wasted hours** - Mathlib takes 10-30 minutes to compile

**This happened on 2025-12-27 morning and caused major session desync.**

### ✅ CORRECT: Sequential Builds

```bash
# Use && to chain builds sequentially
lake build QFD.Module1 && lake build QFD.Module2 && lake build QFD.Module3

# Or run one at a time, wait for completion
lake build QFD.Module.FileName  # Wait for this to finish
lake build QFD.AnotherModule    # Then run this
```

### ❌ WRONG: Parallel Builds

```bash
# NEVER do this - will cause OOM crash!
lake build QFD.Module1 &
lake build QFD.Module2 &
lake build QFD.Module3 &

# NEVER run builds in separate terminal windows simultaneously
# NEVER use parallel execution tools with lake build
```

### Build & Verify

```bash
# Build a specific module (use this after EVERY file modification)
lake build QFD.Module.FileName

# Build entire QFD library
lake build QFD

# Verify paper-ready cosmology proofs
lake build QFD.Cosmology.AxisExtraction && lake build QFD.Cosmology.CoaxialAlignment

# Verify spacetime emergence
lake build QFD.EmergentAlgebra && lake build QFD.SpectralGap
```

### Build Status Check

```bash
# Check for incomplete proofs (sorries)
grep -r "sorry" QFD/**/*.lean

# Count sorries in specific file
grep -c "sorry" QFD/Path/To/File.lean

# Find all theorems (for indexing)
rg -n "^theorem|^lemma" QFD --include="*.lean"
```

### Critical Build Warning

**NEVER RUN MULTIPLE `lake build` COMMANDS IN PARALLEL**. Running multiple builds simultaneously will:
- Cause each to build Mathlib independently
- Result in out-of-memory (OOM) errors and system crashes
- Corrupt build artifacts
- Waste hours of compilation time

✅ **CORRECT**: `lake build QFD.Module1 && lake build QFD.Module2`
❌ **WRONG**: Running builds in separate terminals or using `&` for parallelization

## Architecture: Clifford Algebra Foundation

### Core Geometric Algebra (GA) Module - The Foundation

**Everything builds on `QFD/GA/Cl33.lean`** - the Clifford algebra Cl(3,3) implementation.

**Signature**: `(+,+,+,-,-,-)` - Three spacelike, three timelike dimensions
**Generators**: `e₀, e₁, e₂` (spatial), `e₃` (time), `e₄, e₅` (internal/hidden)

**Key Properties**:
- Anticommutation: `eᵢ * eⱼ = -eⱼ * eᵢ` for i ≠ j
- Squares: `eᵢ * eᵢ = signature(i)` (±1)
- Basis vector: `e i := ι33 (basis_vector i)` where `ι33` is the canonical injection

**Infrastructure Hierarchy** (DO NOT MODIFY - see PROTECTED_FILES.md):
1. `QFD/GA/Cl33.lean` - Base algebra definition
2. `QFD/GA/BasisOperations.lean` - Core lemmas (`basis_sq`, `basis_anticomm`)
3. `QFD/GA/BasisProducts.lean` - Pre-computed product library (207 products)
4. `QFD/GA/BasisReduction.lean` - **AUTOMATION ENGINE** - `clifford_simp` tactic

**Critical Insight**: The `clifford_simp` tactic automates Clifford algebra simplification. Instead of writing 50-line manual calc chains, use:

```lean
-- Before (manual)
calc e 0 * e 3 * e 0
    = e 0 * (e 3 * e 0) := by rw [mul_assoc]
  _ = e 0 * (-(e 0 * e 3)) := by rw [basis_anticomm]
  _ = -(e 0 * e 0 * e 3) := by rw [mul_neg, mul_assoc]
  _ = -(1 * e 3) := by rw [basis_sq]
  _ = -e 3 := by simp

-- After (automated)
by clifford_simp
```

### Spacetime Emergence Architecture

**The Central Theorem**: 4D Minkowski spacetime emerges algebraically from Cl(3,3) when particles have internal rotation `B = e₄ ∧ e₅`.

**Key Files**:
- `EmergentAlgebra.lean` - **Centralizer theorem**: Visible spacetime = elements that commute with B
- `SpectralGap.lean` - Dynamical suppression of extra dimensions (energy gap proof)
- `SpacetimeEmergence_Complete.lean` - Complete proof with all infrastructure

**Proof Flow**:
1. Define internal bivector: `B = e₄ * e₅`
2. Prove spatial generators commute: `[e₀, B] = [e₁, B] = [e₂, B] = 0`
3. Prove time generator commutes: `[e₃, B] = 0`
4. Prove internal generators anticommute: `{e₄, B} = {e₅, B} = 0`
5. Extract signature: `(e₀² = 1, e₁² = 1, e₂² = 1, e₃² = -1)` → Minkowski (+,+,+,-)

### Charge Quantization Architecture

**The Mechanism**: Topological vortex boundary conditions → discrete charge values.

**Key Files**:
- `Charge/Quantization.lean` - Topological charge quantization
- `Charge/Coulomb.lean` - Force law from geometry
- `Soliton/Quantization.lean` - Vortex quantization conditions

**Proof Pattern**: Hard-wall boundary → quantized winding → discrete charge spectrum

### Cosmology Architecture (Paper-Ready)

**The Result**: CMB quadrupole-octupole "Axis of Evil" alignment proven from axisymmetric templates.

**Key Files** (11 theorems, 0 sorries):
- `Cosmology/AxisExtraction.lean` - Quadrupole axis uniqueness (IT.1)
- `Cosmology/OctupoleExtraction.lean` - Octupole axis uniqueness (IT.2)
- `Cosmology/CoaxialAlignment.lean` - Coaxial alignment theorem (IT.4)
- `Cosmology/Polarization.lean` - Sign-flip falsifiability test

**Publication Materials**:
- `Cosmology/CMB_AxisOfEvil_COMPLETE_v1.1.tex` - Complete MNRAS manuscript
- `Cosmology/PAPER_INTEGRATION_GUIDE.md` - LaTeX snippets for papers
- `CITATION.cff` - Software citation metadata

### QM Translation Architecture

**The Innovation**: Eliminates complex numbers from quantum mechanics - phase becomes geometric rotation in Cl(3,3).

**Key Files**:
- `QM_Translation/PauliBridge.lean` - Pauli matrices ↔ Clifford algebra
- `QM_Translation/DiracRealization.lean` - γ-matrices from Cl(3,3) centralizer
- `QM_Translation/SchrodingerEvolution.lean` - Phase as geometric rotation (e^{iθ} → e^{Bθ})
- `QM_Translation/RealDiracEquation.lean` - Mass as internal momentum (E=mc²)

**Proof Pattern**: Complex exponential `e^{iθ}` replaced by bivector exponential `e^{Bθ}` where `B² = -1`.

## Documentation System: The Proof Index

**Problem**: With 369 proven statements, finding "which theorem proves claim X?" requires a traceability system.

**Solution**: Four-file index system that makes the repository self-documenting.

### Core Index Files

1. **`QFD/ProofLedger.lean`** - Master ledger mapping book claims → Lean theorems
   - Organized by book section (Appendix A, Z, Nuclear, Cosmology, etc.)
   - Each claim: reference, plain-English statement, theorem name, file:line, status
   - Tags: `[CLAIM X.Y.Z]`, `[CONCERN_CATEGORY]`

2. **`QFD/CLAIMS_INDEX.txt`** - Grep-able theorem inventory (auto-generated)
   - Format: `File:LineNumber:TheoremName`
   - All 369 proven statements listed
   - Generate: `rg -n "^theorem|^lemma" QFD --include="*.lean"`

3. **`QFD/CONCERN_CATEGORIES.md`** - Critical assumption tracking
   - 5 categories: ADJOINT_POSITIVITY, PHASE_CENTRALIZER, SIGNATURE_CONVENTION, etc.
   - Lists theorems addressing each concern

4. **`QFD/COMPLETE_GUIDE.md`** - Full system documentation

**Usage**:
- Finding a proof: Ctrl+F in `ProofLedger.lean` for claim number
- Listing all theorems: Search `CLAIMS_INDEX.txt`
- Understanding assumptions: Check `CONCERN_CATEGORIES.md`

## AI Assistant Workflow (MANDATORY)

**Located in**: `AI_WORKFLOW.md` (REQUIRED READING before any work)

### Golden Rule

**Write ONE proof → `lake build` → Fix errors → Verify → Next proof**

**NEVER** submit work without successful build verification.

### The Iterative Cycle

```
1. Select ONE theorem with `sorry`
2. Write proof attempt
3. Run `lake build QFD.Module.Name` IMMEDIATELY
4. If errors: Fix the ONE error shown, goto step 3
5. If success: Document, move to next theorem
```

### Common Errors & Solutions

**Error**: `unknown namespace 'QFD.GA.Cl33'`
**Fix**: `Cl33` is a type, not namespace. Use `open QFD.GA` instead.

**Error**: `unfold failed`
**Fix**: Use `simp only [definition]` or `change explicit_expression`

**Error**: `ring tactic failed`
**Fix**: Clifford algebra isn't a ring! Use `clifford_simp` or manual `simp [mul_assoc]`

**Error**: Reserved keyword (lambda, def, theorem)
**Fix**: Rename variable (e.g., `lambda` → `lam`)

### Build Verification Requirements

✅ **SUCCESS** looks like:
```
✔ [3081/3081] Building QFD.Module.Name
```

❌ **FAILURE** looks like:
```
error: QFD/File.lean:82:6: Tactic failed
error: build failed
```

**Completion Criteria**:
- `lake build` shows 0 errors (warnings OK)
- Any `sorry` has documented TODO explaining blocker
- Build log included in report

## Work Queue & Priorities

**Located in**: `WORK_QUEUE.md`

**Current Status** (as of 2025-12-27):
- 19/65 modules building successfully (29%)
- Priority 1: Generations.lean (high-value, unblocks 2 modules)
- Priority 2: Schema.Constraints blockers - **ELIMINATED** ✅
- Untested: 33 modules remaining

**Before starting work**:
1. Read `WORK_QUEUE.md` for current priorities
2. Check if your target file is in `PROTECTED_FILES.md` (DO NOT MODIFY protected files)
3. Follow iterative workflow from `AI_WORKFLOW.md`

## Protected Files (DO NOT MODIFY)

**Located in**: `PROTECTED_FILES.md`

### Absolutely Protected (Core Infrastructure)

- `QFD/GA/Cl33.lean` - Foundation (50+ files depend on it)
- `QFD/GA/BasisOperations.lean` - Core lemmas
- `QFD/GA/BasisReduction.lean` - Automation engine
- `QFD/GA/BasisProducts.lean` - Pre-computed library
- `lakefile.toml` - Build configuration
- `lean-toolchain` - Version specification (4.27.0-rc1)

### Modify With Extreme Caution (Proven Correct)

- `QFD/GA/PhaseCentralizer.lean` (0 sorries + 1 intentional axiom)
- `QFD/Electrodynamics/MaxwellReal.lean` (0 sorries, reference implementation)
- `QFD/QM_Translation/DiracRealization.lean` (0 sorries)
- `QFD/QM_Translation/RealDiracEquation.lean` (0 sorries)

**Rule**: If a file has 0 sorries and isn't in the work queue, don't touch it.

## File Structure

```
Lean4/
├── lakefile.toml              # Build configuration (Lake package manager)
├── lean-toolchain             # Lean version: 4.27.0-rc1
├── README.md                  # Project overview, quick start
├── AI_WORKFLOW.md             # AI assistant workflow (MANDATORY reading)
├── WORK_QUEUE.md              # Prioritized task list
├── PROTECTED_FILES.md         # Files not to modify
├── BUILD_STATUS.md            # Current build status, sorry count
├── CITATION.cff               # Software citation for papers
│
└── QFD/
    ├── ProofLedger.lean       # Master claim → theorem mapping
    ├── PROOF_INDEX.md         # Proof index guide
    ├── CLAIMS_INDEX.txt       # All 369 theorems (grep-able)
    ├── CONCERN_CATEGORIES.md  # Critical assumption tracking
    ├── COMPLETE_GUIDE.md      # Full system documentation
    │
    ├── GA/                    # Geometric Algebra (Clifford Cl(3,3))
    │   ├── Cl33.lean          # Foundation - DO NOT MODIFY
    │   ├── BasisOperations.lean
    │   ├── BasisProducts.lean
    │   ├── BasisReduction.lean # clifford_simp tactic
    │   ├── PhaseCentralizer.lean
    │   ├── Conjugation.lean
    │   └── ...
    │
    ├── Cosmology/             # CMB, supernovae (paper-ready)
    │   ├── AxisExtraction.lean
    │   ├── CoaxialAlignment.lean
    │   ├── CMB_AxisOfEvil_COMPLETE_v1.1.tex
    │   └── PAPER_INTEGRATION_GUIDE.md
    │
    ├── QM_Translation/        # Quantum mechanics from geometry
    │   ├── PauliBridge.lean
    │   ├── DiracRealization.lean
    │   ├── SchrodingerEvolution.lean
    │   └── RealDiracEquation.lean
    │
    ├── Charge/                # Quantization, Coulomb force
    ├── Nuclear/               # Core compression, time cliff
    ├── Lepton/                # Mass spectrum, generations
    ├── Gravity/               # Schwarzschild, geodesics
    ├── Soliton/               # Quantization, vortices
    └── ...
```

## Key Theorems & Locations

**Spacetime Emergence**:
- `emergent_signature_is_minkowski` - `SpacetimeEmergence_Complete.lean:245`
- Centralizer = Minkowski space Cl(3,1)

**Charge Quantization**:
- `unique_vortex_charge` - `Soliton/Quantization.lean:139`
- Hard wall → discrete spectrum

**CMB Axis of Evil**:
- `quadrupole_axis_unique` - `Cosmology/AxisExtraction.lean:260` (IT.1)
- `octupole_axis_unique` - `Cosmology/OctupoleExtraction.lean:214` (IT.2)
- `coaxial_alignment` - `Cosmology/CoaxialAlignment.lean:68` (IT.4)

**Quantum Mechanics**:
- `phase_group_law` - `QM_Translation/SchrodingerEvolution.lean` (e^{iθ} → e^{Bθ})
- `mass_is_internal_momentum` - `QM_Translation/RealDiracEquation.lean`

## Development Environment

**Lean Version**: 4.27.0-rc1 (specified in `lean-toolchain`)
**Build Tool**: Lake (bundled with Lean 4)
**Dependencies**: Mathlib (auto-fetched on first build)
**Disk Space**: ~2 GB (including Mathlib cache)

**Installation**:
```bash
# Install Lean 4 (macOS/Linux)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build project (fetches Mathlib automatically)
cd projects/Lean4
lake build QFD
```

**First build**: 10-30 minutes (fetching & compiling Mathlib)
**Incremental builds**: Seconds to minutes

## Lean 4 Syntax Essentials

### Proof Structure

```lean
theorem name (args : Type) : statement := by
  tactic1
  tactic2
  done
```

### Common Tactics

- `simp` - Simplify using simp lemmas
- `rw [lemma]` - Rewrite using equation
- `exact proof_term` - Provide exact proof
- `apply lemma` - Apply theorem
- `intro h` - Introduce hypothesis
- `have h : P := proof` - Intermediate result
- `calc` - Chain of equalities
- `clifford_simp` - **QFD-specific**: Automate Clifford algebra

### Clifford Algebra Patterns

```lean
-- Anticommutation
have h : e i * e j = -(e j * e i) := basis_anticomm (h_ne : i ≠ j)

-- Squaring
have h : e i * e i = algebraMap ℝ Cl33 (signature33 i) := basis_sq i

-- Automated simplification (PREFERRED)
by clifford_simp
```

## Common Proof Patterns

### Pattern 1: Contradiction via Signature Mismatch

```lean
theorem basis_x_ne_xy : e 0 ≠ e 0 * e 1 := by
  intro h
  have h_sq : (e 0)^2 = (e 0 * e 1)^2 := congrArg (fun x => x^2) h
  rw [sq, sq, e_sq_one 0, e01_sq] at h_sq  -- 1 = -1
  exact one_ne_neg_one h_sq  -- Contradiction
```

### Pattern 2: Centralizer Commutation

```lean
theorem spatial_commutes_with_B : e 0 * B = B * e 0 := by
  unfold B
  calc e 0 * (e 4 * e 5)
      = (e 0 * e 4) * e 5 := by rw [mul_assoc]
    _ = (-(e 4 * e 0)) * e 5 := by rw [basis_anticomm]
    _ = -(e 4 * e 0 * e 5) := by rw [neg_mul]
    _ = -(e 4 * (e 0 * e 5)) := by rw [mul_assoc]
    _ = -(e 4 * (-(e 5 * e 0))) := by rw [basis_anticomm]
    _ = e 4 * e 5 * e 0 := by simp [mul_assoc]
```

### Pattern 3: Energy Positivity via Quadratic Form

```lean
theorem energy_positive (ψ : Multivector) : Energy ψ ≥ 0 := by
  unfold Energy
  apply sum_of_squares_nonneg
  intro i
  exact sq_nonneg (coeff i ψ)
```

## Citation

For papers using this formalization:

```bibtex
@software{qfd_formalization_2025,
  author = {{QFD Formalization Team}},
  title = {{Quantum Field Dynamics: Lean 4 Formalization}},
  year = {2025},
  version = {1.1},
  url = {https://github.com/tracyphasespace/Quantum-Field-Dynamics},
  note = {See QFD/ProofLedger.lean for claim mapping}
}
```

See `CITATION.cff` for complete metadata.

## Key Statistics (as of 2025-12-27)

- **Lean Files**: **180** (was ~71, **+109 NEW FILES created today!** 🎉)
  - Nuclear: 15 new files
  - Electrodynamics: 13 new files
  - Cosmology: 12 new files
  - QM_Translation, Gravity, GA: 11 new files each
  - Weak: 8 new files
  - Lepton: 7 new files
  - Additional domains: 21 new files
- **Proven Theorems**: 364
- **Proven Lemmas**: 118
- **Total Proven**: **482 statements**
- **Definitions**: 316
- **Structures**: 47
- **Build Jobs**: 3165 (was 3081, +84)
- **Sorries**: 16 (in multiple files, most documented)
- **Axioms**: 43 (infrastructure + 2 in Unitarity - disclosed)

## Support

- **Documentation**: `QFD/COMPLETE_GUIDE.md`
- **Theorem Lookup**: Search `QFD/CLAIMS_INDEX.txt`
- **Issues**: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- **License**: MIT
