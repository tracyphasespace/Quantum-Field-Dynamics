# QFD Lean 4 Formalization

**Version 1.3** | **Release Date**: 2025-12-29

Rigorous formalization of Quantum Field Dynamics theorems in Lean 4, covering spacetime emergence, cosmology, nuclear physics, and particle physics.

---

## ⭐ Quick Start - Choose Your Path

### For Reviewers (Verifying Proofs)

**Want to verify the CMB "Axis of Evil" formalization?**

1. **Start here**: [`QFD/ProofLedger.lean`](QFD/ProofLedger.lean) - Claims CO.4-CO.6
2. **Search theorems**: [`QFD/CLAIMS_INDEX.txt`](QFD/CLAIMS_INDEX.txt) - All 575 proven statements
3. **Build & verify**:
   ```bash
   lake build QFD.Cosmology.AxisExtraction QFD.Cosmology.CoaxialAlignment
   ```
4. **Full docs**: [`QFD/Cosmology/README_FORMALIZATION_STATUS.md`](QFD/Cosmology/README_FORMALIZATION_STATUS.md)

**Status**: 11 cosmology theorems, 0 sorry, 1 axiom (disclosed), paper-ready ✓

### For Developers (Understanding the System)

**Want the complete system guide?**

1. **Complete Guide**: [`QFD/COMPLETE_GUIDE.md`](QFD/COMPLETE_GUIDE.md) - Everything in one place
2. **Proof Index**: [`QFD/PROOF_INDEX.md`](QFD/PROOF_INDEX.md) - Quick theorem lookup
3. **Theorem List**: [`QFD/CLAIMS_INDEX.txt`](QFD/CLAIMS_INDEX.txt) - Grep-able reference

### For AI Assistants (Contributing Proofs)

**Want to help complete the formalization?**

**🚨 START HERE - Three Required Documents**:

1. **[`AI_WORKFLOW.md`](AI_WORKFLOW.md)** ⭐ **READ FIRST**
   - Build verification requirements (MANDATORY)
   - ONE proof at a time iterative workflow
   - Common errors and solutions
   - Completion report template

2. **[`CRITICAL_CONSTANTS.md`](CRITICAL_CONSTANTS.md)** ⚠️ **VALIDATION REQUIRED**
   - ⚠️ **alpha_circ = e/(2π) NOT 1/(2π)** - Common AI contamination!
   - All vacuum parameters with Python validation
   - Step-by-step verification protocol
   - **READ THIS before touching any vacuum constants!**

3. **[`WORK_QUEUE.md`](WORK_QUEUE.md)** 📋 **WHAT TO DO**
   - Prioritized task list (65 modules)
   - Detailed task instructions
   - Expected outcomes
   - Build commands

**⚡ USE AUTOMATION**: `clifford_simp` tactic automates Clifford algebra - DON'T write manual expansions!

**Infrastructure**:
- `BasisReduction.lean` - Automation engine (USE THIS!)
- `BasisProducts.lean` - Pre-computed products
- `BasisOperations.lean` - Core lemmas
- `QFD/Vacuum/VacuumParameters.lean` - **AUTHORITATIVE** source for all constants

**Protected Files**: [`PROTECTED_FILES.md`](PROTECTED_FILES.md) - Don't modify these!

**Golden Rule**: Write ONE proof → `lake build` → Fix errors → Verify → Next proof

### For Paper Authors (Using in Publications)

**Want to cite these proofs in a paper?**

1. **Paper Guide**: [`QFD/Cosmology/PAPER_INTEGRATION_GUIDE.md`](QFD/Cosmology/PAPER_INTEGRATION_GUIDE.md)
2. **LaTeX Template**: [`QFD/Cosmology/PAPER_TEMPLATE_WITH_FORMALIZATION.tex`](QFD/Cosmology/PAPER_TEMPLATE_WITH_FORMALIZATION.tex)
3. **Complete Manuscript**: [`QFD/Cosmology/CMB_AxisOfEvil_COMPLETE_v1.1.tex`](QFD/Cosmology/CMB_AxisOfEvil_COMPLETE_v1.1.tex)
4. **Software Citation**: [`CITATION.cff`](CITATION.cff)

---

## What's Formalized

### ✅ Cosmology (Paper-Ready)
**CMB "Axis of Evil" - Quadrupole/Octupole Alignment**
- 11 theorems: axis uniqueness, coaxial alignment, sign-flip falsifier
- 4 core files: AxisExtraction, OctupoleExtraction, CoaxialAlignment, Polarization
- **Status**: 0 sorry, 1 axiom (disclosed), complete documentation
- **Paper**: Complete MNRAS manuscript ready for submission

### ✅ Spacetime Emergence
**Dimensional Reduction from Cl(3,3) to 4D Minkowski**
- EmergentAlgebra.lean - Centralizer theorem (algebraic inevitability)
- SpectralGap.lean - Dynamical suppression of extra dimensions
- ToyModel.lean - Verification via Fourier series
- **Status**: 0 sorry, complete proofs

### ✅ Charge Quantization
**Vacuum Topology → Discrete Charge**
- Quantization.lean - Topological charge quantization
- Coulomb.lean - Force law from geometry
- **Status**: Core theorems proven

### ✅ Nuclear Physics
**Core Compression Law**
- CoreCompression.lean - Mass-radius relation
- TimeCliff.lean - Stability criterion
- **Status**: Primary theorems complete

### ✅ Particle Physics
**Lepton & Neutrino Sector**
- MassSpectrum.lean - Geometric mass hierarchy
- MassFunctional.lean - Mass from geometry (Higgs deletion)
- Topology.lean - Topological protection (matter stability)
- Neutrino oscillation and production mechanisms
- **Status**: Key results formalized

### ✅ Conservation Laws & Black Holes
**Information Paradox Resolution**
- Unitarity.lean - Black hole information conservation (6D unitarity)
- Noether.lean - 6D geometric momentum conservation
- NeutrinoID.lean - Missing energy as geometric rotation
- **Status**: Core theorems complete, 2 axioms in Unitarity

### ✅ Quantum Mechanics Translation
**Dirac Algebra from Geometry**
- DiracRealization.lean - γ-matrices as Cl(3,3) centralizer elements
- PauliBridge.lean - Connection to standard QM formalism
- SchrodingerEvolution.lean - Phase evolution as geometric rotation (eliminates complex i)
- RealDiracEquation.lean - Mass as internal momentum (E=mc² from geometry)
- **Status**: Core proofs complete, 1 documented sorry in SchrodingerEvolution

### 🚧 Additional Domains
- Rift Dynamics (black hole charge eruptions)
- Soliton Physics (quantization, Gaussian/Ricker analysis)
- Gravity (geodesic force, Schwarzschild link)
- Electron Structure (Hill vortex, axis alignment)

---

## Statistics (Updated 2025-12-29)

| Metric | Value |
|--------|-------|
| **Proven Theorems** | 451 |
| **Proven Lemmas** | 124 |
| **Total Proven** | **575 statements** |
| **Definitions** | 409 |
| **Structures** | 53 |
| **Axioms** | 17 (infrastructure + physical hypotheses, all disclosed) |
| **Lean Files** | 215 |
| **Build Status** | ✅ Successful (3089 jobs) |
| **Sorry Count** | 6 actual sorries (20 total mentions including comments, all documented) |

**Recent Additions (2025-12-29)**:
- ✅ Sorry Reduction (23 → 6 actual sorries, 74% reduction)
- ✅ GA/Cl33.lean Complete (basis_isOrtho proven, 0 sorries in foundation)
- ✅ GA/HodgeDual.lean Complete (I₆² = 1 documented axiom from signature formula)
- ✅ Documentation Transparency (TRANSPARENCY.md, professional tone cleanup)
- ✅ Grand Solver Architecture (restored and updated with honest assessment)
- ✅ 27 New Proofs (575 total proven statements, up from 548)

**Additions (2025-12-27)**:
- ✅ Heisenberg Uncertainty (xp_noncomm proven via metric contradiction)
- ✅ Maxwell Geometric Equation (field decomposition complete)
- ✅ Enhanced Conjugation (reverse_B_phase, geometric_norm_sq)
- ✅ Grade Projection (scalar_part, real_energy_density defined)
- ✅ BasisReduction Automation (clifford_simp tactic)

**Previous Additions (2025-12-26)**:
- ✅ Schrödinger Evolution (geometric phase rotation, eliminates complex i)
- ✅ Real Dirac Equation (mass as internal momentum)
- ✅ Black Hole Unitarity Theorem (Information Paradox resolution)
- ✅ Noether Conservation (6D geometric momentum)
- ✅ Dirac Algebra Realization (γ-matrices from Cl(3,3))
- ✅ Vacuum Refraction (CMB power spectrum modulation)
- ✅ Mass Functional (geometric origin of mass)

---

## Documentation Structure

```
Lean4/
├── README.md                    ← You are here
├── COMPLETE_GUIDE.md            ← Full system documentation
├── CITATION.cff                 ← Software citation for papers
├── lakefile.toml                ← Build configuration
├── lean-toolchain               ← Lean version (4.27.0-rc1)
│
└── QFD/
    ├── ProofLedger.lean         ⭐ START HERE (claim → theorem mapping)
    ├── PROOF_INDEX.md           ← Quick theorem lookup guide
    ├── CLAIMS_INDEX.txt         ← Grep-able theorem list (575 proven statements)
    ├── THEOREM_STATEMENTS.txt   ← Complete theorem signatures
    ├── CONCERN_CATEGORIES.md    ← Critical assumptions tracked
    ├── LEAN_PYTHON_CROSSREF.md  ← Lean ↔ Python traceability
    │
    ├── Cosmology/               ← Paper-ready CMB formalization
    │   ├── README_FORMALIZATION_STATUS.md
    │   ├── PAPER_INTEGRATION_GUIDE.md
    │   ├── CMB_AxisOfEvil_COMPLETE_v1.1.tex
    │   ├── AxisExtraction.lean
    │   ├── OctupoleExtraction.lean
    │   ├── CoaxialAlignment.lean
    │   └── Polarization.lean
    │
    ├── [Domain directories...]
    │
    └── archive/                 ← Historical docs (archived 2025-12-25)
        ├── historical_status/
        ├── code_dumps/
        └── old_root_docs/
```

---

## ⚠️ CRITICAL BUILD WARNING ⚠️

**NEVER RUN MULTIPLE INSTANCES OF MATHLIB SIMULTANEOUSLY**

Running multiple `lake build` commands in parallel will cause each to build Mathlib independently, resulting in:
- **Out of Memory (OOM) errors** - System crash likely
- **Build failures** - Corrupted build artifacts
- **Wasted resources** - Hours of unnecessary compilation

**✅ CORRECT**: Run builds sequentially using `&&`:
```bash
lake build QFD.Cosmology.AxisExtraction && lake build QFD.Cosmology.CoaxialAlignment
```

**❌ WRONG**: Never run builds in parallel or multiple terminals simultaneously

---

## Quick Build

```bash
# Build everything
cd projects/Lean4
lake build QFD

# Verify cosmology (paper-ready)
lake build QFD.Cosmology.AxisExtraction QFD.Cosmology.CoaxialAlignment

# Verify spacetime emergence
lake build QFD.EmergentAlgebra QFD.SpectralGap

# Build specific domain
lake build QFD.Charge.Quantization
```

**First build**: 10-30 minutes (fetching Mathlib)
**Incremental builds**: Seconds to minutes

---

## Key Results

### Inference Theorems (IT.1-IT.4)

The cosmology formalization proves four core inference theorems:

- **IT.1**: Quadrupole axis uniqueness (AxisExtraction.lean:260)
- **IT.2**: Octupole axis uniqueness (OctupoleExtraction.lean:214)
- **IT.3**: Monotone invariance (AxisExtraction.lean:152)
- **IT.4**: Coaxial alignment (CoaxialAlignment.lean:68)

These establish that if CMB patterns fit axisymmetric templates with positive amplitude,
the extracted axes are deterministic and co-aligned with the observer's velocity.

### Spacetime Emergence

**Theorem (EmergentAlgebra)**: If a particle has internal rotation B = γ₅ ∧ γ₆ in Cl(3,3),
then the centralizer (visible spacetime) is exactly Cl(3,1) - 4D Minkowski space.

**Theorem (SpectralGap)**: If topology is quantized and centrifugal barrier exists,
then extra dimensions have an energy gap ΔE > 0.

**Result**: 4D Lorentzian spacetime emerges algebraically, extra dimensions are dynamically suppressed.

---

## Requirements

- **Lean**: 4.27.0-rc1 (specified in `lean-toolchain`)
- **Lake**: Build tool (bundled with Lean 4)
- **Mathlib**: Automatically fetched on first build
- **Disk space**: ~2 GB (including Mathlib)

### Installation

```bash
# Install Lean 4 (macOS/Linux)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Clone repository
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics
cd Quantum-Field-Dynamics/projects/Lean4

# Build (fetches Mathlib automatically)
lake build QFD
```

---

## Citation

For papers citing this formalization:

**BibTeX**:
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

See [`CITATION.cff`](CITATION.cff) for complete citation metadata.

---

## Support & Issues

- **Documentation**: See [`COMPLETE_GUIDE.md`](QFD/COMPLETE_GUIDE.md)
- **Theorem lookup**: Search [`CLAIMS_INDEX.txt`](QFD/CLAIMS_INDEX.txt)
- **Issues**: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- **License**: MIT

---

## Version History

**v1.2** (2025-12-26) - Conservation Laws & QM Translation
- **Added**: Schrödinger Evolution (phase as geometric rotation, complex i eliminated)
- **Added**: Real Dirac Equation (mass as internal momentum, E=mc² from geometry)
- **Added**: Black Hole Unitarity Theorem (Information Paradox resolution)
- **Added**: Noether 6D Conservation (geometric momentum)
- **Added**: Dirac Algebra Realization (γ-matrices from Cl(3,3))
- **Added**: Vacuum Refraction (CMB modulation mechanism)
- **Added**: Mass Functional (geometric mass origin)
- **Fixed**: DiracRealization, Noether, RadiativeTransfer compatibility
- **Statistics**: 322 proven theorems/lemmas, 77 files, 3081 build jobs
- **Status**: All critical proofs verified, 26 sorries with documented proof strategies

**v1.1** (2025-12-25) - Complete CMB Formalization
- Added: 11 cosmology theorems (CO.4-CO.6)
- Added: 4 inference theorems (IT.1-IT.4)
- Added: Coaxial alignment theorem
- Added: Complete paper integration materials
- Added: Software citation (CITATION.cff)
- Documentation: 15 files (core + paper)
- Status: 271 theorems, 0 sorry (critical path), paper-ready

**v1.0** (2025-12-17) - Initial Release
- Spacetime emergence proofs (EmergentAlgebra, SpectralGap, ToyModel)
- Charge quantization
- Nuclear physics foundations

---

**Last Updated**: 2025-12-26
**Build Status**: ✅ All proofs verified (3081 jobs)
**Paper Status**: ✅ MNRAS manuscript ready

> **Transparency**: The lepton-soliton model is still exploratory—see `TRANSPARENCY_SUMMARY.md` for the current provenance of β (from α + c₁/c₂), ξ/τ (Stage 2 fits), and α_circ (muon calibration).
