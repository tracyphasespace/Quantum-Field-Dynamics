# Master Refactoring List - For Other AI

**Last Updated**: 2025-12-27
**CRITICAL**: Read BUILD_VERIFICATION_PROTOCOL.md and ITERATIVE_PROOF_WORKFLOW.md before starting
**Total Modules**: 64 from refactored.md + suggestions.md
**Successfully Building**: 9 (14%)
**Need Refactoring**: 55 (86%)

---

## 🚨 MANDATORY WORKFLOW (READ FIRST)

### Build Verification Requirement
**NEVER submit work without successful build verification**

1. **Required Reading Before Starting**:
   - `BUILD_VERIFICATION_PROTOCOL.md` - How to verify builds
   - `ITERATIVE_PROOF_WORKFLOW.md` - ONE proof at a time approach
   - `COMMON_BUILD_ERRORS.md` - Solutions to common errors

2. **Mandatory Workflow**:
   ```
   Write ONE proof → Build → Fix errors → Verify → Next proof
   ```

3. **Completion Criteria**:
   - ✅ `lake build` shows 0 errors (warnings OK)
   - ✅ Build log saved and included in report
   - ✅ Any `sorry` documented with TODO

4. **Red Flag - Do NOT Submit If**:
   - ❌ Haven't run `lake build`
   - ❌ Build has ANY `error:` lines
   - ❌ Tested "in your head" without actual build
   - ❌ Added `sorry` without documentation

**Summary**: If you can't provide a build log showing success, the work is incomplete.

---

## ✅ Successfully Building - NO ACTION NEEDED (9 modules)

These have been removed from the work queue:

1. QFD.Lepton.KoideRelation
2. QFD.Gravity.PerihelionShift
3. QFD.Gravity.SnellLensing
4. QFD.Electrodynamics.ProcaReal
5. QFD.Cosmology.HubbleDrift
6. QFD.GA.Cl33Instances (1 documented sorry - acceptable)
7. QFD.Lepton.FineStructure
8. QFD.Weak.ParityGeometry
9. QFD.Cosmology.CosmicRestFrame

---

## 🔴 PRIORITY 1: Quick Wins - Namespace/Syntax Fixes (4 modules)

### These can be fixed in <30 minutes each

#### 0. QFD.Lepton.Generations ⭐ HIGH VALUE
**Status**: ✅ Builds successfully but has 6 documented sorries
**Issue**: Header claims "0 Sorries" but proof has 6 sorries
**Task**: Complete grade distinctness proof (lines 87, 89, 92, 95, 97, 99)
**Proof Strategy**: Show that basis elements of different grades cannot be equal
  - e₀ (grade 1) ≠ e₀*e₁ (grade 2) - use anticommutativity or grade separation
  - e₀*e₁ (grade 2) ≠ e₀*e₁*e₂ (grade 3) - use grade or algebraic properties
**Impact**: 🔥 UNBLOCKS KoideRelation and FineStructure to truly zero-sorry status
**Difficulty**: Moderate (requires Clifford algebra reasoning)
**Build Command**: `lake build QFD.Lepton.Generations`
**Approach**: Use iterative workflow - complete ONE sorry at a time, build after each

#### 1. QFD.QM_Translation.PauliExclusion
**Error**: `unknown namespace 'QFD.GA.Cl33'`
**Fix**: Change `open QFD.GA.Cl33` → `open QFD.GA`
**Also**: Add `import QFD.GA.Cl33Instances` if using Nontrivial
**Build Command**: `lake build QFD.QM_Translation.PauliExclusion`

#### 2. QFD.Nuclear.YukawaDerivation ⚡ CRITICAL
**Status**: ✅ PARTIALLY COMPLETE - builds with documented sorry
**Error**: Was `lambda` reserved keyword (FIXED)
**Current State**: Main theorem has documented sorry for derivative proof
**Remaining Work**: Complete derivative calculation proof (advanced)
**Blocks**: QFD.Soliton.BreatherModes (still blocked - needs proof completion)
**Build Command**: `lake build QFD.Nuclear.YukawaDerivation`
**Build Status**: ✅ Compiles with warnings (sorry usage documented)

#### 3. QFD.Relativity.LorentzRotors
**Error**: Unknown (needs investigation)
**Action**: Run `lake build QFD.Relativity.LorentzRotors 2>&1 | head -50` to see full error
**Build Command**: `lake build QFD.Relativity.LorentzRotors`

---

## 🟡 PRIORITY 2: Blocked by Schema.Constraints (6 modules)

**Root Cause**: `QFD/Schema/Constraints.lean` has proof errors
**Blocker Errors**:
- Line 177: `unfold ValidParameters` fails
- Line 255: unknown tactic (likely missing linarith import)
- Line 273: No goals to be solved

**Blocked Modules**:
1. QFD.Nuclear.BoundaryCondition
2. QFD.Nuclear.MagicNumbers
3. QFD.Nuclear.DeuteronFit
4. QFD.Cosmology.ZeroPointEnergy
5. QFD.Nuclear.IsomerDecay (stub - no action needed on file itself)
6. QFD.Vacuum.CasimirPressure

**Action for Other AI**:
- **Option A**: Fix Schema.Constraints directly (see instructions below)
- **Option B**: Skip these until Schema.Constraints is fixed by main AI

### How to Fix Schema.Constraints:
```lean
# Line 177 - Change:
unfold ValidParameters NuclearConstraints CosmoConstraints ParticleConstraints
# To:
simp only [ValidParameters, NuclearConstraints, CosmoConstraints, ParticleConstraints]

# Line 255 - Add import if missing:
import Mathlib.Tactic.Linarith

# Line 273 - Remove extra proof step after goal is closed
```

---

## 🟠 PRIORITY 3: Blocked by NeutrinoID/Mathlib Issues (3 modules)

**Root Cause**: NeutrinoID depends on Mathlib Matrix.Determinant which has `gradedModule` error

**Blocked Modules**:
1. QFD.Conservation.NeutrinoMixing
2. QFD.Weak.GeometricBosons
3. QFD.Weak.CPViolation

**Action**:
- Skip for now (Mathlib version issue)
- OR try `lake update mathlib` if you have memory

---

## 🔵 PRIORITY 4: Blocked by SchrodingerEvolution (2 modules)

**Root Cause**: `QFD.QM_Translation.SchrodingerEvolution` has unknown constant error

**Blocked Modules**:
1. QFD.QM_Translation.Zitterbewegung
2. QFD.QM_Translation.LandauLevels

**Error**: `unknown constant 'QFD.QM_Translation.SchrodingerEvolution.phase_group_law'`

**Action**: Investigate SchrodingerEvolution.lean to see what's missing

---

## ⚪ PRIORITY 5: Untested Modules (43 modules)

These haven't been tested yet - run builds to categorize them:

### Gravity (7):
- QFD.Gravity.TorsionContribution
- QFD.Gravity.FrozenStarRadiation
- QFD.Gravity.GravitationalWaves
- QFD.Gravity.InertialInduction
- QFD.Gravity.Ringdown
- QFD.Gravity.UnruhTemperature

### Cosmology (5):
- QFD.Cosmology.ArrowOfTime
- QFD.Cosmology.DarkMatterDensity
- QFD.Cosmology.AxisOfEvil
- QFD.Cosmology.DarkEnergy
- QFD.Cosmology.HubbleTension
- QFD.Cosmology.GZKCutoff

### Electrodynamics (5):
- QFD.Electrodynamics.ComptonScattering
- QFD.Electrodynamics.NoMonopoles
- QFD.Electrodynamics.LambShift
- QFD.Electrodynamics.AharonovBohm
- QFD.Electrodynamics.Birefringence
- QFD.Electrodynamics.CerenkovReal

### QM Translation (5):
- QFD.QM_Translation.BerryPhase
- QFD.QM_Translation.SpinStatistics (blocked by PauliExclusion)
- QFD.QM_Translation.TunnelingTime
- QFD.QM_Translation.QuantumEraser

### Nuclear (5):
- QFD.Nuclear.Confinement
- QFD.Nuclear.StabilityLimit
- QFD.Nuclear.QCDLattice
- QFD.Nuclear.NeutronStarEOS
- QFD.Nuclear.ProtonSpin
- QFD.Nuclear.BarrierTransparency

### Weak (1):
- QFD.Weak.DoubleBetaDecay

### Lepton (2):
- QFD.Lepton.Antimatter
- QFD.Lepton.PairProduction

### Other (13):
- QFD.Unification.FieldGradient
- QFD.Vacuum.Screening
- QFD.Matter.Superconductivity
- QFD.Computing.RotorLogic
- QFD.Relativity.TimeDilationMechanism
- QFD.Thermodynamics.HolographicPrinciple
- QFD.Soliton.BreatherModes (blocked by YukawaDerivation - see Priority 1)
- QFD.Matter.QuantumHall

---

## 📋 Recommended Work Order for Other AI

### Phase 1: Quick Wins (Est. 2-3 hours)
1. ⭐ **Complete Generations.lean** (6 sorries → 0) - **HIGH VALUE** - Unblocks KoideRelation & FineStructure
2. ✅ Fix YukawaDerivation (lambda → lam) - **DONE** (has documented sorry)
3. ⏳ Fix PauliExclusion (namespace) - **IN PROGRESS**
4. ⏳ Test BreatherModes (blocked by YukawaDerivation proof completion)
5. ⏳ Investigate LorentzRotors

### Phase 2: Test Untested Modules (Est. 2-3 hours)
1. Run build tests on all 43 untested modules
2. Categorize by error type (namespace, Schema.Constraints, other)
3. Fix any namespace/quick syntax errors found

### Phase 3: Infrastructure Fixes (Est. 3-5 hours)
1. Fix Schema.Constraints (if comfortable with proof refactoring)
2. OR skip and wait for main AI to fix Schema.Constraints

### Phase 4: Update Master Lists
1. Update refactored.md with test results
2. Remove successfully building modules
3. Report any new error patterns found

---

## 🛠️ Testing Commands

### Quick Test Single Module:
```bash
lake build QFD.Module.Name 2>&1 | tail -10
```

### Test and Save Results:
```bash
lake build QFD.Module.Name 2>&1 | tee test_results_Module.txt
```

### Check for Errors Only:
```bash
lake build QFD.Module.Name 2>&1 | grep "error:" || echo "SUCCESS"
```

---

## 📊 Current Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Successfully Building** | 9 | 14% |
| **Quick Win Fixes Needed** | 4 | 6% |
| **Blocked by Schema.Constraints** | 6 | 9% |
| **Blocked by NeutrinoID** | 3 | 5% |
| **Blocked by SchrodingerEvolution** | 2 | 3% |
| **Untested** | 43 | 67% |
| **TOTAL** | 65 | 100% |

**Note**: Generations.lean was previously counted as "successfully building" but has been moved to Priority 1 due to 6 incomplete proofs.

---

## 🎯 Success Criteria

After Other AI completes work:
- [ ] **Generations.lean** - All 6 sorries completed, KoideRelation & FineStructure at true zero-sorry
- [ ] YukawaDerivation builds successfully (DONE - has documented sorry for advanced proof)
- [ ] BreatherModes unblocked and building
- [ ] PauliExclusion namespace fixed and building
- [ ] All 43 untested modules have been tested and categorized
- [ ] refactored.md updated with new successfully building modules
- [ ] Error patterns documented for any new issues found

---

**Generated**: 2025-12-27 by Claude Code (Main AI)
**For**: Other AI Refactoring Assistant
**Next Review**: After Phase 1 completion
