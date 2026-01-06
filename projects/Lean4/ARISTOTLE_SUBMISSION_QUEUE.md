# Aristotle Submission Queue
## Files Recommended for Review

**Updated**: 2026-01-02
**Status**: Nuclear integration complete, identifying next candidates

---

## High Priority (Ready for Submission)

### 1. TopologicalStability_Refactored.lean ⭐

**Current Status**: 1 sorry (down from 16!)
**Location**: `QFD/Soliton/TopologicalStability_Refactored.lean`
**Why Submit**: Aristotle already created a refactored version with massive improvements
**Aristotle Version**: `Aristotle_In_Progress/TopologicalStability_Refactored_aristotle.lean`
**Action**: Test compile Aristotle version and integrate

**Aristotle Achievements**:
- Reduced 16 sorries → 1 sorry
- Proved `saturated_interior_is_stable`
- 1 axiom: `topological_conservation` (can be proven from Mathlib)
- Status: Ready for immediate integration testing

---

### 2. Cosmology Files (Publication-Ready) ⭐

Aristotle has already reviewed these - just need integration:

#### AxisExtraction_aristotle.lean (540 lines)
**Original**: `QFD/Cosmology/AxisExtraction.lean`
**Aristotle Version**: `QFD/Cosmology/AxisExtraction_aristotle.lean`
**Purpose**: CMB quadrupole axis uniqueness (paper theorem IT.1)
**Action**: Compare with original, test compile, integrate

#### CoaxialAlignment_aristotle.lean (180 lines)
**Original**: `QFD/Cosmology/CoaxialAlignment.lean`
**Aristotle Version**: `QFD/Cosmology/CoaxialAlignment_aristotle.lean`
**Purpose**: CMB Axis-of-Evil alignment (paper theorem IT.4)
**Action**: Compare with original, test compile, integrate

**Impact**: These are paper-ready proofs for MNRAS manuscript

---

### 3. Core Infrastructure Files

Aristotle has reviewed these - need comparison:

#### PhaseCentralizer_aristotle.lean (230 lines)
**Original**: `QFD/GA/PhaseCentralizer.lean`
**Aristotle Version**: `QFD/GA/PhaseCentralizer_aristotle.lean`
**Purpose**: Phase rotor centralization (0 sorries claimed)
**Action**: Compare versions, check for improvements

#### RealDiracEquation_aristotle.lean (180 lines)
**Original**: `QFD/QM_Translation/RealDiracEquation.lean`
**Aristotle Version**: `QFD/QM_Translation/RealDiracEquation_aristotle.lean`
**Purpose**: Dirac equation from geometry (0 sorries)
**Action**: Compare versions, check for improvements

---

## Medium Priority (Files with Sorries)

### Files Currently with Sorries (Need Aristotle Help)

Based on grep results, these files have sorries that Aristotle could address:

1. **QFD/Nuclear/TimeCliff.lean** - Has sorries
   - Note: We have TimeCliff_Complete.lean (0 sorries)
   - Recommendation: Use Complete version, don't resubmit

2. **QFD/Soliton/TopologicalStability.lean** - 16 sorries
   - Note: Refactored version has 1 sorry
   - Recommendation: Use refactored version

3. **Files to investigate for sorries**:
   - Check all QFD/Nuclear/*.lean files
   - Check all QFD/Lepton/*.lean files
   - Check all QFD/Cosmology/*.lean files

---

## Chapter 14 Relevant Files (NEW - For Spherical Harmonic Theory)

These files are relevant to your Chapter 14 work and should be reviewed:

### Nuclear Physics (Decay Modes)

1. **QFD/Nuclear/AlphaNDerivation.lean** (original version)
   - Purpose: Alpha decay geometry
   - Aristotle version: AlphaNDerivation_Complete.lean (14 theorems, 0 sorries) ✅ DONE
   - Status: **Integrated**

2. **QFD/Nuclear/BetaNGammaEDerivation.lean** (original version)
   - Purpose: Beta decay paths
   - Aristotle version: BetaNGammaEDerivation_Complete.lean (21 theorems, 0 sorries) ✅ DONE
   - Status: **Integrated**

3. **QFD/Nuclear/MagicNumbers.lean**
   - Purpose: Shell structure basics
   - Aristotle version: Encountered import error (axioms in CoreCompressionLaw)
   - Action: Fix CoreCompressionLaw axioms first, then resubmit

4. **QFD/Nuclear/CoreCompressionLaw.lean** (29KB)
   - Purpose: Core density saturation
   - Issue: Contains axioms that block MagicNumbers compilation
   - Action: **Submit to Aristotle to eliminate axioms**
   - Priority: HIGH (blocks other work)

### Lepton Physics (Mass Spectrum)

5. **QFD/Lepton/Generations.lean**
   - Purpose: Three lepton families from geometry
   - Current: 0 sorries
   - Action: **Submit for verification pass** (like TimeCliff)

6. **QFD/Lepton/KoideRelation.lean**
   - Purpose: Mass spectrum from S₃ symmetry
   - Current: 0 sorries (3 eliminated Dec 27)
   - Action: **Submit for verification pass**

7. **QFD/Lepton/MassSpectrum.lean**
   - Purpose: Complete mass spectrum derivation
   - Action: Check sorry count, submit if needed

### Soliton Theory (Resonance Foundation)

8. **QFD/Soliton/Quantization.lean** (8KB)
   - Purpose: Topological charge quantization
   - Current: Check sorry count
   - Relevance: Mode quantization for Chapter 14

9. **QFD/Soliton/HardWall.lean**
   - Purpose: Boundary condition enforcement
   - Relevance: Re-187 boundary sensitivity (§14.7.2)

10. **QFD/Soliton/BreatherModes.lean**
    - Purpose: Oscillatory soliton modes
    - Relevance: Harmonic modes for Chapter 14

---

## Low Priority (Already Complete or Not Critical)

- TimeCliff.lean - Have TimeCliff_Complete.lean ✅
- Files with 0 sorries that are already verified
- Documentation files

---

## Recommended Submission Order

### Immediate (This Week):

1. ✅ Test compile `TopologicalStability_Refactored_aristotle.lean`
2. ✅ Test compile `AxisExtraction_aristotle.lean`
3. ✅ Test compile `CoaxialAlignment_aristotle.lean`
4. ✅ Compare and integrate all three if successful

### Next Round (Week 2):

5. **Submit CoreCompressionLaw.lean** to Aristotle
   - Goal: Eliminate axioms blocking MagicNumbers
   - Impact: Unlocks magic number work for Chapter 14

6. **Submit for verification**: Generations.lean, KoideRelation.lean
   - Goal: Verification pass (like TimeCliff)
   - Impact: Confirms lepton mass spectrum proofs

### Future Rounds:

7. Submit remaining soliton files: Quantization, HardWall, BreatherModes
8. Submit any files with >5 sorries found in survey

---

## Integration Workflow

For each Aristotle file:

1. **Test Compile**:
   ```bash
   lake build QFD.Module.FileName_aristotle
   ```

2. **Compare with Original**:
   - Read both versions
   - Count sorries: `grep -c sorry original.lean` vs `grep -c sorry aristotle.lean`
   - Count theorems: `grep -c "^theorem\|^lemma"`
   - Identify improvements

3. **Create Comparison Report**:
   - Document changes
   - List new theorems
   - Note any breaking changes

4. **Integrate if Better**:
   - Copy to production version (remove _aristotle suffix)
   - Test compile production version
   - Update BUILD_STATUS and CITATION
   - Commit with detailed message

5. **Update Proof Counts**:
   - Add new theorem counts to total
   - Update version number
   - Push to GitHub

---

## Success Metrics

**Goal**: Achieve <5 total sorries across entire codebase

**Current Status**:
- Total sorries: ~20 (estimate from survey)
- TopologicalStability: 16 → 1 (if we integrate refactored)
- Remaining: ~4 sorries

**Target**: Get to 0 sorries in all core modules by end of January 2026

**Strategy**: Submit high-sorry-count files to Aristotle, integrate improvements systematically

---

## Notes

**Aristotle UUID Tracking**:
- Each submission gets a UUID
- Track in this file for future reference
- Use UUID to resume sessions if needed

**File Naming Convention**:
- Original: `FileName.lean`
- Aristotle version: `FileName_aristotle.lean`
- Production version after integration: `FileName_Complete.lean` (if major changes)
  OR just `FileName.lean` (if minor improvements)

**Documentation**:
- Every integration gets a report: `ARISTOTLE_[NAME]_INTEGRATION.md`
- Update BUILD_STATUS.md
- Update CITATION.cff version
- Commit message format: "Aristotle integration: [description] (v[version])"
