# Protected Files - DO NOT MODIFY

**For AI Assistants**: These files are foundational infrastructure. **DO NOT EDIT** them.

If you encounter sorries in these files, **SKIP THEM** and report to the human.

---

## 🚫 ABSOLUTELY PROTECTED (Core Infrastructure)

**Never touch these - they are the foundation everything builds on:**

1. **QFD/GA/Cl33.lean** ⛔
   - The Clifford algebra foundation
   - Used by 50+ other files
   - Any breakage cascades everywhere

2. **QFD/GA/BasisOperations.lean** ⛔
   - Core lemmas: `basis_sq`, `basis_anticomm`
   - Used in every proof

3. **QFD/GA/BasisReduction.lean** ⛔ **NEW!**
   - **AUTOMATION ENGINE** - `clifford_simp` tactic
   - Replaces 50-line proofs with one line
   - **DO NOT MODIFY** - Use it, don't change it!

4. **QFD/GA/BasisProducts.lean** ⛔ **NEW!**
   - Pre-computed product library
   - Used by BasisReduction.lean
   - **DO NOT MODIFY** - Import and use lemmas

5. **lakefile.toml** ⛔
   - Build configuration
   - Breaking this breaks everything

6. **lean-toolchain** ⛔
   - Lean version specification
   - Don't change!

7. **QFD/Vacuum/VacuumParameters.lean** ⛔ **CRITICAL**
   - **AUTHORITATIVE SOURCE** for all vacuum constants
   - Contains validated physics from Python verification
   - **NEVER hardcode these constants elsewhere**
   - Critical constants (validated 2025-12-29):
     - `alpha_circ = e/(2π) ≈ 0.4326` ⚠️ **NOT 1/(2π) ≈ 0.159!**
     - `beta = 3.043233053` (Golden Loop)
     - `xi = 1.0` (MCMC validated)
   - **If you modify this file, you MUST re-run Python validation**

---

## ⚠️ MODIFY WITH EXTREME CAUTION (Proven Infrastructure)

**These have 0 sorries and are proven correct. Only modify if you're CERTAIN:**

8. **QFD/GA/PhaseCentralizer.lean**
   - Has 1 intentional sorry (documented axiom)
   - Don't change the proven theorems!

9. **QFD/Electrodynamics/MaxwellReal.lean** ✅ **NEW!**
   - Priority 2 COMPLETE (0 sorries)
   - Maxwell's geometric equation
   - Reference implementation

10. **QFD/GA/Conjugation.lean** ✅ **NEW!**
    - Priority 3 COMPLETE (0 sorries)
    - Reversion operator
    - Reference implementation

11. **QFD/Lepton/AnomalousMoment.lean** ⚠️ **VALIDATED 2025-12-29**
    - Contains V₄ formula with validated constants
    - References `QFD.Vacuum.alpha_circ` (correct)
    - Do NOT hardcode `alpha_circ` locally

12. **QFD/QM_Translation/DiracRealization.lean**
    - Complete, proven (0 sorries)
    - Reference file, don't modify

13. **QFD/QM_Translation/PauliBridge.lean**
    - Complete, proven (0 sorries)
    - Reference file, don't modify

14. **QFD/QM_Translation/RealDiracEquation.lean**
    - Complete, proven (0 sorries)
    - Reference file, don't modify

---

## ✅ SAFE TO WORK ON (Target Files)

**These files NEED work and are safe to modify:**

### 🎯 CURRENT HIGH PRIORITY (Use clifford_simp!):
- ⚡ **QFD/GA/HodgeDual.lean** (1 sorry) - **Priority 5** - Use `clifford_simp` for I6_square!
- ⚠️ **QFD/GA/GradeProjection.lean** (1 sorry) - **Priority 4** - Placeholder, needs design

### Retroactive Simplification (Apply clifford_simp):
- 🔄 **QFD/Electrodynamics/PoyntingTheorem.lean** - Replace manual calc with `clifford_simp`
- 🔄 **QFD/QM_Translation/Heisenberg.lean** - Use automation instead of manual expansion

### Lower Priority:
- ✅ **QFD/GA/MultivectorDefs.lean** (check if still needed)
- ✅ **QFD/GA/MultivectorGrade.lean** (placeholders)

### Medium Priority:
- ✅ **QFD/Conservation/NeutrinoID.lean** (1 sorry)
- ✅ **QFD/Nuclear/TimeCliff.lean** (1 sorry)
- ✅ **QFD/Cosmology/AxisOfEvil.lean** (2 sorries)

### Lower Priority:
- ✅ **QFD/AdjointStability_Complete.lean** (2 sorries)
- ✅ **QFD/BivectorClasses_Complete.lean** (2 sorries)
- ✅ **QFD/SpacetimeEmergence_Complete.lean** (2 sorries)

---

## 📋 Rules for AI Assistants

### DO:
- ✅ Work on files in the "SAFE TO WORK ON" section
- ✅ Add proofs to complete sorries
- ✅ Add helpful comments explaining your proof strategy
- ✅ Test with `lake build` before claiming completion

### DON'T:
- ❌ Modify files in "ABSOLUTELY PROTECTED" section
- ❌ Change working proofs (0 sorry files)
- ❌ Reformat comments in infrastructure files
- ❌ Change imports or file structure
- ❌ Modify theorem statements (only add proofs to sorries)

### IF YOU MUST CHANGE SOMETHING PROTECTED:
1. **Stop immediately**
2. **Document why in comments**
3. **Ask the human for approval first**
4. **Make a backup copy**

---

## 🔍 How to Check if a File is Protected

```bash
# Is this file in the protected list?
grep -i "filename" PROTECTED_FILES.md

# Does this file have 0 sorries? (If yes, don't touch it!)
grep -c "sorry" QFD/Path/To/File.lean

# Is this file imported by many others? (If yes, be careful!)
rg "import.*FileName" QFD -g "*.lean" | wc -l
```

---

## 🚨 What to Do if You Accidentally Modified a Protected File

1. **Stop immediately**
2. **Check if it still builds**: `lake build QFD.GA.Cl33`
3. **If build fails**: Revert your changes with `git checkout QFD/GA/Cl33.lean`
4. **Report to human**: Explain what you changed and why

---

**Last Updated**: 2025-12-26
**Maintained by**: Human oversight, enforced by AI discipline
