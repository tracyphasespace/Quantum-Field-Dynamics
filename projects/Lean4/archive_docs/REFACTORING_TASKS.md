# Refactoring Tasks for Other AI

## Priority 1: YukawaDerivation.lean (Blocks BreatherModes)

**File**: `QFD/Nuclear/YukawaDerivation.lean`

### Issues to Fix:

#### 1. Reserved Keyword Violation ❌ CRITICAL
**Problem**: Using `lambda` as a variable name (reserved keyword in Lean 4)
**Lines**: 47, 56, 57, 72, 73, 74, 86, 92, 112, 134, 136, 139

**Fix**: Rename ALL occurrences of `lambda` to `lam` (or `screening_length`)
- Line 47: `noncomputable def rho_soliton (A lam : ℝ) (r : ℝ) : ℝ :=`
- Line 48: `A * (exp (-lam * r)) / r`
- Line 56: `noncomputable def vacuum_force (k A lam : ℝ) (r : ℝ) : ℝ :=`
- Line 57: `-k * deriv (rho_soliton A lam) r`
- Line 72: `theorem soliton_gradient_is_yukawa (A lam : ℝ) (r : ℝ)`
- Line 74: `F_geometric = -A * (exp (-lam * r)) * (1/r^2 + lam/r)`
- And so on...

**Search and Replace**:
```lean
# Wrong (current)
lambda

# Correct (change to)
lam
```

**Important**: Make sure to update:
- Function signatures
- Function bodies
- Theorem statements
- Proofs
- Comments/documentation

#### 2. Doc-string Formatting Errors
**Lines**: 41, 50

**Problem**: Doc-strings must have a space after `/-!`

**Fix**:
```lean
# Line 41 - BEFORE
/--
The Soliton Density Profile

# Line 41 - AFTER
/--
The Soliton Density Profile
```

```lean
# Line 50 - BEFORE
/--
The Vacuum Force

# Line 50 - AFTER
/--
The Vacuum Force
```

#### 3. Equation Theorem Rewrite Failure
**Line**: 82

**Problem**: `rw [rho_soliton]` fails because equation theorems aren't being generated

**Fix**: Use `unfold rho_soliton` instead:
```lean
# BEFORE (line 82)
rw [rho_soliton]

# AFTER
unfold rho_soliton
```

#### 4. Build Verification
After making changes, verify:
```bash
lake build QFD.Nuclear.YukawaDerivation
lake build QFD.Soliton.BreatherModes  # Should now succeed
```

---

## Priority 2: IsomerDecay.lean (Currently Just a Stub)

**File**: `QFD/Nuclear/IsomerDecay.lean`

### Status: ✅ NO CHANGES NEEDED

This file is intentionally a stub/placeholder. It's blocked by `MagicNumbers` which depends on `Schema.Constraints`, but the file itself has no errors.

**Action**: Leave this file as-is until Schema.Constraints is fixed.

---

## Refactoring Checklist

### For YukawaDerivation.lean:
- [ ] Replace ALL `lambda` → `lam` (check 12+ occurrences)
- [ ] Fix doc-string formatting at lines 41, 50
- [ ] Change `rw [rho_soliton]` to `unfold rho_soliton` at line 82
- [ ] Verify build: `lake build QFD.Nuclear.YukawaDerivation`
- [ ] Verify downstream: `lake build QFD.Soliton.BreatherModes`

### For IsomerDecay.lean:
- [ ] No action required - waiting on Schema.Constraints fix

---

## Expected Outcomes

### After YukawaDerivation refactoring:
- ✅ YukawaDerivation builds with 0 errors
- ✅ BreatherModes is unblocked and should build
- ✅ Nuclear force derivation proofs are complete

### Remaining blockers (not this refactor):
- Schema.Constraints (blocks 5 Nuclear/Cosmo modules)
- NeutrinoID/Mathlib issues (blocks Conservation and Weak modules)

---

## Reference: Common Lean 4 Reserved Keywords

**DO NOT use these as variable names:**
- `lambda` (use `lam`, `λ_val`, or descriptive name)
- `def`, `theorem`, `axiom`, `inductive`
- `if`, `then`, `else`, `match`
- `fun`, `let`, `have`, `show`

**Safe alternatives for physics:**
- λ parameter → `lam`, `wavelength`, `decay_const`
- μ parameter → `mu`, `mass`, `chemical_potential`
- ν parameter → `nu`, `frequency`, `neutrino_flavor`

---

**Generated**: 2025-12-27 by Claude Code
**For**: Other AI refactoring assistant
**Estimated Time**: 15 minutes for complete refactoring
