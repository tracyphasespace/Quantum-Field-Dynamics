# Lean 4 Coding Guide for AI Tools

**Purpose**: Ensure AI-generated Lean code compiles successfully in this project.

---

## ⚠️ CRITICAL: Common AI Code Generation Failures

**Read this FIRST before generating any Lean code!**

Based on analysis of **recurring build failures** (70% initial failure rate), AI tools must change their process to avoid these systematic errors:

### 🚨 Failure Pattern 1: Structure Field Misunderstanding

**Problem**: AI tools incorrectly assume `Quantity` structures work like standard records.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - AI tools generate this invalid syntax
let params : NuclearParams := {
  c1 := { val := 1.0, unit := dimensionless }  -- WRONG: no 'unit' field exists!
}
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Only 'val' field exists, dimension is in the type
let params : NuclearParams := {
  c1 := { val := 1.0 }  -- Type is already 'Unitless' = Quantity dimensionless
}
```

**Why This Happens**: AI training data includes older codebases where units were fields. In THIS project, `Quantity d` is a dependent type where `d : Dimensions` is a type parameter, NOT a field.

**AI Process Change**:
1. When seeing `structure X where field : Quantity d`, recognize `d` is TYPE-LEVEL
2. Never generate `unit := ...` in structure literals
3. Check `QFD/Schema/DimensionalAnalysis.lean:40` for Quantity definition

---

### 🚨 Failure Pattern 2: Namespace Confusion

**Problem**: AI tools incorrectly nest namespaces when types are defined.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - AI tools guess at nested namespaces
open QFD.GA.Cl33              -- ERROR: unknown namespace
def x : QFD.GA.Cl33.Cl33      -- ERROR: unknown constant
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Only open parent namespace
open QFD.GA                   -- Cl33 is defined HERE
def x : Cl33                  -- Now in scope
```

**Why This Happens**: AI assumes file path = namespace hierarchy. In Lean 4, this is false! `QFD/GA/Cl33.lean` defines `QFD.GA.Cl33` as a TYPE, not a namespace.

**AI Process Change**:
1. **NEVER** generate `open X.Y.Z` unless you've verified Z is a declared namespace (not a type)
2. When importing `QFD.GA.Cl33`, always use `open QFD.GA`, not `open QFD.GA.Cl33`
3. Grep for `^namespace` in target file to verify namespace structure

---

### 🚨 Failure Pattern 3: Missing `noncomputable`

**Problem**: AI tools forget that real number operations require `noncomputable`.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - Division, sqrt, trig functions are non-computable
def ratio (x y : ℝ) : ℝ := x / y                    -- ERROR!
def distance (x : ℝ) : ℝ := Real.sqrt x             -- ERROR!
def phase (θ : ℝ) : ℝ := Real.cos θ                 -- ERROR!
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Mark as noncomputable
noncomputable def ratio (x y : ℝ) : ℝ := x / y      -- OK
noncomputable def distance (x : ℝ) : ℝ := Real.sqrt x   -- OK
noncomputable def phase (θ : ℝ) : ℝ := Real.cos θ   -- OK
```

**Why This Happens**: AI assumes Lean can compute on reals like it computes on integers. False! Reals are represented axiomatically.

**AI Process Change**:
1. **ALWAYS** mark definitions involving ℝ as `noncomputable` if they use:
   - Division (`/`)
   - `Real.sqrt`, `Real.exp`, `Real.log`
   - `Real.sin`, `Real.cos`, `Real.tan`
   - Any `Mathlib.Analysis.*` imports
2. Add `noncomputable` as the DEFAULT, remove only if compilation succeeds

---

### 🚨 Failure Pattern 4: Lean 3 vs Lean 4 Syntax

**Problem**: AI training data contains Lean 3 code; this project uses Lean 4.

**What AI Tools Do WRONG** (Lean 3 syntax):
```lean
-- ❌ WRONG - Lean 3 syntax doesn't work in Lean 4
classical.em _                -- Lowercase 'c'
Ex x, P x                     -- Wrong existential symbol
def x := sorry                -- Wrong sorry syntax (sometimes)
```

**What AI Tools MUST Do** (Lean 4 syntax):
```lean
-- ✅ CORRECT - Lean 4 requires these forms
Classical.em _                -- Capital 'C'
∃ x, P x                      -- Unicode exists (type ∃ or use \exists)
def x := sorry                -- Same, but prefer lemmas over defs for sorries
```

**Why This Happens**: Lean 3 is more prevalent in training data. Lean 4 changed many core library namespaces.

**AI Process Change**:
1. **CHECK PROJECT LEAN VERSION FIRST**: lakefile.toml shows `lean4:v4.27.0-rc1`
2. Always capitalize: `Classical`, `Nat`, `Int`, `Real`, `Finset`, etc.
3. Use `∃` not `Ex`, `∀` not `forall`
4. Import paths changed: `Mathlib.Tactic.*` not `tactic.*`

---

### 🚨 Failure Pattern 5: Unicode Character Confusion

**Problem**: AI tools use visually similar but wrong Unicode characters.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - Capital Iota (Greek Ι, U+0399) looks like Latin I
Ι33 (basis_vector 0)          -- ERROR: Unknown identifier

-- ❌ WRONG - Latin 'i' instead of Greek iota
i33 (basis_vector 0)          -- ERROR: Unknown identifier
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Lowercase Greek iota (ι, U+03B9)
ι33 (basis_vector 0)          -- OK: This is the generator map
```

**Why This Happens**: AI doesn't distinguish Unicode visually similar characters. `ι` (Greek) ≠ `i` (Latin) ≠ `Ι` (Greek capital).

**AI Process Change**:
1. When generating Clifford algebra code, **COPY-PASTE** `ι33` from `QFD/GA/Cl33.lean:87`
2. Never type it manually or generate it from scratch
3. Always use lowercase Greek: ι (iota), γ (gamma), μ (mu), ν (nu)
4. Verify with `grep "def ι33"` before using

---

### 🚨 Failure Pattern 6: Proof Tactics for Non-Commutative Algebras

**Problem**: AI tools use `ring` tactic everywhere, but Clifford algebras are non-commutative.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - ring doesn't work in Cl33 (non-commutative)
theorem maxwell : x * y = (1/2) • (x*y + y*x) + ... := by
  ring  -- ERROR: ring made no progress
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Use abel for module operations in non-commutative algebras
theorem maxwell : x * y = (1/2) • (x*y + y*x) + ... := by
  rw [smul_add, smul_sub, add_smul, sub_smul]
  abel  -- OK: abel works on abelian groups (scalar operations)
```

**Why This Happens**: AI assumes all algebras are commutative (ℝ, ℕ, ℤ are). Clifford algebras are NOT.

**AI Process Change**:
1. When working in `Cl33` or any Clifford algebra:
   - **NEVER** use `ring` for products
   - **DO** use `abel` for scalar (`•`) operations
   - **DO** use `ring` ONLY for the scalar coefficients (ℝ)
2. Import `Mathlib.Tactic.Abel` when working with modules
3. Use `mul_assoc`, `mul_comm`, `neg_mul`, `mul_neg` explicitly for products

---

### 🚨 Failure Pattern 7: Incomplete Structure Definitions

**Problem**: AI tools omit required fields when generating structure instances.

**What AI Tools Do WRONG**:
```lean
-- ❌ WRONG - Missing 'std' field
let params : GrandUnifiedParameters := {
  nuclear := ...,
  cosmo := ...,
  particle := ...
}  -- ERROR: Fields missing: `std`
```

**What AI Tools MUST Do**:
```lean
-- ✅ CORRECT - Include ALL required fields
let params : GrandUnifiedParameters := {
  std := { c := {val := 3e8}, G := {val := 6.67e-11}, ... },
  nuclear := ...,
  cosmo := ...,
  particle := ...
}
```

**Why This Happens**: AI tools extract partial examples from documentation and miss required fields.

**AI Process Change**:
1. **ALWAYS** read the structure definition FIRST:
   ```bash
   grep -A 10 "^structure GrandUnifiedParameters" QFD/Schema/Couplings.lean
   ```
2. List ALL fields before generating the instance
3. If a field is unfamiliar, search for its type definition
4. Test: Lean will error with "Fields missing: X" if incomplete

---

## 🔧 AI Pre-Generation Checklist

Before generating ANY Lean code, AI tools MUST:

- [ ] **1. Check Lean version**: Read `lakefile.toml` → Lean 4, not Lean 3
- [ ] **2. Verify namespace**: `grep "^namespace"` in target file
- [ ] **3. Check structure fields**: `grep "^structure TypeName where"` for all fields
- [ ] **4. Review type definitions**: For `Quantity`, `Cl33`, understand type parameters
- [ ] **5. Identify algebra type**: Commutative (ℝ, ℕ) → `ring`; Non-commutative (Cl33) → `abel`
- [ ] **6. Mark noncomputable**: If using ℝ division/sqrt/trig, add `noncomputable`
- [ ] **7. Copy Unicode correctly**: For `ι33`, γ, μ, ν → copy from existing files

---

## 🎯 AI Failure Recovery Process

If generated code fails to build:

### Step 1: Identify Error Type
```bash
lake build QFD.YourModule.YourFile 2>&1 | grep "error:"
```

Match error to patterns above:
- `Fields missing:` → Pattern 7 (Incomplete Structure)
- `unknown namespace` → Pattern 2 (Namespace Confusion)
- `needs 'noncomputable'` → Pattern 3 (Missing noncomputable)
- `unit is not a field` → Pattern 1 (Structure Field Misunderstanding)
- `Unknown identifier Ι33` → Pattern 5 (Unicode Confusion)
- `ring made no progress` → Pattern 6 (Wrong Tactic for Algebra)

### Step 2: Apply Specific Fix
Refer to pattern sections above for exact correction.

### Step 3: Verify Fix
```bash
lake build QFD.YourModule.YourFile
```

### Step 4: Document If New Pattern
If error doesn't match any pattern, add to this guide under new `🚨 Failure Pattern N`.

---

## 📊 Success Metrics

After applying these process changes, AI-generated code should achieve:
- **Target Success Rate**: >90% (currently 70% after manual fixes)
- **Sorries**: Only for unimplemented mathematical proofs (not syntax errors)
- **Build Time**: <3s per module (cached builds)

---

## 1. Common Syntax Errors to Avoid

### ✅ Correct Namespace Usage
```lean
-- ✅ CORRECT
import QFD.GA.Cl33
open QFD.GA  -- Opens the GA namespace, makes Cl33 available

def example : Cl33 := ...

-- ❌ WRONG - Don't nest namespaces incorrectly
open QFD.GA.Cl33  -- ERROR: unknown namespace
def example : QFD.GA.Cl33.Cl33 := ...  -- ERROR: unknown constant
```

### ✅ Unicode Characters (Case-Sensitive)
```lean
-- ✅ CORRECT
ι33 (basis_vector 0)  -- lowercase iota (ι)

-- ❌ WRONG
Ι33 (basis_vector 0)  -- Capital iota (Ι) - different character!
```

### ✅ Existential Quantifier
```lean
-- ✅ CORRECT
∃ x : ℝ, x > 0  -- Use ∃ (exists symbol)

-- ❌ WRONG
Ex x : ℝ, x > 0  -- "Ex" is not a keyword
```

### ✅ Tactic Parentheses
```lean
-- ✅ CORRECT
exact zero_ne_one h
exact absurd h zero_ne_one

-- ❌ WRONG
exact (zero_ne_one) h  -- Don't wrap tactics in parentheses
```

---

## 2. Noncomputable Definitions

Mark definitions using real numbers, division, or square roots as `noncomputable`:

```lean
-- ✅ CORRECT
noncomputable def mass_ratio (m1 m2 : ℝ) : ℝ := m1 / m2
noncomputable def norm_sq (x : ℝ) : ℝ := Real.sqrt x ^ 2

-- ❌ WRONG - Will fail to compile
def mass_ratio (m1 m2 : ℝ) : ℝ := m1 / m2  -- ERROR: needs noncomputable
```

---

## 3. Structure Definitions

Fields cannot reference each other directly. Use explicit annotations:

```lean
-- ✅ CORRECT
structure Config where
  radius : ℝ
  mass : ℝ
  constraint : mass = radius ^ (2 : ℕ)  -- Explicit ℕ annotation on exponent

-- ❌ WRONG - Fields not in scope during definition
structure Config where
  radius mass : ℝ
  constraint : mass = radius ^ 2  -- ERROR: type mismatch on ^
```

---

## 4. Module/Algebra Tactics

For **non-commutative algebras** (like Clifford algebras), use `abel` instead of `ring`:

```lean
-- ✅ CORRECT (for Cl33 proofs)
theorem maxwell_split : x * y = (1/2) • (x*y + y*x) + (1/2) • (x*y - y*x) := by
  rw [smul_add, smul_sub, add_smul, sub_smul]
  abel  -- Use abel for module operations

-- ❌ WRONG
theorem maxwell_split : ... := by
  ring  -- ERROR: ring doesn't work in non-commutative algebras
```

---

## 5. Lean 4 vs Lean 3 Changes

### Classical Logic
```lean
-- ✅ Lean 4
Classical.em _

-- ❌ Lean 3 (doesn't work)
classical.em _  -- lowercase 'c' is wrong
```

### Unused Variables
Remove unused variables from theorem statements:

```lean
-- ✅ CORRECT
theorem result (x : ℝ) (h : x > 0) : x ≠ 0 := by ...

-- ❌ WRONG - Linter warning
theorem result (x y z : ℝ) (h : x > 0) : x ≠ 0 := by ...  -- y, z unused
```

---

## 6. Project-Specific Patterns

### QFD.GA.Cl33 Usage
```lean
import QFD.GA.Cl33
import QFD.GA.BasisOperations

open QFD.GA
open CliffordAlgebra

-- Local helper for basis vectors
private def e (i : Fin 6) : Cl33 := ι33 (basis_vector i)

-- Use generator_squares_to_signature, NOT e_squares_to_signature
have : e 0 * e 0 = algebraMap ℝ Cl33 (signature33 0) :=
  generator_squares_to_signature 0

-- Use generators_anticommute, NOT e_anticommute
have : e 0 * e 1 = - (e 1 * e 0) :=
  add_eq_zero_iff_eq_neg.mp (generators_anticommute 0 1 (by decide))
```

### Handling `0 ≠ 1` Contradictions
```lean
import QFD.GA.Cl33Instances

-- Cl33 now has an explicit Nontrivial instance / lemmas:
intro hzero
have : (0 : Cl33) = 1 := by simpa [hzero]
exact zero_ne_one_Cl33 this
```

---

## 7. Common Error Messages & Fixes

| Error | Fix |
|-------|-----|
| `unknown namespace QFD.GA.Cl33` | Change to `open QFD.GA` |
| `Unknown identifier Ι33` (capital I) | Use lowercase `ι33` |
| `failed to synthesize NeZero 1` | Add sorry with TODO comment for Nontrivial instance |
| `needs noncomputable` | Add `noncomputable` keyword |
| `ring made no progress` | Use `abel` for non-commutative algebras |
| `Unknown identifier Ex` | Use `∃` (exists symbol) |
| `classical.em` not found | Capitalize: `Classical.em` |

---

## 8. Proof Patterns

### Contradiction from 0 = 1
```lean
-- Pattern (when Nontrivial instance is available):
intro hzero
simp only [hzero, zero_mul] at h_square  -- Simplify to 0 = 1
exact absurd h_square zero_ne_one  -- Close with contradiction

-- Current workaround:
sorry  -- TODO: Need Nontrivial Cl33 or algebraMap_injective
```

### Module Scalar Operations
```lean
-- Use explicit module lemmas:
rw [smul_add, smul_sub, add_smul, sub_smul]
rw [smul_smul]  -- For nested scalars
```

### Automation (clifford_simp / clifford_ring)
```lean
import QFD.GA.BasisReduction

example : e 0 * e 3 * e 0 = - e 3 := by
  clifford_simp

example : algebraMap ℝ Cl33 2 * (e 1 * e 1) = 2 := by
  clifford_ring
```
Always try these tactics before expanding long calc chains—the automation handles
sorting, sandwiches, and scalar simplifications automatically.

### Clifford Calc Chains
```lean
-- Use calc for step-by-step algebra:
calc e 0 * e 1 * e 0
    = e 0 * (e 1 * e 0) := by rw [mul_assoc]
  _ = e 0 * (- e 0 * e 1) := by rw [basis_anticomm (by decide)]
  _ = -(e 0 * e 0 * e 1) := by rw [neg_mul, mul_assoc]
  _ = - e 1 := by simp [basis_sq, signature33]
```

---

## 9. Testing Your Code

Before submitting:
1. **Build incrementally**: Use module-level builds like
   `lake build QFD.YourModule.YourFile` (avoid `lake clean` / `lake build QFD`
   which exceed memory on this machine—see `AI_ASSISTANT_QUICK_START.md`).
2. **Check dependencies**: Ensure all imports exist
3. **Verify linters**: Fix warnings about unused variables, long lines
4. **Document sorries**: Add `-- TODO:` comments explaining what's needed

---

## 10. Quick Reference

### Essential Imports for QFD Proofs
```lean
import Mathlib.LinearAlgebra.CliffordAlgebra.Basic
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Tactic.Ring
import QFD.GA.Cl33
import QFD.GA.BasisOperations
```

### Essential Opens
```lean
open QFD.GA
open CliffordAlgebra
```

### Type Annotations
- Exponents: `^ (2 : ℕ)`
- Scalars in Cl33: `algebraMap ℝ Cl33 r`
- Module scalar multiply: `r • x`

---

## Summary Checklist

- [ ] Used lowercase `ι33` not capital `Ι33`
- [ ] Opened `QFD.GA` not `QFD.GA.Cl33`
- [ ] Used `∃` not `Ex` for exists
- [ ] Marked real-valued defs as `noncomputable`
- [ ] Used `abel` not `ring` for Clifford algebra
- [ ] Used `Classical.em` not `classical.em`
- [ ] Removed unused theorem parameters
- [ ] Added `-- TODO:` for all `sorry` statements
- [ ] Tested with `lake build`

---

**Last Updated**: 2025-12-27
**Lean Version**: 4.27.0-rc1
**Mathlib**: Compatible with project lakefile
