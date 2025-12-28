# AI Workflow: Build Verification & Error Solutions

**Last Updated**: 2025-12-27
**Purpose**: Complete guide for AI assistants contributing to Lean4 formalization

---

## 📚 Related Documentation

- **MATHLIB_SEARCH_GUIDE.md** - How to find theorems in Mathlib, handle type system issues
- **COMPLETE_GUIDE.md** - Full system architecture and proof patterns
- **PROTECTED_FILES.md** - Files that should not be modified

---

## 🚨 GOLDEN RULE

**Write ONE proof → `lake build` → Fix errors → Verify → Next proof**

**NEVER submit work without successful build verification.**

---

## Part 1: Iterative Proof Development

### The One-Proof-at-a-Time Cycle

```
Select ONE theorem
    ↓
Write/modify proof
    ↓
lake build IMMEDIATELY
    ↓
✅ Success? → Move to next theorem
❌ Error? → Fix the ONE error → Build again
```

### Step-by-Step Process

**1. Select ONE Theorem**
```lean
-- ✅ Already done
def rho_soliton : ℝ → ℝ := ...

-- 👈 WORK ON THIS ONE
theorem proof_1 : statement := by
  sorry  -- Replace this

-- ⏭️ SKIP FOR NOW
theorem proof_2 : statement := by
  sorry
```

**2. Write the Proof**
```lean
theorem proof_1 : statement := by
  unfold definition
  rw [lemma]
  ring
```

**3. Build Immediately**
```bash
lake build QFD.Module.Name 2>&1 | tee build_log.txt
```

**4. Check Result**

✅ **Success** (no errors):
```
✔ [3081/3081] Building QFD.Module.Name
```
→ Move to next theorem

❌ **Failure** (has errors):
```
error: QFD/Module/Name.lean:82:6: Tactic `unfold` failed
```
→ Read error, fix, rebuild

**5. Repeat Until All Theorems Complete**

---

## Part 2: Build Verification Requirements

### Required for EVERY File Modification

**Step 1**: Make your changes

**Step 2**: Build the specific module
```bash
lake build QFD.Module.FileName 2>&1 | tee build_log.txt
```

**Step 3**: Check build output

✅ **SUCCESS** looks like:
```
✔ [3063/3063] Building QFD.Nuclear.YukawaDerivation
```
OR with warnings (acceptable):
```
warning: declaration uses 'sorry'
```

❌ **FAILURE** looks like:
```
error: QFD/File.lean:82:6: Tactic failed
error: Lean exited with code 1
error: build failed
```

**Step 4**: Fix ALL errors
- If you see ANY `error:` lines, work is NOT complete
- Read error message (line number + problem)
- Fix and return to Step 2

**Step 5**: Test downstream dependencies (if applicable)
```bash
lake build QFD.Module.Dependency
```

### Completion Criteria

**✅ Can Submit When:**
- `lake build` shows 0 errors (warnings OK)
- Build log saved and included
- Any `sorry` documented with TODO

**❌ Do NOT Submit If:**
- Haven't run `lake build`
- Build has ANY `error:` lines
- Tested "in your head" without actual build
- Added `sorry` without documentation

### When to Use `sorry`

**✅ ACCEPTABLE**:
```lean
theorem hard_proof : statement := by
  sorry
  -- TODO: Complete using quotient rule
  -- Attempts: unfold (failed), simp (failed)
  -- Blocker: Mathlib pattern matching issue
```

**❌ UNACCEPTABLE**:
```lean
theorem easy_proof : statement := by
  sorry  -- just to make it compile
```

---

## Part 3: Common Build Errors & Solutions

### Error 1: Reserved Keywords

```
error: expected command
lambda
^
```

**Cause**: `lambda` is a Lean 4 keyword

**Fix**: Rename to `lam`, `wavelength`, etc.

**Avoid**: `lambda`, `def`, `theorem`, `if`, `then`, `else`, `fun`, `let`, `have`

---

### Error 2: Unknown Namespace

```
error: unknown namespace 'QFD.GA.Cl33'
```

**Cause**: `Cl33` is a type, not a namespace

**Fix**:
```lean
-- BEFORE (wrong)
open QFD.GA.Cl33

-- AFTER (correct)
open QFD.GA
```

**Rule**: Only open namespaces, never types

---

### Error 3: Tactic Failures

#### `unfold` failed
```
error: Tactic 'unfold' failed to unfold 'definition'
```

**Solutions**:
```lean
-- Option 1: Use simp only
simp only [definition]

-- Option 2: Use change
change explicit_expression = target

-- Option 3: Documented sorry
sorry -- TODO: unfold fails for noncomputable defs
```

#### `rewrite` pattern not found
```
error: Tactic `rewrite` failed: Did not find occurrence
```

**Solutions**:
```lean
-- Option 1: Use conv
conv_lhs => arg 1; ext x; rw [mul_comm]
rw [lemma]

-- Option 2: Prove intermediate lemma
have h : expr1 = expr2 := by apply lemma; exact proof
rw [h]

-- Option 3: Use calc
calc expr1
    = expr2 := by rw [lemma1]
  _ = expr3 := by ring
```

#### `simp` made no progress
```
error: `simp` made no progress
```

**Fix**: Be more explicit
```lean
-- Instead of: simp
-- Use:
simp only [def1, def2, add_comm]
-- Or:
ring
-- Or:
field_simp
```

---

### Error 4: Import Issues

```
error: unknown constant 'Matrix.det'
```

**Fix**: Add missing import
```lean
import Mathlib.Data.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
```

**Finding imports**: Search Mathlib docs at https://leanprover-community.github.io/mathlib4_docs/

---

### Error 5: Type Mismatches

```
error: Type mismatch
  has type: ℝ → ℝ
  expected: ℝ
```

**Cause**: Forgot to apply function to argument

**Fix**: Add the argument
```lean
-- BEFORE (wrong)
theorem foo : deriv f = 5

-- AFTER (correct)
theorem foo : deriv f x = 5
```

---

### Error 6: Differentiability Proofs

```
error: could not unify
  DifferentiableAt ℝ (HMul.hMul (-lam)) r
```

**Cause**: Incorrect differentiability construction

**Fix**: Build properly for composite functions
```lean
-- For exp(-lam * x):
apply DifferentiableAt.exp
apply DifferentiableAt.const_mul
exact differentiableAt_id

-- For exp(-lam * x) / x:
apply DifferentiableAt.div
· apply DifferentiableAt.exp
  apply differentiableAt_id.const_mul
· exact differentiableAt_id
· exact h_nonzero
```

---

### Error 7: `ring` Fails

```
error: ring tactic failed, ring expressions not equal
```

**Fix**: Use `field_simp` before `ring`
```lean
field_simp
ring
```

---

## Part 4: Finding Mathlib Theorems

**See MATHLIB_SEARCH_GUIDE.md for comprehensive guide**

### When You Need a Mathlib Theorem

**Symptoms**:
- You know a mathematical fact should be true (e.g., "exp(conj z) = conj(exp z)")
- You need to prove standard properties (e.g., cos(x) = Re(exp(ix)))
- You're working with complex numbers, roots of unity, trigonometry, etc.

### Quick Search Process

#### 1. Try `exact?` First (10 seconds)
```lean
theorem my_goal : statement := by
  exact?  -- Let Lean suggest the theorem
```

#### 2. Grep Mathlib Source (1-2 minutes)
```bash
# Find theorems by name pattern
grep -r "theorem.*exp.*conj" .lake/packages/mathlib/Mathlib/Analysis/Complex/

# Find all theorems about a concept
grep -r "theorem.*_re" .lake/packages/mathlib/Mathlib/Data/Complex/Basic.lean

# See context around a theorem
grep -B5 -A5 "theorem_name" .lake/packages/mathlib/Path/To/File.lean
```

#### 3. Check for Scopes and Notation
```bash
# Some identifiers need scopes opened
grep -n "scoped\|open.*Scope" .lake/packages/mathlib/Path/To/File.lean

# Example: 'conj' needs ComplexConjugate scope
# Solution: Add 'open ComplexConjugate' to your namespace
```

### Common Pitfalls

#### Unknown Identifier vs Unknown Constant

```lean
-- ❌ Error: Unknown identifier 'conj'
-- Fix: Add 'open ComplexConjugate'

-- ❌ Error: Unknown constant 'Complex.conj'
-- Fix: It's notation! Use 'conj' not 'Complex.conj'

-- ✅ Correct: Use notation without prefix
have h : conj z = ...

-- ✅ Correct: Use theorem with prefix
simp [Complex.conj_re, Complex.conj_im]
```

#### Type Casts Don't Match

```lean
-- Goal has: Complex.I * ↑(delta + c)
-- Your lemma has: Complex.I * (delta + c)

-- Fix: Explicit rewrite
rw [show Complex.I * ↑(delta + c) = Complex.I * (delta + c) by
      simp [Complex.ofReal_add]]
```

### Essential Mathlib Modules

- **Complex basics**: `Mathlib.Data.Complex.Basic`
- **Complex exponential**: `Mathlib.Analysis.Complex.Exponential`
- **Real trig**: `Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic`
- **Roots of unity**: `Mathlib.RingTheory.RootsOfUnity.Complex`

**See MATHLIB_SEARCH_GUIDE.md for complete reference with examples**

---

## Part 5: Debugging Tactics

```lean
-- See current goal
trace "{goal}"

-- See all hypotheses
trace "{context}"

-- Check type
#check expression

-- Evaluate
#reduce expression
```

---

## Part 5: Completion Report Template

### Good Report
```markdown
## Work Completed: ModuleName

### Files Modified:
- QFD/Section/ModuleName.lean (fixed issue X)

### Build Command:
lake build QFD.Section.ModuleName

### Build Output:
✔ [3081/3081] Building QFD.Section.ModuleName
warning: declaration uses 'sorry' (line 72 - documented)

### Summary:
✅ 0 errors, 1 documented sorry
✅ Completed theorems: proof_1, proof_2
⏳ Remaining: proof_3 (needs advanced technique)

### Iterations:
- Attempt 1: unfold failed → switched to simp only
- Attempt 2: rewrite failed → used calc chain
- Attempt 3: SUCCESS

### Build Log:
Attached: build_ModuleName.txt
```

### Bad Report
```markdown
I made the changes. The code should work now.
```
**Problems**: No verification, no build log, no evidence

---

## Quick Reference

| Error Contains | Solution |
|----------------|----------|
| "expected command" + keyword | Rename reserved keyword |
| "unknown namespace" | Open parent namespace, not type |
| "unfold failed" | Use `simp only` or `change` |
| "rewrite failed" | Use `conv` or prove intermediate |
| "simp made no progress" | Use `simp only [...]` |
| "unknown constant" | Add import |
| "Type mismatch" | Check function application |
| "DifferentiableAt" | Fix composite proof structure |
| "ring failed" | Try `field_simp; ring` |

---

## Summary

**The Workflow**:
1. Pick ONE proof
2. Write attempt
3. `lake build` immediately
4. Fix the ONE error shown
5. Rebuild
6. Repeat until success
7. Move to next proof

**The Rule**:
> If `lake build` shows ANY `error:` lines, your work is NOT complete.

**The Benefit**:
- Immediate feedback
- Isolated debugging
- No cascading failures
- Clear progress
- Self-correcting loop

---

**Generated**: 2025-12-27 by QFD Formalization Team
**Required Reading**: Before starting ANY work on Lean4 formalization
