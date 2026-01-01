# Common Build Errors and Solutions

**Last Updated**: 2025-12-27
**Companion to**: BUILD_VERIFICATION_PROTOCOL.md

This document catalogs actual build errors encountered during QFD formalization and their solutions.

---

## Table of Contents
1. [Reserved Keywords](#reserved-keywords)
2. [Namespace Errors](#namespace-errors)
3. [Tactic Failures](#tactic-failures)
4. [Import Issues](#import-issues)
5. [Type Mismatches](#type-mismatches)
6. [Differentiability Proofs](#differentiability-proofs)

---

## Reserved Keywords

### Error: Using `lambda` as variable name
```
error: QFD/Nuclear/YukawaDerivation.lean:47:41: expected command
```

**Cause**: `lambda` is a Lean 4 keyword (anonymous functions)

**Solution**: Rename to `lam`, `screening_length`, or `decay_const`

**Example Fix**:
```lean
-- BEFORE (fails)
def rho_soliton (A lambda : ℝ) : ℝ := A * exp (-lambda * r) / r

-- AFTER (works)
def rho_soliton (A lam : ℝ) : ℝ := A * exp (-lam * r) / r
```

**Other Reserved Keywords to Avoid**:
- `def`, `theorem`, `axiom`, `inductive`
- `if`, `then`, `else`, `match`, `where`
- `fun`, `let`, `have`, `show`
- `by`, `do`, `return`

**Physics Variable Alternatives**:
- λ (lambda) → `lam`, `wavelength`, `decay_const`
- μ (mu) → `mu`, `mass`, `chemical_potential`
- ν (nu) → `nu`, `frequency`, `flavor`

---

## Namespace Errors

### Error: Unknown namespace for type
```
error: unknown namespace 'QFD.GA.Cl33'
```

**Cause**: `Cl33` is a type, not a namespace. You can't `open` types.

**Solution**: Open the parent namespace instead

**Example Fix**:
```lean
-- BEFORE (fails)
import QFD.GA.Cl33Instances
open QFD.GA.Cl33  -- ERROR: Cl33 is a type!

-- AFTER (works)
import QFD.GA.Cl33Instances
open QFD.GA  -- Open the namespace containing Cl33
```

**Rule**: Only open namespaces (module paths), never types or instances.

---

## Tactic Failures

### Error 1: `unfold` failed to unfold definition
```
error: Tactic 'unfold' failed to unfold 'rho_soliton'
```

**Cause**: Equation theorems not generated for noncomputable definitions

**Solution 1**: Use `simp only [definition_name]`
```lean
simp only [rho_soliton]
```

**Solution 2**: Use `change` to explicitly rewrite
```lean
change deriv (fun x => A * (exp (-lam * x) / x)) r = ...
```

**Solution 3**: Use `rfl` in a `have` statement
```lean
have h : rho_soliton A lam = fun x => A * (exp (-lam * x) / x) := by rfl
rw [h]
```

**Solution 4**: Add documented sorry
```lean
sorry  -- TODO: Complete proof after equation theorem fix
```

### Error 2: `rewrite` failed - pattern not found
```
error: Tactic `rewrite` failed: Did not find an occurrence of the pattern
  deriv (fun x ↦ ?c * ?f x) = fun x ↦ ?c * deriv ?f x
```

**Cause**: Mathlib lemmas have specific pattern requirements

**Solution**: Try alternative approaches
```lean
-- Option 1: Use conv to transform goal first
conv_lhs => arg 1; ext x; rw [mul_comm]
rw [deriv_mul_const]

-- Option 2: Prove intermediate lemma explicitly
have h : deriv (fun x => A * f x) r = A * deriv f r := by
  apply deriv_const_mul
  exact h_diff
rw [h]

-- Option 3: Use calc for step-by-step proof
calc deriv F r
    = deriv (fun x => A * f x) r := rfl
  _ = A * deriv f r := by apply deriv_const_mul; exact h_diff
  _ = target_expression := by ring
```

### Error 3: `simp` made no progress
```
error: `simp` made no progress
```

**Cause**: Simplifier doesn't know how to handle the expression

**Solution**: Be more explicit
```lean
-- Instead of: simp
-- Try:
simp only [definition_name, add_comm, mul_comm]
-- Or:
ring
-- Or:
field_simp
```

---

## Import Issues

### Error: Unknown constant from Mathlib
```
error: unknown constant 'Matrix.det'
```

**Cause**: Missing Mathlib import

**Solution**: Add required import at file top
```lean
import Mathlib.Data.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
```

**Finding the Right Import**:
1. Search Mathlib docs: https://leanprover-community.github.io/mathlib4_docs/
2. Use Lean's autocomplete to see where constant is defined
3. Check similar files that use the same constant

### Error: Circular import
```
error: Circular dependency detected
```

**Solution**: Reorganize dependencies
- Move shared definitions to separate file
- Import only what you need
- Check import graph with `lake exe graph`

---

## Type Mismatches

### Error 1: Function expected, got value
```
error: Type mismatch
  has type: ℝ → ℝ
  expected: ℝ
```

**Cause**: Forgot to apply function to argument

**Solution**: Add the argument
```lean
-- BEFORE (fails)
theorem foo : deriv f = 5

-- AFTER (works)
theorem foo : deriv f x = 5
```

### Error 2: Value expected, got function
```
error: Type mismatch
  has type: ℝ
  expected: ℝ → ℝ
```

**Cause**: Applied too many arguments or forgot lambda

**Solution**: Wrap in lambda or remove argument
```lean
-- Option 1: Wrap in lambda
theorem foo : (fun x => f x) = g

-- Option 2: Use function directly
theorem foo : f = g
```

### Error 3: Definitional equality expected
```
error: 'show' tactic failed, pattern
  expr1
is not definitionally equal to target
  expr2
```

**Cause**: Expressions are propositionally equal but not definitionally equal

**Solution**: Prove equality and rewrite
```lean
-- Instead of: show expr1 = target
-- Use:
have h : expr1 = intermediate := by rfl
rw [h]
-- Then prove: intermediate = target
```

---

## Differentiability Proofs

### Error 1: DifferentiableAt pattern mismatch
```
error: Tactic `apply` failed: could not unify the conclusion
  DifferentiableAt ?𝕜 (-?f) ?x
with the goal
  DifferentiableAt ℝ (HMul.hMul (-lam)) r
```

**Cause**: Incorrect differentiability construction for composite functions

**Solution**: Build differentiability properly
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

### Error 2: Missing differentiability assumption
```
error: Failed to synthesize DifferentiableAt ℝ f x
```

**Solution**: Add hypothesis or prove it
```lean
-- Option 1: Add to theorem statement
theorem foo (h_diff : DifferentiableAt ℝ f x) : ...

-- Option 2: Prove inline
have h_diff : DifferentiableAt ℝ f x := by
  apply DifferentiableAt.div
  · exact diff_proof_numerator
  · exact diff_proof_denominator
  · exact h_nonzero
```

---

## Pattern: Proof Won't Simplify Algebraically

### Error: `ring` fails to close goal
```
error: ring tactic failed, ring expressions not equal
⊢ A * (-lam * exp (-lam * r) / r - exp (-lam * r) / r^2) =
  -A * exp (-lam * r) * (1 / r^2 + lam / r)
```

**Cause**: Expression needs field simplification before ring

**Solution**: Use `field_simp` before `ring`
```lean
field_simp
ring
```

**Alternative**: Use `ring_nf` for normalization
```lean
ring_nf
```

---

## Pattern: Goals with `let` bindings

### Error: Can't rewrite under `let`
```
⊢ let F := deriv f x
  F = target
```

**Solution**: Use `intro` to bring binding into context
```lean
intro F
-- Now goal is: F = target
-- And F is in context as: F : ℝ := deriv f x
```

---

## Debugging Tactics

### See current goal in detail
```lean
trace "{goal}"  -- Shows exact goal structure
```

### See all hypotheses
```lean
trace "{context}"  -- Shows all available facts
```

### Check type of expression
```lean
#check my_expression  -- Shows type
```

### Evaluate expression
```lean
#reduce my_expression  -- Simplifies and shows result
```

---

## When All Else Fails

1. **Simplify the theorem statement**
   - Break complex statement into smaller lemmas
   - Remove unnecessary generality

2. **Add intermediate `have` statements**
   - Prove sub-goals explicitly
   - Build up to final result

3. **Use `sorry` with documentation**
   ```lean
   sorry
   -- TODO: Complete proof
   -- Attempted: tactics list
   -- Blocker: specific issue
   ```

4. **Ask for help**
   - Document exactly what you tried
   - Show the full error message
   - Provide minimal reproducible example

---

## Quick Reference: Error → Solution

| Error Message Contains | Likely Solution |
|------------------------|-----------------|
| "expected command" near keyword | Rename reserved keyword variable |
| "unknown namespace" | Open parent namespace, not type |
| "unfold failed" | Use `simp only [def]` or `change` |
| "rewrite failed: Did not find" | Use `conv`, prove intermediate lemma, or different tactic |
| "simp made no progress" | Use `simp only [...]` with explicit lemmas |
| "unknown constant" | Add missing import |
| "Type mismatch" | Check function application / abstraction |
| "DifferentiableAt" unification | Fix composite differentiability proof structure |
| "ring failed" | Try `field_simp; ring` |

---

**For More Help**:
- BUILD_VERIFICATION_PROTOCOL.md - Required testing procedures
- LEAN_CODING_GUIDE.md - Coding style and best practices
- Mathlib docs - https://leanprover-community.github.io/mathlib4_docs/

**Generated**: 2025-12-27 by QFD Formalization Team
