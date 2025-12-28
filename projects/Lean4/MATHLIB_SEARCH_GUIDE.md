# Mathlib Search Guide: Finding Theorems in Lean 4

**Created**: 2025-12-27
**Context**: Lessons learned from proving Euler's formula in `QFD/Lepton/KoideRelation.lean`

This guide documents how to find theorems, handle type system issues, and work with Mathlib effectively.

---

## Quick Reference: Search Tools

### 1. **Loogle** (Online Search by Type Signature)
- **URL**: https://loogle.lean-lang.org/
- **Use when**: You know what type signature you need but not the theorem name
- **Example**: Search for `ℝ → (ℂ → ℂ)` to find real-to-complex coercion theorems

### 2. **Built-in Tactics** (VSCode Integration)
```lean
-- In a proof, these tactics suggest theorems:
exact?  -- Finds exact proof term for current goal
apply?  -- Finds applicable theorems
rw?     -- Suggests rewrite lemmas
```

**Shortcut**: Ctrl-K Ctrl-S in VSCode to search Mathlib

### 3. **Direct Source Grep** (Most Reliable)
```bash
# Find theorem definitions
grep -r "theorem.*name" .lake/packages/mathlib/Mathlib/Path/To/Module.lean

# Find all theorems about a concept
grep -r "theorem.*cos.*exp" .lake/packages/mathlib/Mathlib/Analysis/Complex/

# Find lemma usage (to understand context)
grep -B5 -A5 "lemma_name" .lake/packages/mathlib/Mathlib/Path/To/Module.lean
```

### 4. **#check and #print** (In Lean REPL)
```lean
#check Complex.exp_conj  -- See type signature
#print Complex.exp_conj  -- See full definition
#check conj              -- Check if identifier exists
```

---

## Case Study: Proving Euler's Formula

**Goal**: Prove `Real.cos x = (Complex.exp (Complex.I * x)).re`

### Step 1: Identify the Challenge
- `Real.cos` and `Complex.exp` are different types
- Need to connect via complex conjugation
- Type casts (`↑`) complicate pattern matching

### Step 2: Find the Right Lemmas

#### Method 1: Grep Mathlib Source
```bash
# Find conjugation lemmas
grep -r "theorem.*_re" .lake/packages/mathlib/Mathlib/Data/Complex/Basic.lean | grep conj

# Result:
# theorem conj_re (z : ℂ) : (conj z).re = z.re :=
# theorem conj_im (z : ℂ) : (conj z).im = -z.im :=
```

#### Method 2: Search for Exponential-Conjugate Connection
```bash
grep -r "theorem.*exp.*conj" .lake/packages/mathlib/Mathlib/Analysis/Complex/Exponential.lean

# Result:
# theorem exp_conj : exp (conj x) = conj (exp x) := by
```

#### Method 3: Understand Scope and Notation
```bash
# Find where 'conj' is defined
grep -n "notation.*conj" .lake/packages/mathlib/Mathlib/Data/Complex/Basic.lean

# Result (line 429):
# notation `conj` in the scope `ComplexConjugate`. -/
```

**Key Discovery**: `conj` is **notation**, not a function! It's defined in the `ComplexConjugate` scope.

### Step 3: Import the Right Scope
```lean
import Mathlib.Analysis.Complex.Exponential
-- At namespace level:
open ComplexConjugate  -- Makes 'conj' notation available
```

### Step 4: Use Notation vs Theorems Correctly
```lean
-- ✅ CORRECT: 'conj' is notation (no prefix)
have h : Complex.exp (-↑x * Complex.I) = conj (Complex.exp (↑x * Complex.I))

-- ✅ CORRECT: Lemmas need Complex. prefix
simp [Complex.conj_ofReal, Complex.conj_I, Complex.conj_re]

-- ❌ WRONG: Don't prefix notation
conj (z)          -- ✅ Correct
Complex.conj (z)  -- ❌ Error: Unknown constant
```

---

## Common Type System Issues

### Issue 1: Cast Matching (`↑(a + b)` vs `↑a + ↑b`)

**Problem**: Lean doesn't automatically recognize `↑(delta + 2*π/3) = ↑delta + ↑(2*π/3)`

**Solution**: Explicit rewrite with `ofReal_add`
```lean
rw [show Complex.I * ↑(delta + c) = Complex.I * (delta + c) by
      simp [Complex.ofReal_add]]
```

**Key Lemma**: `Complex.ofReal_add : ↑(a + b) = ↑a + ↑b`

### Issue 2: `star` vs `conj` in Lean 4

**Problem**: Complex conjugation uses `Star` typeclass, but theorems use `conj` notation

**Hierarchy**:
```lean
-- Definition (StarRing instance)
instance : StarRing ℂ where star z := ⟨z.re, -z.im⟩

-- Notation (in ComplexConjugate scope)
notation "conj" => starRingEnd ℂ

-- Theorems (use notation)
theorem conj_re (z : ℂ) : (conj z).re = z.re
```

**Best Practice**: Use `conj` notation, not `star` directly

### Issue 3: Real vs Complex Cosine

**Problem**: `Real.cos : ℝ → ℝ` is distinct from `Complex.cos : ℂ → ℂ`

**Solution**: Use definitional equality
```lean
conv_lhs => rw [show Real.cos x = (Complex.cos x).re from rfl]
```

**Key Insight**: `Real.cos x` is **defined** as `(Complex.cos ↑x).re`, so `rfl` proves it

---

## Proof Pattern Library

### Pattern 1: Complex Conjugation for Real Arguments

**Goal**: Prove `exp(-↑x * I) = conj(exp(↑x * I))` for real `x`

**Strategy**:
1. Apply conjugation property: `exp(conj z) = conj(exp z)`
2. Show argument matches: `conj(↑x * I) = -↑x * I`
3. Use lemmas: `conj_ofReal` (real coercion commutes) and `conj_I = -I`

```lean
have h : exp (-↑x * I) = conj (exp (↑x * I)) := by
  rw [← Complex.exp_conj]
  congr 1
  simp [Complex.conj_ofReal, Complex.conj_I]
```

### Pattern 2: Roots of Unity Sums

**Goal**: Prove `1 + ω + ω² = 0` where `ω = exp(2πi/3)`

**Strategy**:
1. Show ω is primitive 3rd root: `IsPrimitiveRoot ω 3`
2. Apply Mathlib theorem: `IsPrimitiveRoot.geom_sum_eq_zero`
3. Expand finite sum to explicit form

```lean
lemma sum_third_roots : 1 + omega + omega^2 = 0 := by
  have h := omega_is_primitive_root
  have h_sum : (Finset.range 3).sum (fun i => omega ^ i) = 0 := by
    apply IsPrimitiveRoot.geom_sum_eq_zero h
    norm_num
  have : (Finset.range 3).sum (fun i => omega ^ i) = omega^0 + omega^1 + omega^2 := by
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one]
    ring
  rw [this] at h_sum
  simpa using h_sum
```

### Pattern 3: Factoring Exponential Sums

**Goal**: Show `exp(iδ) + exp(i(δ+2π/3)) + exp(i(δ+4π/3)) = 0`

**Strategy**: Factor out common `exp(iδ)` term

```lean
have h_factor :
  exp (I * delta) * (1 + omega + omega^2) =
  exp (I * delta) + exp (I * (delta + 2*π/3)) + exp (I * (delta + 4*π/3)) := by
    rw [omega, mul_add, mul_add, mul_one]
    congr 1
    · -- exp(iδ) * exp(2πi/3) = exp(i(δ + 2π/3))
      rw [← Complex.exp_add]
      congr 1
      push_cast
      ring_nf
    · -- exp(iδ) * exp(4πi/3) = exp(i(δ + 4π/3))
      rw [sq, mul_comm, mul_assoc, ← Complex.exp_add, ← Complex.exp_add]
      congr 1
      push_cast
      ring_nf
rw [← h_factor, sum_third_roots_eq_zero, mul_zero]
```

**Key Tactics**:
- `push_cast`: Push coercions through operations
- `ring_nf`: Normalize ring expressions (when `ring` fails)
- `congr`: Match goal structure before simplifying arguments

---

## Essential Imports for Complex Proofs

```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic  -- Real trig
import Mathlib.Analysis.Complex.Exponential                    -- Complex exp
import Mathlib.RingTheory.RootsOfUnity.Complex                 -- Roots of unity
import Mathlib.Tactic.Ring                                     -- ring tactic
import Mathlib.Tactic.FieldSimp                                -- field_simp

-- Open necessary scopes
open ComplexConjugate  -- For 'conj' notation
open Real              -- For π, cos, sin without prefix
```

---

## Debugging Checklist

When a proof fails, check:

### 1. **Identifier Unknown**
```
error: Unknown identifier 'conj'
```
**Fix**: Add `open ComplexConjugate`

### 2. **Unknown Constant**
```
error: Unknown constant 'Complex.conj'
```
**Fix**: It's notation! Use `conj` not `Complex.conj`

### 3. **Pattern Not Found**
```
error: Did not find occurrence of pattern
```
**Fix**: Print the goal to see actual syntax
```lean
-- Add this to see the goal:
trace "{goal}"
-- Or use conv mode to explore:
conv_lhs => trace "{goal}"
```

### 4. **Tactic Does Nothing**
```
warning: 'push_cast' tactic does nothing
```
**Fix**: Casts might already be simplified. Check if goal needs it.

### 5. **Ring Tactic Failed**
```
error: ring tactic failed
```
**Fix**: Use `ring_nf` for normal form (doesn't close goal but simplifies)

---

## Advanced: Finding Scope and Notation

### Problem: Theorem exists but identifier unknown

**Example**: You see `conj_re` used in Mathlib but get "Unknown identifier"

**Solution Process**:

1. **Find the source file**:
```bash
grep -r "theorem conj_re" .lake/packages/mathlib/
# Result: Mathlib/Data/Complex/Basic.lean
```

2. **Check for scope declarations**:
```bash
grep -n "scoped\|namespace.*Conj\|open.*Conj" Mathlib/Data/Complex/Basic.lean
# Result: line 43: open ComplexConjugate
```

3. **Understand the notation**:
```bash
sed -n '425,435p' .lake/packages/mathlib/Mathlib/Data/Complex/Basic.lean
# Result: notation `conj` in the scope `ComplexConjugate`
```

4. **Add to your file**:
```lean
open ComplexConjugate  -- Makes 'conj' available
```

---

## When to Use Each Search Method

| Method | Best For | Speed | Reliability |
|--------|----------|-------|-------------|
| **exact?** | Quick proof term finding | ⚡ Fast | Medium (misses some) |
| **Loogle** | Type signature search | ⚡ Fast | Medium (requires exact types) |
| **Grep Mathlib** | Comprehensive search | 🐢 Slow | ✅ High |
| **#check** | Type verification | ⚡ Fast | ✅ High |
| **Read source** | Understanding context | 🐢 Slow | ✅ Highest |

**Recommended Workflow**:
1. Try `exact?` first (10 seconds)
2. If that fails, grep Mathlib (1-2 minutes)
3. Read source context to understand scope/notation (5 minutes)
4. Verify with `#check` before using

---

## Key Mathlib Modules for QFD

### Complex Numbers
- `Mathlib.Data.Complex.Basic` - Core definitions, `conj`, real/imaginary parts
- `Mathlib.Analysis.Complex.Exponential` - `exp`, `sin`, `cos`, `exp_conj`

### Trigonometry
- `Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic` - `Real.cos`, `Real.sin`
- `Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex` - Complex trig

### Roots of Unity
- `Mathlib.RingTheory.RootsOfUnity.Complex` - `IsPrimitiveRoot`, geometric sums
- `Mathlib.Algebra.GeomSum` - Finite geometric series

### Tactics
- `Mathlib.Tactic.Ring` - `ring`, `ring_nf` for polynomial arithmetic
- `Mathlib.Tactic.FieldSimp` - `field_simp` for field operations
- `Mathlib.Tactic.PushNeg` - `push_neg` for negation normalization

---

## Success Metrics: KoideRelation.lean

**Before**: 3 sorries (all mathematical claims assumed)
**After**: 1 sorry (only algebraic simplification remains)

**Proofs completed**:
- ✅ Primitive 3rd root of unity (Mathlib theorem)
- ✅ Sum of 3rd roots = 0 (Mathlib `geom_sum_eq_zero`)
- ✅ Euler's formula cos(x) = Re(exp(ix)) (from complex conjugation)
- ✅ Trigonometric identity cos(δ) + cos(δ+2π/3) + cos(δ+4π/3) = 0

**Key Learning**: The difficulty wasn't the mathematics—it was navigating Lean's type system and finding the right Mathlib lemmas. This guide captures that knowledge for future proofs.

---

## Quick Start Template

Copy this template for your next Mathlib-based proof:

```lean
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.RingTheory.RootsOfUnity.Complex
import Mathlib.Tactic.Ring

namespace YourNamespace

open ComplexConjugate  -- For conj notation
open Real              -- For π, cos, sin

-- Your definitions here

-- Pattern: Prove a helper lemma first
lemma helper_lemma : statement := by
  -- Try exact? first
  exact?
  -- If that fails, build proof manually:
  rw [theorem_from_mathlib]
  simp [other_lemmas]
  ring

-- Main theorem using helper
theorem main_result : statement := by
  rw [helper_lemma]
  -- Continue proof

end YourNamespace
```

---

**Next Steps**: When you encounter a new Mathlib theorem gap, add the search process and solution to this guide.
