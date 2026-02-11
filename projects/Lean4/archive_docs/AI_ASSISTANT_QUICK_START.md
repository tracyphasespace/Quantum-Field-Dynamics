# Quick Start Guide for AI Assistants
## QFD Lean 4 Formalization

**Purpose**: Get AI assistants productive on QFD proofs in 5 minutes
**Audience**: Claude, ChatGPT, or other AI systems tackling sorries
**Last Updated**: 2025-12-27

---

## ⚡ **CRITICAL: USE THE AUTOMATION TOOLS!** ⚡

**DO NOT manually expand Clifford algebra products!** We have automation for this.

### ✅ What You SHOULD Do:

```lean
-- ✅ CORRECT: Use the automation tactic
theorem my_proof : e 0 * e 3 * e 0 = - e 3 := by
  clifford_simp  -- One line!
```

### ❌ What You SHOULD NOT Do:

```lean
-- ❌ WRONG: Manual expansion (50+ lines)
theorem my_proof : e 0 * e 3 * e 0 = - e 3 := by
  calc e 0 * e 3 * e 0
      = e 0 * (e 3 * e 0) := by rw [mul_assoc]
    _ = e 0 * (- e 0 * e 3) := by rw [basis_anticomm (by decide)]
    -- ... 45 more lines ...
```

**The automation tools exist. Use them!**

---

## 🎯 Your Mission

Complete formal Lean 4 proofs that eliminate complex numbers from quantum mechanics by replacing them with geometric algebra (Clifford Algebra Cl(3,3)).

**Key Idea**: The complex "i" is actually a bivector **B = e₄ ∧ e₅** (geometric rotation in internal dimensions).

---

## 📚 Essential Infrastructure (Already Built)

### 0. **AUTOMATION TOOLS** (Use These First!)

**File**: `QFD/GA/BasisReduction.lean` ⚡

```lean
-- Import this for automation
import QFD.GA.BasisReduction

-- Main tactic: Simplifies Clifford algebra expressions automatically
example : e 0 * e 3 * e 0 + e 0 * e 2 * e 0 = - e 3 - e 2 := by
  clifford_simp  -- Handles sorting, squaring, absorption, scalars

-- Extended tactic: Adds ring solver for scalar arithmetic
example : algebraMap ℝ Cl33 2 * (e 1 * e 1) = 2 := by
  clifford_ring
```

**What clifford_simp does**:
- ✅ Sorts basis indices (e₃*e₀ → -e₀*e₃)
- ✅ Applies signature (e₀² → 1, e₃² → -1)
- ✅ Reduces sandwiches (e₀*e₃*e₀ → -e₃)
- ✅ Normalizes scalars
- ✅ Uses pre-computed products from BasisProducts.lean

**File**: `QFD/GA/BasisProducts.lean`

```lean
-- Pre-computed common products (use these if clifford_simp isn't enough)
lemma e0_e3_e0 : e 0 * e 3 * e 0 = - e 3
lemma e0_e2_e0 : e 0 * e 2 * e 0 = - e 2
lemma e3_e0_e3 : e 3 * e 0 * e 3 = e 0
lemma e2_e3_e2 : e 2 * e 3 * e 2 = e 3
lemma e0_e3_e0_e2_e3 : e 0 * e 3 * e 0 * e 2 * e 3 = - e 2  -- For Poynting
lemma e0_e2_e0_e2_e3 : e 0 * e 2 * e 0 * e 2 * e 3 = - e 3  -- For Poynting
```

### 1. Core Lemmas (Your Best Friends)

**File**: `QFD/GA/BasisOperations.lean`

```lean
-- Access basis vectors
def e (i : Fin 6) : Cl33 := ι33 (basis_vector i)

-- Basis vectors square to signature (±1)
theorem basis_sq (i : Fin 6) : e i * e i = algebraMap ℝ Cl33 (signature33 i)

-- Distinct basis vectors anticommute
theorem basis_anticomm {i j : Fin 6} (h : i ≠ j) : e i * e j = - e j * e i
```

**File**: `QFD/GA/PhaseCentralizer.lean`

```lean
-- The geometric "i" - phase rotor bivector
def B_phase : Cl33 := e 4 * e 5

-- THE KEY PROPERTY: B² = -1 (just like complex i!)
theorem phase_rotor_is_imaginary : B_phase * B_phase = -1
```

### 2. Signature Convention

```
e₀² = +1  (space x)
e₁² = +1  (space y)
e₂² = +1  (space z)
e₃² = -1  (time)
e₄² = -1  (internal dim 1)
e₅² = -1  (internal dim 2)
```

Signature: `(+++ ---)`

### 3. Import Template

**EVERY file doing Clifford algebra should import BasisReduction first!**

```lean
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Tactic.Ring
import QFD.GA.Cl33
import QFD.GA.BasisOperations   -- Get e, basis_sq, basis_anticomm
import QFD.GA.BasisReduction    -- ⚡ GET clifford_simp AUTOMATION
import QFD.GA.PhaseCentralizer  -- Get B_phase, phase_rotor_is_imaginary
```

**Use open statements**:
```lean
open QFD.GA
open QFD.GA.BasisReduction  -- Enables clifford_simp tactic
```

---

## 🔧 Common Proof Patterns

### Pattern 0: **USE AUTOMATION FIRST!** ⚡

**ALWAYS try `clifford_simp` before writing manual proofs!**

```lean
-- ✅ ALWAYS TRY THIS FIRST
theorem my_clifford_proof : e 0 * e 3 * e 0 = - e 3 := by
  clifford_simp

-- If clifford_simp doesn't fully solve it, you can combine tactics:
theorem complex_proof : some_expr = target := by
  clifford_simp       -- Simplify Clifford part
  -- Then add manual steps for what remains
  rw [some_lemma]
  ring
```

**Only use manual patterns below if automation doesn't work!**

### Pattern 1: Scalar-Bivector Commutation

**Problem**: Need to swap scalar and bivector in multiplication

```lean
-- Scalars commute with everything
algebraMap ℝ Cl33 (cos a) * B_phase = B_phase * algebraMap ℝ Cl33 (cos a)

-- Tactic:
rw [Algebra.commutes]
```

### Pattern 2: Basis Anticommutation

**Problem**: Need to swap basis vectors

```lean
-- For i ≠ j:  eᵢ * eⱼ = -eⱼ * eᵢ
have h : e 0 * e 3 = - e 3 * e 0 := basis_anticomm (by decide)
```

### Pattern 3: Squaring Basis Vectors

**Problem**: Simplify e² terms

```lean
-- eᵢ² = ±1 depending on signature
have h : e 3 * e 3 = -1 := by
  simpa using basis_sq 3  -- e₃ is timelike, signature = -1
```

### Pattern 4: Bringing B² Together

**Problem**: You have `B * (stuff) * B` and need to use B² = -1

```lean
-- Strategy: Rearrange to get (B * B) * rest
calc B_phase * algebraMap ℝ Cl33 (sin a) * B_phase * algebraMap ℝ Cl33 (sin b)
    = (B_phase * B_phase) * algebraMap ℝ Cl33 (sin a) * algebraMap ℝ Cl33 (sin b) := by
        rw [mul_assoc, mul_assoc]
  _ = (-1 : Cl33) * algebraMap ℝ Cl33 (sin a * sin b) := by
        rw [phase_rotor_is_imaginary, ← map_mul]
  _ = - algebraMap ℝ Cl33 (sin a * sin b) := by rw [neg_one_mul]
```

### Pattern 5: Map Manipulation

**Problem**: Combine or split algebraMap terms

```lean
-- Combine: f(a) * f(b) = f(a*b)
rw [← map_mul]

-- Split: f(a*b) = f(a) * f(b)
rw [map_mul]

-- Subtraction: f(a) - f(b) = f(a - b)
rw [← map_sub]

-- Addition: f(a + b) = f(a) + f(b)
rw [← map_add]
```

---

## 📖 Reference Proofs (Learn from These)

### Example 1: Simple Basis Proof

**File**: `QFD/QM_Translation/Heisenberg.lean`
**Theorem**: `xp_noncomm`

```lean
theorem xp_noncomm : commutator X_op P_op ≠ 0 := by
  unfold commutator X_op P_op
  -- commutator = e₀*e₃ - e₃*e₀
  -- Use basis_anticomm: e₀*e₃ = -e₃*e₀
  -- So: commutator = -e₃*e₀ - e₃*e₀ = -2(e₃*e₀) ≠ 0
  sorry  -- You complete this!
```

**Hint**: Use `basis_anticomm`, then show `-2(e₃*e₀) ≠ 0` by proving `e₃*e₀` is a unit bivector.

### Example 2: Complete Calc Chain

**File**: `QFD/QM_Translation/RealDiracEquation.lean`
**Theorem**: `mass_is_internal_momentum` (0 sorries - COMPLETE!)

Read this file to see how to build multi-step calc chains with justifications.

### Example 3: Documented Strategy (Currently Sorry)

**File**: `QFD/QM_Translation/SchrodingerEvolution.lean`
**Theorem**: `phase_group_law`

Has complete mathematical derivation in comments showing the T1-T4 expansion pattern.

---

## 🎓 Proof Strategy Guide

### When You See a Sorry

1. **Read the comments** - Most sorries have proof strategies documented
2. **Check the type** - What are you trying to prove? Equality? Non-zero? Property?
3. **Unfold definitions** - Use `unfold` to expand custom definitions
4. **Match patterns** - Look for basis products, B², scalar commutation
5. **Use calc chains** - For multi-step proofs, be explicit

### Common Tactics Reference

```lean
-- Expand definitions
unfold GeometricPhase commutator

-- Rewrite with lemmas
rw [basis_anticomm, phase_rotor_is_imaginary]

-- Ring normalization (for commutative subexpressions)
ring
ring_nf

-- Associativity/Commutativity
rw [mul_assoc]      -- (a*b)*c = a*(b*c)
rw [add_assoc]      -- (a+b)+c = a+(b+c)
rw [add_comm]       -- a+b = b+a

-- Negation
rw [neg_mul]        -- -(a*b) = (-a)*b = a*(-b)
rw [neg_one_mul]    -- -1 * a = -a

-- Simplification
simp               -- Applies simplification rules
simp only [h1, h2] -- Selective simplification

-- Congruence (prove parts separately)
congr 1            -- Split goal into subgoals for function arguments

-- Step-by-step calculation
calc a = b := by ...
   _ = c := by ...
   _ = d := by ...
```

---

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Pattern Doesn't Match

**Error**: `Tactic 'rewrite' failed: Did not find an occurrence of the pattern`

**Cause**: Expression structure is different than expected (parentheses, order)

**Solutions**:
- Use `conv` to navigate to exact subexpression
- Use `show` to explicitly state what you're proving
- Add intermediate `have` lemmas to break it down

### Pitfall 2: Non-Commutative Confusion

**Error**: Trying to use `ring` on expressions with bivectors

**Cause**: `ring` only works for commutative rings; Clifford algebra is non-commutative

**Solution**:
- Use `ring` only on the scalar (Real) parts
- Manually handle bivector terms with `mul_assoc`, `basis_anticomm`

### Pitfall 3: Type Mismatch with algebraMap

**Error**: Type mismatch between `ℝ` and `Cl33`

**Solution**: Wrap reals with `algebraMap ℝ Cl33`:
```lean
-- Wrong:
have h : Cl33 := cos a

-- Right:
have h : Cl33 := algebraMap ℝ Cl33 (cos a)
```

### Pitfall 4: Forgetting to Prove i ≠ j

**Error**: Can't use `basis_anticomm` without proving indices are different

**Solution**: Add `(by decide)` or explicit proof:
```lean
have h := basis_anticomm (i := 0) (j := 3) (by decide)
```

---

## 🎯 Recommended Starting Points

### Easiest (15-30 min each):

1. **Heisenberg.lean** - `xp_noncomm`
   - Just basis anticommutation + showing non-zero

2. **Heisenberg.lean** - `uncertainty_is_bivector_area`
   - Commutator algebra: `A*B - B*A = 2(A∧B)` for anticommuting A, B

### Medium (30-60 min):

3. **PoyntingTheorem.lean** - `poynting_is_geometric_product`
   - Complete T1-T4 expansion documented in comments
   - Pattern similar to SchrodingerEvolution

4. **MultivectorDefs.lean** - Any of the 7 sorries
   - Wedge product properties
   - Good for learning non-commutative manipulation

---

## 📋 Checklist Before Starting

- [ ] Read the file header comments (explains what it proves)
- [ ] Check what's imported (tells you available tools)
- [ ] Look for proof strategy comments near the sorry
- [ ] Identify which infrastructure lemmas apply
- [ ] Check a similar completed proof for patterns
- [ ] Start with `unfold` to see what you're working with

---

## 🔍 Quick Reference Commands

```bash
# Build a specific file (incremental only!)
lake build QFD.QM_Translation.Heisenberg

# Find all sorries
grep -r "sorry" QFD --include="*.lean"

# Count sorries in a file
grep -c "sorry" QFD/QM_Translation/Heisenberg.lean

# Search for theorem names
rg "theorem phase_group_law" QFD
```

## ⚠️ CRITICAL: DO NOT CLEAN BUILD

**NEVER** run these commands (they cause OOM on 4GB systems):
```bash
lake clean       # ❌ DON'T RUN THIS
lake build QFD   # ❌ DON'T RUN FULL BUILD
```

**Why**: Mathlib is huge (~2GB compiled). Clean builds:
- Take 30-60 minutes
- Use 4-8GB RAM
- Can cause out-of-memory crashes

**Always use incremental builds**:
```bash
# ✅ GOOD: Build only what you changed
lake build QFD.QM_Translation.Heisenberg

# ✅ GOOD: Build specific module
lake build QFD.GA.MultivectorDefs
```

---

## 💡 Pro Tips

1. **Use calc chains** - They make complex proofs readable and debuggable
2. **Add intermediate `have` lemmas** - Break complex goals into simple steps
3. **Check types early** - Use `#check` to verify expression types
4. **Copy working patterns** - If basis_anticomm worked once, use it again
5. **Document your strategy** - Even if proof fails, comments help next attempt
6. **When stuck, use sorry** - Add detailed comments showing the mathematical derivation

---

## 🎓 Success Criteria

A good proof should:
- ✅ Build without errors (`lake build` succeeds)
- ✅ Use existing infrastructure (don't reinvent lemmas)
- ✅ Have clear intermediate steps (calc or have lemmas)
- ✅ Match established patterns (see completed files)
- ✅ Include brief comments explaining key steps

A good documented sorry should:
- ✅ Have complete mathematical derivation in comments
- ✅ Show T1, T2, T3... term expansion
- ✅ Identify which infrastructure lemmas should apply
- ✅ Build successfully with the sorry in place

---

## 📚 Key Files to Study

**Read these first** (ordered by learning value):

1. `QFD/GA/BasisOperations.lean` - The foundation (50 lines)
2. `QFD/GA/PhaseCentralizer.lean` - The B² = -1 proof (~200 lines)
3. `QFD/QM_Translation/RealDiracEquation.lean` - Complete calc chains (150 lines)
4. `QFD/QM_Translation/SchrodingerEvolution.lean` - Documented strategy pattern (200 lines)

**Current targets** (what needs work):
- `QFD/QM_Translation/Heisenberg.lean` (2 sorries) ⭐ START HERE
- `QFD/Electrodynamics/PoyntingTheorem.lean` (1 sorry)
- `QFD/GA/MultivectorDefs.lean` (7 sorries)
- `QFD/GA/MultivectorGrade.lean` (4 sorries - placeholders)

---

## 🚀 You're Ready!

You now have everything needed to:
- Complete sorries in Heisenberg.lean
- Understand the infrastructure
- Write new proofs following established patterns
- Help eliminate complex numbers from QM! 🎯

**Start with**: `QFD/QM_Translation/Heisenberg.lean` - It's ready for you!

Good luck! The infrastructure is solid, the patterns are proven, and the mathematics is beautiful. 💪
