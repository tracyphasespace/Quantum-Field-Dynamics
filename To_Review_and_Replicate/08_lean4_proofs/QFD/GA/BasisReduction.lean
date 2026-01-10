import Mathlib.Tactic.Ring
import QFD.GA.Cl33
import QFD.GA.BasisOperations
import QFD.GA.BasisProducts

/-!
# Basis Reduction Engine - Complete Automation for Cl(3,3)

**Priority 1 Infrastructure**: Eliminates manual algebraic expansion

**Purpose**: Build a "Symbolic Calculator" that canonizes Clifford products automatically,
turning 50-line manual calc proofs into `by clifford_simp`.

**Design**:
1. **Sorting**: Basis indices in ascending order (e₃*e₀ → -e₀*e₃)
2. **Squaring**: Apply signature (e₀² → 1, e₃² → -1)
3. **Absorption**: Sandwich products (e₀*e₃*e₀ → -e₃)
4. **Scalar Extraction**: Move algebraMap to front, combine coefficients
5. **Specific Products**: Pre-computed patterns from BasisProducts.lean

**Impact**: Retroactively simplifies Heisenberg.lean, PoyntingTheorem.lean, PhaseCentralizer.lean
-/

namespace QFD.GA.BasisReduction

open QFD.GA
open QFD.GA.BasisProducts
open CliffordAlgebra

-----------------------------------------------------------
-- 1. Canonical Sorting (Anticommutation)
-----------------------------------------------------------

/-- Swap basis vectors to ascending order: e_j * e_i = -e_i * e_j when i < j -/
@[simp]
lemma basis_swap_sort {i j : Fin 6} (h : i < j) :
    ι33 (basis_vector j) * ι33 (basis_vector i) =
      - (ι33 (basis_vector i) * ι33 (basis_vector j)) := by
  have h_ne : j ≠ i := ne_of_gt h
  have h_anti := generators_anticommute j i h_ne
  exact add_eq_zero_iff_eq_neg.mp h_anti

/-- Alternative form using e notation -/
@[simp]
lemma e_swap {i j : Fin 6} (h : i < j) :
    e j * e i = - (e i * e j) := by
  unfold e
  exact basis_swap_sort h

-----------------------------------------------------------
-- 2. Squaring Rules (Signature)
-----------------------------------------------------------

/-- Basis vectors square to signature: e_i² = ±1 -/
@[simp]
lemma basis_sq_simplify (i : Fin 6) :
    ι33 (basis_vector i) * ι33 (basis_vector i) =
      algebraMap ℝ Cl33 (signature33 i) :=
  generator_squares_to_signature i

/-- Using e notation -/
@[simp]
lemma e_sq (i : Fin 6) :
    e i * e i = algebraMap ℝ Cl33 (signature33 i) :=
  basis_sq i

-----------------------------------------------------------
-- 3. Absorption Patterns (Sandwich Products)
-----------------------------------------------------------

/-- Sandwich absorption: e_i * e_j * e_i = -σ(i) * e_j when i ≠ j -/
@[simp]
lemma sandwich_absorption {i j : Fin 6} (h : i ≠ j) :
    e i * e j * e i = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by
  calc e i * e j * e i
      = e i * (e j * e i) := by rw [mul_assoc]
    _ = e i * (- e i * e j) := by rw [basis_anticomm h.symm]
    _ = e i * (-(e i * e j)) := by rw [neg_mul]
    _ = -(e i * (e i * e j)) := by rw [mul_neg]
    _ = -(e i * e i * e j) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by rw [basis_sq]

-----------------------------------------------------------
-- 4. Specific Pre-Computed Products
-----------------------------------------------------------

-- Import and expose BasisProducts lemmas as simp rules

@[simp] lemma reduce_e0_e3_e0 : e 0 * e 3 * e 0 = - e 3 := e0_e3_e0
@[simp] lemma reduce_e0_e2_e0 : e 0 * e 2 * e 0 = - e 2 := e0_e2_e0
@[simp] lemma reduce_e3_e0_e3 : e 3 * e 0 * e 3 = e 0 := e3_e0_e3
@[simp] lemma reduce_e2_e3_e2 : e 2 * e 3 * e 2 = - e 3 := e2_e3_e2
@[simp] lemma reduce_e3_e2_e3 : e 3 * e 2 * e 3 = e 2 := e3_e2_e3

-- Quintuple products for Poynting
@[simp] lemma reduce_e0_e3_e0_e2_e3 : e 0 * e 3 * e 0 * e 2 * e 3 = - e 2 :=
  e0_e3_e0_e2_e3

@[simp] lemma reduce_e0_e2_e0_e2_e3 : e 0 * e 2 * e 0 * e 2 * e 3 = - e 3 :=
  e0_e2_e0_e2_e3

-----------------------------------------------------------
-- 5. Scalar Normalization
-----------------------------------------------------------

/-- Scalars commute with basis vectors -/
@[simp]
lemma scalar_basis_commute (c : ℝ) (i : Fin 6) :
    e i * algebraMap ℝ Cl33 c = algebraMap ℝ Cl33 c * e i := by
  rw [Algebra.commutes]

/-- Scalar associativity helper -/
@[simp]
lemma scalar_assoc (c : ℝ) (x y : Cl33) :
    x * (algebraMap ℝ Cl33 c * y) = algebraMap ℝ Cl33 c * (x * y) := by
  rw [← mul_assoc, ← Algebra.commutes, mul_assoc]

-----------------------------------------------------------
-- 6. Signature-Specific Shortcuts
-----------------------------------------------------------

/-- Spatial basis vectors (0,1,2) square to +1 -/
@[simp] lemma e0_square : e 0 * e 0 = 1 := by simp [e_sq, signature33]
@[simp] lemma e1_square : e 1 * e 1 = 1 := by simp [e_sq, signature33]
@[simp] lemma e2_square : e 2 * e 2 = 1 := by simp [e_sq, signature33]

/-- Temporal/internal basis vectors (3,4,5) square to -1 -/
@[simp] lemma e3_square : e 3 * e 3 = -1 := by simp [e_sq, signature33]
@[simp] lemma e4_square : e 4 * e 4 = -1 := by simp [e_sq, signature33]
@[simp] lemma e5_square : e 5 * e 5 = -1 := by simp [e_sq, signature33]

-----------------------------------------------------------
-- 7. THE AUTOMATION TACTIC
-----------------------------------------------------------

/--
**clifford_simp**: Automated Clifford algebra simplification tactic

Applies all canonicalization rules to reduce Clifford products to normal form.

**Example Usage**:
```lean
-- Before: 50 lines of manual calc
theorem poynting_energy : (e 0 * e 3) * e 0 + (e 0 * e 2) * e 0 = - e 3 - e 2 := by
  clifford_simp  -- Done!
```

**Rules Applied**:
- Basis swapping (anticommutation for sorting)
- Signature reduction (e² → ±1)
- Sandwich absorption (eᵢeⱼeᵢ → -σ(i)eⱼ)
- Pre-computed specific products
- Scalar normalization
- Ring arithmetic on scalar parts
-/
syntax "clifford_simp" : tactic

macro_rules
  | `(tactic| clifford_simp) =>
      `(tactic|
        (simp only [
          -- Specific products (highest priority)
          reduce_e0_e3_e0, reduce_e0_e2_e0,
          reduce_e3_e0_e3, reduce_e2_e3_e2, reduce_e3_e2_e3,
          reduce_e0_e3_e0_e2_e3, reduce_e0_e2_e0_e2_e3,
          -- Absorption
          sandwich_absorption,
          -- Sorting
          basis_swap_sort, e_swap,
          -- Squaring
          basis_sq_simplify, e_sq,
          e0_square, e1_square, e2_square,
          e3_square, e4_square, e5_square,
          -- Scalar handling
          scalar_basis_commute, scalar_assoc,
          -- algebraMap properties
          map_zero, map_one, map_add, map_sub, map_neg, map_mul,
          -- Arithmetic (no mul_assoc to avoid interference)
          neg_mul, mul_neg, neg_neg,
          add_comm, add_left_comm, add_assoc,
          mul_comm, mul_left_comm,
          sub_eq_add_neg,
          zero_mul, mul_zero, one_mul, mul_one,
          zero_add, add_zero,
          -- Algebra
          Algebra.smul_def
        ] <;> try ring_nf))

/--
**clifford_ring**: Extended version with aggressive ring normalization

Use when you need full scalar arithmetic simplification.
-/
syntax "clifford_ring" : tactic

macro_rules
  | `(tactic| clifford_ring) =>
      `(tactic| (clifford_simp; ring))

-----------------------------------------------------------
-- 8. Status & Verification
-----------------------------------------------------------

/-- Verification: The tactic works on a simple example -/
example : e 0 * e 3 * e 0 = - e 3 := by clifford_simp

/-- Verification: Multi-term example -/
example : e 0 * e 3 * e 0 + e 0 * e 2 * e 0 = - e 3 - e 2 := by clifford_simp

end QFD.GA.BasisReduction
