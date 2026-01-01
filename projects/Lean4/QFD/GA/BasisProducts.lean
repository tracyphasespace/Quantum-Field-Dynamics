import QFD.GA.BasisOperations

/-!
# Basis Product Library for Cl(3,3)

**Purpose**: Pre-computed products of basis vectors to avoid repetitive calculations.

**Design Philosophy**: Many small, proven bricks > Few large monoliths
- Each lemma is simple and focused
- Easy to verify correctness
- Reusable across proofs
- Fast incremental builds

**Scope**: Common products needed in QFD proofs
- Triple products: eᵢ * eⱼ * eₖ
- Quintuple products: eᵢ * eⱼ * eₖ * eₗ * eₘ
- General patterns: (eᵢeⱼ) * eᵢ = -σ(i)eⱼ

**Usage**: Import this instead of manually computing products
```lean
import QFD.GA.BasisProducts

-- Before: 5 lines of calculation
-- After:
rw [e0_e3_e0]  -- Done!
```

-/

namespace QFD.GA.BasisProducts

open QFD.GA
open CliffordAlgebra

-----------------------------------------------------------
-- 1. Triple Products (Most Common)
-----------------------------------------------------------

/-- e₀ * e₃ * e₀ = -e₃ (sandwich with spatial) -/
lemma e0_e3_e0 : e 0 * e 3 * e 0 = - e 3 := by
  calc e 0 * e 3 * e 0
      = e 0 * (e 3 * e 0) := by rw [mul_assoc]
    _ = e 0 * (- e 0 * e 3) := by rw [basis_anticomm (by decide)]
    _ = e 0 * (-(e 0 * e 3)) := by rw [neg_mul]
    _ = -(e 0 * (e 0 * e 3)) := by rw [mul_neg]
    _ = -(e 0 * e 0 * e 3) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 0) * e 3) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 1 * e 3) := by simp [signature33]
    _ = - e 3 := by simp [map_one, one_mul]

/-- e₀ * e₂ * e₀ = -e₂ (sandwich with spatial) -/
lemma e0_e2_e0 : e 0 * e 2 * e 0 = - e 2 := by
  calc e 0 * e 2 * e 0
      = e 0 * (e 2 * e 0) := by rw [mul_assoc]
    _ = e 0 * (- e 0 * e 2) := by rw [basis_anticomm (by decide)]
    _ = e 0 * (-(e 0 * e 2)) := by rw [neg_mul]
    _ = -(e 0 * (e 0 * e 2)) := by rw [mul_neg]
    _ = -(e 0 * e 0 * e 2) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 0) * e 2) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 1 * e 2) := by simp [signature33]
    _ = - e 2 := by simp [map_one, one_mul]

/-- e₃ * e₀ * e₃ = e₀ (sandwich with time, note sign flip due to signature) -/
lemma e3_e0_e3 : e 3 * e 0 * e 3 = e 0 := by
  calc e 3 * e 0 * e 3
      = e 3 * (e 0 * e 3) := by rw [mul_assoc]
    _ = e 3 * (- e 3 * e 0) := by rw [basis_anticomm (by decide)]
    _ = e 3 * (-(e 3 * e 0)) := by rw [neg_mul]
    _ = -(e 3 * (e 3 * e 0)) := by rw [mul_neg]
    _ = -(e 3 * e 3 * e 0) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 3) * e 0) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 (-1) * e 0) := by simp [signature33]
    _ = -(- e 0) := by simp [map_neg, map_one]
    _ = e 0 := by simp

/-- e₂ * e₃ * e₂ = -e₃ (sandwich spatial with temporal) -/
lemma e2_e3_e2 : e 2 * e 3 * e 2 = - e 3 := by
  calc e 2 * e 3 * e 2
      = e 2 * (e 3 * e 2) := by rw [mul_assoc]
    _ = e 2 * (- e 2 * e 3) := by rw [basis_anticomm (by decide)]
    _ = e 2 * (-(e 2 * e 3)) := by rw [neg_mul]
    _ = -(e 2 * (e 2 * e 3)) := by rw [mul_neg]
    _ = -(e 2 * e 2 * e 3) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 2) * e 3) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 1 * e 3) := by simp [signature33]
    _ = - e 3 := by simp [map_one, one_mul]

/-- e₃ * e₂ * e₃ = e₂ (sandwich temporal with spatial, sign flip) -/
lemma e3_e2_e3 : e 3 * e 2 * e 3 = e 2 := by
  calc e 3 * e 2 * e 3
      = e 3 * (e 2 * e 3) := by rw [mul_assoc]
    _ = e 3 * (- e 3 * e 2) := by rw [basis_anticomm (by decide)]
    _ = e 3 * (-(e 3 * e 2)) := by rw [neg_mul]
    _ = -(e 3 * (e 3 * e 2)) := by rw [mul_neg]
    _ = -(e 3 * e 3 * e 2) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 3) * e 2) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 (-1) * e 2) := by simp [signature33]
    _ = -(- e 2) := by simp [map_neg, map_one]
    _ = e 2 := by simp

-----------------------------------------------------------
-- 2. Quintuple Products (For Poynting Theorem)
-----------------------------------------------------------

/-- e₀ * e₃ * e₀ * e₂ * e₃ = -e₂ (Poynting T2) -/
lemma e0_e3_e0_e2_e3 : e 0 * e 3 * e 0 * e 2 * e 3 = - e 2 := by
  calc e 0 * e 3 * e 0 * e 2 * e 3
      = (e 0 * e 3 * e 0) * e 2 * e 3 := by rw [mul_assoc, mul_assoc]
    _ = (- e 3) * e 2 * e 3 := by rw [e0_e3_e0]
    _ = (- e 3) * (e 2 * e 3) := by rw [mul_assoc]
    _ = -(e 3 * (e 2 * e 3)) := by rw [neg_mul]
    _ = -(e 3 * e 2 * e 3) := by rw [mul_assoc]
    _ = - e 2 := by rw [e3_e2_e3]

/-- e₀ * e₂ * e₀ * e₂ * e₃ = -e₃ (Poynting T4) -/
lemma e0_e2_e0_e2_e3 : e 0 * e 2 * e 0 * e 2 * e 3 = - e 3 := by
  calc e 0 * e 2 * e 0 * e 2 * e 3
      = (e 0 * e 2 * e 0) * e 2 * e 3 := by rw [mul_assoc, mul_assoc]
    _ = (- e 2) * e 2 * e 3 := by rw [e0_e2_e0]
    _ = (- e 2) * (e 2 * e 3) := by rw [mul_assoc]
    _ = -(e 2 * (e 2 * e 3)) := by rw [neg_mul]
    _ = -(e 2 * e 2 * e 3) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 2) * e 3) := by rw [basis_sq]
    _ = -(algebraMap ℝ Cl33 1 * e 3) := by simp [signature33]
    _ = - e 3 := by simp [map_one, one_mul]

-----------------------------------------------------------
-- 3. General Patterns (Reusable Lemmas)
-----------------------------------------------------------

/-- Left contraction: (eᵢeⱼ) * eᵢ = -σ(i)eⱼ when i ≠ j
    This is the key pattern for "sandwiching" operations -/
theorem bivector_left_contract {i j : Fin 6} (h : i ≠ j) :
  (e i * e j) * e i = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by
  calc (e i * e j) * e i
      = e i * (e j * e i) := by rw [mul_assoc]
    _ = e i * (- e i * e j) := by rw [basis_anticomm h.symm]
    _ = e i * (-(e i * e j)) := by rw [neg_mul]
    _ = -(e i * (e i * e j)) := by rw [mul_neg]
    _ = -(e i * e i * e j) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by rw [basis_sq]

/-- Right contraction: eᵢ * (eⱼeᵢ) = -σ(i)eⱼ when i ≠ j
    (Equal to left contraction by associativity) -/
theorem bivector_right_contract {i j : Fin 6} (h : i ≠ j) :
  e i * (e j * e i) = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by
  calc e i * (e j * e i)
      = e i * (- e i * e j) := by rw [basis_anticomm h.symm]
    _ = e i * (-(e i * e j)) := by rw [neg_mul]
    _ = -(e i * (e i * e j)) := by rw [mul_neg]
    _ = -(e i * e i * e j) := by rw [mul_assoc]
    _ = -(algebraMap ℝ Cl33 (signature33 i) * e j) := by rw [basis_sq]

-----------------------------------------------------------
-- 4. Specific Products for Common Use Cases
-----------------------------------------------------------

/-- e₀e₃ * e₀ = -e₃ (bivector left sandwich) -/
lemma e0e3_mul_e0 : (e 0 * e 3) * e 0 = - e 3 := by
  have h := bivector_left_contract (i := 0) (j := 3) (by decide)
  simpa [signature33] using h

/-- e₀e₂ * e₀ = -e₂ (bivector left sandwich) -/
lemma e0e2_mul_e0 : (e 0 * e 2) * e 0 = - e 2 := by
  have h := bivector_left_contract (i := 0) (j := 2) (by decide)
  simpa [signature33] using h

-----------------------------------------------------------
-- 5. Utility: Products with Scalars
-----------------------------------------------------------

/-- Scalar multiple of triple product -/
lemma scalar_triple_product (c : ℝ) (i j k : Fin 6) :
  algebraMap ℝ Cl33 c * (e i * e j * e k) =
  c • (e i * e j * e k) := by
  simp [Algebra.smul_def]

-----------------------------------------------------------
-- 6. Disjoint Bivector Commutation (Neutrino Physics)
-----------------------------------------------------------

/-- Disjoint bivectors commute: (e₀∧e₁) commutes with (e₃∧e₄) -/
lemma e01_commutes_e34 :
    (e 0 * e 1) * (e 3 * e 4) = (e 3 * e 4) * (e 0 * e 1) := by
  have h13 : e 1 * e 3 = -(e 3 * e 1) := by
    simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 3)
  have h03 : e 0 * e 3 = -(e 3 * e 0) := by
    simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 3)
  have h14 : e 1 * e 4 = -(e 4 * e 1) := by
    simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 4)
  have h04 : e 0 * e 4 = -(e 4 * e 0) := by
    simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 4)
  calc
    (e 0 * e 1) * (e 3 * e 4)
        = e 0 * e 1 * e 3 * e 4 := by simp [mul_assoc]
    _ = e 0 * (e 1 * e 3) * e 4 := by simp [mul_assoc]
    _ = e 0 * (-(e 3 * e 1)) * e 4 := by simp [h13]
    _ = -((e 0 * e 3) * e 1 * e 4) := by
          simp [mul_assoc, mul_neg]
    _ = -((-(e 3 * e 0)) * e 1 * e 4) := by
          simp [h03]
    _ = (e 3 * e 0) * e 1 * e 4 := by
          simp [mul_assoc]
    _ = e 3 * e 0 * (e 1 * e 4) := by simp [mul_assoc]
    _ = e 3 * e 0 * (-(e 4 * e 1)) := by
          simp [h14]
    _ = -(e 3 * (e 0 * e 4) * e 1) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 3 * (-(e 4 * e 0)) * e 1) := by
          simp [h04]
    _ = (e 3 * e 4) * (e 0 * e 1) := by simp [mul_assoc]

/-- Disjoint bivectors commute: (e₀∧e₁) commutes with (e₄∧e₅) -/
lemma e01_commutes_e45 :
    (e 0 * e 1) * (e 4 * e 5) = (e 4 * e 5) * (e 0 * e 1) := by
  have h14 : e 1 * e 4 = -(e 4 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 4)
  have h04 : e 0 * e 4 = -(e 4 * e 0) :=
    by simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 4)
  have h15 : e 1 * e 5 = -(e 5 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 5)
  have h05 : e 0 * e 5 = -(e 5 * e 0) :=
    by simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 5)
  calc
    (e 0 * e 1) * (e 4 * e 5)
        = e 0 * e 1 * e 4 * e 5 := by simp [mul_assoc]
    _ = e 0 * (e 1 * e 4) * e 5 := by simp [mul_assoc]
    _ = e 0 * (-(e 4 * e 1)) * e 5 := by simp [h14]
    _ = -((e 0 * e 4) * e 1 * e 5) := by
          simp [mul_assoc, mul_neg]
    _ = -((-(e 4 * e 0)) * e 1 * e 5) := by
          simp [h04]
    _ = (e 4 * e 0) * e 1 * e 5 := by simp [mul_assoc]
    _ = e 4 * e 0 * (e 1 * e 5) := by simp [mul_assoc]
    _ = e 4 * e 0 * (-(e 5 * e 1)) := by simp [h15]
    _ = -(e 4 * (e 0 * e 5) * e 1) := by simp [mul_assoc, mul_neg]
    _ = -(e 4 * (-(e 5 * e 0)) * e 1) := by simp [h05]
    _ = (e 4 * e 5) * (e 0 * e 1) := by simp [mul_assoc]

/-- External trivector squares to scalar: (e₀∧e₁∧e₂)² = -1 -/
lemma e012_sq :
    (e 0 * e 1 * e 2) * (e 0 * e 1 * e 2) =
      algebraMap ℝ Cl33 (-1) := by
  have h20 : e 2 * e 0 = -(e 0 * e 2) :=
    by simpa using basis_anticomm (by decide : (2 : Fin 6) ≠ 0)
  have h21 : e 2 * e 1 = -(e 1 * e 2) :=
    by simpa using basis_anticomm (by decide : (2 : Fin 6) ≠ 1)
  have h10 : e 1 * e 0 = -(e 0 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 0)
  have h0_sq : e 0 * e 0 = algebraMap ℝ Cl33 1 :=
    by simpa [signature33] using basis_sq (0 : Fin 6)
  have h1_sq : e 1 * e 1 = algebraMap ℝ Cl33 1 :=
    by simpa [signature33] using basis_sq (1 : Fin 6)
  have h2_sq : e 2 * e 2 = algebraMap ℝ Cl33 1 :=
    by simpa [signature33] using basis_sq (2 : Fin 6)
  calc
    (e 0 * e 1 * e 2) * (e 0 * e 1 * e 2)
        = e 0 * e 1 * (e 2 * e 0) * e 1 * e 2 := by simp [mul_assoc]
    _ = e 0 * e 1 * (-(e 0 * e 2)) * e 1 * e 2 := by simp [h20]
    _ = -(e 0 * (e 1 * e 0) * e 2 * e 1 * e 2) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 0 * (-(e 0 * e 1)) * e 2 * e 1 * e 2) := by
          simp [h10]
    _ = e 0 * e 0 * (e 1 * e 2 * e 1 * e 2) := by
          simp [mul_assoc, mul_neg]
    _ = (algebraMap ℝ Cl33 1) * (e 1 * e 2 * e 1 * e 2) := by
          simp [h0_sq]
    _ = e 1 * e 2 * e 1 * e 2 := by simp
    _ = e 1 * (e 2 * e 1) * e 2 := by simp [mul_assoc]
    _ = e 1 * (-(e 1 * e 2)) * e 2 := by simp [h21]
    _ = -(e 1 * e 1 * e 2 * e 2) := by
          simp [mul_assoc, mul_neg]
    _ = -((algebraMap ℝ Cl33 1) * (algebraMap ℝ Cl33 1)) := by
          simp [h1_sq, h2_sq]
    _ = algebraMap ℝ Cl33 (-1) := by simp

lemma e345_sq :
    (e 3 * e 4 * e 5) * (e 3 * e 4 * e 5) = algebraMap ℝ Cl33 1 := by
  have h53 : e 5 * e 3 = -(e 3 * e 5) :=
    by simpa using basis_anticomm (by decide : (5 : Fin 6) ≠ 3)
  have h43 : e 4 * e 3 = -(e 3 * e 4) :=
    by simpa using basis_anticomm (by decide : (4 : Fin 6) ≠ 3)
  have h54 : e 5 * e 4 = -(e 4 * e 5) :=
    by simpa using basis_anticomm (by decide : (5 : Fin 6) ≠ 4)
  have h3_sq : e 3 * e 3 = algebraMap ℝ Cl33 (-1) :=
    by simpa [signature33] using basis_sq (3 : Fin 6)
  have h4_sq : e 4 * e 4 = algebraMap ℝ Cl33 (-1) :=
    by simpa [signature33] using basis_sq (4 : Fin 6)
  have h5_sq : e 5 * e 5 = algebraMap ℝ Cl33 (-1) :=
    by simpa [signature33] using basis_sq (5 : Fin 6)
  have h4e5e4 :
      e 4 * e 5 * e 4 = e 5 := by
    calc e 4 * e 5 * e 4
        = e 4 * (e 5 * e 4) := by simp [mul_assoc]
    _ = e 4 * (-(e 4 * e 5)) := by
          simp [h54]
    _ = -(e 4 * e 4 * e 5) := by simp [mul_assoc, mul_neg]
    _ = -((algebraMap ℝ Cl33 (-1)) * e 5) := by simp [h4_sq]
    _ = e 5 := by simp
  have h4e5e4e5 :
      e 4 * e 5 * e 4 * e 5 = algebraMap ℝ Cl33 (-1) := by
    calc e 4 * e 5 * e 4 * e 5
        = (e 4 * e 5 * e 4) * e 5 := by simp [mul_assoc]
    _ = e 5 * e 5 := by simp [h4e5e4]
    _ = algebraMap ℝ Cl33 (-1) := h5_sq
  calc
    (e 3 * e 4 * e 5) * (e 3 * e 4 * e 5)
        = e 3 * e 4 * (e 5 * e 3) * e 4 * e 5 := by simp [mul_assoc]
    _ = e 3 * e 4 * (-(e 3 * e 5)) * e 4 * e 5 := by simp [h53]
    _ = -(e 3 * (e 4 * e 3) * e 5 * e 4 * e 5) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 3 * (-(e 3 * e 4)) * e 5 * e 4 * e 5) := by
          simp [h43]
    _ = e 3 * e 3 * (e 4 * e 5 * e 4 * e 5) := by
          simp [mul_assoc, mul_neg]
    _ = (algebraMap ℝ Cl33 (-1)) *
          (algebraMap ℝ Cl33 (-1)) := by
          simp [h3_sq, h4e5e4e5]
    _ = algebraMap ℝ Cl33 1 := by simp

/-- Spatial and internal trivectors anticommute: e₀∧e₁∧e₂ anti-commutes with e₃∧e₄∧e₅ -/
lemma e012_e345_anticomm :
    (e 0 * e 1 * e 2) * (e 3 * e 4 * e 5) =
      -((e 3 * e 4 * e 5) * (e 0 * e 1 * e 2)) := by
  have h30 : e 0 * e 3 = -(e 3 * e 0) :=
    by simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 3)
  have h31 : e 1 * e 3 = -(e 3 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 3)
  have h32 : e 2 * e 3 = -(e 3 * e 2) :=
    by simpa using basis_anticomm (by decide : (2 : Fin 6) ≠ 3)
  have h40 : e 0 * e 4 = -(e 4 * e 0) :=
    by simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 4)
  have h41 : e 1 * e 4 = -(e 4 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 4)
  have h42 : e 2 * e 4 = -(e 4 * e 2) :=
    by simpa using basis_anticomm (by decide : (2 : Fin 6) ≠ 4)
  have h50 : e 0 * e 5 = -(e 5 * e 0) :=
    by simpa using basis_anticomm (by decide : (0 : Fin 6) ≠ 5)
  have h51 : e 1 * e 5 = -(e 5 * e 1) :=
    by simpa using basis_anticomm (by decide : (1 : Fin 6) ≠ 5)
  have h52 : e 2 * e 5 = -(e 5 * e 2) :=
    by simpa using basis_anticomm (by decide : (2 : Fin 6) ≠ 5)
  calc
    (e 0 * e 1 * e 2) * (e 3 * e 4 * e 5)
        = e 0 * e 1 * e 2 * e 3 * e 4 * e 5 := by simp [mul_assoc]
    _ = e 0 * e 1 * (e 2 * e 3) * e 4 * e 5 := by simp [mul_assoc]
    _ = e 0 * e 1 * (-(e 3 * e 2)) * e 4 * e 5 := by
          simp [h32]
    _ = -(e 0 * (e 1 * e 3) * e 2 * e 4 * e 5) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 0 * (-(e 3 * e 1)) * e 2 * e 4 * e 5) := by
          simp [h31]
    _ = e 0 * e 3 * e 1 * e 2 * e 4 * e 5 := by
          simp [mul_assoc, mul_neg]
    _ = (e 0 * e 3) * e 1 * e 2 * e 4 * e 5 := by
          simp [mul_assoc]
    _ = (-(e 3 * e 0)) * e 1 * e 2 * e 4 * e 5 := by
          simp [h30]
    _ = -(e 3 * e 0 * e 1 * e 2 * e 4 * e 5) := by
          simp [mul_assoc]
    _ = -(e 3 * e 0 * e 1 * (e 2 * e 4) * e 5) := by
          simp [mul_assoc]
    _ = -(e 3 * e 0 * e 1 * (-(e 4 * e 2)) * e 5) := by
          simp [h42]
    _ = e 3 * e 0 * e 1 * e 4 * e 2 * e 5 := by
          simp [mul_assoc, mul_neg]
    _ = e 3 * e 0 * (e 1 * e 4) * e 2 * e 5 := by
          simp [mul_assoc]
    _ = e 3 * e 0 * (-(e 4 * e 1)) * e 2 * e 5 := by
          simp [h41]
    _ = -(e 3 * e 0 * e 4 * e 1 * e 2 * e 5) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 3 * (e 0 * e 4) * e 1 * e 2 * e 5) := by
          simp [mul_assoc]
    _ = -(e 3 * (-(e 4 * e 0)) * e 1 * e 2 * e 5) := by
          simp [h40]
    _ = e 3 * e 4 * e 0 * e 1 * e 2 * e 5 := by
          simp [mul_assoc, mul_neg]
    _ = e 3 * e 4 * e 0 * e 1 * (e 2 * e 5) := by
          simp [mul_assoc]
    _ = e 3 * e 4 * e 0 * e 1 * (-(e 5 * e 2)) := by
          simp [h52]
    _ = -(e 3 * e 4 * e 0 * e 1 * e 5 * e 2) := by
          simp [mul_assoc, mul_neg]
    _ = -(e 3 * e 4 * e 0 * (e 1 * e 5) * e 2) := by
          simp [mul_assoc]
    _ = -(e 3 * e 4 * e 0 * (-(e 5 * e 1)) * e 2) := by
          simp [h51]
    _ = e 3 * e 4 * e 0 * e 5 * e 1 * e 2 := by
          simp [mul_assoc, mul_neg]
    _ = e 3 * e 4 * (e 0 * e 5) * e 1 * e 2 := by
          simp [mul_assoc]
    _ = e 3 * e 4 * (-(e 5 * e 0)) * e 1 * e 2 := by
          simp [h50]
    _ = -(e 3 * e 4 * e 5 * e 0 * e 1 * e 2) := by
          simp [mul_assoc]
    _ = -((e 3 * e 4 * e 5) * (e 0 * e 1 * e 2)) := by
          simp [mul_assoc]

end QFD.GA.BasisProducts
