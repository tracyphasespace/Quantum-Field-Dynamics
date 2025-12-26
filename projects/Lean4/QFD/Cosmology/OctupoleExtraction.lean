-- QFD/Cosmology/OctupoleExtraction.lean
/-
Goal: formalize octupole axis extraction for P₃ patterns.

Design choice:
- P₃ is odd, so the sign-sensitive maximizer is {n}.
- The *axis* used in CMB discussions is signless; formalize maximizers of |P₃|.
  That yields {n, -n}, matching the quadrupole convention.
-/

import QFD.Cosmology.AxisExtraction
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

noncomputable section

open scoped Real BigOperators

namespace QFD.Cosmology

/-- Legendre polynomial P₃ (octupole) -/
def P3 (x : ℝ) : ℝ := (5 * x ^ 3 - 3 * x) / 2

/-- Pattern induced by the l=3 Legendre component along axis n -/
def octPattern (n x : R3) : ℝ := P3 (ip n x)

/-- Signless "axis pattern" (natural for axis-of-evil discussions) -/
def octAxisPattern (n x : R3) : ℝ := |octPattern n x|

/-! ## Algebraic Bound on P₃ -/

/-- Algebraic factorization: 1 - P3(t)^2 factors with (1 - t^2) and always-positive quadratics -/
lemma one_sub_P3_sq_factor (t : ℝ) :
    4 * (1 - (P3 t) ^ 2)
      = (1 - t ^ 2) * (5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2) := by
  simp [P3]
  ring

/-- The quadratics appearing in the factorization are strictly positive for all real t -/
lemma quad_pos_left (t : ℝ) : 0 < (5 * t ^ 2 - 5 * t + 2) := by
  -- Complete the square: 5(t-1/2)^2 + 3/4
  have : (5 * t ^ 2 - 5 * t + 2) = 5 * (t - (1/2)) ^ 2 + (3/4) := by ring
  nlinarith [sq_nonneg (t - (1/2))]

lemma quad_pos_right (t : ℝ) : 0 < (5 * t ^ 2 + 5 * t + 2) := by
  -- Complete the square: 5(t+1/2)^2 + 3/4
  have : (5 * t ^ 2 + 5 * t + 2) = 5 * (t + (1/2)) ^ 2 + (3/4) := by ring
  nlinarith [sq_nonneg (t + (1/2))]

/-- Core inequality: if |t| ≤ 1 then |P3 t| ≤ 1 (no calculus) -/
lemma abs_P3_le_one_of_abs_le_one {t : ℝ} (ht : |t| ≤ 1) : |P3 t| ≤ 1 := by
  have ht2 : 0 ≤ 1 - t ^ 2 := by
    have h1 : t ^ 2 ≤ 1 := by
      calc t ^ 2
          = |t| ^ 2 := by rw [sq_abs]
        _ ≤ 1 ^ 2 := by
          apply sq_le_sq'
          · linarith [abs_nonneg t]
          · exact ht
        _ = 1 := by norm_num
    linarith
  have hposL : 0 ≤ (5 * t ^ 2 - 5 * t + 2) := le_of_lt (quad_pos_left t)
  have hposR : 0 ≤ (5 * t ^ 2 + 5 * t + 2) := le_of_lt (quad_pos_right t)
  have hnonneg4 : 0 ≤ 4 * (1 - (P3 t) ^ 2) := by
    have h_factor := one_sub_P3_sq_factor t
    calc 4 * (1 - (P3 t) ^ 2)
        = (1 - t ^ 2) * (5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2) := h_factor
      _ ≥ 0 := by
          apply mul_nonneg
          · apply mul_nonneg
            · exact ht2
            · exact hposL
          · exact hposR
  have hnonneg : 0 ≤ (1 - (P3 t) ^ 2) := by linarith
  have hsq : (P3 t) ^ 2 ≤ 1 := by linarith
  have : (|P3 t|) ^ 2 ≤ (1 : ℝ) ^ 2 := by
    calc (|P3 t|) ^ 2
        = (P3 t) ^ 2 := by rw [sq_abs]
      _ ≤ 1 := hsq
      _ = 1 ^ 2 := by norm_num
  calc |P3 t|
      ≤ 1 := by
        have := sq_le_sq.mp this
        simpa using this

/-- Lift the bound to inner products of unit vectors: |P3(ip n x)| ≤ 1 -/
lemma abs_octPattern_le_one_of_unit (n x : R3) (hn : IsUnit n) (hx : IsUnit x) :
    |octPattern n x| ≤ 1 := by
  have hip : |ip n x| ≤ 1 := by
    have h := abs_real_inner_le_norm n x
    unfold ip IsUnit at *
    rw [hn, hx] at h
    simpa using h
  exact abs_P3_le_one_of_abs_le_one hip

/-! ## Equality Characterization for P₃ -/

/-- P3 achieves its maximum absolute value 1 exactly when t = ±1 -/
lemma abs_P3_eq_one_iff (t : ℝ) (ht : |t| ≤ 1) :
    |P3 t| = 1 ↔ t = 1 ∨ t = -1 := by
  constructor
  · intro h_abs_eq
    -- |P3 t| = 1 means (P3 t)^2 = 1
    have hsq : (P3 t) ^ 2 = 1 := by
      calc (P3 t) ^ 2
          = |P3 t| ^ 2 := by rw [sq_abs]
        _ = 1 ^ 2 := by rw [h_abs_eq]
        _ = 1 := by norm_num
    -- From factorization: 4(1 - (P3 t)^2) = 0
    have h_factor : 4 * (1 - (P3 t) ^ 2) = 0 := by linarith
    have h_prod : (1 - t ^ 2) * (5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2) = 0 := by
      have := one_sub_P3_sq_factor t
      linarith
    -- The quadratics are always positive, so 1 - t^2 = 0
    have h_quad_nonzero : (5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2) ≠ 0 := by
      apply mul_ne_zero
      · exact ne_of_gt (quad_pos_left t)
      · exact ne_of_gt (quad_pos_right t)
    have h_t2 : 1 - t ^ 2 = 0 := by
      by_contra h_ne
      have h_assoc : (1 - t ^ 2) * ((5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2)) ≠ 0 :=
        mul_ne_zero h_ne h_quad_nonzero
      have : (1 - t ^ 2) * (5 * t ^ 2 - 5 * t + 2) * (5 * t ^ 2 + 5 * t + 2) ≠ 0 := by
        simpa [mul_assoc] using h_assoc
      contradiction
    have : t ^ 2 = 1 := by linarith
    exact sq_eq_one_iff.mp this
  · intro h_pm
    cases h_pm with
    | inl h_pos =>
        rw [h_pos]
        unfold P3
        norm_num
    | inr h_neg =>
        rw [h_neg]
        unfold P3
        norm_num

/-! ## Axis Extraction for Octupole -/

/-- Axis extraction for the signless octupole pattern: maximizers are exactly {n, -n} -/
theorem AxisSet_octAxisPattern_eq_pm (n : R3) (hn : IsUnit n) :
    AxisSet (octAxisPattern n) = {x | x = n ∨ x = -n} := by
  ext x
  unfold AxisSet IsUnit at *
  constructor
  · intro ⟨hx_unit, hx_max⟩
    -- Show that octAxisPattern achieves its maximum 1 at x
    have h_ge : octAxisPattern n n ≤ octAxisPattern n x := hx_max n hn
    have h_le : octAxisPattern n x ≤ 1 := abs_octPattern_le_one_of_unit n x hn hx_unit
    have hnn : octAxisPattern n n = 1 := by
      unfold octAxisPattern octPattern P3 ip
      have h_inner : inner ℝ n n = ‖n‖^2 := real_inner_self_eq_norm_sq n
      rw [h_inner, hn]
      norm_num
    have hxEq1 : octAxisPattern n x = 1 := le_antisymm h_le (by simpa [hnn] using h_ge)
    -- Use equality characterization: |P3(ip n x)| = 1 iff ip n x = ±1
    have h_ip_bounds : |ip n x| ≤ 1 := by
      have h := abs_real_inner_le_norm n x
      simpa [ip, hn, hx_unit] using h
    have h_P3_eq : |P3 (ip n x)| = 1 := by
      unfold octAxisPattern octPattern at hxEq1
      exact hxEq1
    have h_ip : ip n x = 1 ∨ ip n x = -1 := abs_P3_eq_one_iff (ip n x) h_ip_bounds |>.mp h_P3_eq
    -- Convert ip = ±1 to x = ±n
    cases h_ip with
    | inl hip1 =>
        have : n = x := (ip_eq_one_iff_eq n x hn hx_unit).1 hip1
        exact Or.inl this.symm
    | inr hipm1 =>
        have hneg : n = -x := (ip_eq_neg_one_iff_eq_neg n x hn hx_unit).1 hipm1
        have : x = -n := by
          have := neg_eq_iff_eq_neg.mpr hneg
          exact this.symm
        exact Or.inr this
  · intro hx
    cases hx with
    | inl hxn =>
        rw [hxn]
        refine ⟨hn, ?_⟩
        intro y hy
        have : octAxisPattern n y ≤ 1 := abs_octPattern_le_one_of_unit n y hn hy
        have hnn : octAxisPattern n n = 1 := by
          unfold octAxisPattern octPattern P3 ip
          have h_inner : inner ℝ n n = ‖n‖^2 := real_inner_self_eq_norm_sq n
          rw [h_inner, hn]
          norm_num
        linarith [this, hnn]
    | inr hxneg =>
        rw [hxneg]
        have h_neg_unit : IsUnit (-n) := by
          unfold IsUnit at *
          rw [norm_neg]
          exact hn
        refine ⟨h_neg_unit, ?_⟩
        intro y hy
        have : octAxisPattern n y ≤ 1 := abs_octPattern_le_one_of_unit n y hn hy
        have h_neg_n : octAxisPattern n (-n) = 1 := by
          unfold octAxisPattern octPattern P3 ip
          have h_inner_neg : inner ℝ n (-n) = -(inner ℝ n n) := by simp [inner_neg_right]
          have h_inner : inner ℝ n n = ‖n‖^2 := real_inner_self_eq_norm_sq n
          rw [h_inner_neg, h_inner, hn]
          norm_num
        linarith [this, h_neg_n]

/-! ## Model-to-Data Bridge for Octupole -/

/-- Observational fit form for octupole: O(x) = A·|P₃(⟨n,x⟩)| + B -/
def octTempPattern (n : R3) (A B : ℝ) (x : R3) : ℝ :=
  A * octAxisPattern n x + B

/-- Bridge Theorem: Octupole axis extraction for observational fit form -/
theorem AxisSet_octTempPattern_eq_pm (n : R3) (hn : IsUnit n) (A B : ℝ) (hA : 0 < A) :
    AxisSet (octTempPattern n A B) = {x | x = n ∨ x = -n} := by
  have h_aff :
      AxisSet (fun x => A * octAxisPattern n x + B) = AxisSet (octAxisPattern n) := by
    exact AxisSet_affine (octAxisPattern n) A B hA
  calc AxisSet (octTempPattern n A B)
      = AxisSet (fun x => A * octAxisPattern n x + B) := by rfl
    _ = AxisSet (octAxisPattern n) := h_aff
    _ = {x | x = n ∨ x = -n} := AxisSet_octAxisPattern_eq_pm n hn

end QFD.Cosmology
