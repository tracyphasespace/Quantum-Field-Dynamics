-- QFD/Math/ReciprocalIneq.lean
/-
Commit-robust inequality helpers to avoid mathlib lemma-name drift.

These lemmas use only core tactics (field_simp, linarith, nlinarith) rather than
relying on specific mathlib lemma names that may change across versions.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Data.Real.Basic

namespace QFD.Math

open scoped Real

/-- If 0 < a and a ≤ b then 1/b ≤ 1/a -/
lemma one_div_le_one_div_of_le {a b : ℝ} (ha : 0 < a) (hab : a ≤ b) :
    1 / b ≤ 1 / a := by
  have hb : 0 < b := lt_of_lt_of_le ha hab
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hb0 : b ≠ 0 := ne_of_gt hb
  -- Clear denominators: goal becomes a ≤ b
  field_simp [ha0, hb0]
  linarith

/-- If 0 < a and a < b then 1/b < 1/a -/
lemma one_div_lt_one_div_of_lt {a b : ℝ} (ha : 0 < a) (hab : a < b) :
    1 / b < 1 / a := by
  have hb : 0 < b := lt_trans ha hab
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hb0 : b ≠ 0 := ne_of_gt hb
  field_simp [ha0, hb0]
  linarith

/-- If a ≤ 0 and 0 < b then a / b ≤ 0 -/
lemma div_nonpos_of_nonpos_of_pos {a b : ℝ} (ha : a ≤ 0) (hb : 0 < b) :
    a / b ≤ 0 := by
  have hb0 : b ≠ 0 := ne_of_gt hb
  rw [div_eq_mul_inv]
  have hinv_pos : 0 < b⁻¹ := by exact inv_pos.2 hb
  -- a ≤ 0 and 0 < b⁻¹, so a * b⁻¹ ≤ 0 * b⁻¹ = 0
  have : a * b⁻¹ ≤ 0 * b⁻¹ := mul_le_mul_of_nonneg_right ha (le_of_lt hinv_pos)
  simpa using this

/-- Reciprocal reverses inequality: 1 ≤ a⁻¹ iff a ≤ 1 (for a > 0) -/
lemma one_le_inv_iff {a : ℝ} (ha : 0 < a) : (1 ≤ a⁻¹) ↔ a ≤ 1 := by
  have ha0 : a ≠ 0 := ne_of_gt ha
  constructor
  · intro h
    -- 1 ≤ 1/a → a ≤ 1
    have : 1 * a ≤ a⁻¹ * a := by
      exact mul_le_mul_of_nonneg_right h (le_of_lt ha)
    have : a ≤ a⁻¹ * a := by simpa using this
    have : a ≤ 1 := by field_simp [ha0] at this; exact this
    exact this
  · intro h
    -- a ≤ 1 → 1 ≤ 1/a
    field_simp [ha0]
    linarith

/-- If 0 < a ≤ 1 then 1 ≤ a⁻¹ -/
lemma one_le_inv_of_pos_of_le_one {a : ℝ} (ha : 0 < a) (h1 : a ≤ 1) :
    1 ≤ a⁻¹ := by
  exact (one_le_inv_iff ha).2 h1

/-- If 0 < a < 1 then 1 < a⁻¹ -/
lemma one_lt_inv_of_pos_of_lt_one {a : ℝ} (ha : 0 < a) (h1 : a < 1) :
    1 < a⁻¹ := by
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hinv_pos : 0 < a⁻¹ := inv_pos.2 ha
  -- a < 1 and a > 0, so multiplying both sides by a⁻¹ preserves direction
  have h_mul : a * a⁻¹ < 1 * a⁻¹ := mul_lt_mul_of_pos_right h1 hinv_pos
  -- Simplify: a * a⁻¹ = 1
  have h_cancel : a * a⁻¹ = 1 := by field_simp [ha0]
  rw [h_cancel] at h_mul
  simpa using h_mul

/-- Product of two nonpositive numbers is nonnegative -/
lemma mul_nonneg_of_nonpos_of_nonpos {a b : ℝ} (ha : a ≤ 0) (hb : b ≤ 0) :
    0 ≤ a * b := by
  -- Write as (-a) * (-b) where both factors are nonneg
  have ha' : 0 ≤ -a := by linarith
  have hb' : 0 ≤ -b := by linarith
  have : 0 ≤ (-a) * (-b) := mul_nonneg ha' hb'
  calc 0 ≤ (-a) * (-b) := this
    _ = a * b := by ring

end QFD.Math
