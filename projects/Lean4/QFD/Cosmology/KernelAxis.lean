-- QFD/Cosmology/KernelAxis.lean
/-
A commit-robust abstraction:

If a kernel K(u) on u ∈ [0,1] is maximized at u = 1 (with uniqueness),
then the argmax set of x ↦ K(|ip n x|) on the unit sphere is exactly {n, -n}.

This generalizes the P₂ and |P₃| axis extraction pattern and makes the
"robust to kernel family" claim precise.
-/

import QFD.Cosmology.AxisExtraction
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

noncomputable section
open scoped Real

namespace QFD.Cosmology

/-- Kernel property on [0,1]: maximum at 1 with uniqueness. -/
structure KernelMaxAtOne (K : ℝ → ℝ) : Prop where
  le_at_one : ∀ {u}, 0 ≤ u → u ≤ 1 → K u ≤ K 1
  eq_iff_one : ∀ {u}, 0 ≤ u → u ≤ 1 → (K u = K 1 ↔ u = 1)

/-- Kernel pattern built from |ip| (axis-signless by construction). -/
def absKernelPattern (K : ℝ → ℝ) (n x : R3) : ℝ := K (|ip n x|)

/-- Bound: for unit n,x we have |ip n x| ≤ 1. -/
lemma abs_ip_le_one_of_unit (n x : R3) (hn : IsUnit n) (hx : IsUnit x) :
    |ip n x| ≤ 1 := by
  have h : |inner ℝ n x| ≤ ‖n‖ * ‖x‖ := abs_real_inner_le_norm n x
  unfold ip IsUnit at *
  rw [hn, hx] at h
  simpa using h

/-- Value at the axis: |ip n n| = 1 for unit n. -/
lemma abs_ip_nn_eq_one (n : R3) (hn : IsUnit n) : |ip n n| = 1 := by
  have h_inner : inner ℝ n n = ‖n‖^2 := real_inner_self_eq_norm_sq n
  unfold ip IsUnit at *
  rw [h_inner, hn]
  norm_num

/-- Value at the opposite axis: |ip n (-n)| = 1 for unit n. -/
lemma abs_ip_nneg_eq_one (n : R3) (hn : IsUnit n) : |ip n (-n)| = 1 := by
  have h_inner_neg : inner ℝ n (-n) = -(inner ℝ n n) := by simp [inner_neg_right]
  have h_inner : inner ℝ n n = ‖n‖^2 := real_inner_self_eq_norm_sq n
  unfold ip IsUnit at *
  rw [h_inner_neg, h_inner, hn]
  norm_num

/--
**General axis extraction for abs-kernels**

If K has a unique maximum at u=1 on [0,1], then the maximizers of
x ↦ K(|ip n x|) on the unit sphere are exactly {n, -n}.
-/
theorem AxisSet_absKernelPattern_eq_pm
    (K : ℝ → ℝ) (hK : KernelMaxAtOne K)
    (n : R3) (hn : IsUnit n) :
    AxisSet (absKernelPattern K n) = {x | x = n ∨ x = -n} := by
  ext x
  unfold AxisSet IsUnit absKernelPattern at *
  constructor
  · intro ⟨hx_unit, hx_max⟩
    -- Compare to the value at n: f n ≤ f x
    have h_ge : K (|ip n n|) ≤ K (|ip n x|) := hx_max n hn
    -- Bound the x-value by maximality of K on [0,1]
    have hbound : |ip n x| ≤ 1 := abs_ip_le_one_of_unit n x hn hx_unit
    have hle : K (|ip n x|) ≤ K 1 := hK.le_at_one (abs_nonneg (ip n x)) hbound
    have hnn : |ip n n| = 1 := abs_ip_nn_eq_one n hn
    -- So K(|ip n x|) = K 1
    have hxEq : K (|ip n x|) = K 1 := by
      have : K 1 ≤ K (|ip n x|) := by simpa [hnn] using h_ge
      exact le_antisymm hle this
    -- Uniqueness: |ip n x| = 1
    have habsEq : |ip n x| = 1 :=
      (hK.eq_iff_one (abs_nonneg (ip n x)) hbound).1 hxEq
    -- Hence ip n x = ±1
    have hip : ip n x = 1 ∨ ip n x = -1 := by
      -- |a| = 1 ↔ a = 1 ∨ a = -1
      have : |ip n x| = 1 → ip n x = 1 ∨ ip n x = -1 := by
        intro h
        have : (ip n x) ^ 2 = 1 := by
          calc (ip n x) ^ 2
              = |ip n x| ^ 2 := by rw [sq_abs]
            _ = 1 ^ 2 := by rw [h]
            _ = 1 := by norm_num
        exact sq_eq_one_iff.mp this
      exact this habsEq
    -- Convert to x = ±n using your Phase-2 ip lemmas
    cases hip with
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
        -- x = n is an argmax
        rw [hxn]
        refine ⟨hn, ?_⟩
        intro y hy
        have hbound : |ip n y| ≤ 1 := abs_ip_le_one_of_unit n y hn hy
        have hle : K (|ip n y|) ≤ K 1 := hK.le_at_one (abs_nonneg (ip n y)) hbound
        have hnn : |ip n n| = 1 := abs_ip_nn_eq_one n hn
        simpa [hnn] using hle
    | inr hxneg =>
        -- x = -n is an argmax (since |ip n (-n)| = 1)
        rw [hxneg]
        have h_neg_unit : IsUnit (-n) := by
          unfold IsUnit at *
          simpa [norm_neg] using hn
        refine ⟨h_neg_unit, ?_⟩
        intro y hy
        have hbound : |ip n y| ≤ 1 := abs_ip_le_one_of_unit n y hn hy
        have hle : K (|ip n y|) ≤ K 1 := hK.le_at_one (abs_nonneg (ip n y)) hbound
        have hneg : |ip n (-n)| = 1 := abs_ip_nneg_eq_one n hn
        simpa [hneg] using hle

end QFD.Cosmology
