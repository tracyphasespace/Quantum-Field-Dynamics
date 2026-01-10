/-
  Proof: Exponential Bounds Analysis (Starch)

  Description:
  Proves key properties of f(β) = exp(β)/β needed for Golden Loop derivation:
  1. f is strictly increasing on (1, ∞)
  2. For K > e with K ≤ 10, there exists unique β > 1 with f(β) = K

  The QFD target K ≈ 6.891 falls in this range.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Real.Basic
import Mathlib.Topology.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

namespace QFD_Proofs.Starch

open Real Set Filter

/-! ## Section 1: Basic Properties -/

/-- e < 3 -/
lemma exp_one_lt_three : exp 1 < 3 := by
  have h := Real.exp_one_lt_d9; linarith

/-- exp 1 > 2.71 -/
lemma exp_one_gt_271 : exp 1 > 2.71 := by
  have h := Real.exp_one_gt_d9; linarith

/-! ## Section 2: The Golden Function -/

/-- f(β) = exp(β)/β -/
noncomputable def golden_f (β : ℝ) : ℝ := exp β / β

/-- f is positive for β > 0 -/
lemma golden_f_pos (β : ℝ) (hβ : β > 0) : golden_f β > 0 := by
  unfold golden_f; exact div_pos (exp_pos β) hβ

/-- f(1) = e -/
lemma golden_f_at_one : golden_f 1 = exp 1 := by unfold golden_f; simp

/-- f is continuous on (0, ∞) -/
lemma golden_f_continuousOn : ContinuousOn golden_f (Ioi 0) := by
  unfold golden_f
  apply ContinuousOn.div continuousOn_exp continuousOn_id
  intro x hx; exact ne_of_gt hx

/-! ## Section 3: Derivative -/

/-- f is differentiable at β > 0 -/
lemma golden_f_differentiableAt (β : ℝ) (hβ : β > 0) : DifferentiableAt ℝ golden_f β := by
  unfold golden_f
  apply DifferentiableAt.div differentiableAt_exp differentiableAt_id
  exact ne_of_gt hβ

/-- f'(β) = exp(β)(β-1)/β² for β > 0 -/
lemma golden_f_deriv (β : ℝ) (hβ : β > 0) :
    deriv golden_f β = exp β * (β - 1) / β^2 := by
  have hne : β ≠ 0 := ne_of_gt hβ
  -- Use HasDerivAt for exp and id
  have hexp : HasDerivAt exp (exp β) β := Real.hasDerivAt_exp β
  have hid : HasDerivAt id 1 β := hasDerivAt_id β
  -- Quotient rule: (f/g)' = (f'g - fg')/g²
  have hdiv := HasDerivAt.div hexp hid hne
  -- golden_f and (exp / id) are the same function
  have hfunc_eq : golden_f = (exp / id) := by
    ext x
    simp only [golden_f, Pi.div_apply, id]
  -- Therefore their derivatives are equal
  calc deriv golden_f β = deriv (exp / id) β := by rw [hfunc_eq]
    _ = (exp β * id β - exp β * 1) / (id β)^2 := hdiv.deriv
    _ = (exp β * β - exp β * 1) / β^2 := by simp only [id]
    _ = exp β * (β - 1) / β^2 := by ring

/-- f'(β) > 0 for β > 1 -/
lemma golden_f_deriv_pos (β : ℝ) (hβ : β > 1) : deriv golden_f β > 0 := by
  rw [golden_f_deriv β (by linarith)]
  apply div_pos
  · exact mul_pos (exp_pos β) (by linarith)
  · exact sq_pos_of_pos (by linarith)

/-! ## Section 4: Monotonicity -/

/-- f is strictly increasing on (1, ∞) -/
theorem golden_f_strict_mono_on_Ioi_one : StrictMonoOn golden_f (Ioi 1) := by
  apply strictMonoOn_of_deriv_pos (convex_Ioi 1)
  · exact golden_f_continuousOn.mono (fun x hx => lt_trans (by norm_num : (0:ℝ) < 1) hx)
  · intro x hx
    simp only [interior_Ioi, mem_Ioi] at hx
    exact golden_f_deriv_pos x hx

/-! ## Section 5: Uniqueness -/

/-- Root uniqueness: f(β₁) = f(β₂) = K with β₁,β₂ > 1 implies β₁ = β₂ -/
theorem golden_f_unique_root (K : ℝ) (β₁ β₂ : ℝ)
    (hβ₁ : β₁ > 1) (hβ₂ : β₂ > 1)
    (h1 : golden_f β₁ = K) (h2 : golden_f β₂ = K) : β₁ = β₂ := by
  have hmono := golden_f_strict_mono_on_Ioi_one
  by_contra hne
  rcases lt_trichotomy β₁ β₂ with hlt | heq | hgt
  · have hlt_f := hmono hβ₁ hβ₂ hlt
    rw [h1, h2] at hlt_f
    exact lt_irrefl K hlt_f
  · exact hne heq
  · have hlt_f := hmono hβ₂ hβ₁ hgt
    rw [h2, h1] at hlt_f
    exact lt_irrefl K hlt_f

/-! ## Section 6: Existence for QFD Range -/

/-- exp(4) > 50 -/
lemma exp_four_gt_50 : exp 4 > 50 := by
  have h1 : exp 1 > 2.71 := exp_one_gt_271
  have h4 : exp 4 = (exp 1) ^ 4 := by rw [← exp_nat_mul]; norm_num
  -- Use that exp(1) > 2.71 and 2.71^4 > 50
  have h271_pow : (2.71 : ℝ) ^ 4 > 50 := by norm_num
  have h_pos : (0 : ℝ) < 2.71 := by norm_num
  -- exp(1)^4 > 2.71^4 since exp(1) > 2.71 and power is monotone on positives
  have h_exp_pow : (exp 1) ^ 4 > (2.71 : ℝ) ^ 4 := by
    have hexp_pos : 0 < exp 1 := exp_pos 1
    nlinarith [sq_nonneg (exp 1 - 2.71), sq_nonneg (exp 1 + 2.71),
               sq_nonneg ((exp 1)^2 - 2.71^2), sq_nonneg ((exp 1)^2 + 2.71^2)]
  linarith [h4 ▸ h_exp_pow]

/-- f(4) > 12 -/
lemma golden_f_four_gt_12 : golden_f 4 > 12 := by
  unfold golden_f
  have h : exp 4 / 4 > 50 / 4 := div_lt_div_of_pos_right exp_four_gt_50 (by norm_num)
  linarith

/-- K = 6.891 > e -/
lemma K_golden_gt_e : 6.891 > exp 1 := by
  have h := exp_one_lt_three; linarith

/-- Existence: For e < K ≤ 10, there exists β > 1 with f(β) = K -/
theorem golden_f_exists_root (K : ℝ) (hK_low : K > exp 1) (hK_high : K ≤ 10) :
    ∃ β : ℝ, β > 1 ∧ golden_f β = K := by
  have h_low : golden_f 1 < K := by rw [golden_f_at_one]; exact hK_low
  have h_high : golden_f 4 > K := lt_of_le_of_lt hK_high (by linarith [golden_f_four_gt_12])
  have h_cont : ContinuousOn golden_f (Icc 1 4) := by
    apply golden_f_continuousOn.mono
    intro x ⟨h1, _⟩; exact lt_of_lt_of_le (by norm_num) h1
  -- K ∈ (f(1), f(4)), so by IVT there exists c ∈ [1,4] with f(c) = K
  have h_mem : K ∈ Icc (golden_f 1) (golden_f 4) :=
    ⟨le_of_lt h_low, le_of_lt h_high⟩
  obtain ⟨c, hc_mem, hc_eq⟩ := intermediate_value_Icc (by norm_num : (1:ℝ) ≤ 4) h_cont h_mem
  use c
  constructor
  · -- c > 1 (strict inequality)
    by_contra hc_le
    push_neg at hc_le
    have hc_eq_1 : c = 1 := le_antisymm hc_le hc_mem.1
    rw [hc_eq_1, golden_f_at_one] at hc_eq
    linarith
  · exact hc_eq

/-- Main: Unique β > 1 with exp(β)/β = 6.891 -/
theorem exists_unique_golden_beta :
    ∃! β : ℝ, β > 1 ∧ golden_f β = 6.891 := by
  obtain ⟨β, hβ_gt, hβ_eq⟩ := golden_f_exists_root 6.891 K_golden_gt_e (by norm_num)
  use β
  constructor
  · exact ⟨hβ_gt, hβ_eq⟩
  · intro β' ⟨hβ'_gt, hβ'_eq⟩
    symm
    exact golden_f_unique_root 6.891 β β' hβ_gt hβ'_gt hβ_eq hβ'_eq

end QFD_Proofs.Starch
