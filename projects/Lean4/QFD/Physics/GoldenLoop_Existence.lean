/-
  Proof: Existence and Uniqueness of the Golden Loop Solution
  Author: QFD AI Assistant
  Date: January 10, 2026

  Theorem: exists_unique_beta

  Description:
  This proof demonstrates that the core QFD equation defining Vacuum Stiffness (Beta)
  has exactly one valid real solution. This refutes the skeptical claim that Beta
  is an arbitrary free parameter; it is mathematically determined by Alpha.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import QFD.Math.Function_Monotonicity

namespace QFD_Proofs.Existence

open Real Set

-- Redefining constants locally for self-contained proof context
noncomputable def alpha : ℝ := 1 / 137.035999
noncomputable def c1 : ℝ := 0.5 * (1 - alpha)
noncomputable def target_val : ℝ := c1 / (Real.pi^2 * alpha)

-- The core function derived from the Golden Loop: f(x) = x * e^x
noncomputable def golden_function (x : ℝ) : ℝ := x * exp x

/-- alpha is positive -/
lemma alpha_pos : alpha > 0 := by
  unfold alpha
  positivity

/-- alpha < 1 -/
lemma alpha_lt_one : alpha < 1 := by
  unfold alpha
  norm_num

/-- c1 is positive -/
lemma c1_pos : c1 > 0 := by
  unfold c1
  have h := alpha_lt_one
  linarith

/-- target_val is positive -/
lemma target_val_pos : target_val > 0 := by
  unfold target_val
  have h1 : c1 > 0 := c1_pos
  have h2 : Real.pi^2 > 0 := sq_pos_of_pos Real.pi_pos
  have h3 : alpha > 0 := alpha_pos
  have h4 : Real.pi^2 * alpha > 0 := mul_pos h2 h3
  exact div_pos h1 h4

/-- golden_function equals x_exp_x from Function_Monotonicity -/
lemma golden_function_eq : golden_function = fun x => x * exp x := rfl

/-- Lemma 1: The function f(x) = x * e^x is strictly monotonic for x > 0. -/
lemma golden_function_strict_mono : StrictMonoOn golden_function (Ioi 0) := by
  have h := QFD_Proofs.Starch.x_exp_x_monotonic
  convert h using 1

/-- golden_function is continuous -/
lemma golden_function_continuous : Continuous golden_function := by
  unfold golden_function
  exact continuous_id.mul continuous_exp

/-- golden_function at 0 is 0 -/
lemma golden_function_zero : golden_function 0 = 0 := by
  unfold golden_function; simp

/-- golden_function at large values exceeds target_val -/
lemma golden_function_large (M : ℝ) (hM : M > 0) : ∃ x : ℝ, x > 0 ∧ golden_function x > M := by
  -- x * exp(x) grows without bound
  -- For x = M + 1, we have x * exp(x) >= (M+1) * 1 = M + 1 > M since exp(x) >= 1 for x >= 0
  use M + 1
  constructor
  · linarith
  · unfold golden_function
    have hpos : M + 1 > 0 := by linarith
    have h1 : exp (M + 1) ≥ 1 := one_le_exp hpos.le
    have h2 : (M + 1) * exp (M + 1) ≥ (M + 1) * 1 := by
      apply mul_le_mul_of_nonneg_left h1
      linarith
    linarith

/-- Theorem: There exists a unique β > 0 satisfying the Golden Loop. -/
theorem exists_unique_beta : ∃! β : ℝ, β > 0 ∧ golden_function β = target_val := by
  -- Get bounds for IVT
  have htv_pos := target_val_pos
  obtain ⟨upper, hupper_pos, hupper_large⟩ := golden_function_large target_val htv_pos

  -- IVT on [0, upper]
  have h_cont : ContinuousOn golden_function (Icc 0 upper) :=
    golden_function_continuous.continuousOn

  have h_at_zero : golden_function 0 = 0 := golden_function_zero
  have h_low : golden_function 0 ≤ target_val := by rw [h_at_zero]; exact le_of_lt htv_pos
  have h_high : target_val ≤ golden_function upper := le_of_lt hupper_large

  -- target_val ∈ [f(0), f(upper)]
  have h_mem : target_val ∈ Icc (golden_function 0) (golden_function upper) := ⟨h_low, h_high⟩

  obtain ⟨β, hβ_mem, hβ_eq⟩ := intermediate_value_Icc (by linarith : (0:ℝ) ≤ upper) h_cont h_mem

  -- β > 0 (strict inequality since f(0) = 0 < target_val)
  have hβ_pos : β > 0 := by
    by_contra h
    push_neg at h
    have hβ_eq_0 : β = 0 := le_antisymm h hβ_mem.1
    rw [hβ_eq_0, golden_function_zero] at hβ_eq
    linarith

  -- Existence
  use β
  constructor
  · exact ⟨hβ_pos, hβ_eq⟩
  -- Uniqueness via strict monotonicity
  · intro β' ⟨hβ'_pos, hβ'_eq⟩
    by_contra hne
    have hmono := golden_function_strict_mono
    rcases lt_trichotomy β β' with hlt | heq | hgt
    · have h := hmono hβ_pos hβ'_pos hlt
      rw [hβ_eq, hβ'_eq] at h
      exact lt_irrefl _ h
    · exact hne heq.symm
    · have h := hmono hβ'_pos hβ_pos hgt
      rw [hβ_eq, hβ'_eq] at h
      exact lt_irrefl _ h

end QFD_Proofs.Existence