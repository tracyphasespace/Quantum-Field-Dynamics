/-
  Proof: Golden Loop Interval Solver
  Theorem: golden_loop_interval_bound

  Description:
  A rigorous proof that the unique solution β to e^β/β = K exists
  for K in the QFD range. Uses results from Exp_Bounds_Analysis.
  This replaces the 'python_root_finding_beta' axiom.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import QFD.Math.Exp_Bounds_Analysis

namespace QFD_Proofs

open Real Set Filter

/-- The target constant derived from Alpha and c1 -/
noncomputable def K_target : ℝ := 6.890910

/-- The function f(β) = e^β / β - same as golden_f from Exp_Bounds_Analysis -/
noncomputable def f_transcendental (beta : ℝ) : ℝ :=
  (exp beta) / beta

/-- f_transcendental equals golden_f from Exp_Bounds_Analysis -/
lemma f_eq_golden_f : f_transcendental = QFD_Proofs.Starch.golden_f := by
  ext x; rfl

/-- Derivative of f(β) is positive for β > 1 -/
lemma f_strictly_increasing_on_Ioi_one :
    StrictMonoOn f_transcendental (Set.Ioi 1) := by
  rw [f_eq_golden_f]
  exact QFD_Proofs.Starch.golden_f_strict_mono_on_Ioi_one

/-- f is continuous on (0, ∞) -/
lemma f_continuousOn : ContinuousOn f_transcendental (Ioi 0) := by
  rw [f_eq_golden_f]
  exact QFD_Proofs.Starch.golden_f_continuousOn

/-- K_target > e (since K ≈ 6.89 > e ≈ 2.718) -/
lemma K_target_gt_e : K_target > exp 1 := by
  unfold K_target
  have h := QFD_Proofs.Starch.exp_one_lt_three
  linarith

/-- K_target ≤ 10 -/
lemma K_target_le_10 : K_target ≤ 10 := by
  unfold K_target; norm_num

/--
  Main Theorem: There exists a unique β > 1 such that f(β) = K_target.
  This is the Golden Loop root existence and uniqueness.
-/
theorem golden_loop_root_exists :
    ∃ β : ℝ, β > 1 ∧ f_transcendental β = K_target := by
  rw [f_eq_golden_f]
  exact QFD_Proofs.Starch.golden_f_exists_root K_target K_target_gt_e K_target_le_10

/-- Uniqueness: If two values β₁, β₂ > 1 both satisfy f(β) = K_target, they're equal -/
theorem golden_loop_root_unique (β₁ β₂ : ℝ)
    (hβ₁ : β₁ > 1) (hβ₂ : β₂ > 1)
    (h1 : f_transcendental β₁ = K_target) (h2 : f_transcendental β₂ = K_target) :
    β₁ = β₂ := by
  rw [f_eq_golden_f] at h1 h2
  exact QFD_Proofs.Starch.golden_f_unique_root K_target β₁ β₂ hβ₁ hβ₂ h1 h2

/-- Combined existence and uniqueness -/
theorem golden_loop_exists_unique :
    ∃! β : ℝ, β > 1 ∧ f_transcendental β = K_target := by
  obtain ⟨β, hβ_gt, hβ_eq⟩ := golden_loop_root_exists
  use β
  constructor
  · exact ⟨hβ_gt, hβ_eq⟩
  · intro β' ⟨hβ'_gt, hβ'_eq⟩
    symm
    exact golden_loop_root_unique β β' hβ_gt hβ'_gt hβ_eq hβ'_eq

end QFD_Proofs