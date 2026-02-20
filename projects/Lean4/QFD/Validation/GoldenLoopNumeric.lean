import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Linarith
import QFD.Validation.GoldenLoopLocation

namespace QFD.Validation

open Real

/-!
# Golden Loop Numeric Bounds

Isolates the numerical evaluation of the exponential function from the
physical logic. By taking the floating-point bounds as hypotheses, we
completely eliminate the need for global axioms about the root.

Combined with `GoldenLoopLocation.lean` (monotonicity + interval containment)
and `GoldenLoopIVT.lean` (existence via IVT), this closes Axiom #4.

## Proof Strategy

The Golden Loop equation is exp(β)/β = K where K = 2π²/α + 1 ≈ 6.891.
We need: exp(3.028)/3.028 < 6.891 < exp(3.058)/3.058.

These numerical bounds on Real.exp require either:
- LeanCert `interval_bound` (verified floating point)
- Rational Taylor series bounding (tedious but axiom-free)

Here we take them as hypotheses, making the logical structure explicit:
the ONLY gap is numerical evaluation of exp at two points.
-/

/-- Given standard numerical bounds for exp(3.028) and exp(3.058),
    the Golden Loop target K = 6.891 is strictly bounded by the
    interval endpoints exp(a)/a and exp(b)/b.
    Combined with monotonicity, this guarantees |β - 3.043| < 0.015. -/
theorem golden_loop_numerical_bracket
    (h_eval_lo : exp 3.028 < 20.656)
    (h_eval_hi : 21.284 < exp 3.058) :
    exp 3.028 / 3.028 < 6.891 ∧ 6.891 < exp 3.058 / 3.058 := by
  constructor
  · -- exp(3.028) < 20.656 → exp(3.028)/3.028 < 20.656/3.028 < 6.891
    have h1 : exp 3.028 / 3.028 < 20.656 / 3.028 :=
      div_lt_div_of_pos_right h_eval_lo (by norm_num)
    have h2 : (20.656 : ℝ) / 3.028 < 6.891 := by norm_num
    linarith
  · -- 21.284 < exp(3.058) → 6.891 < 21.284/3.058 < exp(3.058)/3.058
    have h1 : (6.891 : ℝ) < 21.284 / 3.058 := by norm_num
    have h2 : 21.284 / 3.058 < exp 3.058 / 3.058 :=
      div_lt_div_of_pos_right h_eval_hi (by norm_num)
    linarith

/-- Full axiom-#4 replacement: combining bracket + location bound.
    The only hypotheses are numerical bounds on exp at two points. -/
theorem beta_location_from_numerical_bounds
    (K β : ℝ)
    (h_root : exp β / β = K)
    (h_K_approx : |K - 6.891| < 0.001)
    (h_beta_ge : 1 ≤ β)
    (h_eval_lo : exp 3.028 < 20.656)
    (h_eval_hi : 21.284 < exp 3.058) :
    2 < β ∧ β < 4 ∧ |β - 3.043| < 0.015 := by
  -- Chain through intermediate rational bounds
  have h_K_lower : exp 3.028 / 3.028 < K := by
    have hK_lo : (6.890 : ℝ) < K := by
      rw [abs_lt] at h_K_approx; linarith
    have h_div_bound : exp 3.028 / 3.028 < 20.656 / 3.028 :=
      div_lt_div_of_pos_right h_eval_lo (by norm_num)
    have h_rat : (20.656 : ℝ) / 3.028 < 6.890 := by norm_num
    linarith
  have h_K_upper : K < exp 3.058 / 3.058 := by
    have hK_hi : K < (6.892 : ℝ) := by
      rw [abs_lt] at h_K_approx; linarith
    have h_div_bound : 21.284 / 3.058 < exp 3.058 / 3.058 :=
      div_lt_div_of_pos_right h_eval_hi (by norm_num)
    have h_rat : (6.892 : ℝ) < 21.284 / 3.058 := by norm_num
    linarith
  have bounds := beta_root_bounds_in_interval K β 3.028 3.058
    (by norm_num) (by norm_num) h_beta_ge h_root h_K_lower h_K_upper
  refine ⟨by linarith [bounds.1], by linarith [bounds.2], ?_⟩
  rw [abs_lt]
  constructor
  · linarith [bounds.1]
  · linarith [bounds.2]

end QFD.Validation
