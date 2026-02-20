import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Linarith

namespace QFD.Validation

open Real

/-!
# Golden Loop Root Uniqueness and Location Bounds

Proves that the Golden Loop function f(x) = exp(x)/x is strictly monotonic
for x ≥ 1, guaranteeing that any root found via IVT is unique and strictly
bounded by its evaluation interval.

## Axiom #4 Status

This file provides the MATHEMATICAL FRAMEWORK for eliminating Axiom #4:
- Monotonicity: exp(x)/x is strictly increasing for x ≥ 1
- Interval containment: if exp(β)/β = K and f(a) < K < f(b), then a < β < b

To fully eliminate Axiom #4, additionally need:
- Numerical bounds: exp(3.028)/3.028 < 6.891 < exp(3.058)/3.058
  (requires LeanCert interval arithmetic or careful rational bounding)
- Combined with GoldenLoopIVT.lean for existence

## Book Reference

- W.9.1 (Root existence and uniqueness)
-/

/-- Algebraic proof that exp(x)/x is strictly increasing for x ≥ 1.

    Key insight: For x ≥ 1 and y > x, we have 1 + (y-x) < exp(y-x)
    (strict convexity of exp). Multiplying by x and using x ≥ 1 gives
    y < x · exp(y-x), which rearranges to exp(x)/x < exp(y)/y. -/
theorem golden_func_strict_mono {x y : ℝ} (hx : 1 ≤ x) (hxy : x < y) :
    exp x / x < exp y / y := by
  have hd : 0 < y - x := sub_pos.mpr hxy
  have he : 1 + (y - x) < exp (y - x) := by
    have := add_one_lt_exp hd.ne'
    linarith
  have hx_pos : 0 < x := by linarith
  have hy_pos : 0 < y := by linarith
  have hexp_x_pos : 0 < exp x := exp_pos x
  rw [div_lt_div_iff₀ hx_pos hy_pos]
  have hy_eq : y = x + (y - x) := by ring
  have h_exp_y : exp y = exp x * exp (y - x) := by
    have : y = x + (y - x) := by linarith
    conv_lhs => rw [this]
    exact exp_add x (y - x)
  have h_ineq1 : x * (1 + (y - x)) < x * exp (y - x) :=
    mul_lt_mul_of_pos_left he hx_pos
  have h_ineq2 : y ≤ x * (1 + (y - x)) := by nlinarith
  have h_ineq3 : y < x * exp (y - x) := lt_of_le_of_lt h_ineq2 h_ineq1
  calc exp x * y = y * exp x := by ring
       _ < (x * exp (y - x)) * exp x := by
           exact mul_lt_mul_of_pos_right h_ineq3 hexp_x_pos
       _ = x * (exp x * exp (y - x)) := by ring
       _ = exp y * x := by rw [← h_exp_y]; ring

/-- If a root exists, it is strictly bounded by the evaluation points.

    Given monotonicity, if f(β) = K and f(a) < K < f(b), then a < β < b.
    Proof by contradiction using strict monotonicity. -/
theorem beta_root_bounds_in_interval (K β a b : ℝ)
    (ha : 1 ≤ a) (hb : 1 ≤ b) (ha_beta : 1 ≤ β)
    (h_root : exp β / β = K)
    (h_K_lower : exp a / a < K)
    (h_K_upper : K < exp b / b) :
    a < β ∧ β < b := by
  constructor
  · by_contra h
    push_neg at h
    rcases eq_or_lt_of_le h with rfl | h_lt
    · rw [h_root] at h_K_lower; linarith
    · have h_mono := golden_func_strict_mono ha_beta h_lt
      rw [h_root] at h_mono; linarith
  · by_contra h
    push_neg at h
    rcases eq_or_lt_of_le h with rfl | h_lt
    · rw [h_root] at h_K_upper; linarith
    · have h_mono := golden_func_strict_mono hb h_lt
      rw [h_root] at h_mono; linarith

/-- The exact location bound for β ≈ 3.043.

    If exp(β)/β = K with K bracketed by f(3.028) and f(3.058),
    then |β - 3.043| < 0.015. -/
theorem beta_location_bound
    (K β : ℝ)
    (h_root : exp β / β = K)
    (h_K_lower : exp 3.028 / 3.028 < K)
    (h_K_upper : K < exp 3.058 / 3.058)
    (h_beta_ge : 1 ≤ β) :
    |β - 3.043| < 0.015 := by
  have ha : (1 : ℝ) ≤ 3.028 := by norm_num
  have hb : (1 : ℝ) ≤ 3.058 := by norm_num
  have bounds := beta_root_bounds_in_interval K β 3.028 3.058 ha hb h_beta_ge
    h_root h_K_lower h_K_upper
  rw [abs_lt]
  constructor
  · linarith [bounds.1]
  · linarith [bounds.2]

end QFD.Validation
