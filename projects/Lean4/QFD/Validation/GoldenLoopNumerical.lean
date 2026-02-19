import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import QFD.Physics.Postulates

/-!
# Golden Loop Numerical Verification

Machine-verified bounds for exp(beta)/beta where beta = 3.043233053.
Proves the key Golden Loop numerical identity without axioms, replacing
`QFD.Physics.beta_satisfies_transcendental`.

## Strategy

We split exp(beta) = exp(3) * exp(delta) where delta = beta - 3 = 0.043233053.

1. **exp(3)**: Bounded via `(exp 1)^3` using Mathlib's 9-digit bounds on exp(1).
2. **exp(delta)**: Lower bound from Taylor T3 via `sum_le_exp_of_nonneg`.
   Upper bound from `exp_bound'` n=4 (Taylor T3 + remainder 5/96).
3. **Combine**: Multiply bounds and divide by beta.
4. **Result**: |exp(beta)/beta - 6.891| < 0.001.

All arithmetic uses coarsened intermediate rational bounds to keep `nlinarith`
tractable, following the pattern in `NumericalConstants.lean`.

## Verified numerical chain

Mathlib: 2.7182818283 < exp(1) < 2.7182818286.
Coarsened: 2.71828 < exp(1) < 2.71829.
Cubed: 271828^3 = 20085496391455552 > 200854 * 10^11 = 20085400000000000.
        271829^3 = 20085718063655789 < 200858 * 10^11 = 20085800000000000.
So: 20.0854 < exp(3) < 20.0858.

delta = 0.043233053: T3 = 1.044181069... (lower), T3 + delta^4*5/96 = 1.044181251... (upper)
So: 1.04418 < exp(delta) < 1.04419.

Product: 200854 * 104418 = 20972772972 > 20972700000 = 209727 * 10^5.
         200858 * 104419 = 20973391502 < 20973400000 = 209734 * 10^5.
So: 20.9727 < exp(beta) < 20.9734.

Ratio:  6890 * 30433 = 209683370 < 209727000. (lower OK)
        209734000 < 209737344 = 6892 * 30432. (upper OK)
So: 6.890 < exp(beta)/beta < 6.892.
-/

open Real Finset

namespace QFD.Validation.GoldenLoopNumerical

noncomputable section

/-! ## Stage 0: Beta decomposition -/

/-- delta = beta - 3 = 0.043233053 -/
private def delta : ℝ := _root_.beta_golden - 3

lemma beta_eq_three_plus_delta : _root_.beta_golden = 3 + delta := by
  unfold delta; ring

lemma delta_val : delta = (43233053 : ℝ) / 1000000000 := by
  unfold delta _root_.beta_golden; norm_num

lemma delta_pos : 0 < delta := by rw [delta_val]; norm_num

lemma delta_lt_one : delta < 1 := by rw [delta_val]; norm_num

lemma delta_nonneg : 0 ≤ delta := le_of_lt delta_pos

lemma delta_le_one : delta ≤ 1 := le_of_lt delta_lt_one

lemma beta_pos : 0 < _root_.beta_golden := by unfold _root_.beta_golden; norm_num

lemma beta_gt : (30432 : ℝ) / 10000 < _root_.beta_golden := by
  unfold _root_.beta_golden; norm_num

lemma beta_lt : _root_.beta_golden < (30433 : ℝ) / 10000 := by
  unfold _root_.beta_golden; norm_num

/-! ## Stage 1: Bounds on exp(3)

exp(3) = (exp 1)^3, bounded using Mathlib's d9 bounds on exp(1). -/

lemma exp_three_eq : exp 3 = (exp 1) ^ 3 := by
  rw [← exp_nat_mul]; norm_num

private lemma exp_one_lower : (271828 : ℝ) / 100000 < exp 1 := by
  linarith [exp_one_gt_d9]

private lemma exp_one_upper : exp 1 < (271829 : ℝ) / 100000 := by
  linarith [exp_one_lt_d9]

/-- exp(3) > 20.0854.
    Proof: (exp 1)^3 > (2.71828)^3 > 20.0854. -/
lemma exp_three_gt : (200854 : ℝ) / 10000 < exp 3 := by
  rw [exp_three_eq]
  have h1 := exp_one_lower
  -- (exp 1)^3 > (271828/100000)^3 by nlinarith with sq_nonneg witnesses
  have h_cube : ((271828 : ℝ) / 100000) ^ 3 < (exp 1) ^ 3 := by
    nlinarith [sq_nonneg (exp 1 - 271828 / 100000),
               sq_nonneg (exp 1 + 271828 / 100000),
               sq_nonneg ((exp 1) ^ 2 - (271828 / 100000) ^ 2),
               sq_nonneg ((exp 1) ^ 2 + (271828 / 100000) ^ 2)]
  -- (271828/100000)^3 > 200854/10000 by norm_num
  -- 271828^3 = 20085496391455552 > 20085400000000000 = 200854 * 10^11
  have h_num : (200854 : ℝ) / 10000 < ((271828 : ℝ) / 100000) ^ 3 := by norm_num
  linarith

/-- exp(3) < 20.0858.
    Proof: (exp 1)^3 < (2.71829)^3 < 20.0858. -/
lemma exp_three_lt : exp 3 < (200858 : ℝ) / 10000 := by
  rw [exp_three_eq]
  have h1 := exp_one_upper
  have h_cube : (exp 1) ^ 3 < ((271829 : ℝ) / 100000) ^ 3 := by
    nlinarith [sq_nonneg (exp 1 - 271829 / 100000),
               sq_nonneg (exp 1 + 271829 / 100000),
               sq_nonneg ((exp 1) ^ 2 - (271829 / 100000) ^ 2),
               sq_nonneg ((exp 1) ^ 2 + (271829 / 100000) ^ 2)]
  -- 271829^3 = 20085718063655789 < 20085800000000000 = 200858 * 10^11
  have h_num : ((271829 : ℝ) / 100000) ^ 3 < (200858 : ℝ) / 10000 := by norm_num
  linarith

/-! ## Stage 2: Bounds on exp(delta)

Lower bound: 1 + x + x^2/2 + x^3/6 <= exp(x) for x >= 0.
This follows the pattern of Mathlib's `quadratic_le_exp_of_nonneg` but with
one more Taylor term, using `sum_le_exp_of_nonneg`.

Upper bound: exp(x) <= (1 + x + x^2/2 + x^3/6) + x^4 * 5/96 for 0 <= x <= 1.
This uses Mathlib's `exp_bound'` with n = 4. -/

/-- The Taylor partial sum for exp at delta, defined via Finset.sum.
    This is exactly the form that `sum_le_exp_of_nonneg` and `exp_bound'` use. -/
private def taylorSum4 : ℝ :=
  ∑ i ∈ range 4, delta ^ i / ↑(i.factorial)

/-- Taylor sum equals its closed form.
    Following the pattern of Mathlib's `quadratic_le_exp_of_nonneg`. -/
private lemma taylorSum4_eq :
    taylorSum4 = 1 + delta + delta ^ 2 / 2 + delta ^ 3 / 6 := by
  unfold taylorSum4
  simp only [sum_range_succ, range_zero, sum_empty, Nat.factorial, Nat.cast_one,
    Nat.cast_ofNat, pow_zero, pow_one]
  push_cast
  ring

/-- Taylor lower bound: taylorSum4 <= exp(delta).
    From Mathlib's `sum_le_exp_of_nonneg` with n = 4. -/
private lemma taylorSum4_le_exp_delta : taylorSum4 ≤ exp delta :=
  sum_le_exp_of_nonneg delta_nonneg 4

/-- Lower bound on exp(delta): exp(delta) > 1.04418.
    Proved via cubic Taylor lower bound at delta = 43233053/10^9. -/
lemma exp_delta_gt : (104418 : ℝ) / 100000 < exp delta := by
  have h := taylorSum4_le_exp_delta
  have h_val : (104418 : ℝ) / 100000 < taylorSum4 := by
    rw [taylorSum4_eq, delta_val]
    norm_num
  linarith

/-- Upper bound on exp(delta): exp(delta) < 1.04419.

    Uses Mathlib's `Real.exp_bound'` (n=4) which gives:
      exp(x) ≤ T3(x) + x^4 * 5/96
    where T3 = 1 + x + x²/2 + x³/6 is the cubic Taylor polynomial.

    Since 5/96 < 1, the remainder is bounded by delta^4.
    After substitution: T3(delta) + delta^4 ≈ 1.044181 + 3.5e-6 ≈ 1.04419 < 1.04419.

    Actually, we skip the coefficient bounding and just evaluate the full expression
    directly using the _ wildcard pattern from Mathlib (avoids form-matching issues). -/
lemma exp_delta_lt : exp delta < (104419 : ℝ) / 100000 := by
  -- Use exp_bound' with n = 4: exp(delta) ≤ T3 + delta^4 * 5/(4!*4).
  -- Pattern from Mathlib's exp_bound_div_one_sub_of_interval' (line 614):
  --   Use _ wildcard to avoid specifying the exact coercion form.
  calc exp delta
      ≤ _ := Real.exp_bound' delta_nonneg delta_le_one (show 0 < 4 by norm_num)
    _ < (104419 : ℝ) / 100000 := by
        -- Reduce the Finset.sum over range 4 and all Nat expressions to rationals.
        -- Same simp strategy as taylorSum4_eq (which already compiles).
        simp only [sum_range_succ, range_zero, sum_empty, Nat.factorial,
          Nat.cast_one, Nat.cast_ofNat, pow_zero, pow_one, zero_add]
        push_cast
        -- Substitute delta = 43233053/10^9 and verify.
        rw [delta_val]
        norm_num

/-! ## Stage 3: Combine to get exp(beta) bounds -/

lemma exp_beta_split : exp _root_.beta_golden = exp 3 * exp delta := by
  rw [beta_eq_three_plus_delta]
  exact exp_add 3 delta

/-- Lower bound: exp(beta) > 20.9727.
    Product of lower bounds: 200854/10000 * 104418/100000 > 209727/10000. -/
lemma exp_beta_gt : (209727 : ℝ) / 10000 < exp _root_.beta_golden := by
  rw [exp_beta_split]
  have h1 := exp_three_gt
  have h2 := exp_delta_gt
  have h_prod : (200854 : ℝ) / 10000 * ((104418 : ℝ) / 100000) ≤ exp 3 * exp delta :=
    mul_le_mul (le_of_lt h1) (le_of_lt h2) (by linarith [exp_pos delta]) (by linarith [exp_pos 3])
  -- 200854 * 104418 = 20972772972; 209727 * 100000 = 20972700000
  -- 20972772972 > 20972700000
  have h_val : (209727 : ℝ) / 10000 <
      (200854 : ℝ) / 10000 * ((104418 : ℝ) / 100000) := by norm_num
  linarith

/-- Upper bound: exp(beta) < 20.9734.
    Product of upper bounds: 200858/10000 * 104419/100000 < 209734/10000. -/
lemma exp_beta_lt : exp _root_.beta_golden < (209734 : ℝ) / 10000 := by
  rw [exp_beta_split]
  have h1 := exp_three_lt
  have h2 := exp_delta_lt
  have h_prod : exp 3 * exp delta ≤
      (200858 : ℝ) / 10000 * ((104419 : ℝ) / 100000) :=
    mul_le_mul (le_of_lt h1) (le_of_lt h2) (by linarith [exp_pos delta]) (by linarith [exp_pos 3])
  -- 200858 * 104419 = 20973391502; 209734 * 100000 = 20973400000
  -- 20973391502 < 20973400000
  have h_val : (200858 : ℝ) / 10000 * ((104419 : ℝ) / 100000) <
      (209734 : ℝ) / 10000 := by norm_num
  linarith

/-! ## Stage 4: exp(beta)/beta bounds -/

/-- Lower bound: exp(beta)/beta > 6.890. -/
lemma ratio_gt : (6890 : ℝ) / 1000 < exp _root_.beta_golden / _root_.beta_golden := by
  rw [lt_div_iff₀ beta_pos]
  have h_exp := exp_beta_gt
  have h_beta := beta_lt
  -- 6890/1000 * beta < 6890/1000 * 30433/10000
  have h_prod : (6890 : ℝ) / 1000 * _root_.beta_golden <
      (6890 : ℝ) / 1000 * ((30433 : ℝ) / 10000) :=
    mul_lt_mul_of_pos_left h_beta (by norm_num : (0 : ℝ) < 6890 / 1000)
  -- 6890 * 30433 = 209683370; 209727 * 1000 = 209727000
  -- 209683370 < 209727000, so 6890*30433/10^7 < 209727/10^4
  have h_val : (6890 : ℝ) / 1000 * ((30433 : ℝ) / 10000) <
      (209727 : ℝ) / 10000 := by norm_num
  linarith

/-- Upper bound: exp(beta)/beta < 6.892. -/
lemma ratio_lt : exp _root_.beta_golden / _root_.beta_golden < (6892 : ℝ) / 1000 := by
  rw [div_lt_iff₀ beta_pos]
  have h_exp := exp_beta_lt
  have h_beta := beta_gt
  -- 6892/1000 * 30432/10000 < 6892/1000 * beta
  have h_prod : (6892 : ℝ) / 1000 * ((30432 : ℝ) / 10000) <
      (6892 : ℝ) / 1000 * _root_.beta_golden :=
    mul_lt_mul_of_pos_left h_beta (by norm_num : (0 : ℝ) < 6892 / 1000)
  -- 6892 * 30432 = 209737344; 209734 * 1000 = 209734000
  -- 209734000 < 209737344, so 209734/10^4 < 6892*30432/10^7
  have h_val : (209734 : ℝ) / 10000 <
      (6892 : ℝ) / 1000 * ((30432 : ℝ) / 10000) := by norm_num
  linarith

/-! ## Stage 5: Main theorem -/

/-- **Golden Loop numerical identity**: |exp(beta)/beta - 6.891| < 0.001.

This theorem eliminates the axiom `QFD.Physics.beta_satisfies_transcendental`
from `Physics/Postulates.lean` by providing a constructive proof using only
Mathlib bounds on exp(1) and standard Taylor estimates.

**Proof chain**:
1. Mathlib d9 bounds: 2.7182818283 < exp(1) < 2.7182818286
2. exp(3) = (exp 1)^3, bounded by cubing coarsened bounds
3. exp(delta) bounded by Taylor T3 lower + T3+remainder upper
4. exp(beta) = exp(3) * exp(delta), bounded by multiplication
5. exp(beta)/beta bounded via conversion to multiplication form
6. |exp(beta)/beta - 6.891| < 0.001 from the two-sided bound

**Zero axioms, zero sorry.**

The tolerance achieved is tighter than stated: the true value of
exp(beta)/beta is approximately 6.89166, giving |6.89166 - 6.891| ~ 0.00066. -/
theorem beta_satisfies_transcendental_proved :
    abs (exp _root_.beta_golden / _root_.beta_golden - 6.891) < 0.001 := by
  rw [abs_lt]
  constructor
  · -- Need: -(0.001) < exp(beta)/beta - 6.891, i.e., 6.890 < exp(beta)/beta
    linarith [ratio_gt]
  · -- Need: exp(beta)/beta - 6.891 < 0.001, i.e., exp(beta)/beta < 6.892
    linarith [ratio_lt]

end

end QFD.Validation.GoldenLoopNumerical
