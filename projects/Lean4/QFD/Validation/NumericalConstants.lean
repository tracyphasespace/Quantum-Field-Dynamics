import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import QFD.Physics.Postulates

/-!
# Numerical Constants Verification

Machine-verified bounds for key QFD numerical identities.
These verify claims in QFD_Edition_v9.9 using only Mathlib bounds on pi and e.

## Verified Identities
1. N_max = 2*pi*beta^3 in (177, 178) -- book Section 8.8.1 line 3815
2. eta = pi^2/beta^2 in (1.065, 1.067) -- book Section 9 line 4459
3. A_crit = 2*e^2*beta^2 in (136, 138) -- book Section 14 line 6140
4. r_0 = pi^2/(beta*e) in (1.19, 1.20) -- book Section 8 line 3751
5. c_2 = 1/beta to 0.01% -- book Section 12

All proofs use: Mathlib pi bounds (d6), exp(1) bounds (d9), norm_num on beta.
Zero sorry, zero axioms.

## Proof Strategy

beta_golden = 3.043233053 is a rational literal (3043233053/10^9 in Lean 4).
We stage the proofs to keep nlinarith's rational arithmetic tractable:

  Stage 1: Bound beta by small rationals (4 digits) via unfold + norm_num
  Stage 2: Bound beta^2 from beta bounds via nlinarith + sq_nonneg
  Stage 3: Bound beta^3 from beta^2 and beta via nlinarith + pow_succ
  Stage 4: Combine with Mathlib pi/e bounds for final results

For division identities (eta, r_0), convert to multiplication via
div_lt_iff / lt_div_iff to avoid division-specific lemmas.
-/

open Real

namespace QFD.Validation.NumericalConstants

-- beta_golden is defined at root level in Postulates.lean:
--   def beta_golden : R := 3.043233053
-- Lean 4 parses this as the rational 3043233053/10^9.

noncomputable section

/-! ## Stage 1: Basic beta bounds

Since beta_golden unfolds to a rational literal, norm_num handles
these directly after unfolding. We use 4-digit truncations as
intermediate stepping stones so that nlinarith in later stages
does not choke on 10^27-scale numerators. -/

/-- beta_golden > 3.0432 (truncation). -/
lemma beta_gt : (30432 : ℝ) / 10000 < _root_.beta_golden := by
  unfold _root_.beta_golden; norm_num

/-- beta_golden < 3.0433 (ceiling). -/
lemma beta_lt : _root_.beta_golden < (30433 : ℝ) / 10000 := by
  unfold _root_.beta_golden; norm_num

/-- beta_golden is positive. -/
lemma beta_pos : (0 : ℝ) < _root_.beta_golden := by
  linarith [beta_gt]

/-! ## Stage 2: Bounds on beta^2

We prove beta^2 bounds from the Stage 1 beta bounds.
The key insight for LOWER bounds: sq_nonneg(beta - a) gives
  beta^2 >= 2*a*beta - a^2
and combined with beta > a, nlinarith deduces beta^2 > a^2.
For UPPER bounds: the product (b - beta)*(b + beta) > 0 gives beta^2 < b^2.

Numerical check:
  (30432/10000)^2 = 9.26106624
  beta^2           = 9.26126741...
  (30433/10000)^2 = 9.26167489

We use 9261/1000 = 9.261 and 9262/1000 = 9.262 as coarsened bounds. -/

/-- beta^2 > 9.261. -/
lemma beta_sq_gt : (9261 : ℝ) / 1000 < _root_.beta_golden ^ 2 := by
  have h1 := beta_gt
  nlinarith [sq_nonneg (_root_.beta_golden - 30432 / 10000)]

/-- beta^2 < 9.262. -/
lemma beta_sq_lt : _root_.beta_golden ^ 2 < (9262 : ℝ) / 1000 := by
  have h1 := beta_lt
  have h2 := beta_pos
  -- (30433/10000 - beta) * (30433/10000 + beta) > 0
  -- => (30433/10000)^2 - beta^2 > 0
  -- => beta^2 < (30433/10000)^2 = 926167489/10^8 < 9262/1000
  nlinarith [mul_pos (show (0 : ℝ) < 30433 / 10000 - _root_.beta_golden by linarith)
                     (show (0 : ℝ) < 30433 / 10000 + _root_.beta_golden by linarith)]

/-- beta^2 is positive. -/
lemma beta_sq_pos : (0 : ℝ) < _root_.beta_golden ^ 2 := by
  have := beta_pos
  positivity

/-! ## Stage 3: Bounds on beta^3

From beta in (3.0432, 3.0433) and beta^2 in (9.261, 9.262),
we get beta^3 = beta^2 * beta.

For the LOWER bound, nlinarith multiplies:
  (beta^2 - 9261/1000) * beta >= 0    => beta^3 >= 9261/1000 * beta
  9261/1000 * (beta - 30432/10000) >= 0  => 9261/1000 * beta >= 9261*30432/10^7
  Combined: beta^3 >= 9261*30432/10^7 = 28.1830752 > 28.183

For the UPPER bound, we use 28188/1000 = 28.188 (slightly loose) since
  9262*30433/10^7 = 28.1870446, which just barely exceeds 28.187. -/

/-- beta^3 > 28.183. -/
lemma beta_cu_gt : (28183 : ℝ) / 1000 < _root_.beta_golden ^ 3 := by
  have h1 := beta_gt
  have h2 := beta_sq_gt
  have hb := beta_pos
  -- Rewrite beta^3 = beta^2 * beta for nlinarith
  have h3 : _root_.beta_golden ^ 3 = _root_.beta_golden ^ 2 * _root_.beta_golden := by ring
  rw [h3]
  nlinarith [mul_pos (show (0 : ℝ) < _root_.beta_golden ^ 2 - 9261 / 1000 by linarith)
                     (show (0 : ℝ) < _root_.beta_golden by linarith)]

/-- beta^3 < 28.188 (slightly loose for nlinarith tractability). -/
lemma beta_cu_lt : _root_.beta_golden ^ 3 < (28188 : ℝ) / 1000 := by
  have h1 := beta_lt
  have h2 := beta_sq_lt
  have hb := beta_pos
  -- Rewrite beta^3 = beta^2 * beta for nlinarith
  have h3 : _root_.beta_golden ^ 3 = _root_.beta_golden ^ 2 * _root_.beta_golden := by ring
  rw [h3]
  -- beta^2 * beta < 28188/1000
  -- From beta^2*(30433/10000 - beta) > 0: beta^2*beta < 30433/10000*beta^2
  -- From beta^2 < 9262/1000: 30433/10000*beta^2 < 30433*9262/10^7 = 28.187 < 28.188
  nlinarith [mul_pos (show (0 : ℝ) < _root_.beta_golden ^ 2 by positivity)
                     (show (0 : ℝ) < 30433 / 10000 - _root_.beta_golden by linarith)]

/-! ## Stage 4: Bounds on pi^2 and e^2

From Mathlib:
  pi_gt_d6 : 3.141592 < pi        pi_lt_d6 : pi < 3.141593
  exp_one_gt_d9 : 2.7182818283 < exp 1
  exp_one_lt_d9 : exp 1 < 2.7182818286

For LOWER bounds on squares, sq_nonneg(x - a) + (x > a) works.
For UPPER bounds on squares, use (b - x)*(b + x) > 0. -/

/-- pi^2 > 9.8696. -/
lemma pi_sq_gt : (98696 : ℝ) / 10000 < π ^ 2 := by
  have hpi := pi_gt_d6
  -- sq_nonneg(pi - 3141592/1000000) gives the cross term
  -- (3141592/1000000)^2 = 9869600294464/10^12 > 98696/10000 = 9869600000000/10^12
  nlinarith [sq_nonneg (π - 3141592 / 1000000)]

/-- pi^2 < 9.8697. -/
lemma pi_sq_lt : π ^ 2 < (98697 : ℝ) / 10000 := by
  have hpi := pi_lt_d6
  have hpi_pos : (0 : ℝ) < π := pi_pos
  -- (3141593/1000000 - pi) * (3141593/1000000 + pi) > 0
  -- => (3141593/1000000)^2 > pi^2
  -- (3141593/1000000)^2 = 9869604007649/10^12 < 98697/10000 = 9869700000000/10^12
  nlinarith [mul_pos (show (0 : ℝ) < 3141593 / 1000000 - π by linarith)
                     (show (0 : ℝ) < 3141593 / 1000000 + π by linarith)]

/-- Intermediate exp(1) bound: exp(1) > 2.71828 (coarsened from d9 for tractability). -/
lemma exp_gt_5d : (271828 : ℝ) / 100000 < exp 1 := by
  have he := exp_one_gt_d9; linarith

/-- Intermediate exp(1) bound: exp(1) < 2.71829 (coarsened from d9 for tractability). -/
lemma exp_lt_5d : exp 1 < (271829 : ℝ) / 100000 := by
  have he := exp_one_lt_d9; linarith

/-- (exp 1)^2 > 7.389.
Uses 5-digit exp bound: (271828/100000)^2 = 7.3890461584 > 7.389. -/
lemma exp_sq_gt : (7389 : ℝ) / 1000 < (exp 1) ^ 2 := by
  have he := exp_gt_5d
  nlinarith [sq_nonneg (exp 1 - 271828 / 100000)]

/-- (exp 1)^2 < 7.39.
Uses 5-digit exp bound: (271829/100000)^2 = 7.3891001241 < 7.39. -/
lemma exp_sq_lt : (exp 1) ^ 2 < (739 : ℝ) / 100 := by
  have he := exp_lt_5d
  have he_pos : (0 : ℝ) < exp 1 := exp_pos 1
  -- (271829/100000 - exp 1) * (271829/100000 + exp 1) > 0
  -- => (271829/100000)^2 > (exp 1)^2
  -- (271829/100000)^2 = 73891001241/10^10 = 7.3891001241 < 7.39
  nlinarith [mul_pos (show (0 : ℝ) < 271829 / 100000 - exp 1 by linarith)
                     (show (0 : ℝ) < 271829 / 100000 + exp 1 by linarith)]

/-! ## Identity 1: N_max = 2*pi*beta^3

Book Section 8.8.1 line 3815: "N_max = 2*pi*beta^3 = 177.087"
We prove: 177 < 2*pi*beta^3 < 178

Numerical check:
  2 * 3.141592 * 28.183 = 177.078... > 177
  2 * 3.141593 * 28.188 = 177.116... < 178 -/

theorem nmax_gt_177 : (177 : ℝ) < 2 * π * _root_.beta_golden ^ 3 := by
  have hpi := pi_gt_d6
  have hb3 := beta_cu_gt
  -- 2 * 3141592/1000000 * 28183/1000 = 177.079 > 177
  nlinarith [mul_pos (show (0 : ℝ) < π - 3141592 / 1000000 by linarith)
                     (show (0 : ℝ) < _root_.beta_golden ^ 3 - 28183 / 1000 by linarith)]

theorem nmax_lt_178 : 2 * π * _root_.beta_golden ^ 3 < (178 : ℝ) := by
  have hpi := pi_lt_d6
  have hb3 := beta_cu_lt
  have hpi_pos : (0 : ℝ) < π := Real.pi_pos
  have hb3_pos : (0 : ℝ) < _root_.beta_golden ^ 3 := by have := beta_pos; positivity
  -- 2 * pi * beta^3 < 2 * pi * 28188/1000 and 2 * pi < 2 * 3.141593
  nlinarith [mul_lt_mul_of_pos_left hb3 (by linarith : (0 : ℝ) < 2 * π)]

/-- **N_max = 2*pi*beta^3 in (177, 178)** -- book Section 8.8.1 -/
theorem nmax_bound :
    (177 : ℝ) < 2 * π * _root_.beta_golden ^ 3 ∧
    2 * π * _root_.beta_golden ^ 3 < (178 : ℝ) :=
  ⟨nmax_gt_177, nmax_lt_178⟩

/-! ## Identity 2: eta = pi^2 / beta^2

Book Section 9 line 4459: "eta = pi^2/beta^2 ~ 1.066"
We prove: 1.065 < pi^2/beta^2 < 1.067

Strategy: convert division to multiplication via lt_div_iff / div_lt_iff.
  1.065 < pi^2/beta^2  iff  1.065 * beta^2 < pi^2  (since beta^2 > 0)
  pi^2/beta^2 < 1.067  iff  pi^2 < 1.067 * beta^2

Numerical check:
  1.065 * 9.262 = 9.86403 < 9.8696 = pi^2_lower   YES
  1.067 * 9.261 = 9.88149 > 9.8697 = pi^2_upper   YES -/

theorem eta_gt : (1065 : ℝ) / 1000 < π ^ 2 / _root_.beta_golden ^ 2 := by
  rw [lt_div_iff₀ beta_sq_pos]
  -- Goal: 1065/1000 * beta^2 < pi^2
  -- From beta^2 < 9262/1000: 1065/1000 * beta^2 < 1065*9262/10^6 = 9864030/10^6
  -- From pi^2 > 98696/10000 = 9869600/10^6
  -- Since 9864030 < 9869600: 1065/1000 * beta^2 < pi^2.
  have hpi2 := pi_sq_gt
  have hb2 := beta_sq_lt
  -- Provide scaled hypothesis as hint
  nlinarith [mul_pos (show (0 : ℝ) < 1065 / 1000 by norm_num)
                     (show (0 : ℝ) < 9262 / 1000 - _root_.beta_golden ^ 2 by linarith)]

theorem eta_lt : π ^ 2 / _root_.beta_golden ^ 2 < (1067 : ℝ) / 1000 := by
  rw [div_lt_iff₀ beta_sq_pos]
  -- Goal: pi^2 < 1067/1000 * beta^2
  -- From beta^2 > 9261/1000: 1067/1000 * beta^2 > 1067*9261/10^6 = 9881487/10^6
  -- From pi^2 < 98697/10000 = 9869700/10^6
  -- Since 9881487 > 9869700: pi^2 < 1067/1000 * beta^2.
  have hpi2 := pi_sq_lt
  have hb2 := beta_sq_gt
  nlinarith [mul_pos (show (0 : ℝ) < 1067 / 1000 by norm_num)
                     (show (0 : ℝ) < _root_.beta_golden ^ 2 - 9261 / 1000 by linarith)]

/-- **eta = pi^2/beta^2 in (1.065, 1.067)** -- book Section 9 -/
theorem eta_bound :
    (1065 : ℝ) / 1000 < π ^ 2 / _root_.beta_golden ^ 2 ∧
    π ^ 2 / _root_.beta_golden ^ 2 < (1067 : ℝ) / 1000 :=
  ⟨eta_gt, eta_lt⟩

/-! ## Identity 3: A_crit = 2 * (exp 1)^2 * beta^2

Book Section 14 line 6140: "A_crit = 2*e^2*beta^2 ~ 136.9"
We prove: 136 < 2*(exp 1)^2*beta^2 < 138

Numerical check:
  2 * 7.389 * 9.261 = 136.859... > 136
  2 * 7.39  * 9.262 = 136.929... < 138 -/

theorem acrit_gt : (136 : ℝ) < 2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 := by
  have he2 := exp_sq_gt
  have hb2 := beta_sq_gt
  -- 2 * 7389/1000 * 9261/1000 = 136.861 > 136
  nlinarith [mul_pos (show (0 : ℝ) < (exp 1) ^ 2 - 7389 / 1000 by linarith)
                     (show (0 : ℝ) < _root_.beta_golden ^ 2 - 9261 / 1000 by linarith)]

theorem acrit_lt : 2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 < (138 : ℝ) := by
  have he2 := exp_sq_lt
  have hb2 := beta_sq_lt
  -- 2 * 739/100 * 9262/1000 = 136.912 < 138
  nlinarith [mul_pos (show (0 : ℝ) < 739 / 100 - (exp 1) ^ 2 by linarith)
                     (show (0 : ℝ) < 9262 / 1000 - _root_.beta_golden ^ 2 by linarith)]

/-- **A_crit = 2*e^2*beta^2 in (136, 138)** -- book Section 14 -/
theorem acrit_bound :
    (136 : ℝ) < 2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 ∧
    2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 < (138 : ℝ) :=
  ⟨acrit_gt, acrit_lt⟩

/-! ## Identity 4: r_0 = pi^2 / (beta * exp 1)

Book Section 8 line 3751: "r_0 = pi^2/(beta*e) ~ 1.193"
We prove: 1.19 < pi^2/(beta * exp 1) < 1.20

Strategy: convert to multiplication.
  1.19 < pi^2/(beta*e)  iff  1.19 * beta * e < pi^2
  pi^2/(beta*e) < 1.20  iff  pi^2 < 1.20 * beta * e

We first establish bounds on the product beta*exp(1), then
use those in the final comparison with pi^2. -/

/-- beta * exp(1) is positive. -/
lemma beta_exp_pos : (0 : ℝ) < _root_.beta_golden * exp 1 :=
  mul_pos beta_pos (exp_pos 1)

/-- beta * exp(1) > 8.272.
From beta > 30432/10000 and exp(1) > 271828/100000:
  30432 * 271828 / 10^9 = 8.2722... > 8.272. -/
lemma beta_exp_gt : (8272 : ℝ) / 1000 < _root_.beta_golden * exp 1 := by
  have hb := beta_gt
  have he := exp_gt_5d
  nlinarith [mul_pos (show (0 : ℝ) < _root_.beta_golden - 30432 / 10000 by linarith)
                     (show (0 : ℝ) < exp 1 - 271828 / 100000 by linarith)]

/-- beta * exp(1) < 8.273.
From beta < 30433/10000 and exp(1) < 271829/100000:
  30433 * 271829 / 10^9 = 8.2725... < 8.273. -/
lemma beta_exp_lt : _root_.beta_golden * exp 1 < (8273 : ℝ) / 1000 := by
  have hb : _root_.beta_golden < 30433 / 10000 := beta_lt
  have he : exp 1 < 271829 / 100000 := exp_lt_5d
  have hb_pos : (0 : ℝ) < _root_.beta_golden := beta_pos
  have he_pos : (0 : ℝ) < exp 1 := Real.exp_pos 1
  nlinarith [mul_lt_mul_of_pos_left he hb_pos]

theorem r0_gt : (119 : ℝ) / 100 < π ^ 2 / (_root_.beta_golden * exp 1) := by
  rw [lt_div_iff₀ beta_exp_pos]
  -- Goal: 119/100 * (beta * exp 1) < pi^2
  -- From beta_exp_lt: beta * exp 1 < 8273/1000
  -- 119/100 * 8273/1000 = 984487/100000 = 9.84487 < 9.8696 = pi_sq_gt. YES.
  have hpi2 := pi_sq_gt
  have hbe := beta_exp_lt
  nlinarith

theorem r0_lt : π ^ 2 / (_root_.beta_golden * exp 1) < (120 : ℝ) / 100 := by
  rw [div_lt_iff₀ beta_exp_pos]
  -- Goal: pi^2 < 120/100 * (beta * exp 1)
  -- From beta_exp_gt: beta * exp 1 > 8272/1000
  -- 120/100 * 8272/1000 = 992640/100000 = 9.9264 > 9.8697 = pi_sq_lt. YES.
  have hpi2 := pi_sq_lt
  have hbe := beta_exp_gt
  nlinarith

/-- **r_0 = pi^2/(beta*e) in (1.19, 1.20)** -- book Section 8 -/
theorem r0_bound :
    (119 : ℝ) / 100 < π ^ 2 / (_root_.beta_golden * exp 1) ∧
    π ^ 2 / (_root_.beta_golden * exp 1) < (120 : ℝ) / 100 :=
  ⟨r0_gt, r0_lt⟩

/-! ## Identity 5: c_2 = 1/beta ~ 0.3286

Book Section 12: c_2 = 1/beta.
Since beta_golden is a rational literal, 1/beta_golden is rational
and this reduces to pure rational arithmetic via norm_num. -/

/-- 1/beta > 0.3285.
Equivalent to: 3285 * beta < 10000, since beta > 0. -/
lemma c2_gt : (3285 : ℝ) / 10000 < 1 / _root_.beta_golden := by
  have hb := beta_pos
  rw [lt_div_iff₀ hb]
  -- Goal: 3285/10000 * beta < 1
  -- i.e., 3285 * beta < 10000
  -- beta < 30433/10000, so 3285 * beta < 3285 * 30433/10000 = 9997240.5/10000 < 1
  -- Actually: 3285/10000 * beta < 3285/10000 * 30433/10000 = 99972405/100000000 < 1
  have h1 := beta_lt
  nlinarith

/-- 1/beta < 0.3287.
Equivalent to: 10000 < 3287 * beta, since beta > 0. -/
lemma c2_lt : 1 / _root_.beta_golden < (3287 : ℝ) / 10000 := by
  have hb := beta_pos
  rw [div_lt_iff₀ hb]
  -- Goal: 1 < 3287/10000 * beta
  -- beta > 30432/10000, so 3287/10000 * beta > 3287 * 30432/10^8 = 100030.../ 10^8 > 1
  have h1 := beta_gt
  nlinarith

/-- **c_2 = 1/beta matches 0.3286 within 0.0001** -- book Section 12.
We express this as |1/beta - 0.3286| < 0.0001. -/
theorem c2_bound : |1 / _root_.beta_golden - (3286 : ℝ) / 10000| < (1 : ℝ) / 10000 := by
  rw [abs_lt]
  exact ⟨by linarith [c2_gt], by linarith [c2_lt]⟩

/-! ## Summary theorem: all five identities verified -/

/-- All five QFD numerical identities verified within stated precision. -/
theorem all_five_identities :
    -- 1. N_max = 2*pi*beta^3 in (177, 178)
    ((177 : ℝ) < 2 * π * _root_.beta_golden ^ 3 ∧
     2 * π * _root_.beta_golden ^ 3 < 178) ∧
    -- 2. eta = pi^2/beta^2 in (1.065, 1.067)
    ((1065 : ℝ) / 1000 < π ^ 2 / _root_.beta_golden ^ 2 ∧
     π ^ 2 / _root_.beta_golden ^ 2 < 1067 / 1000) ∧
    -- 3. A_crit = 2*e^2*beta^2 in (136, 138)
    ((136 : ℝ) < 2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 ∧
     2 * (exp 1) ^ 2 * _root_.beta_golden ^ 2 < 138) ∧
    -- 4. r_0 = pi^2/(beta*e) in (1.19, 1.20)
    ((119 : ℝ) / 100 < π ^ 2 / (_root_.beta_golden * exp 1) ∧
     π ^ 2 / (_root_.beta_golden * exp 1) < 120 / 100) ∧
    -- 5. c_2 = 1/beta within 0.01%
    (|1 / _root_.beta_golden - 3286 / 10000| < 1 / 10000) :=
  ⟨nmax_bound, eta_bound, acrit_bound, r0_bound, c2_bound⟩

end

end QFD.Validation.NumericalConstants
