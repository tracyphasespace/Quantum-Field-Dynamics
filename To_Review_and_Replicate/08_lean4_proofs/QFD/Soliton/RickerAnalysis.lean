import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Basic
import Mathlib.Topology.Order.DenselyOrdered
import Mathlib.Tactic

noncomputable section

namespace QFD.Soliton

open Real Set

/-!
# Ricker Wavelet Analysis (Eliminating HardWall Axioms)

This file provides proofs for the Ricker shape function that eliminate
the three axioms in HardWall.lean:
1. `ricker_shape_bounded`: S(x) ≤ 1
2. `ricker_negative_minimum`: For A < 0, min occurs at x = 0
3. `soliton_always_admissible`: For A > 0, stays above hard wall

We prove these using explicit calculus (no Filters) to maintain stability.
-/

/-! ## 1. The Ricker Shape Function -/

/-- The dimensionless Ricker shape function: S(x) = (1 - x²) exp(-x²/2) -/
def S (x : ℝ) : ℝ := (1 - x^2) * exp (-x^2 / 2)

/-! ## 2. Basic Properties -/

theorem S_at_zero : S 0 = 1 := by
  unfold S
  norm_num

theorem S_even (x : ℝ) : S (-x) = S x := by
  unfold S
  rw [neg_sq]

/-! ## 3. Key Bound: S(x) ≤ 1 -/

/-- The Ricker shape function is bounded above by 1 -/
theorem S_le_one (x : ℝ) : S x ≤ 1 := by
  unfold S
  -- Strategy: Split into cases based on |x|² vs 1
  by_cases h : x^2 ≤ 1
  · -- Case 1: |x| ≤ 1, so (1 - x²) ≥ 0
    have h1 : 1 - x^2 ≤ 1 := by
      have : 0 ≤ x^2 := sq_nonneg _
      linarith
    have h2 : exp (-x^2 / 2) ≤ 1 := by
      rw [exp_le_one_iff]
      have : 0 ≤ x^2 := sq_nonneg _
      linarith
    calc (1 - x^2) * exp (-x^2 / 2)
        ≤ 1 * exp (-x^2 / 2) := by
          apply mul_le_mul_of_nonneg_right h1
          exact le_of_lt (exp_pos _)
      _ ≤ 1 * 1 := by
          apply mul_le_mul_of_nonneg_left h2
          norm_num
      _ = 1 := by norm_num
  · -- Case 2: |x| > 1, so (1 - x²) < 0, hence S < 0 < 1
    have h_neg : 1 - x^2 < 0 := by linarith
    have : (1 - x^2) * exp (-x^2 / 2) < 0 := by
      apply mul_neg_of_neg_of_pos h_neg (exp_pos _)
    linarith

/-! ## 4. Minimum for Negative Amplitudes -/

/-- For A < 0, the minimum of A·S(x) occurs at x = 0 -/
theorem ricker_negative_minimum (A : ℝ) (h_neg : A < 0) (x : ℝ) :
    A ≤ A * S x := by
  have h_S_le : S x ≤ 1 := S_le_one x
  -- Direct algebraic approach: A ≤ A·S(x) iff A - A·S(x) ≤ 0
  -- Factor: A - A·S(x) = A·(1 - S(x))
  -- For A < 0 and 1 - S(x) ≥ 0, we get A·(1 - S(x)) ≤ 0
  rw [← sub_nonpos]
  have h_factor : A - A * S x = A * (1 - S x) := by ring
  rw [h_factor]
  have h_diff_nonneg : 0 ≤ 1 - S x := by linarith [h_S_le]
  apply mul_nonpos_of_nonpos_of_nonneg (le_of_lt h_neg) h_diff_nonneg

/-! ## 5. Critical Points and Minimum Value -/

/-- The derivative of S(x) is S'(x) = -x·exp(-x²/2)·(3 - x²) -/
theorem S_deriv (x : ℝ) :
    HasDerivAt S (- x * exp (-x^2 / 2) * (3 - x^2)) x := by
  unfold S
  -- S(x) = (1 - x²) · exp(-x²/2)
  -- Use product rule: (f·g)' = f'·g + f·g'

  -- Part 1: derivative of (1 - x²)
  have h1 : HasDerivAt (fun x => 1 - x^2) (-2 * x) x := by
    have h_pow : HasDerivAt (fun x => x^2) (2 * x) x := by
      convert hasDerivAt_pow 2 x using 1
      ring
    convert HasDerivAt.sub (hasDerivAt_const x 1) h_pow using 1
    ring

  -- Part 2: derivative of exp(-x²/2)
  have h2 : HasDerivAt (fun x => exp (-x^2 / 2)) (-x * exp (-x^2 / 2)) x := by
    -- Chain rule: d/dx[exp(u)] = exp(u) · u' where u = -x²/2
    have hu : HasDerivAt (fun x => -x^2 / 2) (-x) x := by
      have h_sq : HasDerivAt (fun x => x^2) (2 * x) x := by
        convert hasDerivAt_pow 2 x using 1
        ring
      convert HasDerivAt.div_const (HasDerivAt.neg h_sq) 2 using 1
      ring
    convert HasDerivAt.exp hu using 1
    ring

  -- Apply product rule: (f·g)' = f'·g + f·g'
  convert h1.mul h2 using 1
  -- Simplify: (-2x)·exp(-x²/2) + (1-x²)·(-x·exp(-x²/2))
  --         = -2x·exp(-x²/2) - x·exp(-x²/2) + x³·exp(-x²/2)
  --         = -x·exp(-x²/2)·(2 + 1 - x²)
  --         = -x·exp(-x²/2)·(3 - x²)
  ring

/-- S'(x) = 0 occurs at x = 0 and x = ±√3 -/
theorem S_critical_points (x : ℝ) :
    (- x * exp (-x^2 / 2) * (3 - x^2) = 0) ↔ (x = 0 ∨ x^2 = 3) := by
  constructor
  · intro h
    -- exp(-x²/2) is never zero, so either x = 0 or (3 - x²) = 0
    have h_exp : exp (-x^2 / 2) ≠ 0 := exp_ne_zero _
    by_cases hx : x = 0
    · left; exact hx
    · right
      -- From -x·exp(...)·(3-x²) = 0 and x ≠ 0, exp ≠ 0, we get 3-x² = 0
      have h_mul : -x * (exp (-x^2 / 2) * (3 - x^2)) = 0 := by
        calc -x * (exp (-x^2 / 2) * (3 - x^2))
            = -x * exp (-x^2 / 2) * (3 - x^2) := by ring
          _ = 0 := h
      have h_prod : exp (-x^2 / 2) * (3 - x^2) = 0 := by
        by_contra h_ne
        have h_nx : -x ≠ 0 := by
          intro h_contra
          have : x = 0 := by linarith
          contradiction
        have : -x * (exp (-x^2 / 2) * (3 - x^2)) ≠ 0 :=
          mul_ne_zero h_nx h_ne
        contradiction
      have : (3 - x^2) = 0 := by
        by_contra h_ne
        have : exp (-x^2 / 2) * (3 - x^2) ≠ 0 :=
          mul_ne_zero h_exp h_ne
        contradiction
      linarith
  · intro h
    cases h with
    | inl h =>
        simp [h]
    | inr h =>
        have : 3 - x^2 = 0 := by linarith
        simp [this]

/-- The value S(√3) = -2·exp(-3/2) -/
theorem S_at_sqrt3 : S (Real.sqrt 3) = -2 * exp (-3/2) := by
  unfold S
  have h_sq : (Real.sqrt 3)^2 = 3 := by
    rw [sq_sqrt]; norm_num
  rw [h_sq]
  ring

/-! ## 5.1 Global Lower Bound via Mean Value Monotonicity -/

private lemma deriv_S (x : ℝ) :
    deriv S x = (- x * exp (-x^2 / 2) * (3 - x^2)) := by
  simpa using (S_deriv x).deriv

private lemma S_continuous : Continuous S := by
  -- S(x) = (1 - x^2) * exp(-x^2/2)
  unfold S
  continuity

private lemma S_continuousOn (D : Set ℝ) : ContinuousOn S D :=
  S_continuous.continuousOn

private lemma S_differentiableOn (D : Set ℝ) : DifferentiableOn ℝ S (interior D) := by
  intro x hx
  exact (S_deriv x).differentiableAt.differentiableWithinAt

private lemma negS_continuousOn (D : Set ℝ) : ContinuousOn (fun x => -S x) D := by
  simpa using (S_continuous.neg.continuousOn)

private lemma negS_differentiableOn (D : Set ℝ) :
    DifferentiableOn ℝ (fun x => -S x) (interior D) := by
  intro x hx
  exact (S_deriv x).differentiableAt.neg.differentiableWithinAt

/-- `S` is monotone nondecreasing on `[√3, ∞)` -/
private lemma S_monotoneOn_Ici_sqrt3 :
    MonotoneOn S (Ici (Real.sqrt 3)) := by
  refine monotoneOn_of_deriv_nonneg
      (D := Ici (Real.sqrt 3))
      (convex_Ici (Real.sqrt 3))
      (S_continuousOn _)
      (S_differentiableOn _)
      ?_
  intro x hx
  -- hx : x ∈ interior (Ici √3) = Ioi √3
  have hx' : Real.sqrt 3 < x := by
    simpa [interior_Ici] using hx
  have hs0 : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3
  have hx0 : 0 ≤ x := le_trans hs0 (le_of_lt hx')
  have hxe : 0 ≤ x * exp (-x^2 / 2) :=
    mul_nonneg hx0 (le_of_lt (exp_pos _))

  -- show 3 - x^2 ≤ 0
  have h_le : Real.sqrt 3 ≤ x := le_of_lt hx'
  have habs : |Real.sqrt 3| ≤ |x| := by
    simpa [abs_of_nonneg hs0, abs_of_nonneg hx0] using h_le
  have hsq : (Real.sqrt 3)^2 ≤ x^2 := (sq_le_sq).2 habs
  have hx2 : 3 ≤ x^2 := by
    simpa [sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)] using hsq
  have h3 : 3 - x^2 ≤ 0 := by linarith

  -- derivative is ≥ 0 because: -(nonneg * nonpos) ≥ 0
  have hprod : x * exp (-x^2 / 2) * (3 - x^2) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos hxe h3

  -- rewrite deriv in the file's normal form
  have hrew :
      (- x * exp (-x^2 / 2) * (3 - x^2))
        = - (x * exp (-x^2 / 2) * (3 - x^2)) := by ring

  rw [deriv_S, hrew]
  exact neg_nonneg.2 hprod

/-- `S` is antitone (nonincreasing) on `[0, √3]` -/
private lemma S_antitoneOn_Icc_0_sqrt3 :
    AntitoneOn S (Icc (0 : ℝ) (Real.sqrt 3)) := by
  -- prove monotonicity of -S, then convert
  have hmono : MonotoneOn (fun x => -S x) (Icc (0 : ℝ) (Real.sqrt 3)) := by
    refine monotoneOn_of_deriv_nonneg
        (D := Icc (0 : ℝ) (Real.sqrt 3))
        (convex_Icc (0 : ℝ) (Real.sqrt 3))
        (negS_continuousOn _)
        (negS_differentiableOn _)
        ?_
    intro x hx
    -- hx : x ∈ interior (Icc 0 √3) = Ioo 0 √3
    have hx' : (0 : ℝ) < x ∧ x < Real.sqrt 3 := by
      simpa [interior_Icc] using hx
    have hx0 : 0 ≤ x := le_of_lt hx'.1
    have hs0 : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3

    -- show x^2 ≤ 3 from x ≤ √3
    have h_le : x ≤ Real.sqrt 3 := le_of_lt hx'.2
    have habs : |x| ≤ |Real.sqrt 3| := by
      simpa [abs_of_nonneg hx0, abs_of_nonneg hs0] using h_le
    have hsq : x^2 ≤ (Real.sqrt 3)^2 := (sq_le_sq).2 habs
    have hx2 : x^2 ≤ 3 := by
      simpa [sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)] using hsq
    have h3 : 0 ≤ 3 - x^2 := by linarith [hx2]

    have hxe : 0 ≤ x * exp (-x^2 / 2) :=
      mul_nonneg hx0 (le_of_lt (exp_pos _))
    have hprod : 0 ≤ x * exp (-x^2 / 2) * (3 - x^2) :=
      mul_nonneg hxe h3

    -- compute deriv(-S) via HasDerivAt.neg
    have hderiv_neg :
        deriv (fun x => -S x) x = - (- x * exp (-x^2 / 2) * (3 - x^2)) := by
      simpa using (S_deriv x).neg.deriv

    have hrew :
        - (- x * exp (-x^2 / 2) * (3 - x^2))
          = x * exp (-x^2 / 2) * (3 - x^2) := by ring

    rw [hderiv_neg, hrew]
    exact hprod

  -- convert monotone(-S) to antitone(S)
  intro x hx y hy hxy
  have h := hmono hx hy hxy  -- -S x ≤ -S y
  have h' := neg_le_neg h    -- S x ≥ S y
  simpa [neg_neg] using h'

/-- Global minimum statement: `S(√3) ≤ S(x)` for all `x` -/
theorem S_sqrt3_le (x : ℝ) : S (Real.sqrt 3) ≤ S x := by
  by_cases hx : 0 ≤ x
  · -- x ≥ 0
    by_cases hxs : x ≤ Real.sqrt 3
    · -- x ∈ [0, √3], use antitone with x ≤ √3
      have hxI : x ∈ Icc (0 : ℝ) (Real.sqrt 3) := ⟨hx, hxs⟩
      have hsI : Real.sqrt 3 ∈ Icc (0 : ℝ) (Real.sqrt 3) :=
        ⟨Real.sqrt_nonneg 3, le_rfl⟩
      have hant := S_antitoneOn_Icc_0_sqrt3 hxI hsI hxs
      simpa using hant
    · -- x ≥ √3, use monotone
      have hge : Real.sqrt 3 ≤ x := le_of_not_ge hxs
      have hxI : x ∈ Ici (Real.sqrt 3) := hge
      have hsI : Real.sqrt 3 ∈ Ici (Real.sqrt 3) := show Real.sqrt 3 ≤ Real.sqrt 3 from le_refl _
      have hmono := S_monotoneOn_Ici_sqrt3 hsI hxI hge
      simpa using hmono
  · -- x < 0, use evenness: S is even, so S(√3) ≤ S(-x) by positive case
    have hx' : 0 ≤ -x := by linarith
    by_cases hxs : -x ≤ Real.sqrt 3
    · have hxI : (-x) ∈ Icc (0 : ℝ) (Real.sqrt 3) := ⟨hx', hxs⟩
      have hsI : Real.sqrt 3 ∈ Icc (0 : ℝ) (Real.sqrt 3) :=
        ⟨Real.sqrt_nonneg 3, le_rfl⟩
      have hant := S_antitoneOn_Icc_0_sqrt3 hxI hsI hxs
      simpa [S_even x] using hant
    · have hge : Real.sqrt 3 ≤ -x := le_of_not_ge hxs
      have hxI : (-x) ∈ Ici (Real.sqrt 3) := hge
      have hsI : Real.sqrt 3 ∈ Ici (Real.sqrt 3) := show Real.sqrt 3 ≤ Real.sqrt 3 from le_refl _
      have hmono := S_monotoneOn_Ici_sqrt3 hsI hxI hge
      simpa [S_even x] using hmono

/-- Pointwise lower bound: `S(x) ≥ -2*exp(-3/2)` -/
theorem S_lower_bound (x : ℝ) : (-2 * exp (-3/2)) ≤ S x := by
  have h := S_sqrt3_le x
  -- S(√3) = -2*exp(-3/2)
  simpa [S_at_sqrt3] using h

/-! ## 6. Admissibility Lemmas (Replacing Axioms) -/

/-- Replaces `axiom ricker_shape_bounded` -/
theorem ricker_shape_bounded : ∀ x, S x ≤ 1 := S_le_one

/-- Replaces `axiom ricker_negative_minimum` -/
theorem ricker_negative_min (A : ℝ) (h : A < 0) : ∀ R, A ≤ A * S R :=
  ricker_negative_minimum A h

/-- Replaces `axiom soliton_always_admissible` (with required amplitude bound). -/
theorem soliton_always_admissible_aux
    (A v₀ : ℝ) (h_pos : 0 < A) (h_v₀ : 0 < v₀)
    (h_bound : A < v₀ * exp (3/2) / 2) :
    ∀ x, -v₀ < A * S x := by
  intro x

  -- 1) Use global lower bound on S
  have hS : (-2 * exp (-3/2)) ≤ S x := S_lower_bound x
  have hA0 : 0 ≤ A := le_of_lt h_pos
  have hmul_le : A * (-2 * exp (-3/2)) ≤ A * S x :=
    mul_le_mul_of_nonneg_left hS hA0

  -- 2) Convert amplitude bound into: -v₀ < A * (-2*exp(-3/2))
  have hposC : 0 < (2 * exp (-3/2)) := by
    have : 0 < exp (-3/2) := exp_pos _
    nlinarith

  have hmul_lt :
      A * (2 * exp (-3/2))
        < (v₀ * exp (3/2) / 2) * (2 * exp (-3/2)) :=
    mul_lt_mul_of_pos_right h_bound hposC

  have hexp : exp (3/2) * exp (-3/2) = (1 : ℝ) := by
    -- exp(a)*exp(b)=exp(a+b)
    have : (3/2 : ℝ) + (-3/2) = 0 := by ring
    calc exp (3/2) * exp (-3/2)
        = exp ((3/2 : ℝ) + (-3/2)) := by
            simpa [Real.exp_add] using (Real.exp_add (3/2 : ℝ) (-3/2 : ℝ)).symm
      _ = exp 0 := by simp [this]
      _ = 1 := by simp

  have hRHS :
      (v₀ * exp (3/2) / 2) * (2 * exp (-3/2)) = v₀ := by
    -- cancel /2 with *2 and cancel exp factors
    calc (v₀ * exp (3/2) / 2) * (2 * exp (-3/2))
        = v₀ * exp (3/2) * (1/2) * 2 * exp (-3/2) := by ring
      _ = v₀ * exp (3/2) * exp (-3/2) * ((1/2) * 2) := by ring
      _ = v₀ * exp (3/2) * exp (-3/2) * 1 := by norm_num
      _ = v₀ * (exp (3/2) * exp (-3/2)) := by ring
      _ = v₀ * 1 := by rw [hexp]
      _ = v₀ := by ring

  have hAv0 : A * (2 * exp (-3/2)) < v₀ := by
    simpa [hRHS, mul_assoc, mul_left_comm, mul_comm] using hmul_lt

  have hneg : -v₀ < A * (-2 * exp (-3/2)) := by
    -- A*(2*e) < v₀  =>  -v₀ < -A*(2*e) = A*(-2*e)
    have : -v₀ < -(A * (2 * exp (-3/2))) := by
      simpa using (neg_lt_neg hAv0)
    -- rewrite -(A*(2*e)) as A*(-2*e)
    simpa [mul_assoc, mul_left_comm, mul_comm] using this

  -- 3) Chain strict < then ≤
  exact lt_of_lt_of_le hneg hmul_le

end QFD.Soliton

end
