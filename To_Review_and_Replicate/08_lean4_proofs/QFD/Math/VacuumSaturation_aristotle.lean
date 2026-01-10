/-
This file was edited by Aristotle.

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: d8bd3544-a9d7-48c1-803d-e60fce7784ca
-/

import Mathlib


/- Aristotle failed to load this code into its environment. Double check that the syntax is correct.

Unexpected name `QFD` after `end`: The current section is unnamed

Hint: Delete the name `QFD` to end the current unnamed scope; outer named scopes can then be closed using additional `end` command(s):
  end ̵Q̵F̵D̵-/
set_option autoImplicit false

namespace QFD

/-!
# VacuumSaturation

This module captures the "granularity wall" as a mathematically explicit saturation
barrier.

Instead of adding ad-hoc high-order polynomial terms (ρ⁶, ρ⁸, ...), we model the
high-density regime with a saturation potential

  V_sat(ρ) = ρ / (1 - ρ/ρ_max)

Key properties proven here:
1. In the continuum regime (ρ ≪ ρ_max), V_sat is well-behaved.
2. As ρ approaches ρ_max from below, V_sat diverges (an asymptotic wall).
3. If density scales like Q / R³, then a density cap ρ < ρ_max implies a hard
   lower bound on the radius R (a "pixel size" effect).
-/

noncomputable section

/-- Saturation potential with asymptotic wall at ρ = ρ_max. -/
def V_sat (ρ ρ_max : ℝ) : ℝ :=
  ρ / (1 - ρ / ρ_max)

namespace Vsat

/-- Helpful algebraic rewrite: V_sat(ρ) = ρ * ρ_max / (ρ_max - ρ). -/
lemma V_sat_eq_mul_div (ρ ρ_max : ℝ) (hmax : ρ_max ≠ 0) :
    V_sat ρ ρ_max = (ρ * ρ_max) / (ρ_max - ρ) := by
  unfold V_sat
  -- 1 - ρ/ρ_max = (ρ_max - ρ)/ρ_max
  have h : (1 - ρ / ρ_max) = (ρ_max - ρ) / ρ_max := by
    field_simp [hmax]
    ring
  -- substitute and simplify
  simp [h, div_div, hmax, sub_eq_add_neg, add_comm, add_left_comm, add_assoc, mul_assoc]
  ring

/-- In the low-density regime (ρ ≤ ρ_max/2), saturation is controlled: V_sat(ρ) ≤ 2ρ. -/
lemma V_sat_le_two_mul (ρ ρ_max : ℝ) (hmax : ρ_max > 0)
    (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ ρ_max / 2) :
    V_sat ρ ρ_max ≤ 2 * ρ := by
  unfold V_sat
  have hden_pos : 0 < (1 - ρ / ρ_max) := by
    have : ρ / ρ_max ≤ (ρ_max / 2) / ρ_max := by
      gcongr
    -- (ρ_max/2)/ρ_max = 1/2
    have : ρ / ρ_max ≤ (1 : ℝ) / 2 := by
      simpa [div_div, hmax.ne', one_div, mul_assoc] using this
    have : (ρ / ρ_max) < 1 := by
      have : (1 : ℝ) / 2 < 1 := by norm_num
      exact lt_of_le_of_lt this this
    linarith
  have hden_ge : (1 - ρ / ρ_max) ≥ (1 : ℝ) / 2 := by
    -- from ρ/ρ_max ≤ 1/2
    have : ρ / ρ_max ≤ (1 : ℝ) / 2 := by
      have : ρ ≤ ρ_max / 2 := hρ
      have : ρ / ρ_max ≤ (ρ_max / 2) / ρ_max := by
        gcongr
      -- simplify
      simpa [div_div, hmax.ne', one_div, mul_assoc] using this
    linarith
  -- divide by a larger denominator gives a smaller quotient
  have : ρ / (1 - ρ / ρ_max) ≤ ρ / ((1 : ℝ) / 2) := by
    exact div_le_div_of_nonneg_left hρ0 (by linarith [hden_pos]) hden_ge
  -- simplify right side
  simpa [V_sat, div_div] using this

/-- The asymptotic wall: for any target M > 0, sufficiently close to ρ_max we exceed M. -/
theorem saturation_wall
    (ρ ρ_max M : ℝ)
    (hmax : ρ_max > 0)
    (hM : M > 0)
    (hρ : ρ_max / 2 ≤ ρ)
    (hρlt : ρ < ρ_max)
    (hclose : ρ_max - ρ < ρ_max^2 / (2 * M)) :
    V_sat ρ ρ_max > M := by
  -- rewrite into (ρ*ρ_max)/(ρ_max-ρ)
  have hmax0 : ρ_max ≠ 0 := ne_of_gt hmax
  have hden_pos : 0 < ρ_max - ρ := sub_pos.mpr hρlt
  have hnum_lb : ρ_max^2 / 2 ≤ ρ * ρ_max := by
    -- since ρ ≥ ρ_max/2 and ρ_max>0
    have : (ρ_max / 2) * ρ_max ≤ ρ * ρ_max := by
      gcongr
    -- (ρ_max/2)*ρ_max = ρ_max^2/2
    simpa [pow_two, mul_assoc, mul_left_comm, mul_comm, div_eq_mul_inv] using this
  have hVsat : V_sat ρ ρ_max = (ρ * ρ_max) / (ρ_max - ρ) := by
    simpa [V_sat_eq_mul_div ρ ρ_max hmax0]
  -- lower bound V_sat by (ρ_max^2/2)/(ρ_max-ρ)
  have hVsat_lb : (ρ_max^2 / 2) / (ρ_max - ρ) ≤ V_sat ρ ρ_max := by
    -- same positive denominator; compare numerators
    rw [hVsat]
    exact div_le_div_of_le (le_of_lt hden_pos) hnum_lb
  -- it suffices to show (ρ_max^2/2)/(ρ_max-ρ) > M
  have hcore : (ρ_max^2 / 2) / (ρ_max - ρ) > M := by
    -- from ρ_max - ρ < ρ_max^2/(2M)
    have hδpos : 0 < ρ_max^2 / (2 * M) := by
      have : 0 < ρ_max^2 := sq_pos_of_ne_zero hmax0
      nlinarith
    -- reciprocals flip
    have hrec : (1 / (ρ_max^2 / (2 * M))) < 1 / (ρ_max - ρ) :=
      one_div_lt_one_div_of_lt hden_pos hclose
    have hposfac : 0 < ρ_max^2 / 2 := by
      have : 0 < ρ_max^2 := sq_pos_of_ne_zero hmax0
      nlinarith
    have : (ρ_max^2 / 2) * (1 / (ρ_max^2 / (2 * M))) < (ρ_max^2 / 2) * (1 / (ρ_max - ρ)) :=
      mul_lt_mul_of_pos_left hrec hposfac
    -- simplify left product to M, right to (ρ_max^2/2)/(ρ_max-ρ)
    have hleft : (ρ_max^2 / 2) * (1 / (ρ_max^2 / (2 * M))) = M := by
      field_simp [hmax0, ne_of_gt hM]
      ring
    -- finalize
    have hright : (ρ_max^2 / 2) * (1 / (ρ_max - ρ)) = (ρ_max^2 / 2) / (ρ_max - ρ) := by
      simp [div_eq_mul_inv, mul_assoc]
    -- convert
    nlinarith [this, hleft, hright]
  -- combine lower bound with core
  exact lt_of_lt_of_le hcore hVsat_lb

end Vsat

/-! ## Radius lower bounds from a density cap -/

/-- A simple density–radius scaling (e.g. compression): ρ(R) = Q / R³. -/
def ρ_of_R (Q R : ℝ) : ℝ := Q / (R ^ 3)

/--
If density obeys ρ(R) = Q / R³ and we require ρ(R) < ρ_max, then R³ > Q / ρ_max.
This is the algebraic "granularity wall": a maximum packing density implies a minimum radius.
-/
theorem radius_cube_lower_bound
    (Q R ρ_max : ℝ)
    (hQ : Q > 0)
    (hR : R > 0)
    (hmax : ρ_max > 0)
    (hcap : ρ_of_R Q R < ρ_max) :
    R ^ 3 > Q / ρ_max := by
  unfold ρ_of_R at hcap
  -- Q / R^3 < ρ_max  ⇒  Q < ρ_max * R^3  ⇒  Q/ρ_max < R^3
  have hR3pos : R ^ 3 > 0 := by
    have : 0 < R := hR
    nlinarith
  have : Q < ρ_max * (R ^ 3) := by
    -- multiply both sides by R^3 > 0
    have := (div_lt_iff hR3pos).1 hcap
    simpa [mul_comm, mul_left_comm, mul_assoc] using this
  have : Q / ρ_max < R ^ 3 := by
    -- divide by ρ_max > 0
    have := (div_lt_iff hmax).2 this
    -- (Q/ρ_max) < (ρ_max*R^3)/ρ_max = R^3
    simpa [div_eq_mul_inv, mul_assoc] using this
  exact this

end QFD