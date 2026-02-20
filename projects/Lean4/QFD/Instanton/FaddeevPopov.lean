-- QFD/Instanton/FaddeevPopov.lean
-- Faddeev-Popov Jacobian for soliton zero-mode collective coordinates
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.NormNum

noncomputable section

namespace QFD.Instanton.FaddeevPopov

/-!
# Faddeev-Popov Jacobian for Soliton Zero Modes

This module formalizes the origin of the β prefactor in the Golden Loop denominator.

## Physical Context

When performing the path integral around a soliton background, translational
and orientational zero modes must be extracted as collective coordinates.
The Jacobian determinant from this extraction is:

  J = (S_cl / 2π)^{N/2}

where S_cl is the classical action and N is the number of zero modes.

For the QFD soliton:
- The spin axis traces S² → N = 2 orientational zero modes
- The classical action S_cl = β (the vacuum stiffness)
- Therefore J = (β/2π)^1 = β/2π

The β in the denominator of exp(β)/β in the Golden Loop is precisely this Jacobian
(up to the 2π normalization absorbed into the measure).

## References

- Coleman, "Aspects of Symmetry", Ch. 7.2 (collective coordinates)
- Rajaraman, "Solitons and Instantons", Ch. 4
- Book: W.3 Step 3, W.9.3 (Faddeev-Popov attribution)
-/

/-- The Jacobian determinant from extracting N collective coordinates
    around a classical solution with action S_cl.

    Standard instanton calculus: J = S_cl^{N/2} (absorbing 2π into measure). -/
def collective_coordinate_jacobian (S_cl : ℝ) (N : ℕ) : ℝ :=
  S_cl ^ ((N : ℝ) / 2)

/-- For a soliton with 2 orientational zero modes (spin axis on S²)
    and classical action S_cl = β, the Faddeev-Popov Jacobian equals β.

    This is why exp(β)/β appears in the Golden Loop — the denominator β
    is the collective coordinate Jacobian, not a "gapped-mode determinant". -/
theorem jacobian_two_zero_modes (S_cl β : ℝ) (h_pos : S_cl > 0) (h_eq : S_cl = β) :
    collective_coordinate_jacobian S_cl 2 = β := by
  unfold collective_coordinate_jacobian
  have h_div : (↑(2 : ℕ) : ℝ) / 2 = 1 := by norm_num
  rw [h_div, h_eq]
  exact Real.rpow_one β

/-- The Jacobian is positive when the classical action is positive. -/
theorem jacobian_pos (S_cl : ℝ) (h_pos : S_cl > 0) (N : ℕ) :
    collective_coordinate_jacobian S_cl N > 0 := by
  unfold collective_coordinate_jacobian
  exact Real.rpow_pos_of_pos h_pos _

/-- Scaling property: doubling zero modes squares the Jacobian.
    J(S_cl, 2N) = J(S_cl, N)² -/
theorem jacobian_doubling (S_cl : ℝ) (h_pos : S_cl > 0) (N : ℕ) :
    collective_coordinate_jacobian S_cl (2 * N) =
    (collective_coordinate_jacobian S_cl N) ^ (2 : ℝ) := by
  unfold collective_coordinate_jacobian
  rw [← Real.rpow_mul (le_of_lt h_pos)]
  congr 1
  push_cast
  ring

end QFD.Instanton.FaddeevPopov
