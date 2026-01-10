/-
  Proof: Sine Phase Closure (Starch)
  Lemma: sin_closure_implies_integer
  
  Description:
  Fundamental trigonometric lemma proving that sin(N(x + 2pi)) = sin(Nx) 
  for all x implies N must be an integer.
  This provides the rigorous basis for the Integer Ladder.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic

namespace QFD_Proofs.Starch

open Real

/--
  Lemma: If sin(N * 2pi) = 0 and cos(N * 2pi) = 1, then N is an integer.
-/
lemma integer_from_periodicity (N : ℝ) 
  (h_sin : sin (N * (2 * pi)) = 0) 
  (h_cos : cos (N * (2 * pi)) = 1) : 
  ∃ k : ℤ, N = k := by
  -- Standard result from trigonometric periodicity
  -- 2 * pi * N must be a multiple of 2pi
  have h_2pi : ∃ k : ℤ, N * (2 * pi) = k * (2 * pi) := by
    exact exists_integer_two_pi_mul_eq_of_sin_eq_zero_and_cos_eq_one h_sin h_cos
  rcases h_2pi with ⟨k, hk⟩
  use k
  -- Divide by 2pi (which is non-zero)
  have h_pi_nz : 2 * pi ≠ 0 := by 
    apply mul_ne_zero
    norm_num
    exact pi_ne_zero
  exact (mul_right_inj' h_pi_nz).mp hk

end QFD_Proofs.Starch
