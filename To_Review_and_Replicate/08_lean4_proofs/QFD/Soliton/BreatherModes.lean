import QFD.Nuclear.YukawaDerivation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Soliton Breather Modes

The Yukawa model supplies the restoring force; here we record a canonical
time-dependent radius that exhibits a periodic breather solution.
-/

namespace QFD.Soliton.BreatherModes

open scoped Real

/-- A simple harmonic breather mode centred at `R₀` with amplitude `δ`. -/
noncomputable def breathing_mode (R₀ δ β : ℝ) (t : ℝ) : ℝ :=
  R₀ + δ * Real.sin (β * t)

/--
Given a positive frequency `β`, the breather returns to its initial radius with
period `2π / β`, demonstrating a genuine oscillatory mode.
-/
theorem breathing_mode_exists (R₀ δ β : ℝ) (hβ : 0 < β) :
    ∃ T > 0, ∀ t, breathing_mode R₀ δ β (t + T) = breathing_mode R₀ δ β t := by
  refine ⟨(2 * Real.pi) / β, ?_, ?_⟩
  · exact div_pos Real.two_pi_pos hβ
  · intro t
    have hβ' : β ≠ 0 := ne_of_gt hβ
    have hperiod : β * ((2 * Real.pi) / β) = 2 * Real.pi := by
      simp [div_eq_mul_inv, hβ']
    have hsin :
        Real.sin (β * (t + (2 * Real.pi) / β)) = Real.sin (β * t) := by
      have := Real.sin_add (β * t) (β * ((2 * Real.pi) / β))
      simpa [mul_add, hperiod, Real.sin_two_pi, Real.cos_two_pi] using this
    simp [breathing_mode, hsin, mul_add, add_comm, add_left_comm, add_assoc]

end QFD.Soliton.BreatherModes
