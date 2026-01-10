/-!
# Validation Lockstep Summary

This file collects the bridging lemmas to state, in one place, that the Lean
model and the validation scripts are forced to use the same constants,
predictions, and cosmology relations.
-/

import QFD.Validation.BackbonePredictionBridge
import QFD.Validation.IntegerLadderConstraint
import QFD.Validation.KdVRedshiftBridge

namespace CodexProofs

/--
If the Lean backbone only admits integer solutions, then the script-side
predictions automatically exclude half-integers (the core integer ladder claim).
-/
theorem integer_ladder_locked
    (hint : ∀ A, ∃ n : ℤ, leanPrediction A = n) :
    ∀ A, ¬ ∃ k : ℤ, scriptPrediction A = (k : ℝ) + (1 / 2 : ℝ) := by
  intro A
  have legend :=
    (no_half_integer_predictions
      (Φ := fun A => leanPrediction A) (hint := hint)) A
  intro hx
  apply legend
  rcases hx with ⟨k, hk⟩
  refine ⟨k, ?_⟩
  simpa [script_vs_lean A, scriptPrediction, leanPrediction] using hk

/--
Restatement of the cosmology bridge: supplying the drag hypothesis instantly
implies the temperature scaling used in the scripts.
-/
theorem cosmology_locked
    (drag : KdVDrag) {E0 E T0 : ℝ} (hE0 : E0 > 0)
    (hE : E = E0 * Real.exp (-κ * D)) :
    let z := E0 / E - 1
    in T0 / (1 + z) = T0 * Real.exp (-κ * D) :=
  temperature_scaling (drag := drag) hE0 hE

end CodexProofs
