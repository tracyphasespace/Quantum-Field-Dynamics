/-!
# Backbone Prediction Bridge

This module instantiates the soliton backbone with the shared constants and
proves that the Lean and Python prediction functions are identical.
-/

import QFD.Validation.DerivedConstantConsistency
import QFD.Validation.SolitonBackboneFromModel

namespace CodexProofs

open CodexProofs

/-- The Lean-side model using the derived constants. -/
def leanSolitonModel : SolitonModel :=
  { c1 := c1Lean, c2 := c2Lean }

/-- Prediction function as implemented in the Python validators. -/
@[simp] def scriptPrediction (A : ℝ) : ℝ :=
  c1Script * A^(2/3 : ℝ) + c2Script * A

/-- Prediction function restated via the Lean model. -/
@[simp] def leanPrediction (A : ℝ) : ℝ :=
  backboneCharge leanSolitonModel A

lemma script_vs_lean (A : ℝ) :
    scriptPrediction A = leanPrediction A := by
  simp [scriptPrediction, leanPrediction, leanSolitonModel,
    backboneCharge, c1_matches_python, c2_matches_python]

/--
Instantiating `A` with any mass number immediately shows the code path used by
`analysis/nuclear/scripts/integer_ladder_test.py` matches the Lean backbone.
-/
theorem prediction_agrees_for_all (A : ℝ) :
    c1Script * A^(2/3 : ℝ) + c2Script * A =
      c1Lean * A^(2/3 : ℝ) + c2Lean * A := by
  simpa [scriptPrediction, leanPrediction] using script_vs_lean A

end CodexProofs

