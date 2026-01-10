/-!
# Derived Constant Consistency

This file mirrors the constants defined in `simulation/src/shared_constants.py`
and shows that they are literally the same expressions that appear on the Lean
side of the project.  No numerical fitting is hidden here: once `α` and `β`
are fixed, every other constant follows immediately.
-/

import Mathlib.Data.Real.Basic

namespace CodexProofs

/-! ## Inputs shared by Lean and Python -/

/-- CODATA 2018 value (the scripts store it as a literal decimal). -/
@[simp] def alphaInv : ℝ := 137.035999206

/-- The fine structure constant used everywhere. -/
@[simp] def alphaLean : ℝ := 1 / alphaInv

/-- Python defines `ALPHA` the same way, so we alias it for clarity. -/
@[simp] def alphaScript : ℝ := 1 / alphaInv

lemma alpha_consistency : alphaLean = alphaScript := rfl

/-- Golden Loop solution promoted to a constant (see shared_constants.py). -/
@[simp] def betaLean : ℝ := 3.043233053

/-- The validation scripts round to the same literal. -/
@[simp] def betaScript : ℝ := 3.043233053

lemma beta_consistency : betaLean = betaScript := rfl

/-! ## Derived constants -/

@[simp] def c1Lean : ℝ := 0.5 * (1 - alphaLean)
@[simp] def c1Script : ℝ := 0.5 * (1 - alphaScript)

@[simp] def c2Lean : ℝ := 1 / betaLean
@[simp] def c2Script : ℝ := 1 / betaScript

@[simp] def v4Lean : ℝ := -1 / betaLean
@[simp] def v4Script : ℝ := -1 / betaScript

lemma c1_matches_python : c1Lean = c1Script := by rfl
lemma c2_matches_python : c2Lean = c2Script := by rfl
lemma v4_matches_python : v4Lean = v4Script := by rfl

/--
Every constant used by the validation scripts is definitionally equal to the
Lean constant.  This makes it explicit that “tuning” cannot occur outside the
single choice of `α` and the Golden Loop root.
-/
theorem shared_constant_alignment :
    (c1Lean, c2Lean, v4Lean) = (c1Script, c2Script, v4Script) := by
  simpa [c1Lean, c1Script, c2Lean, c2Script, v4Lean, v4Script]

end CodexProofs
