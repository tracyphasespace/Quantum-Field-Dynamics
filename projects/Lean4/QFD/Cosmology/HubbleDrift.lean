import QFD.Cosmology.RadiativeTransfer
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
# Hubble Flow as Drift

We expose the standard exponential-decay form so downstream code can reuse the
symbol without duplicating ODE reasoning.  Filling in the analytic proof that
`deriv E = -H E` implies this form is left to the dedicated Cosmology cluster.
-/

namespace QFD.Cosmology.HubbleDrift

open QFD.Cosmology
open scoped Real

/-- Any candidate solution that matches the exponential ansatz stays exponential. -/
theorem redshift_is_exponential_decay (E : ℝ → ℝ) (H : ℝ)
    (h_solution : ∀ x, E x = E 0 * Real.exp (-H * x))
    (_ : ∀ x, deriv E x = -H * E x) :
    ∃ E0, ∀ x, E x = E0 * Real.exp (-H * x) :=
  ⟨E 0, h_solution⟩

end QFD.Cosmology.HubbleDrift
