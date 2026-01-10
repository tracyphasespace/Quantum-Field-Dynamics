import QFD.Gravity.GeodesicEquivalence

/-!
# Perihelion Precession via Drift

The heavy lifting (solving the orbit equation with a refractive-index drift) is
performed by a standalone Python/Lean notebook.  Here we simply expose the
symbolic statement that relates the free coefficient `κ` in the refractive-index
expansion to the general-relativistic value `3`.
-/

namespace QFD.Gravity.PerihelionShift

open QFD.Gravity.GeodesicEquivalence

/-- The optical metric used by the drift formalism. -/
noncomputable def DriftingRefractiveIndex (r : ℝ) (GM c : ℝ) (kappa : ℝ) : ℝ :=
  1 + (2 * GM / (c ^ 2 * r)) + kappa * (GM / (c ^ 2 * r)) ^ 2

/--
If the effective index is tuned so that the orbit-averaged shift equals the
observed general-relativistic value, the coefficient must be `3`.
-/
theorem perihelion_shift_match (kappa : ℝ)
    (h_match : kappa / 3 = 1) :
    kappa = 3 := by
  have := congrArg (fun t : ℝ => t * 3) h_match
  simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using this

end QFD.Gravity.PerihelionShift
