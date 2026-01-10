/-!
# KdV Interaction → Redshift Bridge

This module reframes the photon KdV interaction postulate as the exponential
energy decay law used in the cosmology validation scripts.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Basic

namespace CodexProofs

variables {κ D : ℝ}

/-- Abstract statement of the KdV drag postulate. -/
structure KdVDrag where
  κ_pos : κ > 0
  /-- Energy along a geodesic loses a fraction proportional to κ. -/
  energy_law :
    ∀ {E0 : ℝ}, E0 > 0 → ∃ f : ℝ → ℝ,
      (∀ d, f d = E0 * Real.exp (-κ * d))

/--
Given the drag law, the photon redshift relation `ln (1+z) = κ D` follows by
unwinding the exponential.
-/
theorem redshift_from_kdv
    (drag : KdVDrag) {E0 E : ℝ} (hE0 : E0 > 0)
    (hE : E = E0 * Real.exp (-κ * D)) :
    Real.log (E0 / E) = κ * D := by
  classical
  have hExp_ne : Real.exp (-κ * D) ≠ 0 := by
    exact (Real.exp_ne_zero _)
  have hE_ne : E ≠ 0 := by
    have := mul_ne_zero (ne_of_gt hE0) hExp_ne
    simpa [hE] using this
  have hratio : E0 / E = Real.exp (κ * D) := by
    field_simp [hE, hE_ne, hExp_ne, Real.exp_neg, mul_comm, mul_left_comm,
      mul_assoc]
  have : Real.log (Real.exp (κ * D)) = κ * D := Real.log_exp (κ * D)
  simpa [hratio] using this

/--
Expressed in the language of cosmology scripts: define `z := E0 / E - 1`,
then `log (1 + z) = κ D`.
-/
theorem cosmology_form (drag : KdVDrag) {E0 E : ℝ} (hE0 : E0 > 0)
    (hE : E = E0 * Real.exp (-κ * D)) :
    Real.log (1 + (E0 / E - 1)) = κ * D := by
  have := redshift_from_kdv (κ := κ) (D := D) drag hE0 hE
  simpa using this

/--
If photon energy tracks temperature linearly, the drag law implies the
cosmology script’s temperature scaling `T = T₀ / (1 + z)`.
-/
theorem temperature_scaling
    (drag : KdVDrag) {E0 E T0 : ℝ} (hE0 : E0 > 0)
    (hE : E = E0 * Real.exp (-κ * D)) :
    let z := E0 / E - 1
    in T0 / (1 + z) = T0 * Real.exp (-κ * D) := by
  intro z
  have hE_pos : E > 0 := by
    have hexp := Real.exp_pos (-κ * D)
    have := mul_pos hE0 hexp
    simpa [hE] using this
  have hz : z = E0 / E - 1 := rfl
  have hden_eq : 1 + z = E0 / E := by
    simp [hz, add_comm, add_left_comm, add_assoc, sub_eq_add_neg]
  have : E0 / E > 0 := div_pos hE0 hE_pos
  have hden_ne : 1 + z ≠ 0 := by
    have hzpos : 1 + z > 0 := by simpa [hden_eq] using this
    exact ne_of_gt hzpos
  simp [hz, hden_eq, hden_ne, hE, add_comm, add_left_comm, add_assoc]

end CodexProofs
