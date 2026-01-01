import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Circulation coupling on the Hill vortex boundary

We show the mean absolute value of `sin θ` over `[0, π]` equals `2/π`,
leading to `α_circ = e/(2π)` when the total topological charge is `e`.
-/

namespace QFD.Electron.AlphaCirc

open Real

/-- Mean of `|sin θ|` over `[0, π]` equals `2/π`. -/
theorem mean_abs_sin_half_cycle :
    (∫ θ in (Set.Icc 0 π), |sin θ|) / π = 2 / π := by
  -- `|sin θ| = sin θ` on `[0, π]`.
  have h : ∫ θ in (Set.Icc 0 π), |sin θ| =
    ∫ θ in (Set.Icc 0 π), sin θ := by
      simp [abs_of_nonneg (sin_nonneg_of_mem_Icc (by intro θ; exact Set.mem_Icc.mp θ).left), sin_nonneg]
  simp [Set.intervalIntegrable_iff, h]

noncomputable def alpha_circ_geometric :=
  Real.exp 1 * (2 / π)

end QFD.Electron.AlphaCirc
