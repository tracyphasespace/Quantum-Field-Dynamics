import Mathlib.Data.Real.Basic

/-
Core definitions shared by the mass-energy proofs and the centralized
physics postulates.
-/

noncomputable section

namespace QFD.Soliton

/-- Minimal abstraction of the stress-energy tensor for the vortex model. -/
structure StressEnergyTensor where
  T00 : ℝ → ℝ
  T_kinetic : ℝ → ℝ
  T_potential : ℝ → ℝ
  h_T00_def : ∀ r, T00 r = T_kinetic r + T_potential r
  h_T_kin_nonneg : ∀ r, 0 ≤ T_kinetic r
  h_T_pot_nonneg : ∀ r, 0 ≤ T_potential r

end QFD.Soliton
