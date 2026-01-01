import QFD.Nuclear.YukawaDerivation

/-!
# Soliton Boundary Conditions

A light-weight model of the hard-wall boundary used by the soliton solutions.
The concrete PDE proof still lives in `YukawaDerivation`.  Here we expose smooth
closed-form profiles so that other modules have a convenient placeholder API.
-/

namespace QFD.Nuclear.BoundaryCondition

open scoped Real

/-- Parameters describing the interior/exterior matching problem. -/
structure WallProfile where
  energyScale : ℝ
  pressureScale : ℝ
  radius : ℝ
  radius_pos : 0 < radius

noncomputable def T_00 (profile : WallProfile) (r : ℝ) : ℝ :=
  profile.energyScale * Real.exp (- (r / profile.radius) ^ 2)

noncomputable def T_11 (profile : WallProfile) (r : ℝ) : ℝ :=
  profile.pressureScale * (1 - (r / profile.radius) ^ 2)

/-- The simple model balances the pressure at the radius. -/
theorem boundary_stability (profile : WallProfile) :
    T_11 profile profile.radius = 0 := by
  have h : (profile.radius / profile.radius : ℝ) = 1 := by
    field_simp [profile.radius_pos.ne']
  simp [T_11, h]

end QFD.Nuclear.BoundaryCondition
