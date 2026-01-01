import QFD.Lepton.FineStructure

/-!
# Gravity from Vacuum Stiffness

Define the dimensionless gravitational coupling:
  α_G = G * m_p^2 / (ℏ c)
and the geometric factor:
  ξ_QFD = α_G * (L0 / l_p)^2.
-/

namespace QFD.Gravity

open Real

noncomputable def G_Target : ℝ := 6.67430e-11
noncomputable def protonMass : ℝ := 1.672619e-27
noncomputable def speedOfLight : ℝ := 299792458
noncomputable def planckConst : ℝ := 1.054571817e-34
noncomputable def planckLength : ℝ := 1.616255e-35
noncomputable def protonChargeRadius : ℝ := 0.8414e-15

noncomputable def alphaG : ℝ :=
  G_Target * (protonMass^2) / (planckConst * speedOfLight)

noncomputable def xi_qfd : ℝ :=
  alphaG * (protonChargeRadius / planckLength)^2

end QFD.Gravity
