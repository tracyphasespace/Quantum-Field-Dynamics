import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic

noncomputable section

namespace QFD.Lepton.Structure

variable (k_e : ℝ)  -- Coulomb Constant
variable (q   : ℝ)  -- Elementary Charge

structure VortexElectron where
  mass    : ℝ
  charge  : ℝ
  radius  : ℝ
  h_rad   : radius > 0

namespace Interaction

def ShieldingFactor (e : VortexElectron) (r : ℝ) : ℝ :=
  if r >= e.radius then
    1.0
  else
    (r / e.radius) ^ 3

def VortexForce (e : VortexElectron) (r : ℝ) (r_pos : r > 0) : ℝ :=
  let Q_eff := e.charge * ShieldingFactor e r
  k_e * (q * Q_eff) / r ^ 2

theorem external_is_classical_coulomb
  (e : VortexElectron) (r : ℝ) (hr : r >= e.radius) (hr_pos : r > 0) :
  VortexForce k_e q e r hr_pos = k_e * (q * e.charge) / r ^ 2 := by
  unfold VortexForce ShieldingFactor
  rw [if_pos hr]
  norm_num

theorem internal_is_zitterbewegung
  (e : VortexElectron) (r : ℝ) (hr : r < e.radius) (hr_pos : r > 0) :
  ∃ (k_spring : ℝ), VortexForce k_e q e r hr_pos = k_spring * r := by
  unfold VortexForce ShieldingFactor
  rw [if_neg (not_le_of_gt hr)]
  let k_spring := k_e * q * e.charge / e.radius ^ 3
  use k_spring
  dsimp
  field_simp
  ring

end Interaction
end QFD.Lepton.Structure
