
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.GCongr

set_option autoImplicit false

namespace QFD

noncomputable def V_sat (ρ ρ_max : ℝ) : ℝ :=
  ρ / (1 - ρ / ρ_max)

namespace Vsat

lemma V_sat_eq_mul_div (ρ ρ_max : ℝ) (hmax : ρ_max ≠ 0) :
    V_sat ρ ρ_max = (ρ * ρ_max) / (ρ_max - ρ) := by
  unfold V_sat
  field_simp [hmax]

lemma V_sat_le_two_mul (ρ ρ_max : ℝ) (hmax : ρ_max > 0)
    (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ ρ_max / 2) :
    V_sat ρ ρ_max ≤ 2 * ρ := by
  unfold V_sat
  have hden_pos : 0 < 1 - ρ / ρ_max := by linarith [(div_le_div_right hmax).mpr hρ, by norm_num]
  rw [div_le_iff hden_pos, mul_comm]
  gcongr
  linarith

theorem saturation_wall
    (ρ ρ_max M : ℝ)
    (hmax : ρ_max > 0)
    (hM : M > 0)
    (hρ : ρ_max / 2 ≤ ρ)
    (hρlt : ρ < ρ_max)
    (hclose : ρ_max - ρ < ρ_max^2 / (2 * M)) :
    V_sat ρ ρ_max > M := by
  rw [V_sat_eq_mul_div _ _ hmax.ne.symm]
  rw [div_gt_iff (by linarith)]
  gcongr
  linarith

end Vsat

noncomputable def ρ_of_R (Q R : ℝ) : ℝ := Q / (R ^ 3)

theorem radius_cube_lower_bound
    (Q R ρ_max : ℝ)
    (hQ : Q > 0)
    (hR : R > 0)
    (hmax : ρ_max > 0)
    (hcap : ρ_of_R Q R < ρ_max) :
    R ^ 3 > Q / ρ_max := by
  unfold ρ_of_R at hcap
  rwa [div_lt_iff (by linarith), mul_comm, div_lt_iff hmax] at hcap

end QFD
