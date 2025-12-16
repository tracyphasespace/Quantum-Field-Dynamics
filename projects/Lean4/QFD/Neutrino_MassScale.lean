import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Ring

noncomputable section

namespace QFD.Neutrino

/-!
# Gate N-L6: Mass Scale Estimate (Dimensional Analysis)

Focuses on structural API. This defines the mass m_ν as a derived quantity
of the R_proton / λ_electron geometric ratio.

The key result is that if the proton is smaller than the electron Compton wavelength,
the neutrino mass MUST be strictly positive but suppressed relative to the electron mass.

This version uses the correct Mathlib lemmas for power inequalities.
-/

/--
Inputs to the QFD Geometric Mass Theory.
We strictly enforce positivity of all physical scales.
-/
structure MassContext where
  m_e : ℝ        -- Electron Mass (in eV or consistent units)
  R_p : ℝ        -- Proton Radius
  lambda_e : ℝ   -- Electron Reduced Compton Wavelength

  h_me_pos : 0 < m_e
  h_Rp_pos : 0 < R_p
  h_lam_pos : 0 < lambda_e

/--
The Geometric Coupling Efficiency (ε).
Defined as the volumetric ratio of the Proton to the Electron.
ε ≈ (R_p / λ_e)³
-/
def coupling_efficiency (ctx : MassContext) : ℝ :=
  (ctx.R_p / ctx.lambda_e) ^ (3 : ℕ)

/--
The Derived Neutrino Mass.
m_ν = ε · m_e
-/
def neutrino_mass (ctx : MassContext) : ℝ :=
  coupling_efficiency ctx * ctx.m_e

/--
**Theorem N-L6**: Neutrino Mass Hierarchy.

If the proton is smaller than the electron (R_p < λ_e),
the neutrino mass MUST be strictly positive and strictly less than the electron mass.

This is the geometric suppression mechanism: m_ν = (R_p/λ_e)³ · m_e < m_e.
-/
theorem neutrino_mass_hierarchy (ctx : MassContext) (h_scale : ctx.R_p < ctx.lambda_e) :
    0 < neutrino_mass ctx ∧ neutrino_mass ctx < ctx.m_e := by
  unfold neutrino_mass coupling_efficiency

  have h_ratio_pos : 0 < ctx.R_p / ctx.lambda_e :=
    div_pos ctx.h_Rp_pos ctx.h_lam_pos

  have h_pow_pos : 0 < (ctx.R_p / ctx.lambda_e) ^ (3 : ℕ) :=
    pow_pos h_ratio_pos 3

  constructor

  -- 1. Positivity: 0 < m_ν
  · exact mul_pos h_pow_pos ctx.h_me_pos

  -- 2. Hierarchy: m_ν < m_e
  · have h_ratio_lt_1 : ctx.R_p / ctx.lambda_e < 1 := by
      rw [div_lt_one ctx.h_lam_pos]
      exact h_scale

    -- For 0 < x < 1, prove x^2 < 1 first
    have h_sq_lt_one : (ctx.R_p / ctx.lambda_e) ^ 2 < 1 := by
      calc (ctx.R_p / ctx.lambda_e) ^ 2
          = (ctx.R_p / ctx.lambda_e) * (ctx.R_p / ctx.lambda_e) := by ring
        _ < (ctx.R_p / ctx.lambda_e) * 1 := mul_lt_mul_of_pos_left h_ratio_lt_1 h_ratio_pos
        _ = ctx.R_p / ctx.lambda_e := by ring
        _ < 1 := h_ratio_lt_1

    -- Then prove x^3 < 1
    have h_pow_lt_1 : (ctx.R_p / ctx.lambda_e) ^ 3 < 1 := by
      calc (ctx.R_p / ctx.lambda_e) ^ 3
          = (ctx.R_p / ctx.lambda_e) * (ctx.R_p / ctx.lambda_e) ^ 2 := by ring
        _ < (ctx.R_p / ctx.lambda_e) * 1 := mul_lt_mul_of_pos_left h_sq_lt_one h_ratio_pos
        _ = ctx.R_p / ctx.lambda_e := by ring
        _ < 1 := h_ratio_lt_1

    -- Multiply both sides by m_e > 0
    calc coupling_efficiency ctx * ctx.m_e
        < 1 * ctx.m_e := mul_lt_mul_of_pos_right h_pow_lt_1 ctx.h_me_pos
      _ = ctx.m_e := by ring

end QFD.Neutrino
