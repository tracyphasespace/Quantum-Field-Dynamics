import QFD.Charge.Vacuum
import QFD.Charge.Potential
import Mathlib.Analysis.Calculus.Deriv.Basic

noncomputable section

namespace QFD.Charge

open Real Filter
open scoped Topology

/-!
# Gate C-L3: Virtual Force (Derivation of Coulomb's Law)
-/

/-- Charge density field definition. -/
def charge_density_field (ctx : VacuumContext) (sign : PerturbationSign) (k : ℝ) (r : ℝ) : ℝ :=
  ctx.rho_vac + (sign_value sign) * (k / r)

/-- Time Metric field definition. -/
def charge_metric_field (ctx : VacuumContext) (sign : PerturbationSign) (k : ℝ) (r : ℝ) : ℝ :=
  time_metric ctx (charge_density_field ctx sign k r)

/--
**Theorem C-L3A**: Inverse Square Force Law.
-/
theorem inverse_square_force (ctx : VacuumContext) (sign : PerturbationSign) (k : ℝ) (r : ℝ)
    (hr : r ≠ 0) (hk : 0 < k) :
    deriv (charge_metric_field ctx sign k) r = (sign_value sign) * (ctx.alpha * k) / r ^ 2 := by
  unfold charge_metric_field time_metric charge_density_field

  -- We are deriving: 1 - α * (ρ_vac + s * (k/r) - ρ_vac)
  -- Simplifies to: 1 - α * s * k * r⁻¹

  -- Use Filter.EventuallyEq with a local neighborhood equality to simplify BEFORE deriving
  have h_simp : (fun x => 1 - ctx.alpha * (ctx.rho_vac + sign_value sign * (k / x) - ctx.rho_vac))
              =ᶠ[𝓝 r] (fun x => 1 - (ctx.alpha * sign_value sign * k) * x⁻¹) := by
    filter_upwards with x
    ring

  rw [h_simp.deriv_eq]

  -- Now derive the simplified form: 1 - C * x⁻¹ using HasDerivAt
  have h_const : HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 r := hasDerivAt_const r (1 : ℝ)

  have h_inv : HasDerivAt (fun x : ℝ => x⁻¹) (-1 / r^2) r := by
    simpa using (hasDerivAt_id r).inv hr

  have h_scaled : HasDerivAt (fun x : ℝ => (ctx.alpha * sign_value sign * k) * x⁻¹)
      ((ctx.alpha * sign_value sign * k) * (-1 / r^2)) r := by
    simpa using h_inv.const_mul (ctx.alpha * sign_value sign * k)

  have h_main : HasDerivAt (fun x : ℝ => 1 - (ctx.alpha * sign_value sign * k) * x⁻¹)
      (0 - (ctx.alpha * sign_value sign * k) * (-1 / r^2)) r := by
    simpa using h_const.sub h_scaled

  have := h_main.deriv
  simp only [sub_zero, neg_mul, mul_neg, neg_neg] at this
  convert this using 1
  field_simp [hr]
  ring

/--
**Theorem C-L3B**: Interaction Sign Rule.
-/
theorem interaction_sign_rule (sign1 sign2 : PerturbationSign) :
    let product := (sign_value sign1) * (sign_value sign2)
    (sign1 = sign2 → product = 1) ∧ (sign1 ≠ sign2 → product = -1) := by
  constructor
  · intro h
    subst h
    cases sign1 <;> simp [sign_value]
  · intro h
    cases sign1 <;> cases sign2
    · contradiction
    · simp [sign_value]
    · simp [sign_value]
    · contradiction

/--
**Theorem C-L3C**: Coulomb Force.
-/
theorem coulomb_force (ctx : VacuumContext) (sign1 sign2 : PerturbationSign) (k : ℝ) (r : ℝ)
    (hr : r ≠ 0) (hk : 0 < k) :
    ∃ C : ℝ, deriv (charge_metric_field ctx sign1 k) r * (sign_value sign2) =
    C * ((sign_value sign1) * (sign_value sign2)) / r ^ 2 ∧ C = ctx.alpha * k := by
  use ctx.alpha * k
  refine ⟨?_, rfl⟩
  rw [inverse_square_force ctx sign1 k r hr hk]
  field_simp


end QFD.Charge
