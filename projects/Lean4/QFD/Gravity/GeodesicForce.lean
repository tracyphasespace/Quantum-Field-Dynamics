import QFD.Gravity.TimeRefraction
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

namespace QFD.Gravity

/-!
# Gate G-L2: Radial Force from Time Potential (No Filters)

We avoid the full variational/geodesic derivation here (which is heavier),
and formalize the stable, spherically-symmetric proxy:

* Define radial force magnitude: `F(r) := - dV/dr`
* Since `V(r) = -(c²/2) κ ρ(r)` exactly, we get:

  F(r) = (c²/2) κ ρ'(r)

This is the kernel-checked "force = time-gradient" statement in 1D radial form.
-/

/-- Radial force magnitude (1D proxy for spherical symmetry): `F := - dV/dr`. -/
def radialForce (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) : ℝ :=
  - deriv (timePotential ctx rho) r

/-- General force law, assuming `ρ` has a derivative at `r`. -/
theorem radialForce_eq
    (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ)
    (rho' : ℝ) (h : HasDerivAt rho rho' r) :
    radialForce ctx rho r = (ctx.c ^ 2) / 2 * ctx.kappa * rho' := by
  unfold radialForce
  -- Rewrite V as a constant multiple of rho.
  let A : ℝ := (-(ctx.c ^ 2) / 2) * ctx.kappa
  have hV : timePotential ctx rho = fun x => A * rho x := by
    funext x
    -- timePotential = -(c^2)/2 * (kappa * rho x)
    -- and A is defined as (-(c^2)/2)*kappa
    simp [A, timePotential_eq, mul_assoc, mul_left_comm, mul_comm]
  rw [hV]
  -- Differentiate A * rho using HasDerivAt scaling.
  have h_scaled : HasDerivAt (fun x => A * rho x) (A * rho') r :=
    h.const_mul A
  have h_deriv : deriv (fun x => A * rho x) r = A * rho' := by
    simpa using h_scaled.deriv
  rw [h_deriv]
  -- Now simplify: - (A * rho') = (c^2)/2 * kappa * rho'
  simp [A]
  ring

/-- Point-mass density ansatz: `ρ(r) = M / r`. -/
def rhoPointMass (M : ℝ) (r : ℝ) : ℝ := M / r

/-- Derivative of `M/r` at `r ≠ 0` using HasDerivAt only. -/
lemma hasDerivAt_rhoPointMass (M : ℝ) {r : ℝ} (hr : r ≠ 0) :
    HasDerivAt (rhoPointMass M) (-M / r ^ 2) r := by
  -- Start with derivative of inv: x ↦ x⁻¹
  have h_inv : HasDerivAt (fun x : ℝ => x⁻¹) (-1 / r ^ 2) r := by
    simpa using (hasDerivAt_id r).inv hr
  -- Scale by M: x ↦ M * x⁻¹
  have h_mul : HasDerivAt (fun x : ℝ => M * x⁻¹) (M * (-1 / r ^ 2)) r :=
    h_inv.const_mul M
  -- Rewrite M / x as M * x⁻¹
  -- And normalize M * (-1 / r^2) to -M / r^2
  simpa [rhoPointMass, div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using h_mul

/-- Inverse-square force for the point-mass ansatz, with `r ≠ 0`. -/
theorem inverse_square_force
    (ctx : GravityContext) (M : ℝ) (r : ℝ) (hr : r ≠ 0) :
    radialForce ctx (rhoPointMass M) r =
      - (ctx.c ^ 2) / 2 * ctx.kappa * M / r ^ 2 := by
  -- Apply the general law with rho' = -M/r^2
  have hρ : HasDerivAt (rhoPointMass M) (-M / r ^ 2) r :=
    hasDerivAt_rhoPointMass (M := M) hr
  -- radialForce = (c^2)/2*kappa*rho' = (c^2)/2*kappa*(-M/r^2)
  rw [radialForce_eq (ctx := ctx) (rho := rhoPointMass M) (r := r) (rho' := (-M / r ^ 2)) hρ]
  -- Simplify: (c^2)/2 * kappa * (-M/r^2) = -(c^2)/2 * kappa * M/r^2
  ring

end QFD.Gravity
