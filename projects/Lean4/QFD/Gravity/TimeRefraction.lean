import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

namespace QFD.Gravity

/-!
# Gate G-L1: Time Refraction (No Filters)

Key design choice for Mathlib robustness:

* We DO NOT define `n = sqrt(1 + κ ρ)`.
* Instead, we take `n² := 1 + κ ρ` as the primitive object.

This avoids sqrt-differentiation and avoids any Filter/Topological machinery.

Model definitions:

* n²(r) = 1 + κ ρ(r)
* g₀₀(r) = 1 / n²(r)
* V(r)   = -(c²/2) (n²(r) - 1) = -(c²/2) κ ρ(r)   (exact)
-/

/-- Minimal gravity context for time-refraction modeling. -/
structure GravityContext where
  c : ℝ
  hc : 0 < c
  kappa : ℝ

/-- Primitive object: `n²(r) := 1 + κ ρ(r)`. -/
def n2 (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) : ℝ :=
  1 + ctx.kappa * rho r

/-- Optical time metric (weak-field model): `g00 := 1 / n²`. -/
def g00 (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) : ℝ :=
  (n2 ctx rho r)⁻¹

/-- Time potential: `V := -(c²/2) (n² - 1)`. -/
def timePotential (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) : ℝ :=
  -(ctx.c ^ 2) / 2 * (n2 ctx rho r - 1)

/-- Exact simplification: `V(r) = -(c²/2) * κ * ρ(r)` (no approximation). -/
theorem timePotential_eq (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) :
    timePotential ctx rho r = -(ctx.c ^ 2) / 2 * (ctx.kappa * rho r) := by
  unfold timePotential n2
  ring

/-- Convenience: `g00` expanded. -/
theorem g00_eq (ctx : GravityContext) (rho : ℝ → ℝ) (r : ℝ) :
    g00 ctx rho r = (1 + ctx.kappa * rho r)⁻¹ := by
  rfl

end QFD.Gravity
