import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Linarith

noncomputable section

namespace QFD.Charge

/-!
# Gate C-L1: Vacuum Floor and Refractive Polarity

This file defines the vacuum structure and the mechanism of Time Refraction.
Critically, it introduces **Polarity**:
* **Source (+)**: High density (Nucleus). Slows time (n > 1).
* **Sink (-)**: Low density (Electron). Speeds time (n < 1).
-/

/-- The Sign of the Density Perturbation. -/
inductive PerturbationSign
| Source -- Positive density perturbation (Pressure / Nucleus)
| Sink   -- Negative density perturbation (Void / Electron)
deriving DecidableEq, Repr

/-- Helper: Convert sign to real scalar (+1 or -1). -/
def sign_value (s : PerturbationSign) : ℝ :=
  match s with
  | PerturbationSign.Source => 1
  | PerturbationSign.Sink => -1

/-- The Context defining the vacuum properties. -/
structure VacuumContext where
  /-- The vacuum density floor. Must be strictly positive. -/
  rho_vac : ℝ
  h_rho_vac_pos : 0 < rho_vac

  /-- The refractive coupling constant alpha. Must be positive. -/
  alpha : ℝ
  h_alpha_pos : 0 < alpha

/--
The signed density perturbation δρ.
-/
def delta_rho (sign : PerturbationSign) (magnitude : ℝ) : ℝ :=
  (sign_value sign) * magnitude

/--
The Total Density ρ_total = ρ_vac + δρ.
Must satisfy the Cavitation Limit (ρ_total ≥ 0).
-/
def total_density (ctx : VacuumContext) (sign : PerturbationSign) (magnitude : ℝ) : ℝ :=
  ctx.rho_vac + delta_rho sign magnitude

/--
The Time Metric Coupling Function `g00(ρ)`.
Standard QFD: Higher density = Slower time (Dilation).
g00 ≈ 1 - α(ρ - ρ_vac)
-/
def time_metric (ctx : VacuumContext) (rho_tot : ℝ) : ℝ :=
  1 - ctx.alpha * (rho_tot - ctx.rho_vac)

/--
**Theorem C-L1**: Polarity Time Effect.
Prove that Sources slow time (g00 < 1) and Sinks speed up time (g00 > 1),
relative to the vacuum (g00 = 1).
-/
theorem polarity_time_effect
  (ctx : VacuumContext) (mag : ℝ) (h_mag_pos : 0 < mag) :
  let rho_source := total_density ctx PerturbationSign.Source mag
  let rho_sink := total_density ctx PerturbationSign.Sink mag
  (time_metric ctx rho_source < 1) ∧ (time_metric ctx rho_sink > 1) := by
  unfold total_density delta_rho time_metric sign_value
  constructor
  · -- Source: 1 - α*mag < 1
    have h := mul_pos ctx.h_alpha_pos h_mag_pos
    linarith
  · -- Sink: 1 + α*mag > 1
    have h := mul_pos ctx.h_alpha_pos h_mag_pos
    linarith

end QFD.Charge
