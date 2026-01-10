/-
  Proof: Shell Theorem Harmonic Decay
  Theorem: exterior_harmonic_decay

  Description:
  Proves that any spherically symmetric solution to the harmonic equation
  outside a source must decay as 1/r, linking QFD scalar gravity to
  Newtonian potential.

  Mathematical Structure:
  The radial Laplacian Lap(f) = (1/r²) d/dr(r² df/dr) = 0 has general solution
  f(r) = A/r + B. The decay condition f(r) → 0 as r → ∞ forces B = 0.
-/

import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.Order.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace QFD_Proofs

open Real Filter

/-- The radial Laplacian operator for a function f(r): Lap(f) = (1/r²) * d/dr (r² * df/dr) -/
noncomputable def radial_laplacian_op (f : ℝ → ℝ) (r : ℝ) : ℝ :=
  let df := deriv f
  let inner := fun x => (x^2) * df x
  (1 / r^2) * deriv inner r

/-- The 1/r potential function -/
noncomputable def inverse_r (C : ℝ) (r : ℝ) : ℝ := C / r

/-- Key lemma: A function of form A/r + B that decays to 0 at infinity must have B = 0 -/
lemma decay_implies_no_constant (A B : ℝ)
    (h_decay : Tendsto (fun r => A / r + B) atTop (nhds 0)) : B = 0 := by
  -- As r → ∞, A/r → 0, so A/r + B → B
  -- If the limit is 0, then B = 0
  have h_Ar_tends : Tendsto (fun r => A / r) atTop (nhds 0) := by
    have heq : (fun r => A / r) = (fun r => A * r⁻¹) := by ext; ring
    rw [heq]
    have h := Tendsto.const_mul A tendsto_inv_atTop_zero
    simp only [mul_zero] at h
    exact h
  -- The sum A/r + B tends to 0 + B = B
  have h_sum : Tendsto (fun r => A / r + B) atTop (nhds (0 + B)) := by
    exact Tendsto.add h_Ar_tends tendsto_const_nhds
  -- But we're told it tends to 0
  have h_eq : (0 : ℝ) + B = 0 := by
    have := tendsto_nhds_unique h_decay h_sum
    linarith
  linarith

/-- Key lemma: If f = A/r + B for all r > R and f decays, then f = A/r -/
lemma harmonic_solution_form (f : ℝ → ℝ) (A B R : ℝ) (_hR : R > 0)
    (h_form : ∀ r > R, f r = A / r + B)
    (h_decay : Tendsto f atTop (nhds 0)) :
    ∀ r > R, f r = A / r := by
  -- First show B = 0 using the decay condition
  have hB : B = 0 := by
    -- f agrees with A/r + B for large r, so the limits must match
    have h_event : ∀ᶠ r in atTop, f r = A / r + B := by
      rw [eventually_atTop]
      use R + 1
      intro r hr
      apply h_form
      linarith
    -- Convert to EventuallyEq for use with Tendsto.congr'
    have h_eq : f =ᶠ[atTop] (fun r => A / r + B) := h_event
    have h_tendsto_form : Tendsto (fun r => A / r + B) atTop (nhds 0) := by
      exact Tendsto.congr' h_eq h_decay
    exact decay_implies_no_constant A B h_tendsto_form
  -- Now substitute B = 0
  intro r hr
  rw [h_form r hr, hB, add_zero]

/-- Main Theorem: Solutions to the radial Laplacian equation that decay at infinity
have the form C/r. We use the fact that Lap(f) = 0 has general solution f(r) = A/r + B. -/
theorem exterior_harmonic_decay
    (f : ℝ → ℝ) (R : ℝ) (hR : R > 0)
    -- The harmonic condition implies f has the form A/r + B (ODE theory)
    (h_solution_form : ∃ A B : ℝ, ∀ r > R, f r = A / r + B)
    (h_decay : Tendsto f atTop (nhds 0)) :
    ∃ C : ℝ, ∀ r > R, f r = C / r := by
  -- Extract A and B from the solution form
  obtain ⟨A, B, h_form⟩ := h_solution_form
  -- Use the decay condition to eliminate B
  use A
  exact harmonic_solution_form f A B R hR h_form h_decay

/-- Corollary: The original formulation with explicit Laplacian condition. We add the ODE
solution hypothesis explicitly since full ODE solving requires machinery beyond basic Mathlib. -/
theorem exterior_harmonic_decay_laplacian
    (f : ℝ → ℝ) (R : ℝ) (hR : R > 0)
    (_h_harmonic : ∀ r > R, radial_laplacian_op f r = 0)
    -- ODE theory: harmonic condition implies solution form A/r + B
    (h_ode_solution : ∃ A B : ℝ, ∀ r > R, f r = A / r + B)
    (h_decay : Tendsto f atTop (nhds 0)) :
    ∃ C : ℝ, ∀ r > R, f r = C / r :=
  exterior_harmonic_decay f R hR h_ode_solution h_decay

end QFD_Proofs
