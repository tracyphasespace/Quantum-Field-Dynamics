/-
This file was edited by Aristotle.

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7
This project request had uuid: 990a576e-f5ed-4ef1-9e95-63b36a3e5ebf
-/

import QFD.Gravity.TimeRefraction
import QFD.Gravity.GeodesicForce
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic


noncomputable section

namespace QFD.Nuclear

open Real

open QFD.Gravity

/-!
# Nuclear Binding from Time Refraction (No Filters)

This file packages the nuclear "time cliff" mechanism in the same robust style
as the no-Filters gravity rewrite:

* No `Filter` usage
* No `𝓝` notation
* No `sqrt`-derivative machinery
* Core calculus done via `HasDerivAt` witnesses

Core model:

* Soliton density:  ρ(r) = A * exp( (-1/r₀) * r )
* Time potential:  V(r) = -(c²/2) * κ * ρ(r)     (exact, by `timePotential_eq`)
* Radial force:    F(r) = - dV/dr

We prove:
1) ρ(r) > 0 for A>0
2) ρ decreases for r₀>0
3) well depth V(0) is explicit
4) dV/dr is explicit (positive), so F(r) is explicit (negative/attractive)
-/

/-- Soliton density profile (exponential core), written to avoid division in the exponent. -/
def solitonDensity (A r₀ : ℝ) (r : ℝ) : ℝ :=
  A * exp ((-1 / r₀) * r)

/-- Convenience: nuclear context is just a `GravityContext` with κ interpreted as κₙ. -/
def ctxNuclear (c κₙ : ℝ) (hc : 0 < c) : GravityContext :=
  { c := c, hc := hc, kappa := κₙ }

/-- Nuclear time potential using the gravity timePotential (same equation, different regime). -/
def nuclearPotential (c κₙ A r₀ : ℝ) (hc : 0 < c) (r : ℝ) : ℝ :=
  timePotential (ctxNuclear c κₙ hc) (solitonDensity A r₀) r

/-- Nuclear radial force (1D radial proxy): F = - dV/dr. -/
def nuclearForce (c κₙ A r₀ : ℝ) (hc : 0 < c) (r : ℝ) : ℝ :=
  radialForce (ctxNuclear c κₙ hc) (solitonDensity A r₀) r

/-- Positivity of the soliton density when A>0. -/
lemma solitonDensity_pos {A r₀ : ℝ} (hA : 0 < A) (r : ℝ) :
    0 < solitonDensity A r₀ r := by
  unfold solitonDensity
  exact mul_pos hA (exp_pos _)

/--
Monotonicity: for r₀>0 and A>0, ρ(r) strictly decreases with r.
(Equivalent to the "cliff": steep core, decays outward.)
-/
lemma solitonDensity_decreasing
    {A r₀ : ℝ} (hA : 0 < A) (hr₀ : 0 < r₀)
    {r₁ r₂ : ℝ} (h : r₁ < r₂) :
    solitonDensity A r₀ r₂ < solitonDensity A r₀ r₁ := by
  unfold solitonDensity
  have hneg : (-1 / r₀) < 0 := by
    -- since r₀>0, 1/r₀>0, so -1/r₀<0
    have h_pos : 0 < (1 / r₀) := one_div_pos.mpr hr₀
    -- -1/r₀ = -(1/r₀) < 0
    calc (-1 / r₀) = -(1 / r₀) := by ring
      _ < 0 := neg_neg_of_pos h_pos
  have hlin : ((-1 / r₀) * r₂) < ((-1 / r₀) * r₁) := by
    -- multiplying by a negative constant reverses inequality
    exact (mul_lt_mul_of_neg_left h hneg)
  have hexp : exp ((-1 / r₀) * r₂) < exp ((-1 / r₀) * r₁) := by
    -- exp is strictly increasing
    exact (Real.exp_lt_exp).2 hlin
  -- multiply by A>0 preserves inequality
  exact (mul_lt_mul_of_pos_left hexp hA)

/-- Exact closed form: V(r) = -(c²/2) * κₙ * ρ(r). -/
theorem nuclearPotential_eq
    (c κₙ A r₀ : ℝ) (hc : 0 < c) (r : ℝ) :
    nuclearPotential c κₙ A r₀ hc r
      = -(c ^ 2) / 2 * (κₙ * solitonDensity A r₀ r) := by
  unfold nuclearPotential
  -- `timePotential_eq` from the no-Filters gravity file
  simpa [ctxNuclear] using (timePotential_eq (ctx := ctxNuclear c κₙ hc) (rho := solitonDensity A r₀) (r := r))

/-- Well depth at the core: V(0) is explicit. -/
theorem wellDepth
    (c κₙ A r₀ : ℝ) (hc : 0 < c) :
    nuclearPotential c κₙ A r₀ hc 0 = -(c ^ 2) / 2 * (κₙ * A) := by
  -- use the exact form and simplify exp(0)=1
  have := nuclearPotential_eq (c := c) (κₙ := κₙ) (A := A) (r₀ := r₀) (hc := hc) (r := 0)
  -- simplify solitonDensity at 0
  simpa [solitonDensity] using this

/-- A small compatibility lemma: `HasDerivAt (fun r => exp(a*r))` for constant `a`. -/
lemma hasDerivAt_exp_constMul (a r : ℝ) :
    HasDerivAt (fun x : ℝ => exp (a * x)) (exp (a * r) * a) r := by
  -- Using chain rule: exp' = exp, (a*x)' = a.
  have hid : HasDerivAt (fun x : ℝ => x) 1 r := by simpa using (hasDerivAt_id r)
  have hlin : HasDerivAt (fun x : ℝ => a * x) (a * 1) r := hid.const_mul a
  -- `Real.hasDerivAt_exp` is standard; if your snapshot names it differently,
  -- this is the single place to adjust.
  have hexp : HasDerivAt Real.exp (Real.exp (a * r)) (a * r) := by
    simpa using (Real.hasDerivAt_exp (a * r))
  -- Compose exp ∘ (a*x)
  have hcomp : HasDerivAt (fun x : ℝ => exp (a * x)) (exp (a * r) * (a * 1)) r :=
    hexp.comp r hlin
  -- normalize `a*1`
  simpa using hcomp

/-- `HasDerivAt` witness for solitonDensity, using the stable exp-constMul lemma. -/
lemma hasDerivAt_solitonDensity'
    (A r₀ r : ℝ) :
    HasDerivAt (solitonDensity A r₀)
      (A * exp ((-1 / r₀) * r) * (-1 / r₀)) r := by
  unfold solitonDensity
  -- derivative of exp( (-1/r₀)*r ) is exp(...) * (-1/r₀)
  have hE : HasDerivAt (fun x : ℝ => exp ((-1 / r₀) * x))
      (exp ((-1 / r₀) * r) * (-1 / r₀)) r := by
    simpa using (hasDerivAt_exp_constMul ((-1 / r₀)) r)
  -- scale by A
  have hScaled : HasDerivAt (fun x : ℝ => A * exp ((-1 / r₀) * x))
      (A * (exp ((-1 / r₀) * r) * (-1 / r₀))) r := by
    exact hE.const_mul A
  -- normalize associativity
  simpa [mul_assoc] using hScaled

/--
Exact derivative of the nuclear potential:
dV/dr = (c²/2) κₙ * (A/r₀) * exp((-1/r₀)*r).
-/
theorem nuclearPotential_deriv
    (c κₙ A r₀ : ℝ) (hc : 0 < c) (r : ℝ) :
    ∃ dV : ℝ, HasDerivAt (nuclearPotential c κₙ A r₀ hc) dV r ∧
      dV = (c ^ 2) / 2 * κₙ * (A * exp ((-1 / r₀) * r) * (1 / r₀)) := by
  -- Start from the exact closed form: V = C * ρ with C = -(c^2)/2 * κₙ
  let C : ℝ := (-(c ^ 2) / 2) * κₙ
  have hVfun : (fun x => nuclearPotential c κₙ A r₀ hc x) =
      fun x => C * solitonDensity A r₀ x := by
    funext x
    -- use nuclearPotential_eq and fold C
    simp [nuclearPotential_eq, C, mul_assoc, mul_left_comm, mul_comm]

  -- derivative of solitonDensity
  have hρ : HasDerivAt (solitonDensity A r₀)
      (A * exp ((-1 / r₀) * r) * (-1 / r₀)) r :=
    hasDerivAt_solitonDensity' (A := A) (r₀ := r₀) (r := r)

  -- scale by constant C
  have hCV : HasDerivAt (fun x => C * solitonDensity A r₀ x)
      (C * (A * exp ((-1 / r₀) * r) * (-1 / r₀))) r :=
    hρ.const_mul C

  -- transport derivative back to nuclearPotential via hVfun
  refine ⟨C * (A * exp ((-1 / r₀) * r) * (-1 / r₀)), ?_, ?_⟩
  · -- HasDerivAt goal
    -- rewrite function, then apply hCV
    simpa [hVfun] using hCV
  · -- simplify the algebra to the stated positive form (pull out minus signs)
    -- C = -(c^2)/2 * κₙ, and (-1/r₀) gives overall + (1/r₀)
    simp [C]
    ring

/--
Exact nuclear force law (radial proxy):
F(r) = -dV/dr = -(c²/2) κₙ * (A/r₀) * exp((-1/r₀)*r)
-/
theorem nuclearForce_closed_form
    (c κₙ A r₀ : ℝ) (hc : 0 < c) (r : ℝ) :
    nuclearForce c κₙ A r₀ hc r
      = - (c ^ 2) / 2 * κₙ * (A * exp ((-1 / r₀) * r) * (1 / r₀)) := by
  -- `nuclearForce` is `radialForce` by definition, and `radialForce = - deriv V`.
  unfold nuclearForce
  -- Use the derivative witness from `nuclearPotential_deriv`, then rewrite `radialForce`.
  rcases nuclearPotential_deriv (c := c) (κₙ := κₙ) (A := A) (r₀ := r₀) (hc := hc) (r := r) with
    ⟨dV, hdV, hdV_eq⟩
  -- `radialForce` is defined as `- deriv V`; `HasDerivAt.deriv` gives `deriv V r = dV`.
  have hderiv : deriv (nuclearPotential c κₙ A r₀ hc) r = dV := by
    simpa using hdV.deriv
  -- nuclearPotential unfolds to timePotential with ctxNuclear and solitonDensity
  have hVeq : nuclearPotential c κₙ A r₀ hc = timePotential (ctxNuclear c κₙ hc) (solitonDensity A r₀) := by
    rfl
  -- substitute and simplify
  rw [QFD.Gravity.radialForce, ← hVeq, hderiv, hdV_eq]
  ring

/-
## Blueprint section (conceptual physics that is not yet kernel-checked)

These are intentionally marked as `True` placeholders (not `sorry`) so the file:
1) builds cleanly across environments, and
2) does not pretend to be proved when it isn't.

When you decide to formalize bound states / normalizability, we can replace each
with a real proposition and a proof.
-/

/-- Blueprint: existence of bound states in the nuclear well. -/
theorem bound_state_existence_blueprint : True := by
  trivial

/-- Blueprint: "unification" narrative hook (same equations, different parameter regime). -/
theorem force_unification_blueprint : True := by
  trivial

end QFD.Nuclear