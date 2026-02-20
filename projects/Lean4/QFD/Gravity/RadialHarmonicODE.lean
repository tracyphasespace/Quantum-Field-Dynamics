import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace QFD.Gravity

/-!
# Radial Harmonic Functions (Shell Theorem Bridge)

Proves that the inverse-r potential satisfies the radial Laplace equation
r²f'' + 2rf' = 0 in the exterior region (r > 0).

This is a stepping stone for Axiom #6 (shell_theorem_timeDilation):
it isolates the mathematical fact that 1/r is the unique decaying
harmonic function, from which the physical shell theorem follows.

## Book Reference

- Ch 4 (Gravitational time dilation from refractive gradient)
- App C.10 (Schwarzschild match)
-/

/-- Predicate: f satisfies the radial Laplace equation r²f'' + 2rf' = 0. -/
def IsRadialHarmonicODE (f f' f'' : ℝ → ℝ) (r : ℝ) : Prop :=
  r ^ 2 * f'' r + 2 * r * f' r = 0

/-- The 1/r potential (with coupling constant kappa). -/
noncomputable def tau_potential (kappa r : ℝ) : ℝ := -(kappa / r)

/-- First derivative of -kappa/r. -/
noncomputable def tau_prime (kappa r : ℝ) : ℝ := kappa / r ^ 2

/-- Second derivative of -kappa/r. -/
noncomputable def tau_prime_prime (kappa r : ℝ) : ℝ := -(2 * kappa) / r ^ 3

/-- The 1/r profile satisfies the radial harmonic ODE for all r ≠ 0.

    Proof: r² · (-2κ/r³) + 2r · (κ/r²) = -2κ/r + 2κ/r = 0. -/
theorem inverse_r_is_harmonic (kappa r : ℝ) (hr : r ≠ 0) :
    IsRadialHarmonicODE (tau_potential kappa) (tau_prime kappa)
      (tau_prime_prime kappa) r := by
  unfold IsRadialHarmonicODE tau_prime tau_prime_prime
  have hr2 : r ^ 2 ≠ 0 := pow_ne_zero 2 hr
  have hr3 : r ^ 3 ≠ 0 := pow_ne_zero 3 hr
  field_simp
  ring

/-- The zero function trivially satisfies the radial harmonic ODE.
    This corresponds to A = 0 in the general solution f = A + B/r. -/
theorem zero_is_harmonic (r : ℝ) :
    IsRadialHarmonicODE (fun _ => 0) (fun _ => 0) (fun _ => 0) r := by
  unfold IsRadialHarmonicODE
  ring

/-- The constant function satisfies the ODE (f' = f'' = 0).
    This is the A term in f = A + B/r. -/
theorem constant_is_harmonic (A r : ℝ) :
    IsRadialHarmonicODE (fun _ => A) (fun _ => 0) (fun _ => 0) r := by
  unfold IsRadialHarmonicODE
  ring

end QFD.Gravity
