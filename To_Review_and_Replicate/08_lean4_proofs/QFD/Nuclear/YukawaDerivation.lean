import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.ZPow
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

noncomputable section

/-!
# The Yukawa Derivation (Vacuum Pressure)

**Status**: ✅ COMPLETE (0 Sorries)
**Date**: 2025-12-31

## Physical Interpretation
Standard Physics: The Strong Force is a fundamental interaction mediated by
exchange particles (pions/gluons).
QFD: The "Strong Force" is the restoring force (Pressure Gradient) of the
vacuum density near a soliton core.

## Main Results
- `soliton_gradient_is_yukawa`: Derivative of soliton density yields Yukawa functional form
- `magnitude_match`: Geometric and textbook forces match (opposite sign conventions)
-/

namespace QFD.Nuclear.YukawaDerivation

open Real

/--
The Soliton Density Profile $\rho(r)$.
Describes how the time-flow density concentrates near a nucleon core.
A: Amplitude (Soliton Height)
lam: Vacuum Stiffness (renamed from lambda to avoid keyword collision)
-/
def rho_soliton (A lam : ℝ) (r : ℝ) : ℝ :=
  A * (exp (-lam * r)) / r

/--
The Vacuum Force $F(r)$.
Force is proportional to negative gradient of density (restoring force).
-/
def vacuum_force (k A lam : ℝ) (r : ℝ) : ℝ :=
  -k * deriv (rho_soliton A lam) r

/--
The "Textbook" Yukawa Force Law.
Derived from $V = -g^2 \frac{e^{-mr}}{r}$.
-/
def textbook_yukawa_force (g_sq m : ℝ) (r : ℝ) : ℝ :=
  -g_sq * (exp (-m * r)) * (1 / r^2 + m / r)

/--
**Theorem: The Soliton Gradient is the Yukawa Force**
Prove that taking the derivative of the geometric Soliton Density
yields the exact functional form of the Strong Force.
-/
theorem soliton_gradient_is_yukawa (A lam : ℝ) (r : ℝ) (h_r : r ≠ 0) :
  deriv (rho_soliton A lam) r = -A * (exp (-lam * r)) * (1 / r^2 + lam / r) := by
  unfold rho_soliton

  -- Step 1: Derivative of numerator A * exp(-lam * x)
  have h_num : HasDerivAt (fun x => A * exp (-lam * x))
    (-A * lam * exp (-lam * r)) r := by
    -- First get derivative of exp(-lam * x)
    have h_exp : HasDerivAt (fun x => exp (-lam * x))
      (-lam * exp (-lam * r)) r := by
      have h_inner : HasDerivAt (fun x => -lam * x) (-lam) r := by
        convert (hasDerivAt_id r).const_mul (-lam) using 1
        simp
      have h_comp := HasDerivAt.comp r (Real.hasDerivAt_exp (-lam * r)) h_inner
      convert h_comp using 1
      ring
    -- Apply constant multiple rule
    have h_const : HasDerivAt (fun x => A * exp (-lam * x))
      (A * (-lam * exp (-lam * r))) r :=
      h_exp.const_mul A
    ring_nf at h_const ⊢
    exact h_const

  -- Step 2: Derivative of denominator x
  have h_denom : HasDerivAt (fun x => x) 1 r := hasDerivAt_id r

  -- Step 3: Apply quotient rule
  have h_quot : HasDerivAt (fun x => (A * exp (-lam * x)) / x)
    ((-A * lam * exp (-lam * r) * r - A * exp (-lam * r) * 1) / r^2) r :=
    h_num.div h_denom h_r

  -- Step 4: Simplify to target form
  have h_simp : (-A * lam * exp (-lam * r) * r - A * exp (-lam * r) * 1) / r^2 =
    -A * exp (-lam * r) * (1 / r^2 + lam / r) := by
    field_simp [h_r]
    ring

  rw [h_quot.deriv]
  exact h_simp

/--
**Corollary: Magnitude Identification**
By comparing the geometric gradient with the Yukawa form, we identify physical constants.
The force magnitudes match (signs differ due to attractive vs restoring force convention).
-/
theorem magnitude_match (g_sq m k A lam : ℝ) (r : ℝ) (h_r : r ≠ 0)
  (h_mass : lam = m)
  (h_coupling : k * A = g_sq) :
  vacuum_force k A lam r = -textbook_yukawa_force g_sq m r := by
  unfold vacuum_force textbook_yukawa_force
  rw [soliton_gradient_is_yukawa A lam r h_r, ←h_mass, ←h_coupling]
  simp only [neg_mul, neg_neg]
  ring

end QFD.Nuclear.YukawaDerivation
