-- QFD/Cosmology/AchromaticDrag.lean
-- Formal proof that QFD vacuum drag is achromatic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

noncomputable section

namespace QFD.Cosmology.AchromaticDrag

open Real

/-!
# QFD Achromatic Drag: Formal Proof

This module formalizes the proof that the QFD vacuum drag mechanism produces
an energy-independent (achromatic) redshift.

## Physical Setup

The QFD interaction Lagrangian L'_{int,drag} yields:
  - Cross-section σ(E) proportional to E  (photon field amplitude ~ √E)
  - Energy transfer per scatter: ΔE = k_B T_CMB (fixed by bath temperature)

## Key Theorem

Given σ(E) = σ₀ · E and ΔE = constant:

    dE/dx = -n · σ(E) · ΔE = -(n · σ₀ · ΔE) · E = -α_drag · E

This is a first-order linear ODE with solution:

    E(x) = E₀ · exp(-α_drag · x)

The resulting redshift z = E₀/E(D) - 1 = exp(α_drag · D) - 1 is independent
of E₀, proving achromaticity.

## References

- QFD Book v8.5, §9 (drag mechanism), Appendix C.4.2 (cross-section)
- Python: projects/astrophysics/achromaticity/achromaticity_derivation.py
-/

/-- The drag coefficient α_drag, defined as a positive constant
    independent of photon energy.  In QFD: α_drag = K_J / c. -/
structure DragParams where
  alpha_drag : ℝ
  h_pos : 0 < alpha_drag

/-- The cross-section is proportional to photon energy.
    σ(E) = σ₀ · E where σ₀ > 0 is a constant. -/
structure CrossSectionLinear where
  sigma_0 : ℝ
  h_sigma_pos : 0 < sigma_0

/-- The bath sets a fixed energy transfer per scatter.
    ΔE = k_B T_CMB, independent of the photon energy. -/
structure BathTransfer where
  delta_E : ℝ
  h_delta_pos : 0 < delta_E

/-- The number density of scatterers in the vacuum. -/
structure VacuumDensity where
  n_bath : ℝ
  h_n_pos : 0 < n_bath

/-- Energy loss rate from linear cross-section + fixed bath transfer.
    dE/dx = -n · σ₀ · ΔE · E = -α_drag · E -/
def energy_loss_rate (n : VacuumDensity) (σ : CrossSectionLinear)
    (bath : BathTransfer) (E : ℝ) : ℝ :=
  -(n.n_bath * σ.sigma_0 * bath.delta_E) * E

/-- The drag coefficient constructed from microscopic parameters. -/
def drag_coeff (n : VacuumDensity) (σ : CrossSectionLinear)
    (bath : BathTransfer) : ℝ :=
  n.n_bath * σ.sigma_0 * bath.delta_E

/-- The drag coefficient is positive (since all factors are positive). -/
theorem drag_coeff_pos (n : VacuumDensity) (σ : CrossSectionLinear)
    (bath : BathTransfer) : 0 < drag_coeff n σ bath := by
  unfold drag_coeff
  exact mul_pos (mul_pos n.h_n_pos σ.h_sigma_pos) bath.h_delta_pos

/-- The energy loss rate equals -α_drag · E. -/
theorem loss_rate_is_linear (n : VacuumDensity) (σ : CrossSectionLinear)
    (bath : BathTransfer) (E : ℝ) :
    energy_loss_rate n σ bath E = -(drag_coeff n σ bath) * E := by
  unfold energy_loss_rate drag_coeff
  ring

/-- Solution of dE/dx = -α · E is E(x) = E₀ · exp(-α · x). -/
def energy_solution (E₀ : ℝ) (α : ℝ) (x : ℝ) : ℝ :=
  E₀ * exp (-α * x)

/-- The solution satisfies the initial condition E(0) = E₀. -/
theorem energy_solution_initial (E₀ α : ℝ) :
    energy_solution E₀ α 0 = E₀ := by
  unfold energy_solution
  simp [mul_zero, exp_zero, mul_one]

/-- The energy solution is always positive when E₀ > 0. -/
theorem energy_solution_pos (E₀ α x : ℝ) (h : 0 < E₀) :
    0 < energy_solution E₀ α x := by
  unfold energy_solution
  exact mul_pos h (exp_pos _)

/-- The redshift from energy loss: z(D) = E₀/E(D) - 1. -/
def redshift (E₀ α D : ℝ) : ℝ :=
  E₀ / (energy_solution E₀ α D) - 1

/-- The redshift simplifies to exp(α·D) - 1. -/
theorem redshift_is_exp (E₀ α D : ℝ) (hE : E₀ ≠ 0) :
    redshift E₀ α D = exp (α * D) - 1 := by
  unfold redshift energy_solution
  -- E₀ / (E₀ * exp(-α*D)) - 1 = exp(α*D) - 1
  have hexp : exp (-α * D) ≠ 0 := exp_ne_zero _
  have hprod : E₀ * exp (-α * D) ≠ 0 := mul_ne_zero hE hexp
  congr 1
  -- Goal: E₀ / (E₀ * exp (-α * D)) = exp (α * D)
  rw [div_eq_iff hprod]
  -- Goal: E₀ = exp (α * D) * (E₀ * exp (-α * D))
  have h1 : exp (α * D) * exp (-α * D) = 1 := by
    rw [← exp_add]; simp [add_neg_cancel]
  symm
  calc exp (α * D) * (E₀ * exp (-α * D))
      = E₀ * (exp (α * D) * exp (-α * D)) := by ring
    _ = E₀ * 1 := by rw [h1]
    _ = E₀ := by ring

/-!
## The Achromaticity Theorem

The central result: redshift z depends only on (α_drag, D), NOT on E₀.
Two photons with different energies E₁ and E₂ traversing the same distance D
experience the SAME redshift z.
-/

/-- **Achromaticity Theorem**: The redshift is independent of initial photon energy.
    For any two photon energies E₁ ≠ 0, E₂ ≠ 0, the redshift is identical. -/
theorem achromaticity (E₁ E₂ α D : ℝ) (hE₁ : E₁ ≠ 0) (hE₂ : E₂ ≠ 0) :
    redshift E₁ α D = redshift E₂ α D := by
  rw [redshift_is_exp E₁ α D hE₁, redshift_is_exp E₂ α D hE₂]

/-- **Spectral Line Preservation**: The ratio of two photon energies is preserved
    through QFD drag.  E₁(D)/E₂(D) = E₁(0)/E₂(0). -/
theorem spectral_ratio_preserved (E₁ E₂ α D : ℝ) (hE₂ : E₂ ≠ 0) :
    energy_solution E₁ α D / energy_solution E₂ α D = E₁ / E₂ := by
  unfold energy_solution
  have hexp : exp (-α * D) ≠ 0 := exp_ne_zero _
  field_simp

/-!
## Contrast: Why Fractional Loss Is Chromatic

If instead ΔE ∝ E (fractional loss per scatter), then:
  dE/dx = -β · E²
which gives E(x) = E₀/(1 + β·E₀·x), making z dependent on E₀.
This is CHROMATIC and inconsistent with observations.
-/

/-- Fractional-loss (chromatic) solution: E(x) = E₀ / (1 + β·E₀·x). -/
def chromatic_solution (E₀ β x : ℝ) : ℝ :=
  E₀ / (1 + β * E₀ * x)

/-- Chromatic redshift depends on E₀ (NOT achromatic). -/
def chromatic_redshift (E₀ β D : ℝ) : ℝ :=
  E₀ / (chromatic_solution E₀ β D) - 1

/-- The chromatic redshift simplifies to β·E₀·D (depends on E₀!).
    This proves the chromatic model is E₀-dependent, unlike QFD. -/
theorem chromatic_redshift_depends_on_energy (E₀ β D : ℝ)
    (hE : E₀ ≠ 0) (hD : 1 + β * E₀ * D ≠ 0) :
    chromatic_redshift E₀ β D = β * E₀ * D := by
  unfold chromatic_redshift chromatic_solution
  -- E₀ / (E₀ / (1 + β*E₀*D)) - 1 = β*E₀*D
  have hdiv : E₀ / (1 + β * E₀ * D) ≠ 0 := div_ne_zero hE hD
  field_simp
  ring

/-- Two different energies give DIFFERENT chromatic redshifts (hence chromatic). -/
theorem chromatic_is_not_achromatic (β D : ℝ) (E₁ E₂ : ℝ)
    (hβ : β ≠ 0) (hD : D ≠ 0) (hE_neq : E₁ ≠ E₂) :
    β * E₁ * D ≠ β * E₂ * D := by
  intro h
  have h1 : β * E₁ = β * E₂ := by
    exact mul_right_cancel₀ hD h
  have h2 : E₁ = E₂ := mul_left_cancel₀ hβ h1
  exact hE_neq h2

/-!
## Connection to K_J

The drag coefficient α_drag = K_J / c where K_J is the QFD vacuum
refraction parameter.  For small z:

    z ≈ α_drag · D = (K_J / c) · D

This matches the Hubble law z ≈ (H₀/c) · D, with K_J replacing H₀.
-/

/-- For positive x, exp(x) > 1 + x (strict convexity of exp).
    Used in the small-z Hubble law approximation. -/
theorem exp_gt_one_add (x : ℝ) (hx : 0 < x) :
    1 + x < exp x := by
  linarith [add_one_lt_exp (ne_of_gt hx)]

end QFD.Cosmology.AchromaticDrag
