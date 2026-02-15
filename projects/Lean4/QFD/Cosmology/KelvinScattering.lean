-- QFD/Cosmology/KelvinScattering.lean
-- Kelvin wave scattering: ω ∝ k² dispersion → σ_nf ∝ √E cross-section
-- Bridges AchromaticDrag (σ_fwd ∝ E) and chromatic dimming (σ_nf ∝ √E)
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Mul        -- deriv_div_const
import Mathlib.Analysis.SpecialFunctions.Sqrt      -- deriv_sqrt, sq_sqrt, mul_self_sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real  -- rpow_neg, rpow_add, sqrt_eq_rpow
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

noncomputable section

namespace QFD.Cosmology.KelvinScattering

open Real

/-!
# Kelvin Wave Scattering in the QFD Vacuum

This module formalizes the derivation from quadratic Kelvin wave dispersion
(ω ∝ k²) to the non-forward scattering cross-section σ_nf ∝ √E.

## Physical Setup

The QFD vacuum is a superfluid with Kelvin wave excitations obeying:
  E = κ k²  (quadratic dispersion, 2 thermodynamic DOF)

Inverting: k(E) = √(E/κ), so the 1D density of states is:
  ρ(E) = dk/dE = 1/(2√κ) · E^{-1/2}

Fermi's Golden Rule with derivative coupling |M|² ∝ E gives:
  σ_nf = |M|² · ρ(E) ∝ E · E^{-1/2} = E^{1/2}

## Key Results

1. `density_of_states_1D`: dk/dE = (1/(2√κ)) · E^{-1/2}
2. `cross_section_scales_sqrt_E`: σ_nf = C · √E  with C > 0

## References

- QFD Book v8.5, Appendix C.4.3 (Dual Vertices)
- Python: projects/astrophysics/golden-loop-sne/golden_loop_sne.py (v2)
-/

/-- Vacuum parameters: stiffness κ and coupling strength.
    Positivity fields eliminate boilerplate in downstream proofs. -/
structure KelvinParams where
  k_kappa : ℝ        -- Dispersion constant κ
  m_coupling : ℝ     -- Matrix element coupling strength
  h_kappa_pos : 0 < k_kappa
  h_coupling_pos : 0 < m_coupling

/-- Quadratic dispersion: E = κ k². -/
def energy_from_momentum (k : ℝ) (p : KelvinParams) : ℝ :=
  p.k_kappa * k ^ 2

/-- Inverse relation: k = √(E/κ). -/
def momentum_from_energy (E : ℝ) (p : KelvinParams) : ℝ :=
  sqrt (E / p.k_kappa)

/-- Matrix element squared |M|² ∝ E (derivative coupling). -/
def matrix_element_squared (E : ℝ) (p : KelvinParams) : ℝ :=
  p.m_coupling * E

/-- Non-forward cross-section: σ_nf = |M|² · ρ(E) where ρ = dk/dE. -/
def cross_section_nf (E : ℝ) (p : KelvinParams) : ℝ :=
  matrix_element_squared E p * deriv (fun x => momentum_from_energy x p) E

/--
**Density of States in 1D.**

From k = √(E/κ), differentiation gives:
  dk/dE = (1/(2√κ)) · E^{-1/2}

Proof strategy: use `deriv_sqrt` to compute d/dE[√(E/κ)],
factor √(E/κ) = √E/√κ via `sqrt_div`, then close with
`mul_self_sqrt` to resolve (√κ)² = κ.
-/
theorem density_of_states_1D (p : KelvinParams) (E : ℝ) (hE : 0 < E) :
    deriv (fun x => momentum_from_energy x p) E =
      (1 / (2 * sqrt p.k_kappa)) * E ^ (-(1 / 2 : ℝ)) := by
  unfold momentum_from_energy
  have hfE : E / p.k_kappa ≠ 0 := ne_of_gt (div_pos hE p.h_kappa_pos)
  -- Step 1: Apply deriv_sqrt: d/dx[√(f(x))] = f'(x) / (2√(f(x)))
  rw [deriv_sqrt (differentiableAt_fun_id.div_const _) hfE]
  -- Step 2: Simplify d/dE[E/κ] = 1/κ
  have h_lin : deriv (fun x : ℝ => x / p.k_kappa) E = 1 / p.k_kappa := by
    rw [deriv_div_const]; simp
  rw [h_lin]
  -- Step 3: Factor √(E/κ) = √E / √κ
  rw [sqrt_div hE.le]
  -- Goal: 1/κ / (2 * (√E / √κ)) = (1/(2√κ)) * E^(-1/2)
  -- Step 4: Convert E^(-1/2) to (√E)⁻¹ to stay in sqrt-land
  rw [rpow_neg hE.le, ← sqrt_eq_rpow, inv_eq_one_div]
  -- Goal: 1/κ / (2 * (√E / √κ)) = (1/(2√κ)) * (1/√E)
  -- Step 5: Algebraic closure
  have h_sqrtκ : sqrt p.k_kappa ≠ 0 := (sqrt_pos.mpr p.h_kappa_pos).ne'
  have h_sqrtE : sqrt E ≠ 0 := (sqrt_pos.mpr hE).ne'
  have hκ : p.k_kappa ≠ 0 := p.h_kappa_pos.ne'
  field_simp
  -- field_simp leaves (√κ)·(√κ) vs κ; resolve with mul_self_sqrt
  nlinarith [mul_self_sqrt p.h_kappa_pos.le]

/--
**Cross-section Scaling Law.**

Combining |M|² ∝ E with ρ(E) ∝ E^{-1/2}:
  σ_nf = |M|² · ρ(E) = C · E^{1/2} = C · √E

This is the "smoking gun" of QFD cosmology: chromatic dimming
scales as √E, explaining supernova light curve erosion.
-/
theorem cross_section_scales_sqrt_E (p : KelvinParams) (E : ℝ) (hE : 0 < E) :
    ∃ C > 0, cross_section_nf E p = C * sqrt E := by
  use p.m_coupling / (2 * sqrt p.k_kappa)
  constructor
  · exact div_pos p.h_coupling_pos (mul_pos two_pos (sqrt_pos.mpr p.h_kappa_pos))
  · unfold cross_section_nf matrix_element_squared
    rw [density_of_states_1D p E hE]
    -- Goal: m·E · ((1/(2√κ)) · E^(-1/2)) = (m/(2√κ)) · √E
    -- Combine E · E^(-1/2) = √E via rpow arithmetic
    have hcomb : E * E ^ (-(1 / 2 : ℝ)) = sqrt E := by
      nth_rw 1 [← rpow_one E]
      rw [← rpow_add hE, sqrt_eq_rpow]
      norm_num
    -- Separate terms, apply combination, close with ring
    calc p.m_coupling * E * (1 / (2 * sqrt p.k_kappa) * E ^ (-(1 / 2 : ℝ)))
        = p.m_coupling * (1 / (2 * sqrt p.k_kappa)) * (E * E ^ (-(1 / 2 : ℝ))) := by ring
      _ = p.m_coupling * (1 / (2 * sqrt p.k_kappa)) * sqrt E := by rw [hcomb]
      _ = p.m_coupling / (2 * sqrt p.k_kappa) * sqrt E := by ring

end QFD.Cosmology.KelvinScattering
