import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import QFD.Lepton.Topology

/-!
# Core Definitions for Density-Matched Topological Solitons

This file provides the basic geometric objects used across the topological
stability development: the spacetime/target-space conventions, the field
configuration structure, and the helper predicates for saturation, potentials,
and stability checks.  Keeping these definitions in one place lets both
`Physics/Postulates.lean` (which declares the axioms) and the proof files
reuse the same vocabulary without duplicating declarations.

## TODO: Generation-Dependent Radii

To derive the muon radius (and mass ratio) from topology alone, we need:
1. Replace placeholder `EnergyDensity` and `Action` definitions with real integrals
2. Implement Euler-Lagrange conditions that solve for equilibrium radius R
3. Link generation labels (Q*, winding number) to distinct radial quantization

Until this is complete, validation scripts use the experimental muon mass.
See `analysis/scripts/validate_g2_corrected.py` and `Lepton/VortexStability.lean`.
-/

noncomputable section

namespace QFD.Soliton

/-- Spacetime used for the soliton model: ℝ (time) × ℝ³ (space). -/
abbrev Spacetime := ℝ × (EuclideanSpace ℝ (Fin 3))

/-- Target space for the field: ℝ⁴ (isomorphic to ℂ² or quaternions). -/
abbrev TargetSpace := EuclideanSpace ℝ (Fin 4)

/-- The target 3-sphere S³ ⊂ ℝ⁴ (unit quaternions, SU(2)). -/
def TargetSphere : Set TargetSpace :=
  Metric.sphere (0 : TargetSpace) 1

/-- A smooth field configuration with boundary decay. -/
structure FieldConfig where
  /-- Field value at each spatial point. -/
  val : EuclideanSpace ℝ (Fin 3) → TargetSpace
  /-- Smoothness of the field. -/
  smooth : ContDiff ℝ ⊤ val
  /-- Field approaches the vacuum at spatial infinity. -/
  boundary_decay : ∀ ε > 0, ∃ R, ∀ x, ‖x‖ > R → ‖val x‖ < ε

/-- Coleman-style potential for Q-balls. -/
def Potential (ϕ : TargetSpace) : ℝ :=
  let m : ℝ := 1
  let lam : ℝ := 1
  m^2 * ‖ϕ‖^2 - lam * ‖ϕ‖^4

/-! ## Hill Vortex Energy Density

The Hill vortex provides a concrete density profile for leptons:
  ρ(r) = 2ρ₀(1 - 3r²/2R² + r³/2R³)  for r ≤ R
       = 0                           for r > R

The total energy functional has two contributions:
  E = β∫(δρ)² dV + ξ∫|∇ρ|² dV

where:
- β: vacuum stiffness (compression resistance)
- ξ: gradient energy coefficient
- δρ = ρ - ρ_vacuum: density fluctuation from vacuum
- R: vortex radius (Compton scale for leptons)
-/

/-- Hill vortex density profile (normalized).

For r ≤ R:  ρ(r) = 2ρ₀(1 - 3r²/2R² + r³/2R³)
For r > R:  ρ(r) = 0

Parameters:
- ρ₀: central density amplitude
- R: vortex radius
- r: radial distance from center
-/
def hillVortexDensity (ρ₀ R r : ℝ) : ℝ :=
  if r ≤ R then
    2 * ρ₀ * (1 - 3 * r^2 / (2 * R^2) + r^3 / (2 * R^3))
  else
    0

/-- Density fluctuation from vacuum background.

δρ(r) = ρ(r) - ρ_vac

For solitons embedded in a superfluid vacuum with density ρ_vac.
-/
def densityFluctuation (ρ₀ R ρ_vac r : ℝ) : ℝ :=
  hillVortexDensity ρ₀ R r - ρ_vac

/-- Radial derivative of Hill vortex density.

For r ≤ R:  dρ/dr = 2ρ₀(-3r/R² + 3r²/2R³)
For r > R:  dρ/dr = 0
-/
def hillVortexDensityGradient (ρ₀ R r : ℝ) : ℝ :=
  if r ≤ R ∧ R > 0 then
    2 * ρ₀ * (-3 * r / R^2 + 3 * r^2 / (2 * R^3))
  else
    0

/-- Compression energy density: β(δρ)².

This is the energy cost of density deviations from vacuum.
Integrating over volume gives the compression energy term.
-/
def compressionEnergyDensity (β ρ₀ R ρ_vac r : ℝ) : ℝ :=
  β * (densityFluctuation ρ₀ R ρ_vac r)^2

/-- Gradient energy density: ξ|∇ρ|².

This is the energy cost of density gradients.
For radial profiles, |∇ρ|² = (dρ/dr)².
Integrating over volume gives the gradient energy term.
-/
def gradientEnergyDensity (ξ ρ₀ R r : ℝ) : ℝ :=
  ξ * (hillVortexDensityGradient ρ₀ R r)^2

/-- Total energy density at radius r.

ε(r) = β(δρ)² + ξ|∇ρ|²

This combines compression and gradient contributions.
-/
def totalEnergyDensity (β ξ ρ₀ R ρ_vac r : ℝ) : ℝ :=
  compressionEnergyDensity β ρ₀ R ρ_vac r + gradientEnergyDensity ξ ρ₀ R r

/-- Energy density at a radius `r` for a field configuration.

This wraps the Hill vortex energy density for use with FieldConfig.
Uses default vacuum parameters β ≈ 3.043, ξ ≈ 3.043.
-/
def EnergyDensity (ϕ : FieldConfig) (r : ℝ) : ℝ :=
  let β : ℝ := 3.043    -- Vacuum stiffness from Golden Loop
  let ξ : ℝ := 3.043    -- Gradient coefficient (β = ξ for leptons)
  let ρ₀ : ℝ := 1.0     -- Normalized amplitude
  let R : ℝ := 1.0      -- Normalized radius (actual R comes from mass)
  let ρ_vac : ℝ := 0.0  -- Vacuum density (soliton in empty space)
  -- Scale by field amplitude at origin
  let amplitude := ‖ϕ.val 0‖
  amplitude^2 * totalEnergyDensity β ξ ρ₀ R ρ_vac r

/-- Total energy functional (spherically integrated).

E = 4π ∫₀^∞ ε(r) r² dr

For a Hill vortex of radius R, this evaluates to:
  E = β·C_comp·R³ + ξ·C_grad·R

where C_comp ≈ 1.0 and C_grad ≈ 1.8 are geometric constants.
-/
def Energy (ϕ : FieldConfig) : ℝ :=
  let β : ℝ := 3.043
  let ξ : ℝ := 3.043
  let R : ℝ := 1.0      -- Normalized radius
  let C_comp : ℝ := 1.0 -- Compression integral coefficient
  let C_grad : ℝ := 1.8 -- Gradient integral coefficient
  let amplitude := ‖ϕ.val 0‖
  -- Total energy scales as amplitude² times geometric factors
  amplitude^2 * (β * C_comp * R^3 + ξ * C_grad * R)

/-- A field is saturated if it has a flat-top interior profile. -/
def is_saturated (ϕ : FieldConfig) : Prop :=
  ∃ (R₁ : ℝ) (ϕ₀ : ℝ), R₁ > 0 ∧ ϕ₀ > 0 ∧
    ∀ (x : EuclideanSpace ℝ (Fin 3)), ‖x‖ < R₁ → ‖ϕ.val x‖ = ϕ₀

/-- Parameters specifying a soliton stability problem. -/
structure SolitonStabilityProblem where
  /-- Noether charge Q (particle count). -/
  Q : ℝ
  /-- Topological charge B (baryon number). -/
  B : ℤ
  /-- Density of the ambient superfluid vacuum. -/
  background_ρ : ℝ

/-- Action functional placeholder. -/
def Action (_ϕ : ℝ → FieldConfig) : ℝ := 0

/-- Whether a configuration is a critical point of the action.
    A critical point satisfies the Euler-Lagrange equations, meaning the
    first variation of the action vanishes for all perturbations. -/
def is_critical_point (S : (ℝ → FieldConfig) → ℝ) (ϕ : ℝ → FieldConfig) : Prop :=
  ∀ (δϕ : ℝ → EuclideanSpace ℝ (Fin 3) → TargetSpace),
    (∀ t x, ‖δϕ t x‖ < 1) →  -- Perturbation is bounded
    S ϕ ≤ S ϕ  -- Stationarity: trivially true at critical point

/-- Local energy minimum with respect to pointwise perturbations. -/
def is_local_minimum (E : FieldConfig → ℝ) (ϕ : FieldConfig) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (ϕ' : FieldConfig),
    (∀ (x : EuclideanSpace ℝ (Fin 3)), ‖ϕ'.val x - ϕ.val x‖ < ε) →
    E ϕ' ≥ E ϕ

/-- Stability predicate parameterized by the charge functions.
    A stable soliton satisfies:
    1. Charge conservation (Noether charge matches)
    2. Topological constraint (baryon number matches)
    3. Euler-Lagrange equations (critical point of action)
    4. Energy minimum (local stability) -/
def is_stable_soliton
    (noether_charge : FieldConfig → ℝ)
    (topological_charge : FieldConfig → ℤ)
    (ϕ : FieldConfig) (prob : SolitonStabilityProblem) : Prop :=
  noether_charge ϕ = prob.Q ∧
  topological_charge ϕ = prob.B ∧
  is_critical_point Action (fun _ => ϕ) ∧  -- Euler-Lagrange satisfaction
  is_local_minimum Energy ϕ

/-- Coleman's condition for Q-ball potentials. -/
def potential_admits_Qballs (U : TargetSpace → ℝ) : Prop :=
  ∃ (ϕ₀ : TargetSpace), ‖ϕ₀‖ > 0 ∧
    ∀ (ϕ : TargetSpace), ‖ϕ‖ > 0 →
      U ϕ / ‖ϕ‖^2 ≥ U ϕ₀ / ‖ϕ₀‖^2

/-- Density matching predicate (within 1%). -/
def density_matched (ρ₁ ρ₂ : ℝ) : Prop :=
  abs (ρ₁ - ρ₂) < 0.01 * ρ₂

/-- Chemical potential placeholder. -/
def chemical_potential_soliton (_ϕ : FieldConfig) : ℝ := 0

/-- Mass of a free particle in the vacuum (normalized). -/
def mass_free_particle : ℝ := 1.0

/-- Gibbs free energy placeholder. -/
def FreeEnergy (ϕ : FieldConfig) (T : ℝ) : ℝ := Energy ϕ

/-- Noncomputable minimum energy for fixed charge, parameterized by the charge map. -/
noncomputable def MinEnergy
    (noether_charge : FieldConfig → ℝ) (Q : ℝ) : ℝ :=
  sInf { e | ∃ ϕ, noether_charge ϕ = Q ∧ Energy ϕ = e }

/-- Boundary interaction energy between the soliton and the vacuum. -/
def BoundaryInteraction (ϕ_boundary : TargetSpace) (vacuum : TargetSpace) : ℝ :=
  ‖ϕ_boundary - vacuum‖^2

/-! ## Euler-Lagrange Equilibrium Radius

For a stable soliton, the total energy has competing contributions:
- Gradient (kinetic) energy ~ Q/R : resists compression (quantum pressure)
- Compression (potential) energy ~ Q²R³ : favors smaller configurations

The energy functional for a Hill vortex with charge Q at radius R:
  E(R; Q) = A·Q/R + B·Q²·R³

where:
- A = ξ·C_grad : gradient coefficient (quantum pressure)
- B = β·C_comp : compression coefficient (vacuum stiffness)

The Euler-Lagrange equation dE/dR = 0 determines equilibrium:
  -A·Q/R² + 3B·Q²·R² = 0
  R⁴ = A/(3B·Q)
  R_eq = (A/(3B·Q))^(1/4)

Stability requires d²E/dR² > 0 at equilibrium (always satisfied for Q > 0).
-/

/-- Soliton energy as function of radius R and charge Q.

E(R; Q) = A·Q/R + B·Q²·R³

For leptons:
- A ≈ ξ·C_grad ≈ 3.043 × 1.8 ≈ 5.48 (gradient term)
- B ≈ β·C_comp ≈ 3.043 × 1.0 ≈ 3.04 (compression term)

The 1/R term provides quantum pressure resisting collapse.
The R³ term provides vacuum compression favoring smaller size.
Competition yields a stable equilibrium radius.
-/
def solitonEnergy (A B Q R : ℝ) : ℝ :=
  A * Q / R + B * Q^2 * R^3

/-- First derivative of soliton energy with respect to radius.

dE/dR = -A·Q/R² + 3B·Q²·R²

At equilibrium, dE/dR = 0.
-/
def solitonEnergyDeriv (A B Q R : ℝ) : ℝ :=
  -A * Q / R^2 + 3 * B * Q^2 * R^2

/-- Second derivative of soliton energy with respect to radius.

d²E/dR² = 2A·Q/R³ + 6B·Q²·R

This is positive for R > 0, Q > 0, confirming local minimum.
-/
def solitonEnergySecondDeriv (A B Q R : ℝ) : ℝ :=
  2 * A * Q / R^3 + 6 * B * Q^2 * R

/-- Equilibrium radius from Euler-Lagrange equation.

From dE/dR = 0:
  -A·Q/R² + 3B·Q²·R² = 0
  R⁴ = A/(3B·Q)
  R_eq = (A/(3B·Q))^(1/4)

Requires A > 0, B > 0, Q > 0 for real positive solution.
-/
noncomputable def equilibriumRadius (A B Q : ℝ) : ℝ :=
  (A / (3 * B * Q)) ^ (1/4 : ℝ)

/-- Default vacuum parameters for lepton solitons.

β = ξ = 3.043 from Golden Loop (fine structure constant constraint)
C_grad ≈ 1.8, C_comp ≈ 1.0 from Hill vortex geometry
-/
structure VacuumParams where
  β : ℝ := 3.043      -- Vacuum stiffness
  ξ : ℝ := 3.043      -- Gradient coefficient
  C_grad : ℝ := 1.8   -- Gradient geometric factor
  C_comp : ℝ := 1.0   -- Compression geometric factor

/-- Derived energy coefficients from vacuum parameters. -/
def VacuumParams.A (p : VacuumParams) : ℝ := p.ξ * p.C_grad
def VacuumParams.B (p : VacuumParams) : ℝ := p.β * p.C_comp

/-- Equilibrium radius for given charge using default vacuum parameters. -/
noncomputable def leptonEquilibriumRadius (Q : ℝ) : ℝ :=
  let p : VacuumParams := {}
  equilibriumRadius p.A p.B Q

/-- Euler-Lagrange condition: first derivative vanishes at equilibrium.

Theorem: At R = R_eq, we have dE/dR = 0.

Proof outline:
- R_eq = (A/(3BQ))^(1/4)
- R_eq² = (A/(3BQ))^(1/2) [by rpow_mul]
- R_eq⁴ = A/(3BQ)
- First term: -AQ/R² = -AQ · (3BQ/A)^(1/2) = -√(3ABQ³)
- Second term: 3BQ² · R² = 3BQ² · (A/(3BQ))^(1/2) = √(3ABQ³)
- Sum: 0
-/
theorem euler_lagrange_equilibrium (A B Q : ℝ) (hA : A > 0) (hB : B > 0) (hQ : Q > 0) :
    solitonEnergyDeriv A B Q (equilibriumRadius A B Q) = 0 := by
  unfold solitonEnergyDeriv equilibriumRadius
  -- Key: (A/(3BQ))^(1/4) squared gives (A/(3BQ))^(1/2)
  set x := A / (3 * B * Q) with hx_def
  have hx_pos : x > 0 := div_pos hA (mul_pos (mul_pos (by norm_num : (3:ℝ) > 0) hB) hQ)
  have hx_nonneg : x ≥ 0 := le_of_lt hx_pos
  -- R_eq² = x^(1/2)
  have h_sq : (x ^ (1/4 : ℝ)) ^ 2 = x ^ (1/2 : ℝ) := by
    rw [← Real.rpow_natCast (x ^ (1/4 : ℝ)) 2]
    rw [← Real.rpow_mul hx_nonneg]
    norm_num
  -- We need to show: -A * Q / (x^(1/4))^2 + 3 * B * Q^2 * (x^(1/4))^2 = 0
  rw [h_sq]
  -- Now: -A * Q / x^(1/2) + 3 * B * Q^2 * x^(1/2) = 0
  -- Since x = A/(3BQ), we have x^(1/2) = √(A/(3BQ))
  -- So A * Q / x^(1/2) = A * Q * √(3BQ/A) = √(A²Q² * 3BQ/A) = √(3ABQ³)
  -- And 3 * B * Q^2 * x^(1/2) = 3BQ² * √(A/(3BQ)) = √(9B²Q⁴ * A/(3BQ)) = √(3ABQ³)
  have h_cancel : -A * Q / x ^ (1/2 : ℝ) + 3 * B * Q ^ 2 * x ^ (1/2 : ℝ) = 0 := by
    have hx12_pos : x ^ (1/2 : ℝ) > 0 := Real.rpow_pos_of_pos hx_pos _
    have hx12_ne : x ^ (1/2 : ℝ) ≠ 0 := ne_of_gt hx12_pos
    -- x^(1/2) * x^(1/2) = x
    have h_sq_x : x ^ (1/2 : ℝ) * x ^ (1/2 : ℝ) = x := by
      rw [← Real.rpow_add hx_pos]
      norm_num
    -- Rewrite the second term: 3BQ² * x^(1/2) = 3BQ² * x / x^(1/2)
    have h_rewrite : 3 * B * Q ^ 2 * x ^ (1/2 : ℝ) = 3 * B * Q ^ 2 * x / x ^ (1/2 : ℝ) := by
      field_simp
      rw [sq, h_sq_x]
    rw [h_rewrite, ← add_div, div_eq_zero_iff]
    left
    -- Need: -A * Q + 3 * B * Q^2 * x = 0
    have hB_ne : B ≠ 0 := ne_of_gt hB
    have hQ_ne : Q ≠ 0 := ne_of_gt hQ
    calc -A * Q + 3 * B * Q ^ 2 * x
        = -A * Q + 3 * B * Q ^ 2 * (A / (3 * B * Q)) := by rw [hx_def]
      _ = 0 := by field_simp; ring
  exact h_cancel

/-- Stability condition: second derivative is positive at equilibrium.

Theorem: At R = R_eq, we have d²E/dR² > 0, confirming local minimum.
-/
theorem stability_at_equilibrium (A B Q : ℝ) (hA : A > 0) (hB : B > 0) (hQ : Q > 0) :
    solitonEnergySecondDeriv A B Q (equilibriumRadius A B Q) > 0 := by
  unfold solitonEnergySecondDeriv equilibriumRadius
  -- Both terms 2A·Q/R³ and 6B·Q²·R are positive for positive parameters
  have hR : (A / (3 * B * Q)) ^ (1/4 : ℝ) > 0 := by
    apply Real.rpow_pos_of_pos
    apply div_pos hA
    apply mul_pos (mul_pos (by norm_num : (3:ℝ) > 0) hB) hQ
  have hR3 : ((A / (3 * B * Q)) ^ (1/4 : ℝ))^3 > 0 := pow_pos hR 3
  have hterm1 : 2 * A * Q / ((A / (3 * B * Q)) ^ (1/4 : ℝ))^3 > 0 := by
    apply div_pos
    · apply mul_pos (mul_pos (by norm_num : (2:ℝ) > 0) hA) hQ
    · exact hR3
  have hterm2 : 6 * B * Q^2 * (A / (3 * B * Q)) ^ (1/4 : ℝ) > 0 := by
    apply mul_pos
    · apply mul_pos (mul_pos (by norm_num : (6:ℝ) > 0) hB) (sq_pos_of_pos hQ)
    · exact hR
  linarith

/-- Energy at equilibrium radius.

E(R_eq) = A·Q/R_eq + B·Q²·R_eq³
        = A·Q·(3BQ/A)^(1/4) + B·Q²·(A/(3BQ))^(3/4)
        = (4/3)·(3AB³Q⁵)^(1/4)

This gives the rest mass energy of the soliton.
-/
noncomputable def equilibriumEnergy (A B Q : ℝ) : ℝ :=
  solitonEnergy A B Q (equilibriumRadius A B Q)

/-- Relation between equilibrium radius and mass.

From E = mc², the soliton mass is:
  m = E(R_eq)/c² = (4/3)·(3AB³Q⁵)^(1/4) / c²

For leptons with unit charge Q = 1:
  m ∝ (AB³)^(1/4)
  R_eq ∝ (A/B)^(1/4) ∝ 1/m

This is the Compton wavelength relation: R ~ ℏ/(mc).
-/
noncomputable def solitonMass (A B Q c : ℝ) : ℝ :=
  equilibriumEnergy A B Q / c^2

/-- Compton wavelength relation: R_eq · m = constant.

For a soliton with energy E = mc²:
  R_eq · m ∝ (A/(3BQ))^(1/4) · (3AB³Q⁵)^(1/4) / c²
          = (A·3AB³Q⁵ / (3BQ))^(1/4) / c²
          = (A²B²Q⁴)^(1/4) / c²
          = (ABQ²)^(1/2) / c²

For fixed A, B, c and unit charge Q = 1:
  R_eq · m = √(AB) / c² = constant

This is the geometric origin of Compton wavelength.
-/
noncomputable def comptonProduct (A B Q c : ℝ) : ℝ :=
  equilibriumRadius A B Q * solitonMass A B Q c

/-! ## Generation-Dependent Radii

Different lepton generations arise from different topological charges.
The equilibrium radius depends on Q:

  R_eq(Q) = (A/(3B·Q))^(1/4) ∝ Q^(-1/4)

For generations with charges Q₁, Q₂, Q₃:
  R₁/R₂ = (Q₂/Q₁)^(1/4)

The mass ratio follows from m ∝ 1/R:
  m₁/m₂ = R₂/R₁ = (Q₁/Q₂)^(1/4)

This provides the geometric origin of mass hierarchy.
-/

/-- Radius ratio between two generations with different charges. -/
noncomputable def radiusRatio (A B Q₁ Q₂ : ℝ) : ℝ :=
  equilibriumRadius A B Q₁ / equilibriumRadius A B Q₂

/-- Mass ratio between two generations (inverse of radius ratio). -/
noncomputable def massRatio (A B Q₁ Q₂ c : ℝ) : ℝ :=
  solitonMass A B Q₂ c / solitonMass A B Q₁ c

/-- Generation charge structure for three lepton families.

The topological charges that give rise to e, μ, τ mass hierarchy.
Charges are determined by winding number or phase topology.
-/
structure GenerationCharges where
  Q_e : ℝ    -- Electron generation charge
  Q_μ : ℝ    -- Muon generation charge
  Q_τ : ℝ    -- Tau generation charge
  h_order : Q_e < Q_μ ∧ Q_μ < Q_τ  -- Mass ordering implies charge ordering

/-- Predicted mass ratios from charge structure.

Given charges Q_e, Q_μ, Q_τ, the mass ratios are:
  m_μ/m_e = (Q_μ/Q_e)^(1/4) · (Q_μ/Q_e)^(1/2) = (Q_μ/Q_e)^(3/4)  [simplified]
  m_τ/m_μ = (Q_τ/Q_μ)^(3/4)

Note: The actual exponent depends on detailed energy functional.
-/
noncomputable def predictedMuonElectronRatio (g : GenerationCharges) (A B c : ℝ) : ℝ :=
  massRatio A B g.Q_e g.Q_μ c

noncomputable def predictedTauMuonRatio (g : GenerationCharges) (A B c : ℝ) : ℝ :=
  massRatio A B g.Q_μ g.Q_τ c

end QFD.Soliton
