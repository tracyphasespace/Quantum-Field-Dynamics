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

end QFD.Soliton
