import QFD.Schema.DimensionalAnalysis

noncomputable section

namespace QFD.Schema

open Dimensions

/-!
# Grand Solver Parameter Schema

This file defines the complete set of adjustable parameters for the
QFD Grand Solver. It unites the Nuclear, Cosmological, and Particle
domains into a single optimization space.

## Schema Structure
1. **StandardConstants**: Fixed physics (c, G, h, e)
2. **NuclearParams**: Binding energy and shell couplings
3. **CosmoParams**: Expansion and dark sector couplings
4. **ParticleParams**: Fermion masses and geometric couplings
5. **GrandUnifiedParameters**: The union of all sets
-/

/-- Fixed Fundamental Constants (SI Units) -/
structure StandardConstants where
  c : Velocity
  G : Quantity ⟨3, -1, -2, 0⟩ -- m³/kg/s²
  hbar : Action
  e : Charge
  k_B : Quantity ⟨2, 1, -2, 0⟩ -- J/K (simplified)

/-- Nuclear & Nuclide Parameters (Core Compression) -/
structure NuclearParams where
  c1 : Unitless -- Surface term (A^(2/3))
  c2 : Unitless -- Volume term (A)
  V4 : Energy -- Potential depth (Nuclear well)
  k_c2 : Mass -- Mass scale for binding
  alpha_n : Unitless -- Nuclear fine structure equivalent
  beta_n : Unitless -- Asymmetry coupling
  gamma_e : Unitless -- Geometric shielding factor

/-- Cosmological Parameters (Time Refraction) -/
structure CosmoParams where
  k_J : Quantity ⟨1, 0, -1, 0⟩ -- km/s/Mpc (H0 equivalent)
  eta_prime : Unitless -- Conformal time derivative scale
  A_plasma : Unitless -- SNe plasma dispersion coupling
  rho_vac : Density -- Vacuum energy density
  w_dark : Unitless -- Equation of state param

/-- Particle & Lepton Parameters (Soliton Geometry) -/
structure ParticleParams where
  g_c : Unitless -- Geometric charge coupling (0 ≤ g_c ≤ 1)
  V2 : Energy -- Weak potential scale
  lambda_R : Unitless -- Ricker wavelet width coupling
  mu_e : Mass -- Electron mass seed
  mu_nu : Mass -- Neutrino mass seed

/-- The Grand Solver State Vector
    This structure represents one point in the high-dimensional optimization space.
-/
structure GrandUnifiedParameters where
  std : StandardConstants
  nuclear : NuclearParams
  cosmo : CosmoParams
  particle : ParticleParams

/-! ## Helper: Parameter Flat-Packing
    Functions to serialize/deserialize this structure for the Python optimizer.
-/

def count_parameters : Nat :=
  7 + 5 + 5 -- Nuclear + Cosmo + Particle (Standard are fixed)

end QFD.Schema
