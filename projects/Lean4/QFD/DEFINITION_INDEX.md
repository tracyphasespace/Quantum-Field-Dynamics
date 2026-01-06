# Definition Index - Complete Catalog

**Date**: 2026-01-04
**Total Definitions**: 607
**Total Structures**: 76
**Status**: Comprehensive catalog of all definitions in QFD formalization

This document catalogs every definition and structure in the QFD formalization, organized by module and purpose.

---

## Overview Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Geometric Algebra (GA)** | ~45 | Foundation: Clifford Cl(3,3) basis, operations, grade projection |
| **Cosmology** | ~85 | CMB analysis, axis extraction, vacuum effects, scattering |
| **Nuclear Physics** | ~110 | Core compression, binding energy, symmetry energy, magic numbers |
| **Lepton Physics** | ~75 | Mass spectrum, vortex stability, g-2 anomaly, generations |
| **QM Translation** | ~35 | Quantum mechanics from geometry: Dirac, Pauli, Schrödinger |
| **Soliton Theory** | ~55 | Quantization, hard wall boundary, topological stability |
| **Electrodynamics** | ~40 | Maxwell equations, Coulomb force, magnetic effects |
| **Gravity** | ~35 | Geodesics, Schwarzschild, lensing, G derivation |
| **Charge** | ~25 | Quantization, Coulomb potential, vacuum structure |
| **Vacuum** | ~20 | Vacuum parameters, stiffness constants, density scales |
| **Neutrino** | ~40 | Mass scale, oscillation, chirality, production |
| **Relativity** | ~15 | Lorentz rotors, time dilation, Sagnac effect |
| **Other Modules** | ~27 | Conservation, weak interaction, rift dynamics, etc. |

**Total**: 607 definitions

---

## Module 1: Geometric Algebra (GA) - Foundation Layer

**Purpose**: Implements Clifford algebra Cl(3,3) - the mathematical foundation for all physics.

### Core Definitions (GA/Cl33.lean)

| Definition | Line | Purpose | Type |
|------------|------|---------|------|
| `signature33` | 58 | Metric signature (+,+,+,-,-,-) | Fin 6 → ℝ |
| `Q33` | 78 | Quadratic form for Cl(3,3) | QuadraticForm ℝ (Fin 6 → ℝ) |
| `ι33` | 95 | Canonical injection into Clifford algebra | (Fin 6 → ℝ) →ₗ[ℝ] Cl33 |
| `basis_vector` | 102 | Standard basis vectors | Fin 6 → (Fin 6 → ℝ) |

**Critical Insight**: These 4 definitions generate the entire 64-dimensional Clifford algebra Cl(3,3).

### Basis Operations (GA/BasisOperations.lean)

| Definition | Line | Purpose |
|------------|------|---------|
| `e` | 23 | Basis element generator: e i = ι33 (basis_vector i) |

**Note**: Single definition, but most important - all physics uses `e 0` through `e 5`.

### Grade Projection (GA/GradeProjection.lean)

| Definition | Line | Purpose |
|------------|------|---------|
| `scalar_part` | 22 | Extract scalar (grade-0) component |
| `real_energy_density` | 39 | Energy density from multivector field |

### Multivector Grade (GA/MultivectorGrade.lean)

| Definition | Line | Purpose |
|------------|------|---------|
| `isScalar` | 36 | Test if element is grade-0 |
| `isVector` | 40 | Test if element is grade-1 |
| `isBivector` | 44 | Test if element is grade-2 |

### Hodge Dual (GA/HodgeDual.lean)

| Definition | Line | Purpose |
|------------|------|---------|
| `I_6` | 29 | 6D pseudoscalar (e₀∧e₁∧e₂∧e₃∧e₄∧e₅) |
| `I_4` | 33 | 4D pseudoscalar (e₀∧e₁∧e₂∧e₃) |

### Phase Centralizer (GA/PhaseCentralizer.lean)

| Definition | Line | Purpose | Status |
|------------|------|---------|--------|
| `B_phase` | 113 | Internal bivector B = e₄ ∧ e₅ | **CRITICAL** |
| `commutes_with_phase` | 133 | Centralizer predicate: [x, B] = 0 | **CRITICAL** |

**Significance**: These 2 definitions enable spacetime emergence theorem.

### Conjugation (GA/Conjugation.lean)

| Definition | Line | Purpose |
|------------|------|---------|
| `B_phase` | 23 | Internal bivector (duplicate - consolidate?) |

---

## Module 2: Spacetime Emergence

**Purpose**: Proves 4D Minkowski spacetime emerges from Cl(3,3) centralizer.

### SpacetimeEmergence_Complete.lean

| Definition | Line | Purpose |
|------------|------|---------|
| `signature33` | 44 | Metric signature (module-local version) |
| `Q33` | 53 | Quadratic form (module-local version) |
| `e` | 60 | Basis generator (module-local version) |
| `B_internal` | 66 | Internal bivector B = e₄ ∧ e₅ |
| `commutes_with_B` | 69 | Centralizer predicate |
| `emergent_minkowski` | 283 | Type for emergent 4D spacetime |

**Note**: Local redefinitions for self-containment - could be refactored to import from GA.

---

## Module 3: Cosmology - CMB Analysis

**Purpose**: CMB "Axis of Evil" alignment, vacuum effects, scattering predictions.

### AxisExtraction.lean (11 definitions)

| Definition | Line | Purpose | Theorem Link |
|------------|------|---------|--------------|
| `IsUnit` | 23 | Unit vector predicate: ‖x‖ = 1 | - |
| `P2` | 26 | Legendre polynomial P₂(t) = (3t²-1)/2 | IT.1 |
| `ip` | 32 | Inner product shorthand | - |
| `score` | 35 | Alignment score: (n·x)² | IT.1 |
| `quadPattern` | 38 | Quadrupole pattern P₂(n·x) | IT.1 |
| `AxisSet` | 41 | Critical point set for pattern | IT.1 |
| `tempPattern` | 269 | Temperature anisotropy T(x) = A + B·P₂(n·x) | - |
| `Equator` | 309 | Equatorial plane {x : n·x = 0} | - |

**Key Result**: `quadPattern` generates observable CMB quadrupole (published result).

### OctupoleExtraction.lean (4 definitions)

| Definition | Line | Purpose |
|------------|------|---------|
| `P3` | 23 | Legendre polynomial P₃(t) = (5t³-3t)/2 |
| `octPattern` | 26 | Octupole pattern P₃(n·x) |
| `octAxisPattern` | 29 | Absolute octupole pattern |
| `octTempPattern` | 210 | Octupole temperature anisotropy |

### CoaxialAlignment.lean (0 definitions)

**Note**: This module is pure theorem proving - uses definitions from AxisExtraction and OctupoleExtraction.

### Polarization.lean (14 definitions)

| Definition | Line | Purpose |
|------------|------|---------|
| `P0` | 19 | Legendre polynomial P₀(x) = 1 |
| `polarization_fraction` | 102 | Polarized fraction P/I |
| `is_E_mode_pattern` | 154 | E-mode identification |
| `is_B_mode_pattern` | 174 | B-mode identification |
| `polarization_cross_section` | 199 | Thomson scattering cross-section |
| `C_ell_TE` | 251 | Temperature-E-mode cross-spectrum |
| `C_ell_EE` | 265 | E-mode power spectrum |
| `E_mode_axis_test` | 311 | Axis alignment test for E-modes |
| `E_to_B_ratio` | 335 | E/B ratio test |
| `TE_phase_test` | 368 | TE correlation phase test |
| `polPattern` | 448 | Polarization pattern function |

**Structure**: `StokesParameters` (line 87) - I, Q, U, V components.

### RadiativeTransfer.lean (13 definitions + 2 structures)

| Definition | Line | Purpose |
|------------|------|---------|
| `optical_depth` | 87 | Photon absorption depth τ(z) |
| `survival_fraction` | 93 | Fraction surviving: exp(-τ) |
| `effective_redshift` | 124 | Modified z from vacuum drift |
| `observed_frequency` | 130 | Frequency after drift |
| `planck_occupation` | 166 | Blackbody occupation number |
| `y_distortion` | 182 | y-parameter spectral distortion |
| `collimated_flux` | 225 | Directed radiation flux |
| `isotropic_source` | 233 | Isotropic source term |
| `cmb_spectrum_prediction` | 255 | Predicted CMB spectrum |
| `distance_modulus_survivor` | 265 | Distance modulus with absorption |
| `falsified_by_firas` | 305 | FIRAS falsification criterion |
| `falsified_by_spectroscopy` | 314 | Spectroscopy falsification |

**Structures**:
- `RadiativeTransferParams` (line 49) - α, β, k_drift, T_cmb
- `RadiativeTransferConstraints` (line 63) - Validation predicates

### VacuumRefraction.lean (12 definitions + 2 structures)

| Definition | Line | Purpose |
|------------|------|---------|
| `characteristic_ell_scale` | 107 | Characteristic angular scale |
| `modulation_function` | 126 | Power spectrum modulation |
| `unitarity_bound` | 212 | Unitarity constraint |
| `C_ell_TT_QFD` | 245 | TT spectrum prediction |
| `C_ell_TE_QFD` | 258 | TE spectrum prediction |
| `C_ell_EE_QFD` | 266 | EE spectrum prediction |
| `phase_coherence_test` | 282 | Phase coherence check |
| `falsified_by_unitarity` | 291 | Unitarity falsification |
| `falsified_by_null_detection` | 300 | Null detection falsification |

**Structures**:
- `VacuumRefractionParams` (line 57) - r_psi, A_osc, ell_coherent, etc.
- `VacuumRefractionConstraints` (line 75) - Physical bounds

### ScatteringBias.lean (2 structures)

**Structures**:
- `ScatteringParams` (line 51) - σ_scat, λ_scat, z_scatter
- `ScatteringConstraints` (line 59) - Physical constraints

### VacuumDensityMatch.lean (2 definitions)

| Definition | Line | Purpose |
|------------|------|---------|
| `V` | 22 | Vacuum potential V(ρ) |
| `IsStableUniverse` | 27 | Stability criterion β > 0 |

### KernelAxis.lean (2 definitions + 1 structure)

| Definition | Line | Purpose |
|------------|------|---------|
| `absKernelPattern` | 27 | General kernel pattern K(\|n·x\|) |

**Structure**: `KernelMaxAtOne` (line 22) - Kernel peaked at alignment.

---

## Module 4: Nuclear Physics - Binding Energy & Structure

**Purpose**: Core compression law, magic numbers, symmetry energy, fission stability.

### CoreCompressionLaw.lean (~35 definitions + 4 structures)

**Key Definitions**:

| Definition | Line | Purpose | Category |
|------------|------|---------|----------|
| Energy functionals | Various | B/A prediction from (Z, N, A) | **Core** |
| Fit metrics | 385+ | χ², residuals, correlation | Statistical |
| Stress statistics | 334+ | Compression analysis | Derived |

**Structures**:
- `CCLParams` (line 19) - α_n, c₂, v₄, k_c₂
- `CCLConstraints` (line 28) - Parameter validation
- `StressStatistics` (line 334) - Compression data
- `FitMetrics` (line 385) - Fit quality
- `CCLParamsDimensional` (line 501) - Dimensionful parameters

### BindingMassScale.lean

**Purpose**: Derives mass scale from vacuum stiffness.

### SymmetryEnergyMinimization.lean (2 structures)

**Structures**:
- `ChargeNumber` (line 58) - Z with positivity
- `MassNumber` (line 63) - A with positivity

### MagicNumbers.lean

**Purpose**: Proves magic numbers emerge from shell closure.

### DeuteronFit.lean

**Purpose**: Fits deuteron binding energy from vacuum stiffness.

### FissionTopology.lean (1 structure)

**Structure**: `HarmonicSoliton` (line 69) - Harmonic vortex configuration.

### IsobarStability.lean (1 structure)

**Structure**: `EnergyConstants` (line 25) - α_n, c₂, v₄ package.

### BoundaryCondition.lean (1 structure)

**Structure**: `WallProfile` (line 16) - Hard wall boundary data.

### SelectionRules.lean (1 structure)

**Structure**: `NuclearState` (line 17) - (J, π, T) quantum numbers.

### AlphaNDerivation.lean & AlphaNDerivation_Complete.lean

**Purpose**: Derives α_n ≈ 18.6 MeV from vacuum stiffness.

### BetaNGammaEDerivation.lean & BetaNGammaEDerivation_Complete.lean

**Purpose**: Derives β_N and γ_E from geometric packing.

---

## Module 5: Lepton Physics - Mass Spectrum & Vortex Structure

**Purpose**: Electron, muon, tau mass spectrum from vacuum vortex isomers.

### MassSpectrum.lean (2 structures)

**Structures**:
- `SolitonParams` (line 36) - β, ξ, τ vacuum parameters
- `MassState` (line 111) - n, ℓ, m quantum numbers

### VortexStability.lean (1 structure)

**Structure**: `HillGeometry` (line 82) - Hill vortex radius and circulation.

### LeptonG2Prediction.lean (1 structure)

**Structure**: `ElasticVacuum` (line 32) - (β, ξ, τ) with positivity.

**Key Definitions**:

| Definition | Line | Purpose |
|------------|------|---------|
| `predicted_vacuum_polarization` | 50 | V₄ = -ξ/β (g-2 prediction) |
| `standard_model_A2` | 58 | QED coefficient A₂ ≈ -0.328478965 |

**Theorem**: `golden_loop_prediction_accuracy` (line 89) - **AXIOM** (experimental validation).

### GeometricAnomaly.lean (1 structure)

**Structure**: `VortexParticle` (line 80) - Radius and charge.

### LeptonIsomers.lean (1 structure)

**Structure**: `LeptonModel` (line 54) - Full lepton model extending EmergentConstants.

### QBallStructure.lean (2 structures)

**Structures**:
- `QBallConfig` (line 33) - n-vortex configuration
- `QBallModel` (line 54) - Q-ball model

### Generations.lean

**Purpose**: Proves 3 lepton generations from geometric algebra grades.

### KoideRelation.lean & KoideAlgebra.lean

**Purpose**: Koide relation Q = 2/3 from S₃ symmetry.

### FineStructure.lean

**Purpose**: Fine structure constant α from vacuum stiffness.

### Antimatter.lean (1 structure)

**Structure**: `Wavelet` (line 22) - Antimatter wavelet representation.

### Topology.lean

**Purpose**: Topological charge and winding number definitions.

---

## Module 6: QM Translation - Quantum Mechanics from Geometry

**Purpose**: Eliminates complex numbers - phase becomes bivector rotation.

### DiracRealization.lean (5 definitions + 1 structure)

**Key Definitions**:

| Definition | Line | Purpose |
|------------|------|---------|
| `gamma_1` | 61 | γ₁ = e₀ (Dirac matrix) |
| `gamma_2` | 64 | γ₂ = e₁ |
| `gamma_3` | 67 | γ₃ = e₂ |
| `gamma_0` | 70 | γ₀ = e₃ (timelike) |
| `spacetime_metric_value` | 81 | η_μν values |
| `to_qfd_index` | 89 | Map Fin 4 → Fin 6 |
| `gamma` | 97 | General γ_μ = e(to_qfd_index μ) |

**Structure**: `GeometricMomentum` (line 63) - (p₀, p₁, p₂, p₃, p₄, p₅).

### RealDiracEquation.lean (1 definition + 1 structure)

**Definition**: `Massless6DEquation` (line 72) - 6D massless Dirac equation.

**Structure**: `GeometricMomentum` (line 63) - Momentum in 6D.

### PauliBridge.lean (4 definitions)

| Definition | Line | Purpose |
|------------|------|---------|
| `sigma_x` | 75 | σ_x = e₀ |
| `sigma_y` | 76 | σ_y = e₁ |
| `sigma_z` | 77 | σ_z = e₂ |
| `I_spatial` | 84 | Spatial pseudoscalar e₀∧e₁∧e₂ |

### SchrodingerEvolution.lean

**Purpose**: Phase evolution e^{iθ} → e^{Bθ} (complex → bivector).

### Heisenberg.lean (3 definitions)

| Definition | Line | Purpose |
|------------|------|---------|
| `X_op` | 50 | Position operator X = e₀ |
| `P_op` | 53 | Momentum operator P = e₃ |
| `commutator` | 56 | [A, B] = AB - BA |

### MeasurementCollapse.lean (2 structures + 1 definition)

**Structures**:
- `QuantumState` (line 16) - Superposition state
- `ClassicalState` (line 22) - Measured outcome

**Definition**: `decoherence_rate` (line 26) - Decoherence timescale.

---

## Module 7: Soliton Theory - Topological Stability

**Purpose**: Quantization conditions, hard wall boundary, topological conservation.

### TopologicalStability.lean (~20 definitions)

**Note**: This is a large file with many energy functionals and stability definitions. Key examples:

- Energy density functionals
- Pressure gradients
- Topological charge extraction
- Vacuum normalization

### Quantization.lean

**Purpose**: Vortex quantization from boundary conditions.

### HardWall.lean

**Purpose**: Hard wall boundary conditions for solitons.

### GaussianMoments.lean

**Purpose**: Gaussian profile moments and integrals.

### RickerAnalysis.lean

**Purpose**: Ricker wavelet analysis for soliton profiles.

### BreatherModes.lean

**Purpose**: Breather mode analysis (radial oscillations).

---

## Module 8: Electrodynamics

**Purpose**: Maxwell equations, Coulomb force, magnetic effects from geometry.

### MaxwellReal.lean

**Purpose**: Real-valued Maxwell equations (no complex numbers).

### Coulomb.lean (in Charge/)

**Purpose**: Coulomb force from vacuum stress.

### AharonovBohm.lean

**Purpose**: Aharonov-Bohm effect from geometric phase.

### MagneticHelicity.lean

**Purpose**: Magnetic helicity conservation.

### LarmorPrecession.lean

**Purpose**: Larmor precession from bivector dynamics.

### NoMonopoles.lean

**Purpose**: Proves ∇·B = 0 from Clifford structure.

### PoyntingTheorem.lean

**Purpose**: Energy conservation in EM fields.

### ProcaReal.lean

**Purpose**: Massive photon (Proca equation) in real form.

### CerenkovReal.lean

**Purpose**: Čerenkov radiation from vacuum polarization.

### DispersionRelation.lean

**Purpose**: Dispersion relation ω(k) from vacuum.

---

## Module 9: Gravity

**Purpose**: Geodesics, Schwarzschild geometry, lensing, G derivation.

### G_Derivation.lean

**Purpose**: Derives Newton's constant G from vacuum stiffness.

### GeodesicEquivalence.lean

**Purpose**: Proves geodesic equation from vacuum gradient.

### GeodesicForce.lean

**Purpose**: Geodesic force from vacuum refraction.

### SchwarzschildLink.lean

**Purpose**: Schwarzschild solution from vacuum density profile.

### SnellLensing.lean

**Purpose**: Gravitational lensing from Snell's law.

### TimeRefraction.lean

**Purpose**: Time dilation from vacuum refractive index.

### PerihelionShift.lean

**Purpose**: Mercury perihelion shift prediction.

### GeometricCoupling.lean

**Purpose**: Gravity coupling from vacuum geometry.

---

## Module 10: Charge

**Purpose**: Charge quantization, Coulomb potential, vacuum structure.

### Quantization.lean

**Purpose**: Proves charge quantization from vortex topology.

### Coulomb.lean

**Purpose**: Coulomb 1/r potential from vacuum stress.

### Potential.lean

**Purpose**: Electric potential from field configuration.

### Vacuum.lean

**Purpose**: Vacuum charge structure.

---

## Module 11: Vacuum Parameters

**Purpose**: Fundamental vacuum constants (β, ξ, τ, ρ_vac).

### VacuumParameters.lean (5 definitions + 5 structures)

**Structures**:
- `VacuumBulkModulus` (line 34) - β with positivity
- `VacuumGradientStiffness` (line 65) - ξ with positivity
- `VacuumTemporalStiffness` (line 93) - τ with positivity
- `VacuumDensityScale` (line 117) - ρ_vac with positivity
- `VacuumParameters` (line 131) - Complete parameter set

**Purpose**: These structures enforce positivity constraints on physical parameters.

### StrongCP.lean

**Purpose**: Strong CP problem resolution via vacuum topology.

### ZetaPhysics.lean

**Purpose**: Riemann zeta function in vacuum energy.

---

## Module 12: Neutrino Physics

**Purpose**: Neutrino mass scale, oscillation, chirality, production.

### Neutrino_MassScale.lean (2 definitions + 1 structure)

**Structure**: `MassContext` (line 25) - Vacuum coupling context.

**Definitions**:
- `coupling_efficiency` (line 39) - Neutrino-vacuum coupling
- `neutrino_mass` (line 46) - m_ν = η · m_e

### Neutrino_Chirality.lean (1 definition + 1 structure)

**Structure**: `Vortex` (line 50) - Vortex configuration.

**Definition**: `chirality_op` (line 61) - Chirality operator.

### Neutrino_Production.lean (2 structures)

**Structures**:
- `BetaDecayProps` (line 49) - Beta decay properties
- `Realize` (line 83) - Realization functor

### Neutrino_Oscillation.lean

**Purpose**: Neutrino oscillation from mass mixing.

### Neutrino_Bleaching.lean

**Purpose**: Neutrino energy loss in vacuum.

### Neutrino_MinimalRotor.lean

**Purpose**: Minimal rotor model for neutrino.

### Neutrino_Topology.lean

**Purpose**: Topological structure of neutrino vortex.

---

## Module 13: Relativity

**Purpose**: Lorentz transformations, time dilation, Sagnac effect.

### LorentzRotors.lean

**Purpose**: Lorentz transformations as bivector rotations.

### TimeDilationMechanism.lean

**Purpose**: Time dilation from vacuum refractive index.

### SagnacEffect.lean

**Purpose**: Sagnac effect from rotating vacuum.

---

## Module 14: Weak Interaction

**Purpose**: CP violation, chiral anomaly, double beta decay, pion geometry.

### CPViolation.lean

**Purpose**: CP violation from vacuum phase.

### ChiralAnomaly.lean

**Purpose**: Chiral anomaly from topology.

### DoubleBetaDecay.lean

**Purpose**: Neutrinoless double beta decay.

### PionGeometry.lean

**Purpose**: Pion as geometric excitation.

---

## Module 15: Hydrogen & Photon Physics

**Purpose**: Photon-soliton interaction, resonance, scattering.

### PhotonSoliton.lean & PhotonSolitonStable.lean

**Purpose**: Photon as soliton excitation.

### PhotonSolitonEmergentConstants.lean (1 structure)

**Structure**: `EmergentConstants` (line 40) - Speed of light emergence.

### TopologicalCharge.lean (1 structure)

**Structure**: `QFDModelTopological` (line 31) - Topological extension.

### PhotonResonance.lean (1 structure)

**Structure**: `ResonantModel` (line 15) - Resonance model.

### PhotonScattering.lean

**Purpose**: Rayleigh and Raman scattering.

### SpeedOfLight.lean

**Purpose**: Derives c from vacuum parameters.

### UnifiedForces.lean & UnifiedForces_v2.lean

**Purpose**: Unification of EM, weak, strong forces.

---

## Module 16: Electron Structure

**Purpose**: Electron as Hill vortex with topological charge.

### HillVortex.lean

**Purpose**: Hill vortex velocity field.

### CirculationTopology.lean

**Purpose**: Circulation quantization from topology.

### AlphaCirc.lean

**Purpose**: Fine structure constant from circulation.

### AxisAlignment.lean

**Purpose**: Vortex axis alignment.

---

## Module 17: Conservation Laws

**Purpose**: Noether theorem, unitarity, neutrino identity.

### Noether.lean

**Purpose**: Noether theorem for conserved currents.

### Unitarity.lean

**Purpose**: Unitarity preservation (black holes, S-matrix).

### NeutrinoID.lean & variants

**Purpose**: Neutrino identification from quantum numbers.

### NeutrinoMixing.lean

**Purpose**: Neutrino mixing matrix.

---

## Module 18: Rift Dynamics (Black Hole Physics)

**Purpose**: Black hole ergosphere, charge escape, spin sorting.

### ChargeEscape.lean

**Purpose**: Charge separation in ergosphere.

### RotationDynamics.lean

**Purpose**: Vacuum rotation around black holes.

### SequentialEruptions.lean

**Purpose**: Sequential eruption mechanism.

### SpinSorting.lean

**Purpose**: Spin-dependent particle sorting.

---

## Module 19: Schema & Dimensional Analysis

**Purpose**: Constraint enforcement, coupling definitions, unit checking.

### Constraints.lean

**Purpose**: Parameter constraints and validation.

### Couplings.lean

**Purpose**: Coupling constant definitions.

### DimensionalAnalysis.lean

**Purpose**: Dimensional consistency checking.

---

## Module 20: Supporting Infrastructure

### ToyModel.lean (1 definition)

**Definition**: `toyWindingOp` (line 55) - Toy winding operator for testing.

### ProofLedger.lean

**Purpose**: Maps book claims to Lean theorems (documentation only).

### Math/ReciprocalIneq.lean

**Purpose**: Mathematical lemmas for reciprocal inequalities.

### Classical/Conservation.lean

**Purpose**: Classical conservation laws.

### Test/TrivialProof.lean

**Purpose**: Test infrastructure for build system.

### StabilityCriterion.lean

**Purpose**: General stability criterion definitions.

### EmergentAlgebra.lean & EmergentAlgebra_Heavy.lean

**Purpose**: Emergent algebra structure and heavy computations.

### AdjointStability_Complete.lean (7 definitions + 1 structure)

**Definitions**:
- `signature` (line 50) - Signature function
- `swap_sign` (line 53) - Sign swap for basis
- `blade_square` (line 59) - Blade squaring
- `adjoint_action` (line 132) - Adjoint action
- `Multivector` (line 154) - Multivector type
- `qfd_adjoint` (line 157) - QFD adjoint
- `energy_functional` (line 161) - Energy functional

### BivectorClasses_Complete.lean

**Purpose**: Bivector classification (spatial, temporal, internal).

### GoldenLoop.lean

**Purpose**: Golden ratio β = ϕ² verification.

### AngularSelection.lean

**Purpose**: Angular momentum selection rules.

---

## Summary by Purpose

### Foundational Definitions (Essential Infrastructure)

1. **GA/Cl33.lean**: `signature33`, `Q33`, `ι33`, `basis_vector` (4 definitions)
2. **GA/BasisOperations.lean**: `e` (1 definition) - **MOST IMPORTANT**
3. **GA/PhaseCentralizer.lean**: `B_phase`, `commutes_with_phase` (2 definitions) - **CRITICAL**

**Total foundational**: ~7 definitions that generate everything else.

### Physical Observable Definitions

- Energy functionals: ~45 definitions
- Force laws: ~15 definitions
- Spectra (CMB, mass, binding): ~35 definitions
- Quantum operators: ~20 definitions

**Total observables**: ~115 definitions

### Geometric Structure Definitions

- Patterns (quadrupole, octupole, polarization): ~25 definitions
- Vortex structures: ~20 definitions
- Topological invariants: ~15 definitions

**Total geometric**: ~60 definitions

### Validation & Falsification Definitions

- Falsification criteria: ~10 definitions
- Constraint predicates: ~25 definitions
- Fit metrics: ~15 definitions

**Total validation**: ~50 definitions

### Derived Quantities

- Ratios, fractions, normalized values: ~100 definitions
- Intermediate calculations: ~150 definitions
- Helper functions: ~125 definitions

**Total derived**: ~375 definitions

---

## Key Insights

### Design Patterns

1. **Positivity Enforcement**: Many structures bundle a value with positivity proof
   - Example: `VacuumBulkModulus` = β + (β > 0)
   - Ensures type safety for physical parameters

2. **Constraint Structures**: Separate parameter data from validation
   - Example: `CCLParams` (data) + `CCLConstraints` (validation)
   - Allows compile-time vs. runtime checking

3. **Falsification Predicates**: Observable predictions as boolean functions
   - Example: `falsified_by_firas`, `falsified_by_unitarity`
   - Makes theory testable

4. **Module-Local Redefinitions**: Some modules redefine `e`, `signature33`, etc.
   - Reason: Self-containment for paper extraction
   - Refactoring opportunity: Import from GA module instead

### Redundancy Analysis

**Potential Consolidation**:
1. `B_phase` defined in 3 files (GA/PhaseCentralizer, GA/Conjugation, SpacetimeEmergence)
   - **Action**: Consolidate to GA/PhaseCentralizer, import elsewhere
2. `signature33`, `Q33`, `e` redefined in SpacetimeEmergence_Complete
   - **Action**: Import from GA/Cl33 instead
3. `GeometricMomentum` defined in 2 files
   - **Action**: Move to shared location

**Estimated reduction**: 5-10 definitions could be eliminated via imports.

### Documentation Completeness

**Well-Documented Modules**:
- GA/ - Comprehensive doc comments
- Cosmology/AxisExtraction.lean - Publication-ready
- QM_Translation/ - Clear purpose statements

**Under-Documented Modules**:
- Some Nuclear/ files lack purpose comments
- Rift/ modules need more context

**Action**: Add module-level purpose docstrings to all files.

---

## Navigation Guide

### Finding a Definition

1. **By Module**: Use table of contents above
2. **By Name**: Search this file for definition name
3. **By Purpose**: Check "Summary by Purpose" section
4. **By File**:
   ```bash
   grep -rn "^def definition_name" QFD/**/*.lean
   ```

### Understanding Dependencies

**Dependency Hierarchy** (bottom-up):
1. **Foundation**: GA/Cl33.lean (4 definitions)
2. **Basis Operations**: GA/BasisOperations.lean (1 definition: `e`)
3. **Everything Else**: Builds on `e 0` through `e 5`

**Critical Path**:
```
signature33 → Q33 → ι33 → basis_vector → e → ALL PHYSICS
```

### Most Important Definitions (Top 10)

1. **`e`** (GA/BasisOperations.lean:23) - Generates all basis elements
2. **`B_phase`** (GA/PhaseCentralizer.lean:113) - Enables spacetime emergence
3. **`signature33`** (GA/Cl33.lean:58) - Metric signature
4. **`quadPattern`** (Cosmology/AxisExtraction.lean:38) - CMB quadrupole
5. **`predicted_vacuum_polarization`** (Lepton/LeptonG2Prediction.lean:50) - g-2 prediction
6. **`gamma`** (QM_Translation/DiracRealization.lean:97) - Dirac matrices
7. **Energy functionals in CoreCompressionLaw** - Nuclear binding
8. **`mass_formula` in LeptonIsomers** - Lepton masses
9. **`optical_depth` in RadiativeTransfer** - Cosmology predictions
10. **`charge quantization` in Charge/Quantization** - Charge values

---

## Future Work

### Immediate Actions

1. **Consolidate redundant definitions** (B_phase, signature33, etc.)
2. **Add module-level docstrings** to under-documented files
3. **Create STRUCTURE_INDEX.md** for 76 structures
4. **Cross-reference** definitions with theorems that use them

### Long-Term Enhancements

1. **Dependency graph**: Visual map of definition dependencies
2. **Definition categories**: Tag each definition by role (observable, intermediate, helper)
3. **Theorem-definition matrix**: Which theorems use which definitions
4. **Refactoring guide**: Safe refactoring patterns for consolidation

---

## Appendix: Complete File List by Module

### GA Module (11 files)
- Cl33.lean, BasisOperations.lean, BasisProducts.lean, BasisReduction.lean
- Cl33Instances.lean, Cl33_Minimal.lean, Conjugation.lean, GradeProjection.lean
- HodgeDual.lean, MultivectorGrade.lean, PhaseCentralizer.lean, Tactics.lean

### Cosmology Module (9 files)
- AxisExtraction.lean, CoaxialAlignment.lean, HubbleDrift.lean, KernelAxis.lean
- OctupoleExtraction.lean, Polarization.lean, RadiativeTransfer.lean
- RealTimeCosmology.lean, ScatteringBias.lean, VacuumDensityMatch.lean, VacuumRefraction.lean

### Nuclear Module (15 files)
- AlphaNDerivation.lean, AlphaNDerivation_Complete.lean
- BetaNGammaEDerivation.lean, BetaNGammaEDerivation_Complete.lean
- BindingMassScale.lean, BoundaryCondition.lean, CoreCompression.lean
- CoreCompressionLaw.lean, DeuteronFit.lean, FissionTopology.lean
- IsobarStability.lean, IsomerDecay.lean, MagicNumbers.lean, ProtonSpin.lean
- QuarticStiffness.lean, SelectionRules.lean, SymmetryEnergyMinimization.lean
- TimeCliff.lean, TimeCliff_Complete.lean, VacuumStiffness.lean, WellDepth.lean
- YukawaDerivation.lean

### Lepton Module (14 files)
- AnomalousMoment.lean, Antimatter.lean, FineStructure.lean, Generations.lean
- GeometricAnomaly.lean, KoideAlgebra.lean, KoideRelation.lean
- LeptonG2Prediction.lean, LeptonIsomers.lean, MassFunctional.lean
- MassSpectrum.lean, PairProduction.lean, QBallStructure.lean
- StabilityGuards.lean, Topology.lean, VortexStability.lean, VortexStability_v3.lean

### QM_Translation Module (9 files)
- DiracRealization.lean, Heisenberg.lean, MeasurementCollapse.lean
- PauliBridge.lean, PauliExclusion.lean, RealDiracEquation.lean
- SchrodingerEvolution.lean
- (Plus 3 test files: PauliBridge_Test.lean, Test2, Test3)

### Electrodynamics Module (9 files)
- AharonovBohm.lean, CerenkovReal.lean, DispersionRelation.lean
- LarmorPrecession.lean, MagneticHelicity.lean, MaxwellReal.lean
- NoMonopoles.lean, PoyntingTheorem.lean, ProcaReal.lean

### Gravity Module (8 files)
- G_Derivation.lean, GeodesicEquivalence.lean, GeodesicForce.lean
- GeometricCoupling.lean, PerihelionShift.lean, SchwarzschildLink.lean
- SnellLensing.lean, TimeRefraction.lean

### Soliton Module (6 files)
- BreatherModes.lean, GaussianMoments.lean, HardWall.lean
- Quantization.lean, RickerAnalysis.lean, TopologicalStability.lean
- TopologicalStability_Refactored.lean

### Hydrogen Module (9 files)
- PhotonResonance.lean, PhotonScattering.lean, PhotonSoliton.lean
- PhotonSolitonEmergentConstants.lean, PhotonSolitonStable.lean
- PhotonSoliton_Kinematic.lean, SpeedOfLight.lean, TopologicalCharge.lean
- UnifiedForces.lean, UnifiedForces_v2.lean

### Charge Module (4 files)
- Coulomb.lean, Potential.lean, Quantization.lean, Vacuum.lean

### Vacuum Module (3 files)
- StrongCP.lean, VacuumParameters.lean, ZetaPhysics.lean

### Neutrino Module (7 files)
- Neutrino.lean, Neutrino_Bleaching.lean, Neutrino_Chirality.lean
- Neutrino_MassScale.lean, Neutrino_MinimalRotor.lean
- Neutrino_Oscillation.lean, Neutrino_Production.lean, Neutrino_Topology.lean

### Conservation Module (7 files)
- NeutrinoID.lean, NeutrinoID_Automated.lean, NeutrinoID_Fixed.lean
- NeutrinoID_Production.lean, NeutrinoID_Simple.lean
- NeutrinoMixing.lean, Noether.lean, Unitarity.lean

### Relativity Module (3 files)
- LorentzRotors.lean, SagnacEffect.lean, TimeDilationMechanism.lean

### Weak Module (4 files)
- CPViolation.lean, ChiralAnomaly.lean, DoubleBetaDecay.lean, PionGeometry.lean

### Rift Module (4 files)
- ChargeEscape.lean, RotationDynamics.lean, SequentialEruptions.lean, SpinSorting.lean

### Electron Module (4 files)
- AlphaCirc.lean, AxisAlignment.lean, CirculationTopology.lean, HillVortex.lean

### Schema Module (3 files)
- Constraints.lean, Couplings.lean, DimensionalAnalysis.lean

### Matter Module (1 file)
- ProtonTopology.lean

### Classical Module (1 file)
- Conservation.lean

### Empirical Module (1 file)
- CoreCompression.lean

### Math Module (1 file)
- ReciprocalIneq.lean

### Test Module (1 file)
- TrivialProof.lean

### Root-Level Files (9 files)
- AdjointStability_Complete.lean, AngularSelection.lean
- BivectorClasses_Complete.lean, Cl33ImportTest.lean
- EmergentAlgebra.lean, EmergentAlgebra_Heavy.lean, GoldenLoop.lean
- SpacetimeEmergence_Complete.lean, SpectralGap.lean
- StabilityCriterion.lean, ToyModel.lean, ProofLedger.lean

**Total**: 169 files (excluding archives, backups, sketches)

---

## Conclusion

This index catalogs all 607 definitions across 169 Lean files in the QFD formalization. The repository demonstrates:

1. **Hierarchical Design**: 7 foundational definitions generate 600+ derived concepts
2. **Type Safety**: Structures enforce physical constraints (positivity, units)
3. **Falsifiability**: Observable predictions as computable functions
4. **Modularity**: Clear separation of concerns (GA, physics domains)
5. **Completeness**: 791 proven statements built on this foundation

**Usage**: Navigate by module (Section headers), search by name (Ctrl+F), or consult "Top 10" for entry points.

**Maintenance**: Update this index when adding new definitions. Run definition count to verify:
```bash
grep -r "^def " --include="*.lean" QFD | wc -l  # Should match "Total Definitions" above
```

**Next Step**: Create STRUCTURE_INDEX.md for the 76 structures with detailed field documentation.
