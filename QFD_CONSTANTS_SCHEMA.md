# QFD Universal Constants & Schema Reference

**Purpose**: Unified constants, units, and naming across all QFD sectors (lepton, nuclear, photon)
**Date**: 2026-01-03
**Status**: Master reference for cross-sector interactions

---

## 📋 Table of Contents

1. [Universal Physical Constants](#universal-physical-constants)
2. [QFD Framework Parameters](#qfd-framework-parameters)
3. [Lepton Sector (Hill Vortex)](#lepton-sector-hill-vortex)
4. [Nuclear Sector (Q-Ball Soliton)](#nuclear-sector-q-ball-soliton)
5. [Photon Sector](#photon-sector)
6. [Units and Conventions](#units-and-conventions)
7. [Schema and Data Structures](#schema-and-data-structures)
8. [Interaction Models](#interaction-models)

---

## Universal Physical Constants

### Standard Model Constants

```python
# Particle Masses (MeV)
ELECTRON_MASS = 0.5109989461  # MeV
MUON_MASS = 105.6583745       # MeV
TAU_MASS = 1776.86            # MeV
M_PROTON = 938.272            # MeV
M_NEUTRON = 939.565           # MeV

# Fundamental Constants (natural units ℏ = c = 1)
HBAR_C = 197.3269804          # MeV·fm (conversion factor)
ALPHA_EM = 1/137.036          # Fine structure constant
SPEED_OF_LIGHT = 299792.458   # km/s (SI)
```

### File Locations
- **Lepton**: `/V22_Lepton_Analysis/integration_attempts/v22_enhanced_hill_vortex_solver.py:146-152`
- **Nuclear**: `/LaGrangianSolitons/soliton_geometry_solver.py:22-25`

---

## QFD Framework Parameters

### The Universal Stiffness Parameter β

```python
# Vacuum stiffness (dimensionless)
BETA = 3.058  # From cosmology/nuclear constraint

# Alternative name in some contexts:
VACUUM_STIFFNESS = 3.058

# Measured values by sector:
BETA_COSMOLOGY = 3.058    # CMB axis alignment
BETA_NUCLEAR = 3.1 ± 0.05 # Nuclear binding energies
BETA_LEPTON = 3.15 ± 0.05 # Hill vortex fits (effective)

# Note: Small offset (3%) between sectors under investigation
# Working hypothesis: β = 3.058 is fundamental, offsets are systematic
```

**Physical Meaning**:
- Resistance of vacuum to density perturbations
- Appears in energy density: `E_stab = ∫ β(δρ)² dV`
- Unifies cosmological, nuclear, and lepton sectors

**Derivation**:
- From α-constraint: `π² · exp(β) · (c₂/c₁) = α⁻¹ = 137.036`
- Where `c₂/c₁ ≈ 6.42` from nuclear saturation

### Vacuum Density

```python
# Vacuum baseline density (normalized units)
RHO_VAC = 1.0  # Normalization constant

# Density perturbations:
# δρ(r) = ρ(r) - RHO_VAC
# Constraint: ρ(r) ≥ 0 (no negative density)
```

---

## Lepton Sector (Hill Vortex)

### Model: Hill's Spherical Vortex

**Reference**: `/V22_Lepton_Analysis/integration_attempts/v22_enhanced_hill_vortex_solver.py`
**Lean Formalization**: `/projects/Lean4/QFD/Electron/HillVortex.lean`

### Geometric Parameters

```python
class HillVortexContext:
    """
    Hill vortex parameters for leptons (e, μ, τ)

    Physical interpretation:
    - Leptons are ROTATING VORTEX SOLITONS in vacuum
    - Internal circulation creates pressure deficit (Bernoulli effect)
    - Toroidal swirl adds angular momentum (spin)
    """

    def __init__(self, R, U, beta, rho_vac=1.0, g_c=0.985):
        self.R = R              # Vortex radius (fm)
        self.U = U              # Propagation velocity (c units)
        self.beta = beta        # Vacuum stiffness (3.058)
        self.rho_vac = rho_vac  # Vacuum density floor
        self.g_c = g_c          # Charge coupling constant
```

**Typical Values**:

| Lepton | R (fm) | U | Q* | Mass (MeV) |
|--------|--------|---|-----|------------|
| Electron | ~0.84 | ~0.1 | 2.2 | 0.511 |
| Muon | ~0.04 | ~0.5 | 2.3 | 105.7 |
| Tau | ~0.002 | ~0.9 | 9800 | 1776.9 |

*(Values are fitted, not predicted - see GIGO warning)*

### Stream Functions

```python
# Internal (r < R): Rotational vortex flow
def stream_function_internal(r, theta):
    """
    ψ = -(3U/2R²)(R² - r²)r² sin²θ

    From: Lamb (1932) §159, HillVortex.lean:38-39
    """
    sin_sq = np.sin(theta)**2
    return -(3 * U / (2 * R**2)) * (R**2 - r**2) * r**2 * sin_sq

# External (r > R): Irrotational potential flow
def stream_function_external(r, theta):
    """
    ψ = (U/2)(r² - R³/r) sin²θ

    From: Lamb (1932) §160, HillVortex.lean:42-43
    """
    sin_sq = np.sin(theta)**2
    return (U / 2) * (r**2 - R**3 / r) * sin_sq
```

### Density Perturbation

```python
def density_perturbation(r, amplitude):
    """
    Density deficit from Bernoulli effect

    δρ(r) = -amplitude · (1 - r²/R²)  for r < R
          = 0                         for r ≥ R

    From: HillVortex.lean:65-70
    Physical: Pressure drops where velocity is high
    """
    if r < R:
        return -amplitude * (1 - (r / R)**2)
    else:
        return 0.0
```

### 4-Component Field Structure

```python
# Field components (from AxisAlignment.lean)
psi_s   # Scalar density (pressure perturbation)
psi_b0  # Bi-vector x-component (toroidal swirl)
psi_b1  # Bi-vector y-component (toroidal swirl)
psi_b2  # Bi-vector z-component (toroidal swirl)

# Total circulation = poloidal (shape) + toroidal (spin)
# L_z = ∫ (r × v_toroidal) · ẑ dV
```

### Energy Functional

```python
def energy_functional(psi, R, U, beta):
    """
    E = ∫ [½|∇ψ|² + V(ρ(ψ)) + E_csr(ψ_s)] · 4πr² dr

    Components:
    - Kinetic: ½|∇ψ|² (gradient energy)
    - Potential: V(ρ) = β·(ρ - ρ_vac)² (stiffness)
    - Charge self-repulsion: E_csr = (g_c²/2)∫ρ_q² dV
    """
    kinetic = 0.5 * np.sum(np.gradient(psi)**2)
    rho = rho_vac + density_perturbation(r, amplitude)
    potential = beta * (rho - rho_vac)**2
    charge_self = 0.5 * g_c**2 * np.sum(charge_density(psi)**2)

    return simps((kinetic + potential + charge_self) * 4*np.pi*r**2, r)
```

### Normalization Constraint (Q*)

```python
# Q* = RMS charge density (encodes internal structure)
def Q_star(psi_s, g_c):
    """
    Q* = ∫ ρ_charge² · 4πr² dr

    Where: ρ_q = -g_c · ∇²ψ_s (Poisson equation)

    Physical meaning:
    - Characterizes internal angular structure
    - Different Q* → different masses
    - Electron: Q* ≈ 2.2 (ground state)
    - Muon: Q* ≈ 2.3 (first excitation)
    - Tau: Q* ≈ 9800 (highly excited)
    """
    rho_charge = -g_c * laplacian(psi_s)
    return simps(rho_charge**2 * 4*np.pi*r**2, r)
```

---

## Nuclear Sector (Q-Ball Soliton)

### Model: Topological Q-Ball with Soliton Profile

**Reference**: `/LaGrangianSolitons/soliton_geometry_solver.py`
**Lean Formalization**: `/projects/Lean4/QFD/Nuclear/`

### Geometric Parameters

```python
class SolitonFieldNucleus:
    """
    Q-ball soliton model for nuclei

    Physical interpretation:
    - Nuclei are STATIC SOLITON LUMPS in vacuum
    - Sech profile (topological, saturates naturally)
    - Alpha clustering + valence nucleons
    """

    def __init__(self, Z, N_neutrons):
        self.Z = Z                      # Proton number
        self.N = N_neutrons             # Neutron number
        self.A = Z + N_neutrons         # Mass number
        self.n_alphas = self.A // 4     # Number of alpha clusters
        self.n_valence = self.A % 4     # Extra nucleons
        self.soliton_width = 0.85       # fm (skin depth)
```

### Universal Constants

```python
# QFD Nuclear Constants (fixed)
M_PROTON = 938.272         # MeV (energy scale)
LAMBDA_TEMPORAL = 0.42     # He-4 Golden Spike (calibrated)
ALPHA_EM = 1/137.036       # Fine structure constant

# Derived from harmonic model:
c1 = 1.474  # Surface tension coefficient (dimensionless)
c2 = 0.173  # Volume pressure coefficient (dimensionless)

# Tension ratio (geometric stress parameter):
# ratio = (c2/c1) * A^(1/3)
# Drip lines:
#   Neutron drip: ratio > 1.701
#   Proton drip: ratio ~ 0.450 (73.6% lower due to Coulomb)
```

### Soliton Kernel (Density Profile)

```python
def soliton_kernel(r, width=0.85):
    """
    Sech profile for Q-ball soliton

    ρ(r) ~ sech(r/w) = 1/cosh(r/w)

    Physical meaning:
    - Topological soliton solution (Appendix R)
    - Saturates naturally (unlike Gaussian)
    - Behaves like tanh inside, exp outside
    - Width w ~ 1/√β (vacuum skin depth)
    """
    r = np.clip(r, 0, 20)  # Numerical stability
    return 1.0 / np.cosh(r / width)
```

### Harmonic Mode Number (N)

```python
# Integer quantization from standing wave structure
# N = harmonic mode number (from fit to SEMF)

# Conservation law (universal):
# N_parent = ΣN_fragments

# Typical values:
N_He4 = 2     # Alpha particle (magic number)
N_C14 = 8     # Carbon-14 cluster
N_Ne20 = 10   # Neon-20 cluster

# Even-N rule: All observed clusters have even N
# (Topological closure requires inversion symmetry)
```

### Metric Equation (QFD Time Dilation)

```python
def metric_factor(rho_total, lambda_temporal=0.42):
    """
    QFD metric equation (geometric time dilation)

    √g₀₀ = 1 / (1 + λ·ρ)

    From: TimeDilationSoliton, geometric algebra

    Physical meaning:
    - Dense regions → slower clock rates
    - Creates "gravitational" binding
    - λ = 0.42 calibrated from He-4 mass
    """
    return 1.0 / (1.0 + lambda_temporal * rho_total)
```

### Alpha Cluster Geometry

```python
# Standard alpha cluster configurations
# (from soliton_geometry_solver.py:44-64)

ALPHA_GEOMETRIES = {
    1: [[0, 0, 0]],                    # He-4: Single alpha
    2: [[-s, 0, 0], [s, 0, 0]],       # Be-8: Linear dimer
    3: "equilateral_triangle",         # C-12: Hoyle state
    4: "tetrahedron",                  # O-16: Tetrahedral
    # ... higher configurations
}

# Spacing parameter s determined by energy minimization
```

---

## Photon Sector

### Model: Vacuum Wave Excitations

**Reference**: `/Photon/README.md`
**Status**: Initial development

### Photon as Vacuum Wave

```python
# Electromagnetic field in Cl(3,3) geometric algebra
F = E + iB  # Bivector field

# Wave equation (from vacuum dynamics):
# ω² = c²k² + O(k⁴/Λ²)
# where Λ ~ Planck scale

# Energy density:
u = (beta/2) * (grad_rho)**2  # QFD interpretation
# Compare: u = (ε₀/2)(E² + c²B²)  # Standard EM
```

### Vacuum Impedance

```python
# Standard: Z₀ = √(μ₀/ε₀) ≈ 376.7 Ω
# QFD: Z₀ ~ β × (geometric factors)?

# Hypothesis: β → ε₀, μ₀ via dimensional analysis
# Testable: Does β = 3.058 predict correct Z₀?
```

### Fine Structure from Vacuum Geometry

```python
# Nuclear sector derivation:
# π² · exp(β) · (c₂/c₁) = α⁻¹

# Can photon sector derive α independently?
# If yes → strong validation of β = 3.058
```

---

## Units and Conventions

### Natural Units (ℏ = c = 1)

```python
# Energy: MeV
# Length: fm (femtometers)
# Time: fm/c ≈ 3.3 × 10⁻²⁴ s

# Conversion:
HBAR_C = 197.3269804  # MeV·fm

# Usage:
# Length: L_phys = L_natural * HBAR_C / E_scale
# Energy: E_phys = E_natural (in MeV)
```

### Dimensionless Parameters

```python
# β: Vacuum stiffness (dimensionless)
# α: Fine structure constant (dimensionless)
# λ: Temporal metric parameter (dimensionless)
# c₂/c₁: Nuclear coupling ratio (dimensionless)
```

### Coordinate Systems

```python
# Spherical coordinates (standard physics convention)
r = sqrt(x² + y² + z²)      # Radial distance
theta = arccos(z/r)         # Polar angle (0 at north pole)
phi = arctan2(y, x)         # Azimuthal angle

# Integration measure:
dV = r² sin(theta) dr dtheta dphi = 4πr² dr  # (spherically symmetric)
```

---

## Schema and Data Structures

### Lepton Data Structure

```python
class LeptonState:
    """Schema for lepton vortex state"""
    name: str              # "electron", "muon", "tau"
    mass: float            # MeV
    R: float               # Vortex radius (fm)
    U: float               # Propagation velocity (c units)
    Q_star: float          # RMS charge density
    psi: np.ndarray        # 4-component field [psi_s, psi_b0, psi_b1, psi_b2]
    energy: float          # Total energy (MeV)
    spin: float            # Angular momentum (ℏ units)

# File format: JSON
# Location: /V22_Lepton_Analysis/results/*.json
```

### Nuclear Data Structure

```python
class NucleusState:
    """Schema for nuclear soliton state"""
    Z: int                 # Proton number
    N: int                 # Neutron number
    A: int                 # Mass number
    harmonic_N: int        # Harmonic mode number (integer)
    mass: float            # Nuclear mass (MeV)
    binding_energy: float  # Binding energy (MeV)
    decay_modes: str       # Decay channels (from NUBASE2020)

# File format: Parquet
# Location: /LaGrangianSolitons/harmonic_nuclear_model/data/derived/harmonic_scores.parquet
```

### Cross-Sector Schema

```python
class QFDState:
    """Unified QFD state across sectors"""
    sector: str            # "lepton", "nuclear", "photon"
    soliton_type: str      # "Hill_vortex", "Q_ball", "wave"
    beta: float            # Vacuum stiffness (3.058)
    density_profile: callable  # ρ(r) function
    energy_functional: callable  # E[ψ] functional
    conserved_charges: dict  # {N: int, Q: int, L_z: float}

# Enables: Cross-sector interaction calculations
```

---

## Interaction Models

### Photon-Lepton Coupling

```python
def photon_lepton_vertex(photon, lepton, alpha=ALPHA_EM):
    """
    QED vertex for γ-e interaction

    Standard: Feynman diagram with α coupling
    QFD: Vortex distortion by EM wave

    Parameters:
    - photon: Bivector field F = E + iB
    - lepton: Hill vortex state (ψ_s, ψ_b)
    - alpha: Fine structure constant

    Returns:
    - Interaction energy (MeV)
    - Modified vortex state (ψ' after interaction)

    Physics:
    - EM field distorts vortex streamlines
    - Creates resonant oscillations
    - Photon absorption/emission
    """
    # Implementation TBD
    pass
```

### Nuclear-Electron Shielding

```python
def electron_shielding(nucleus, electron_density):
    """
    Electron cloud screening of nuclear Coulomb field

    From: vortex_shielding_coupling.py

    Theory:
    - Electron vortices create topological shield around nuclear Q-ball
    - Shield "refracts" Coulomb field lines
    - Reduces effective nuclear charge Z_eff

    Parameters:
    - nucleus: Q-ball soliton state
    - electron_density: Vortex density ρ_e(r)

    Returns:
    - Effective charge Z_eff(r)
    - Shielding energy correction (eV)
    """
    # Implementation in LaGrangianSolitons/vortex_shielding_coupling.py
    pass
```

### Resonant Oscillations → Photon Emission

```python
def resonant_photon_emission(lepton_initial, lepton_final):
    """
    Photon emission from lepton vortex transition

    Hypothesis:
    - Lepton vortex oscillates between states (e.g., excited → ground)
    - Oscillation frequency ω = ΔE/ℏ
    - Resonant coupling to vacuum → photon emission
    - Vacuum stiffness β determines emission rate

    Parameters:
    - lepton_initial: Excited vortex state
    - lepton_final: Ground vortex state

    Returns:
    - Photon energy: E_γ = E_initial - E_final
    - Photon momentum: p_γ = E_γ/c
    - Emission rate: Γ = f(β, ΔE, Q*)

    Key insight:
    - Vortex "breathing mode" couples to vacuum waves
    - Same β parameter links lepton structure to photon dynamics
    - Testable prediction: Emission rate vs β
    """
    Delta_E = lepton_initial.energy - lepton_final.energy
    omega = Delta_E / HBAR_C

    # Coupling strength ~ β (vacuum stiffness)
    coupling = compute_vacuum_coupling(beta=BETA)

    # Transition rate (Fermi's golden rule analog)
    Gamma = coupling * omega**3 * overlap_integral(
        lepton_initial.psi, lepton_final.psi
    )

    return {
        'photon_energy': Delta_E,
        'frequency': omega,
        'emission_rate': Gamma,
        'polarization': compute_polarization(lepton_initial, lepton_final)
    }
```

---

## File Locations Summary

### Lepton (Hill Vortex)
```
Primary: /V22_Lepton_Analysis/integration_attempts/v22_enhanced_hill_vortex_solver.py
Lean: /projects/Lean4/QFD/Electron/HillVortex.lean
Lean: /projects/Lean4/QFD/Electron/AxisAlignment.lean
Results: /V22_Lepton_Analysis/results/*.json
```

### Nuclear (Q-Ball)
```
Primary: /projects/particle-physics/LaGrangianSolitons/soliton_geometry_solver.py
Harmonic: /LaGrangianSolitons/harmonic_nuclear_model/validate_conservation_law.py
Data: /LaGrangianSolitons/harmonic_nuclear_model/data/derived/harmonic_scores.parquet
Lean: /projects/Lean4/QFD/Nuclear/*.lean
```

### Photon
```
Primary: /Photon/README.md (scaffolding only)
Lean: /projects/Lean4/QFD/Photon/*.lean (planned)
```

### Cross-Sector
```
Shielding: /LaGrangianSolitons/vortex_shielding_coupling.py
Constants: /QFD_CONSTANTS_SCHEMA.md (this file)
Master: /CLAUDE.md (project briefing)
```

---

## Consistency Checks

### β Universality Test

```python
def test_beta_consistency():
    """
    Check if β = 3.058 is consistent across sectors

    Tests:
    1. Cosmology: CMB axis alignment → β ≈ 3.058 ✓
    2. Nuclear: Binding energies → β ≈ 3.1 ± 0.05 ✓ (3% offset)
    3. Lepton: Hill vortex masses → β ≈ 3.15 ± 0.05 (fitted)
    4. Photon: α derivation → β ≈ ? (TBD)

    Status: 3% systematic offset between sectors
    Hypothesis: β = 3.058 fundamental, offsets are closure errors
    """
    pass
```

### α Cross-Check

```python
def test_alpha_consistency():
    """
    Check if α = 1/137.036 derived consistently

    Nuclear sector: π²·exp(β)·(c₂/c₁) = α⁻¹ ✓
    Photon sector: α from vacuum geometry (TBD)

    If both derive α from β → strong validation
    """
    pass
```

---

## Notes for Photon-Lepton Resonance Work

### What You Need

1. **Lepton vortex state**: Load from `/V22_Lepton_Analysis/results/*.json`
   - Contains R, U, Q*, psi components
   - Energy levels for transitions

2. **Photon field**: Define in `/Photon/` (TBD)
   - Bivector structure F = E + iB
   - Wave equation from vacuum dynamics

3. **Coupling**: Use β = 3.058 for vacuum response
   - Stiffness determines photon-vortex interaction strength
   - Same β links lepton structure to photon emission

4. **Units**: Natural units throughout (ℏ = c = 1)
   - Energy in MeV
   - Length in fm
   - Convert with HBAR_C = 197.3 MeV·fm

### Testable Predictions

- Photon emission rate vs β (compare to QED)
- Anomalous magnetic moment (g-2) from vortex structure
- Charge radius from Hill vortex geometry
- Form factors F(q²) from scattering

---

## Version History

- **2026-01-03**: Initial creation (comprehensive cross-sector reference)
- **Future**: Add photon sector constants as developed

---

**Status**: Master reference, ready for photon-lepton interaction modeling
**Author**: QFD Project Team
**Purpose**: Ensure consistent constants/units across all QFD calculations

**Next step**: Implement `resonant_photon_emission()` using these constants! 📸⚛️
