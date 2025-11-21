# QFD Unified Schema V2.0

**Version:** 2.0.0
**Date:** 2025-11-21
**Status:** UNIFIED STANDARD

---

## Purpose

This schema consolidates all variable naming conventions, parameter definitions, and coupling coefficients across the entire QFD ecosystem:

- **Nuclear Solver** (`qfd_lib/nuclear/`)
- **Cosmology/Supernova** (`SupernovaSrc/v20/`)
- **Nuclide Prediction** (`nuclide-prediction/`)
- **Lepton Physics** (`qfd_lib/leptons/`)

**Goal:** Single source of truth for all variable names, types, and units to enable cross-project compatibility.

---

## 1. Fundamental QFD Couplings (The "DNA")

These ~15 dimensionless parameters govern all QFD physics.

### 1.1 Potential Couplings

| Canonical Name | Symbol | Type | Description | Domains |
|---------------|--------|------|-------------|---------|
| `V2` | V₂ | `float` | Quadratic potential (mass scale) | Nuclear, Lepton |
| `V4` | V₄ | `float` | Quartic potential (self-interaction) | Nuclear, Lepton |
| `V6` | V₆ | `float` | Sextic potential (stability) | Nuclear, Lepton |
| `V8` | V₈ | `float` | Octic potential (high-energy cutoff) | Nuclear, Lepton |

### 1.2 Rotor Kinetic/Potential Couplings

| Canonical Name | Symbol | Type | Description | Domains |
|---------------|--------|------|-------------|---------|
| `lambda_R1` | λᴿ¹ | `float` | Rotor kinetic (spin stiffness) | Nuclear, Lepton |
| `lambda_R2` | λᴿ² | `float` | Rotor kinetic (spin inertia) | Nuclear, Lepton |
| `lambda_R3` | λᴿ³ | `float` | Rotor potential well depth | Nuclear, Lepton |
| `lambda_R4` | λᴿ⁴ | `float` | Rotor anharmonicity | Nuclear, Lepton |

### 1.3 Interaction Couplings

| Canonical Name | Symbol | Type | Description | Domains |
|---------------|--------|------|-------------|---------|
| `k_J` | kⱼ | `float` | Universal J·A interaction | Nuclear, Cosmology |
| `k_c2` | k_c² | `float` | Charge geometry coupling | Nuclear, Lepton |
| `k_EM` | k_EM | `float` | EM kinetic coupling | All |
| `k_csr` | k_csr | `float` | Core-surface-rotor coupling | Lepton |

### 1.4 Vacuum and Gravity Couplings

| Canonical Name | Symbol | Type | Description | Domains |
|---------------|--------|------|-------------|---------|
| `xi` | ξ | `float` | Vacuum coupling (ψ_s-EM) | Cosmology |
| `g_c` | g_c | `float` | Geometric charge coupling | All |
| `eta_prime` | η' | `float` | Photon self-interaction (FDR) | Cosmology |

---

## 2. Nuclear Domain Parameters

### 2.1 Effective Nuclear Couplings (Genesis Constants)

| Canonical Name | Symbol | Type | Default | Description |
|---------------|--------|------|---------|-------------|
| `alpha` | α | `float` | 3.50 | Coulomb + J·A coupling strength |
| `beta` | β | `float` | 3.90 | Kinetic term weight |
| `gamma_e` | γₑ | `float` | 5.50 | Electron field coupling |
| `eta` | η | `float` | 0.05 | Gradient term weight |
| `kappa_time` | κ_t | `float` | 3.2 | Temporal evolution stiffness |

**Reference Values (Hydrogen):**
- `alpha_H = 3.50`
- `beta_H = 3.90`
- `gamma_e_H = 5.50`

### 2.2 Nucleus Properties

| Canonical Name | Symbol | Type | Description |
|---------------|--------|------|-------------|
| `A` | A | `int` | Mass number (nucleons) |
| `Z` | Z | `int` | Atomic number (protons) |
| `N` | N | `int` | Neutron number (A - Z) |
| `Q` | Q | `int` | Charge number |
| `N_e` | Nₑ | `int` | Number of electrons |
| `mass_amu` | m | `float` | Nuclear mass in AMU |

### 2.3 Nuclear Observables

| Canonical Name | Symbol | Type | Unit | Description |
|---------------|--------|------|------|-------------|
| `E_bind_MeV` | E_B | `float` | MeV | Binding energy |
| `E_model_MeV` | E_mod | `float` | MeV | Model prediction |
| `E_exp_MeV` | E_exp | `float` | MeV | Experimental value |
| `R_charge_fm` | R_c | `float` | fm | RMS charge radius |
| `R_matter_fm` | R_m | `float` | fm | RMS matter radius |

### 2.4 Energy Breakdown Components

| Canonical Name | Symbol | Type | Unit | Description |
|---------------|--------|------|------|-------------|
| `T_N` | T_N | `float` | MeV | Nuclear kinetic energy |
| `T_e` | T_e | `float` | MeV | Electron kinetic energy |
| `V_coul` | V_C | `float` | MeV | Coulomb energy |
| `V4_N_eff` | V₄_N | `float` | MeV | Nuclear quartic potential |
| `V4_e` | V₄_e | `float` | MeV | Electron quartic potential |
| `V_time_balance` | V_τ | `float` | MeV | Time balance energy |

### 2.5 Solver State Fields

| Canonical Name | Type | Description |
|---------------|------|-------------|
| `psi_N` | `Tensor` | Nuclear field amplitude |
| `psi_e` | `Tensor` | Electron field amplitude |
| `rho_N` | `Tensor` | Nuclear charge density |
| `rho_e` | `Tensor` | Electron charge density |
| `phi_N` | `Tensor` | Nuclear potential |
| `phi_e` | `Tensor` | Electron potential |

---

## 3. Cosmology/Supernova Domain Parameters

### 3.1 Redshift Components

| Canonical Name | Symbol | Type | Description |
|---------------|--------|------|-------------|
| `z_obs` | z_obs | `float` | Observed heliocentric redshift |
| `z_cosmo` | z_c | `float` | Cosmological redshift (tired light) |
| `z_plasma` | z_p | `float` | Plasma Veil component |
| `z_FDR` | z_F | `float` | Vacuum Sear (FDR) component |
| `z_bbh` | z_bbh | `float` | BBH gravitational redshift |
| `z_total` | z_tot | `float` | Total observed redshift |

**Formula:**
```python
(1 + z_total) = (1 + z_cosmo) * (1 + z_bbh) - 1.0
```

### 3.2 Supernova Light Curve Parameters

| Canonical Name | Symbol | Type | Unit | Description |
|---------------|--------|------|------|-------------|
| `t0` | t₀ | `float` | MJD | Explosion time |
| `ln_A` | ln(A) | `float` | — | Log amplitude scaling |
| `A_plasma` | A_p | `float` | — | Plasma opacity amplitude |
| `beta` | β | `float` | — | Opacity wavelength dependence |
| `eta_prime` | η' | `float` | — | FDR opacity (photon self-interaction) |
| `A_lens` | A_L | `float` | — | BBH lensing amplitude |

### 3.3 Physical Constants (Cosmology)

| Canonical Name | Symbol | Type | Unit | Value |
|---------------|--------|------|------|-------|
| `C_KM_S` | c | `float` | km/s | 299792.458 |
| `K_J_BASELINE` | k_J⁰ | `float` | km/s/Mpc | 70.0 |
| `H0_kms_Mpc` | H₀ | `float` | km/s/Mpc | 70.0 |
| `alpha_0` | α₀ | `float` | Mpc⁻¹ | H₀/c |

### 3.4 BBH (Binary Black Hole) Parameters

| Canonical Name | Symbol | Type | Unit | Default |
|---------------|--------|------|------|---------|
| `P_orb` | P | `float` | days | 22.0 |
| `phi_0` | φ₀ | `float` | rad | 0.0 |
| `L_peak` | L_p | `float` | erg/s | 1e43 |

---

## 4. Nuclide Prediction Domain

### 4.1 Core Compression Law Parameters

| Canonical Name | Symbol | Type | Description |
|---------------|--------|------|-------------|
| `c1` | c₁ | `float` | Surface effects coefficient (A^(2/3)) |
| `c2` | c₂ | `float` | Core compression coefficient (A) |

**Model:**
```python
Q = c1 * A**(2/3) + c2 * A
```

### 4.2 Nuclide Properties

| Canonical Name | Symbol | Type | Description |
|---------------|--------|------|-------------|
| `A` | A | `int` | Mass number |
| `Q` | Q | `int` | Charge number |
| `Stable` | — | `bool` | Stability flag (1=stable, 0=unstable) |

### 4.3 Fit Statistics

| Canonical Name | Type | Description |
|---------------|------|-------------|
| `r_squared` | `float` | R² goodness of fit |
| `rmse` | `float` | Root mean square error |
| `residuals` | `ndarray` | Q_obs - Q_pred |

---

## 5. Numerical/Solver Configuration

### 5.1 Grid Parameters

| Canonical Name | Symbol | Type | Unit | Description |
|---------------|--------|------|------|-------------|
| `N_r` | N_r | `int` | — | Radial grid points |
| `N_theta` | N_θ | `int` | — | Angular grid points |
| `R_max_fm` | R_max | `float` | fm | Maximum radial extent |
| `dr_fm` | Δr | `float` | fm | Radial spacing |
| `grid_points` | N | `int` | — | Total grid points |
| `max_radius` | R_max | `float` | fm | Maximum radius |
| `grid_dx` | Δx | `float` | fm | Grid spacing |

### 5.2 Iteration Parameters

| Canonical Name | Symbol | Type | Description |
|---------------|--------|------|-------------|
| `iters_outer` | N_out | `int` | Outer iterations |
| `iters_inner` | N_in | `int` | Inner iterations |
| `lr_N` | η_N | `float` | Nuclear learning rate |
| `lr_e` | η_e | `float` | Electron learning rate |
| `dt` | Δt | `float` | Time step (imaginary time) |

### 5.3 Convergence Criteria

| Canonical Name | Type | Default | Description |
|---------------|------|---------|-------------|
| `tol_energy_rel` | `float` | 1e-7 | Relative energy tolerance |
| `spectral_cutoff` | `float` | 0.36 | Spectral cutoff |
| `early_stop_vir` | `float` | 0.20 | Early stop virial threshold |

---

## 6. Result/Output Schema

### 6.1 Success Flags

| Canonical Name | Type | Description |
|---------------|------|-------------|
| `converged` | `bool` | Solver converged |
| `virial_ok` | `bool` | Virial constraint satisfied |
| `bound_ok` | `bool` | System is bound (E < 0) |
| `penalties_ok` | `bool` | All penalties below threshold |
| `physical_success` | `bool` | All criteria satisfied |

### 6.2 Penalty Terms

| Canonical Name | Type | Description |
|---------------|------|-------------|
| `penalty_Q` | `float` | Charge penalty |
| `penalty_B` | `float` | Baryon number penalty |
| `penalty_center` | `float` | Center-of-mass penalty |

### 6.3 Virial Metrics

| Canonical Name | Type | Description |
|---------------|--------|-------------|
| `virial` | `float` | Virial residual |
| `virial_tol_pass` | `float` | Tolerance threshold used |

---

## 7. Mapping Legacy Names

### 7.1 Nuclear Solver Mappings

| Legacy Name | Canonical Name | Notes |
|------------|---------------|-------|
| `Z` | `Z` | Keep (atomic number) |
| `charge` | `Q` | Use in JSON output |
| `electrons` | `N_e` | Use in JSON output |
| `gamma_e_target` | `gamma_e` | Genesis constant target |
| `E_model` | `E_model_MeV` | Add unit suffix |

### 7.2 Cosmology Mappings

| Legacy Name | Canonical Name | Notes |
|------------|---------------|-------|
| `k_J_correction` | `k_J` | Cosmic drag correction |
| `wavelength_nm` | `wavelength_nm` | Keep |
| `flux_jy` | `flux_Jy` | Jansky flux |

---

## 8. Python Schema Definitions

### 8.1 Core Data Classes

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class QFDCouplings:
    """Fundamental QFD coupling constants."""
    V2: float = 0.0
    V4: float = 11.0
    V6: float = 0.0
    V8: float = 0.0
    k_J: float = 70.0
    k_c2: float = 0.5
    k_EM: float = 1.0
    k_csr: float = 0.0
    xi: float = 0.0
    g_c: float = 0.985
    eta_prime: float = 0.0

@dataclass
class NuclearParams:
    """Effective nuclear coupling parameters."""
    alpha: float = 3.50
    beta: float = 3.90
    gamma_e: float = 5.50
    eta: float = 0.05
    kappa_time: float = 3.2

@dataclass
class NucleusProperties:
    """Target nucleus specification."""
    A: int  # Mass number
    Z: int  # Atomic number
    N_e: int = None  # Electrons (default = Z)
    mass_amu: float = None

    def __post_init__(self):
        if self.N_e is None:
            self.N_e = self.Z

@dataclass
class CosmologyParams:
    """Supernova/cosmology model parameters."""
    t0: float  # Explosion time (MJD)
    ln_A: float  # Log amplitude
    A_plasma: float  # Plasma opacity
    beta: float  # Opacity wavelength dependence
    eta_prime: float  # FDR opacity
    A_lens: float = 0.0  # BBH lensing amplitude
    k_J_correction: float = 0.0  # Cosmic drag correction

@dataclass
class NuclideParams:
    """Core Compression Law parameters."""
    c1: float  # Surface coefficient
    c2: float  # Core compression coefficient

@dataclass
class SolverConfig:
    """Numerical solver configuration."""
    grid_points: int = 96
    max_radius: float = 14.0
    grid_dx: float = 0.29
    iters_outer: int = 800
    iters_inner: int = 100
    tol_energy_rel: float = 1e-7
    spectral_cutoff: float = 0.36
    lr_N: float = 0.01
    lr_e: float = 0.01

@dataclass
class QFDResult:
    """Standard result schema for all QFD outputs."""
    # Core physics
    alpha: float
    beta: float
    gamma_e: float

    # Nucleus
    A: int
    Z: int
    N_e: int
    mass_amu: float

    # Numerics
    grid_points: int
    max_radius: float
    iters_outer: int

    # Results
    converged: bool
    E_model_MeV: float
    virial: float

    # Breakdown
    T_N: float
    T_e: float
    V_coul: float
    V4_N_eff: float
    V4_e: float

    # Penalties
    penalty_Q: float
    penalty_B: float
    penalty_center: float

    # Flags
    virial_ok: bool
    bound_ok: bool
    penalties_ok: bool
    physical_success: bool

    # Metadata
    timestamp: str
    solver_version: str = "v2.0"
```

### 8.2 Translation Functions

```python
def translate_legacy_params(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Translate legacy parameter names to unified schema."""
    mapping = {
        'Z': 'Z',
        'charge': 'Z',
        'electrons': 'N_e',
        'N_e': 'N_e',
        'gamma_e_target': 'gamma_e',
        'E_model': 'E_model_MeV',
    }

    result = {}
    for key, value in legacy.items():
        canonical = mapping.get(key, key)
        result[canonical] = value
    return result

def genesis_constants() -> NuclearParams:
    """Return the Genesis Constants (hydrogen reference)."""
    return NuclearParams(
        alpha=3.50,
        beta=3.90,
        gamma_e=5.50,
        eta=0.05,
        kappa_time=3.2
    )
```

---

## 9. Unit Conventions

### 9.1 Mandatory Unit Suffixes

| Quantity | Suffixes |
|----------|----------|
| Energy | `_eV`, `_keV`, `_MeV`, `_GeV`, `_J` |
| Mass | `_eV`, `_MeV`, `_GeV`, `_kg`, `_amu` |
| Length | `_fm`, `_pm`, `_nm`, `_m` |
| Distance | `_Mpc`, `_kpc`, `_pc`, `_m` |
| Velocity | `_km_s`, `_m_s`, `_c` |
| Hubble | `_kms_Mpc` |

### 9.2 Examples

```python
# CORRECT
E_bind_MeV = 28.296
mass_e_eV = 510998.95
R_charge_fm = 0.8414
H0_kms_Mpc = 70.0
L_Mpc = 100.0

# INCORRECT
energy = 28.296    # Missing unit
mass = 511000      # Missing unit
radius = 0.84      # Missing unit
```

---

## 10. File Naming Conventions

### 10.1 Configuration Files

```
qfd_unified_params.json
qfd_nuclear_genesis.json
qfd_cosmology_config.json
```

### 10.2 Output Files

```
<domain>_<purpose>_<timestamp>.<ext>

nuclear_fit_results_20251121.csv
cosmology_sn_lightcurve_20251121.json
nuclide_compression_fit.json
```

### 10.3 Reports

```
QFD_NUCLEAR_VALIDATION_REPORT.md
QFD_COSMOLOGY_HUBBLE_FIT.md
```

---

## 11. Implementation Checklist

### Phase 1: Core Unification (Immediate)

- [ ] Create `qfd_lib/core/unified_schema.py` with dataclasses
- [ ] Add translation functions for legacy compatibility
- [ ] Update v20 pipeline to use canonical names
- [ ] Update nuclide-prediction to use canonical names

### Phase 2: Observable Naming

- [ ] Add unit suffixes to all energy variables
- [ ] Add unit suffixes to all length variables
- [ ] Update JSON output formats

### Phase 3: Validation

- [ ] Create naming validator tool
- [ ] Run on all projects
- [ ] Fix violations

---

## 12. Cross-Reference: Project-Specific Usage

### v20 Supernova (Cosmology)

Primary parameters: `t0`, `ln_A`, `A_plasma`, `beta`, `eta_prime`, `A_lens`, `k_J_correction`

Redshifts: `z_obs`, `z_cosmo`, `z_bbh`, `z_total`

### Nuclide Prediction

Primary parameters: `c1`, `c2`

Observables: `A`, `Q`, `Stable`, `r_squared`, `rmse`

### Nuclear Solver

Primary parameters: `alpha`, `beta`, `gamma_e`, `eta`, `kappa_time`

Properties: `A`, `Z`, `N_e`, `mass_amu`

Results: `E_model_MeV`, `virial`, `T_N`, `T_e`, `V_coul`

---

**END OF QFD UNIFIED SCHEMA V2.0**
