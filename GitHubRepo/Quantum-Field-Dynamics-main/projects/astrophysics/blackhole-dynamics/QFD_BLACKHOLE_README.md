# QFD Black Hole Dynamics

**Version:** 1.1 (Unified Astrophysical Emitter)
**Status:** ✅ Production Ready
**Branch:** qfd-blackhole/rift-mechanism
**Date:** 2025-10-01

---

## Quick Start

### Run Validation Tests

```bash
cd /path/to/blackhole-dynamics
python test_qfd_blackhole.py
```

**Expected Output:**
```
✓ ALL TESTS PASSED (10/10 categories)
Total execution time: ~4-5 seconds
```

### Basic Usage Example

```python
from qfd_blackhole import (
    QFDBlackHoleSoliton,
    BinaryBlackHoleSystem,
    calculate_jet_total_redshift
)
import numpy as np

# Create binary black hole system
bh1 = QFDBlackHoleSoliton(mass=10.0, soliton_radius=2.0)
bh2 = QFDBlackHoleSoliton(mass=5.0, soliton_radius=1.5)
system = BinaryBlackHoleSystem(bh1, bh2, separation=20.0)

# Calculate observable jet redshift (all 5 mechanisms)
result = calculate_jet_total_redshift(
    jet_position=np.array([15.0, 0, 0]),
    jet_velocity=np.array([0.3, 0, 0]),      # 0.3c
    wavelength_nm=656.3,                      # H-alpha
    time_since_ejection_days=10.0,
    jet_flux_erg_cm2_s=1e12,
    distance_from_bh_cm=1e15,
    bh_system=system,
    observer_distance_Mpc=100.0
)

print(f"Total redshift: z = {result['z_total']:.6f}")
print(f"Dominant mechanism: Gravitational ({result['contributions']['gravitational_percent']:.1f}%)")
```

---

## What This Is

**QFD Black Hole Dynamics** is a complete, production-ready implementation of black holes according to the QFD (Quantum Field Dynamics) Prime Directive.

### Three Core Mechanisms

1. **Deformable Soliton Surface & Gravitational Rift**
   - Black holes are finite-density solitons (NO singularities)
   - Surface deforms under tidal forces (NOT rigid event horizon)
   - Rift forms at L1 Lagrange point between binary BHs
   - Dynamic escape channel for matter ejection

2. **Stratified Ejection Cascade**
   - Matter stratified by binding energy inside BH
   - Sequential ejection: Leptons → Hydrogen → Helium → Heavy
   - Mass-selective nozzle effect
   - Information conservation (matter in = matter out)

3. **Tidal Torque & Angular Momentum Generation**
   - Differential gravity from companion BH imparts torque
   - Generates angular momentum in ejected jets
   - Seeds galactic rotation
   - Black hole recoil from momentum conservation

### NEW: Unified Astrophysical Emitter (v1.1)

Integrates **all five QFD redshift mechanisms** for observable jets:

1. **Gravitational Redshift** - Escape from potential well
2. **Relativistic Doppler** - High-velocity jet motion
3. **Plasma Veil** - Wavelength-dependent scattering (β = 0.8)
4. **Vacuum Sear** - Flux-dependent near-source effect (γ = 1.0)
5. **Cosmological Baseline** - QFD tired light to observer

**Multiplicative Combination:**
```
(1 + z_total) = (1+z_grav) × (1+z_doppler) × (1+z_plasma) × (1+z_FDR) × (1+z_cosmo)
```

---

## Files in This Directory

### QFD Implementation (USE THESE)

**Core Code:**
- `qfd_blackhole.py` (1100+ lines) - Complete QFD physics implementation
- `test_qfd_blackhole.py` (900+ lines) - Comprehensive validation suite

**Documentation:**
- `QFD_BLACKHOLE_IMPLEMENTATION.md` (1300+ lines) - Full technical documentation
- `QFD_BLACKHOLE_SUMMARY.md` (500+ lines) - Executive summary with results
- `QFD_BLACKHOLE_README.md` (this file) - Quick start guide

### Legacy Code (IGNORE FOR QFD WORK)

- `main.py`, `simulation.py`, `core.py`, `visualization.py` - Old particle simulation
- `README.md` - Legacy documentation (CUDA-based, not QFD)
- `config.py` - Legacy configuration

**Note:** The legacy code uses different physics (generic scalar fields, not QFD). For QFD black holes, use `qfd_blackhole.py` only.

---

## Requirements

**Python:** 3.10+

**Dependencies:**
```bash
pip install numpy scipy matplotlib
```

**For integration with supernova physics:**
Ensure you have access to:
```
../redshift-analysis/RedShift-Enhanced/qfd_lib/
```

This provides:
- `qfd_redshift.py` - Cosmological baseline
- `qfd_supernova.py` - Plasma Veil + Vacuum Sear

---

## Validation Results

### ✓ ALL TESTS PASSED (10/10 categories)

1. **Soliton Structure** - No singularities, Φ(0) finite
2. **Rift Mechanism** - L1 point found, barrier decreases with separation
3. **Stratified Ejection** - Leptons escape first, correct binding hierarchy
4. **Tidal Torque** - Angular momentum grows, torque verified
5. **QFD Constraints** - All Prime Directive requirements met
6. **Conservation Laws** - Mass and momentum conserved
7. **Edge Cases** - Numerically stable for all parameters
8. **Performance** - 32M potential evaluations/second
9. **Jet Redshift (3 mechanisms)** - Grav + Doppler + Cosmo validated
10. **Unified Model (5 mechanisms)** - Full integration with near-source effects

### Typical Results (100 Mpc observer, 0.3c jet)

```
z_gravitational:  4.291  (83.4% contribution)
z_doppler:        0.363  (15.5% contribution)
z_cosmological:   0.024  (1.2% contribution)
z_plasma:         ~1e-5  (negligible at distance)
z_FDR:            ~1e-9  (negligible at distance)
z_total:          6.381
```

---

## Key Functions

### Basic Black Hole Creation

```python
from qfd_blackhole import QFDBlackHoleSoliton

# Single black hole
bh = QFDBlackHoleSoliton(
    mass=10.0,              # Solar masses
    soliton_radius=2.0,     # Soliton core radius
    position=[0, 0, 0]      # 3D position
)

# Evaluate potential
r = 5.0
phi = bh.potential(r)      # Returns finite value (not -∞)
```

### Binary System & Rift

```python
from qfd_blackhole import BinaryBlackHoleSystem

system = BinaryBlackHoleSystem(bh1, bh2, separation=20.0)

# Rift properties
print(f"L1 point: {system.L1_point}")
print(f"Barrier: {system.rift_barrier_height()}")
print(f"Width: {system.rift_width()}")
```

### Stratified Plasma

```python
from qfd_blackhole import StratifiedPlasma

plasma = StratifiedPlasma(
    total_mass=1.0,
    composition={'leptons': 0.001, 'hydrogen': 0.7,
                 'helium': 0.28, 'heavy': 0.019}
)

# What escapes at given barrier?
sequence = plasma.ejection_sequence(barrier_energy=-2.0)
```

### Jet Redshift (3 Mechanisms)

```python
from qfd_blackhole import calculate_jet_redshift

z_total, z_grav, z_doppler, z_cosmo = calculate_jet_redshift(
    jet_position=np.array([15.0, 0, 0]),
    jet_velocity=np.array([0.3, 0, 0]),
    bh_system=system,
    observer_distance_Mpc=100.0
)
```

### Unified Emitter (5 Mechanisms)

```python
from qfd_blackhole import calculate_jet_total_redshift

result = calculate_jet_total_redshift(
    jet_position=np.array([15.0, 0, 0]),
    jet_velocity=np.array([0.3, 0, 0]),
    wavelength_nm=656.3,
    time_since_ejection_days=10.0,
    jet_flux_erg_cm2_s=1e12,
    distance_from_bh_cm=1e15,
    bh_system=system,
    observer_distance_Mpc=100.0
)

# Access all components
print(result['z_total'])
print(result['z_gravitational'])
print(result['contributions'])  # Percentage breakdown
```

---

## QFD Prime Directive Compliance

### ✓ Required Features (All Implemented)

- ✅ **Finite density everywhere** - Φ(r=0) = -10.0 (not -∞)
- ✅ **Deformable surface** - Tidal Love number k_tide > 0
- ✅ **Information conservation** - Mass tracking in ejection
- ✅ **Rift mechanism** - L1 point escape channel

### ✗ Forbidden Concepts (All Avoided)

- ❌ **NO singularities** - Soliton has finite core density
- ❌ **NO one-way horizon** - Surface is deformable
- ❌ **NO information loss** - Matter re-ejected via Rift
- ❌ **NO accretion-only jets** - Rift is primary mechanism

---

## Observable Predictions

### 1. Binary Black Hole Mergers (LIGO/Virgo + EM)

- **Pre-merger brightening** as Rift barrier lowers
- **Stratified spectral lines** (H-alpha before He lines)
- **Total ejected mass** ~ 0.1-1% of M_BH
- **Angular momentum** in jets correlates with binary parameters

### 2. AGN Jets

- **Jet L correlation:** L_jet ∝ M₁ × M₂ / D² × t_eject
- **Wavelength-dependent redshift** (Plasma Veil: β = 0.8)
- **Flux-dependent redshift** (Vacuum Sear: γ = 1.0)
- **Spectroscopic information recovery** from interior composition

### 3. Galactic Dynamics

- **Rotation seeding** from central SMBH binary
- **Bulge rotation speed** ∝ √(L_ejected / R_bulge)
- **BH mass - rotation correlations**

---

## Version History

**v1.0** (Commits: 915ecad, a797cb1, 19f6c4e)
- Three core mechanisms (Rift, Ejection, Torque)
- 8 test categories
- Complete documentation

**v1.1** (Commit: 63bb6e4, ca1b7a9) ✅ CURRENT
- Added unified astrophysical emitter model
- 5 redshift mechanisms (grav + Doppler + plasma + FDR + cosmo)
- 10 test categories (added Tests 9 & 10)
- Cross-module integration (blackhole + redshift + supernova)

---

## Git Status

**Branch:** `qfd-blackhole/rift-mechanism`
**Commits:** 5 total (915ecad → ca1b7a9)
**Status:** Clean, all changes committed

**To merge to main:**
```bash
git checkout main
git merge qfd-blackhole/rift-mechanism
```

---

## For AI Continuation After Reboot

**What was built:** Complete QFD black hole implementation with unified redshift model

**Key files to review:**
1. `QFD_BLACKHOLE_SUMMARY.md` - Quick overview with results
2. `QFD_BLACKHOLE_IMPLEMENTATION.md` - Full technical details
3. `qfd_blackhole.py` - Source code (1100+ lines)
4. `test_qfd_blackhole.py` - Run this first to verify

**Status:** ✅ Complete and validated (10/10 tests passing)

**Next tasks (if needed):**
- Merge to main branch
- Generate visualization plots
- Apply to observational AGN data
- Extend to supermassive binaries (M ~ 10⁶-10⁹ M☉)

**Integration points:**
- Imports `qfd_redshift.calculate_redshift()` from `../redshift-analysis/RedShift-Enhanced/qfd_lib/`
- Imports `qfd_supernova.{calculate_z_plasma, calculate_z_FDR}` from same location
- All three modules (blackhole, redshift, supernova) use multiplicative redshift formula

**Performance:** Fully optimized NumPy vectorization, 32M potential evaluations/second

---

## Contact / References

**QFD Framework Documentation:**
- Baseline redshift: `../redshift-analysis/RedShift-Enhanced/`
- Supernova physics: `../redshift-analysis/RedShift-Enhanced/qfd_lib/qfd_supernova.py`
- Black hole dynamics: This directory

**Prime Directives:**
- Black Hole: See commit message 915ecad or `QFD_BLACKHOLE_IMPLEMENTATION.md`
- Redshift: See `../redshift-analysis/` documentation
- Supernova: See `../redshift-analysis/RedShift-Enhanced/QFD_SUPERNOVA_IMPLEMENTATION.md`

---

**Last Updated:** 2025-10-01
**Version:** 1.1
**Status:** ✅ Production Ready
