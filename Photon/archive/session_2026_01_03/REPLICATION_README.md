# QFD Photon Sector: Numerical Calculations

**Purpose**: Reproduce dimensional analysis calculations for QFD hypothesis  
**Date**: 2026-01-03  
**Status**: Numerical validation only - no experimental tests

---

## What This Code Does

This directory contains Python scripts that:

1. **Numerically integrate** a Hill Vortex velocity field to calculate geometric shape factor
2. **Perform dimensional analysis** on ℏ = Γ·λ·L₀·c formula
3. **Predict a length scale** L₀ from known constants (ℏ, c, Γ, λ)
4. **Validate kinematic relations** (E = pc, k·λ = 2π) to machine precision

**What it does NOT do**:
- Derive c or ℏ from first principles
- Experimentally validate L₀ predictions
- Test against spectroscopy data
- Prove any "Theory of Everything" claims

---

## Installation

```bash
# Clone repository
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics
cd Quantum-Field-Dynamics/Photon

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- numpy, scipy, matplotlib

---

## Running the Calculations

### Quick Start: Run All Scripts
```bash
python run_all.py
```

This executes all numerical calculations in order and saves results to `results/`.

### Individual Scripts

**1. Hill Vortex Integration**
```bash
python analysis/integrate_hbar.py
```
- **Purpose**: Calculate geometric shape factor from Hill Vortex velocity field
- **Method**: Triple integral over spherical coordinates using scipy.integrate
- **Output**: Γ_vortex = 1.6919 ± 10⁻¹⁵
- **Time**: ~1 second

**2. Dimensional Audit**
```bash
python analysis/dimensional_audit.py
```
- **Purpose**: Invert ℏ = Γ·λ·L₀·c to predict L₀
- **Method**: Arithmetic with known constants
- **Output**: L₀ = 0.125 fm
- **Time**: <1 second

**3. Emergent Constants Demo**
```bash
python analysis/derive_constants.py
```
- **Purpose**: Show dimensional relationships in natural units
- **Method**: c = √(β/ρ), ℏ from angular momentum
- **Output**: c, ℏ in natural units (c=1 normalized)
- **Time**: <1 second
- **Note**: This uses natural units where c=1, so "deriving c" is circular

**4. Soliton Balance Simulation**
```bash
python analysis/soliton_balance_simulation.py
```
- **Purpose**: Validate kinematic relations (E=pc, etc.)
- **Method**: Direct calculation with schema constants
- **Output**: 7/7 tests passed (ε < 10⁻¹⁶)
- **Time**: ~1 second

---

## Expected Results

### Numerical Outputs

**File**: `results/hill_vortex_integration.txt`
```
Geometric Angular Momentum (L): 1.4794
Implied Planck Constant (ℏ = 2L): 2.9587
Ratio ℏ/c: 1.6919
Integration Error: < 1e-15
```

**File**: `results/dimensional_audit.txt`
```
Γ_vortex = 1.6919 (geometric shape factor)
L₀ = 0.125 fm (predicted length scale)
Predicted ℏ = 1.054571817e-34 J·s
Measured ℏ  = 1.054571817e-34 J·s
Relative error: 0.0 (machine precision)
```

### Validation Tests

All kinematic relations should pass with relative error < 10⁻¹⁵:
- ✅ k·λ = 2π (geometric identity)
- ✅ p = ℏk (momentum definition)
- ✅ E = ℏω (energy definition)
- ✅ E = pc (energy-momentum relation)
- ✅ ω = ck (dispersion relation)

---

## Interpreting the Results

### What Γ = 1.6919 Means
- This is the shape factor from numerical integration of Hill Vortex
- **Assumption**: Electron is modeled as Hill Vortex (not experimentally proven)
- **Calculation**: Correct given the velocity field formula
- **Status**: Numerically validated, physically speculative

### What L₀ = 0.125 fm Means
- This is the length scale predicted from ℏ = Γ·λ·L₀·c
- **Assumption**: λ = 1 AMU is correct mass scale (not derived)
- **Comparison**: Nuclear hard core ~ 0.3-0.5 fm (literature)
- **Status**: Same order of magnitude, not precise match

### What "Emergent c and ℏ" Means
- In natural units (where c=1), formulas relate β to wave speed
- **Circular reasoning**: Natural units already assume c=1
- **Limitation**: Cannot predict SI value of c without knowing ρ in kg/m³
- **Status**: Dimensional analysis, not derivation from first principles

---

## Limitations and Caveats

### Assumptions
1. Hill Vortex is correct model for electron (not proven)
2. λ = 1 AMU is correct mass scale (not derived)
3. Velocity profile is analytical approximation (not from first principles)
4. Integration domain chosen by hand (r < R)

### Circular Reasoning
- Used known ℏ to predict L₀
- Cannot claim ℏ "emerges" when we used measured ℏ as input
- This is postdiction, not prediction

### Lacking Experimental Validation
- L₀ = 0.125 fm not confirmed by experiment
- No comparison to spectroscopy data
- No tests of mechanistic resonance framework

---

## Testable Predictions (Not Yet Tested)

### 1. Nucleon Form Factor
**Prediction**: Structure at q ~ 1/L₀ = 8 fm⁻¹ (E ~ 310 MeV)
**Test**: Deep inelastic scattering at this energy
**Status**: Not yet performed

### 2. Spectral Linewidth
**Prediction**: Δω·Δt ≥ n where n ~ L₀/λ
**Test**: Ultra-short pulse laser linewidths
**Status**: Not yet performed

### 3. Stokes Shift
**Prediction**: Maximum redshift = 0.69 × E_excitation
**Test**: High-energy fluorescence spectroscopy
**Status**: Not yet performed

---

## File Structure

```
Photon/
├── README.md                           (this file)
├── requirements.txt                    (Python dependencies)
├── run_all.py                          (execute all calculations)
│
├── analysis/
│   ├── integrate_hbar.py              (Hill Vortex integration)
│   ├── dimensional_audit.py           (L₀ prediction)
│   ├── derive_constants.py            (natural units demo)
│   └── soliton_balance_simulation.py  (kinematic validation)
│
├── docs/
│   ├── EMERGENT_CONSTANTS.md          (hypothesis framework)
│   ├── MECHANISTIC_RESONANCE.md       (absorption model)
│   ├── SOLITON_MECHANISM.md           (physical narrative)
│   └── SCIENTIFIC_AUDIT_2026_01_03.md (honest assessment)
│
├── results/
│   └── (generated by run_all.py)
│
└── RESULTS.md                          (validation summary)
```

---

## Scientific Status

**What we claim**:
- ✅ Hill Vortex integration yields Γ = 1.6919 (numerically)
- ✅ Dimensional formula ℏ = Γ·λ·L₀·c predicts L₀ = 0.125 fm
- ✅ This is within order of magnitude of nuclear scales
- ✅ Kinematic relations validated to machine precision

**What we do NOT claim**:
- ❌ c or ℏ "emerge" from first principles
- ❌ L₀ is experimentally confirmed
- ❌ This is a "Theory of Everything"
- ❌ QFD replaces Standard Model

**Honest framing**: Interesting hypothesis with testable predictions, awaiting experimental validation.

---

## Citation

If you use this code, please cite:

```
QFD Photon Sector Numerical Calculations (2026)
https://github.com/tracyphasespace/Quantum-Field-Dynamics
```

And note that results are **numerical calculations only**, not experimental validations.

---

## Contact

Questions or issues: Open an issue on GitHub

---

**Last Updated**: 2026-01-03  
**Status**: Numerical calculations complete, experimental tests pending
