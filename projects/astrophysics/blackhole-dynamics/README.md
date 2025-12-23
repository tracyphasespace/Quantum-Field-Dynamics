# QFD Binary Black Hole Dynamics

**Quantum Field Dynamics simulation of material escape from binary black hole systems**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-2.3.2+-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project simulates and visualizes how material can escape from binary black hole systems in the **Quantum Field Dynamics (QFD)** framework. Unlike General Relativity where escape is impossible, QFD treats the event horizon as an apparent potential barrier rather than a geometric singularity, allowing for rare but natural escape events.

### Key Result

Material escapes through a **sequential four-mechanism causal chain**:

```
L1 Gatekeeper (60%) → Rotation Elevator (28%) → Thermal Discriminator (3%) → Coulomb Ejector (12%)
                                ↓
                        ~1% escape probability
```

## Physics Background

### QFD vs General Relativity

| Aspect | General Relativity | QFD |
|--------|-------------------|-----|
| Event horizon | Physical boundary | Mathematical surface (potential barrier) |
| Singularity | Yes (infinite curvature) | No (continuous field) |
| Escape | Impossible (need v > c) | Natural but rare (~1%) |
| Binary effect | None | Creates L1 saddle point |
| Mechanisms | - | Four sequential mechanisms |

### The Sequential Mechanism

**Step 1: L1 Gatekeeper (~60%)**
- Binary configuration creates Lagrange point L1 (saddle point)
- Counter-pull from companion BH lowers gravitational barrier
- Opens directional "spillway" for escape

**Step 2: Rotation Elevator (~28%)**
- Disk rotation at Ω ~ 0.5 c/r_g (up to 1000 Hz)
- Centrifugal force: a_c = Ω²r lifts matter to L1 threshold
- Frame dragging provides additional boost

**Step 3: Thermal Discriminator (~3% energy, 100% trigger)**
- Maxwell-Boltzmann tail: v_th ∝ 1/√m
- Electrons escape first (43× faster than protons)
- BH acquires positive charge → activates Coulomb mechanism

**Step 4: Coulomb Ejector (~12%)**
- Electric repulsion: F = kQq/r²
- Final kick pushes positive ions over barrier
- Completes the escape trajectory

### Three Fates

Material crossing the Schwarzschild surface can:
- **~1% Escapes**: High velocity in binary COM frame → reaches infinity
- **Some % Captured**: Crosses rift but falls into companion BH
- **~99% Falls Back**: High v_local but low v_COM → returns to origin

**Critical insight**: High velocity in local frame ≠ high velocity in system frame!

## Installation

### Requirements

- Python 3.13+ (3.10+ should work)
- CUDA 12.x (optional, for GPU acceleration)
- See `requirements.txt` for full dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/blackhole-dynamics.git
cd blackhole-dynamics

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration
pip install cupy-cuda12x==13.5.1
```

## Usage

### Quick Start

```python
from rift.elliptical_orbit_eruption import run_eccentric_binary_simulation

# Simulate 100 M☉ binary with e=0.9 eccentricity
run_eccentric_binary_simulation()
```

### Key Simulations

**1. Elliptical Orbit Eruption** - Main visualization
```bash
python -m rift.elliptical_orbit_eruption
```
Generates: `validation_plots/15_eccentric_orbit_eruption.png`
- Shows orbit, separation, energy budget, and flare rate
- Demonstrates sequential mechanism contributions

**2. Parameter Space Analysis**
```bash
python -m rift.parameter_space_analysis
```
Generates: Parameter space plots showing escape vs separation, spin, temperature

**3. Mass-Dependent Ejection**
```bash
python -m rift.mass_dependent_ejection
```
Shows how lighter elements (H, He) escape preferentially

**4. Astrophysical Scaling**
```bash
python -m rift.astrophysical_scaling
```
Demonstrates mechanism contributions across mass ranges (10 M☉ to 10⁹ M☉)

## Project Structure

```
blackhole-dynamics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git exclusions
├── rift/                              # Main Python package
│   ├── __init__.py
│   ├── core_3d.py                     # Core 3D dynamics
│   ├── elliptical_orbit_eruption.py   # Main eccentric binary simulation
│   ├── parameter_space_analysis.py    # Parameter space exploration
│   ├── mass_dependent_ejection.py     # Mass-dependent escape analysis
│   ├── astrophysical_scaling.py       # Mass scaling (stellar to galactic)
│   ├── binary_rift_simulation.py      # Binary rift core simulation
│   ├── rotation_dynamics.py           # Rotational kinematics
│   └── visualization.py               # Plotting utilities
├── docs/                              # Documentation
│   ├── QFD_RIFT_COMPREHENSIVE_REFERENCE.md  # Complete physics reference
│   ├── CORRECT_RIFT_PHYSICS.md              # Physics explanation
│   ├── PARADIGM_SHIFT.md                    # GR vs QFD comparison
│   ├── PARAMETER_SPACE_RESULTS.md           # Parameter scan results
│   └── archive/                             # Historical status files
└── validation_plots/                  # Generated visualizations
```

## Key Results

### Energy Budget (100 M☉ binary at 300 r_g)

```
Reference barrier (single BH):     1.2×10¹⁵ J

Sequential contributions:
  1. L1 Gatekeeper:        7.2×10¹⁴ J  (60%)
  2. Rotation Elevator:    3.4×10¹⁴ J  (28%)
  3. Thermal Discriminator: 3.6×10¹³ J   (3%)
  4. Coulomb Ejector:      1.4×10¹⁴ J  (12%)
  ─────────────────────────────────────
  Total:                   1.2×10¹⁵ J  (≈100%)

Escape probability: ~1% (when perfectly aligned)
```

### Mass Scaling

| Mass Range | Critical Separation | Dominant Mechanisms |
|------------|-------------------|---------------------|
| 10 M☉ (stellar) | ~3,000 km | All four critical |
| 100 M☉ (stellar) | ~30,000 km | Balanced baseline |
| 10⁶ M☉ (SMBH) | ~30 million km | L1 + Rotation only |
| 10⁹ M☉ (massive SMBH) | ~0.2 AU | Pure gravitational |

**Key insight**: Electromagnetic and thermal effects dominate for stellar-mass BHs, but become negligible for supermassive BHs where pure gravity dominates.

## Documentation

See [`docs/`](docs/) directory for detailed documentation:

- **[QFD_RIFT_COMPREHENSIVE_REFERENCE.md](docs/QFD_RIFT_COMPREHENSIVE_REFERENCE.md)** - Complete physics reference with mass scaling
- **[CORRECT_RIFT_PHYSICS.md](docs/CORRECT_RIFT_PHYSICS.md)** - Detailed physics explanation
- **[PARADIGM_SHIFT.md](docs/PARADIGM_SHIFT.md)** - Why QFD predicts escape where GR cannot
- **[PARAMETER_SPACE_RESULTS.md](docs/PARAMETER_SPACE_RESULTS.md)** - Parameter exploration results

## Contributing

This is a research project. Contributions, issues, and feature requests are welcome!

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qfd_blackhole_dynamics,
  title = {QFD Binary Black Hole Dynamics},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/blackhole-dynamics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on Quantum Field Dynamics (QFD) framework
- Simulations use NumPy, SciPy, and Matplotlib
- GPU acceleration via CuPy (optional)

## Contact

For questions or collaboration: [Your contact information]

---

**Status**: ✅ Physics corrected and validated (v2.0, 2025-12-22)

**Last Updated**: 2025-12-22
