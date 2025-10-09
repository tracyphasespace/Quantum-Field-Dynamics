# NuclideModel - QFD Nuclear Structure Calculator

**Quantum Field Dynamics (QFD) solver for nuclear mass predictions using soliton field theory.**

Version: 1.0 (Phase 9 with AME2020 calibration)
Date: October 2025
License: MIT

---

## Overview

NuclideModel is a field-theoretic approach to nuclear structure that models nuclei as **coherent soliton configurations** rather than collections of individual nucleons. The solver computes nuclear masses by minimizing coupled field equations for charge-rich, charge-poor, and electron fields with self-consistent Coulomb, cohesion, and surface interactions.

### Key Features

- **Soliton-based nuclear model**: No particle-based assumptions (no neutrons/protons)
- **Self-consistent field equations**: Coupled charge-rich/charge-poor/electron fields
- **AME2020 calibrated parameters**: Validated against 3558 experimental nuclear masses
- **Charge asymmetry energy**: QFD-native symmetry energy (not SEMF)
- **Physics-driven calibration**: Optimized on magic numbers and shell closures
- **Virial constraint**: Ensures physical field configurations (|virial| < 0.18)

### Performance (Trial 32 Calibration)

| Mass Region | Representative Nuclei | Typical Error |
|-------------|----------------------|---------------|
| Light (A<60) | He-4, C-12, O-16, Si-28, Ca-40 | **< 1%** |
| Medium (60≤A<120) | Fe-56, Ni-62, Sn-100 | **2-3%** |
| Heavy (A≥120) | Pb-208, Au-197, U-238 | **7-9%** |

**Note**: Heavy isotope systematic underbinding indicates missing physics (surface effects, pairing, or regional parameter variation).

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/NuclideModel.git
cd NuclideModel

# Install dependencies
pip install -r requirements.txt
```

### Run Single Nucleus

```bash
# Compute O-16 using Trial 32 parameters
python src/qfd_solver.py \
  --A 16 --Z 8 \
  --grid-points 48 --iters-outer 360 \
  --param-file results/trial32_params.json \
  --emit-json --out-json O16_result.json

# Check result
jq '.E_model, .virial_abs, .physical_success' O16_result.json
```

### Run Meta-Optimization

```bash
# Optimize parameters against physics-driven calibration set
python src/qfd_metaopt_ame2020.py \
  --n-calibration 30 \
  --max-iter 100 \
  --out-json my_optimization.json
```

---

## Experimental Features 🧪

**NEW**: Advanced features with 4-10× speedup are available in `experimental/` directory:
- ✨ **Parallel meta-optimizer** (4 workers, adaptive iterations)
- ✨ **Phase 11 solver** with self-repulsion term (may fix heavy nuclei underbinding)
- ✨ **Environment-based configuration** (.env files)

⚠️ **Status**: Experimental - not yet production-ready. See `experimental/README.md` for details.

For production work, use the stable code in `src/` (below).

---

## Repository Structure

```
NuclideModel/
├── src/                               # Stable v1.0 code
│   ├── qfd_solver.py              # Phase 9 solver (stable)
│   └── qfd_metaopt_ame2020.py     # Meta-optimizer (serial)
├── experimental/                  # 🧪 v2.0-alpha features (6-10× faster!)
│   ├── qfd_solver_v11.py          # Phase 11 + self-repulsion
│   ├── qfd_metaopt_v15.py         # Parallel optimizer
│   ├── .env.example               # Environment config
│   └── README.md                  # Experimental docs
├── data/
│   ├── ame2020_system_energies.csv    # 3558 experimental masses
│   ├── stable_isotopes.csv            # 254 stable isotopes
│   └── magic_numbers.json             # Shell closure reference
├── results/
│   ├── trial32_params.json            # Best calibrated parameters
│   ├── trial32_ame2020_test.json      # Full Trial 32 validation
│   └── phase_comparison/              # Phase 9 vs Phase 10 analysis
├── examples/
│   ├── run_he4.sh                     # He-4 example
│   ├── run_pb208.sh                   # Pb-208 example
│   └── sweep_magic_numbers.py         # Magic number sweep
├── docs/
│   ├── PHYSICS_MODEL.md               # QFD field theory overview
│   ├── PARAMETERS.md                  # Parameter descriptions
│   ├── CALIBRATION_GUIDE.md           # How to recalibrate
│   └── FINDINGS.md                    # Research findings summary
├── requirements.txt
├── LICENSE
└── README.md (this file)
```

---

## Physics Model

### Field Variables

- **ψ_N**: Nuclear (nucleon) field - charge-rich/charge-poor soliton density
- **ψ_e**: Electron field - bound electron density
- **B**: Rotor field - angular momentum / deformation

### Energy Functional

```
E_total = T_N + T_e + T_rotor             (kinetic)
        + V2_N + V4_N + V6_N              (cohesion: φ²-φ⁴-φ⁶)
        + V2_e + V4_e + V6_e              (electron self-interaction)
        + V_iso                            (isospin-like charge asymmetry)
        + V_rotor                          (angular momentum penalty)
        + V_surf                           (surface tension)
        + V_coul                           (Coulomb: spectral solver)
        + V_sym                            (QFD charge asymmetry energy)
```

### Charge Asymmetry Energy (V_sym)

```
V_sym = c_sym × (N-Z)² / A^(1/3)
```

This is **NOT** the SEMF asymmetry term - it arises from QFD field surface effects for charge-imbalanced configurations.

Calibrated value: **c_sym = 25.0 MeV**

---

## Parameter Reference

### Trial 32 Parameters (Phase 9 + AME2020)

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| c_v2_base | 2.201711 | MeV | Baseline cohesion strength |
| c_v2_iso | 0.027035 | MeV | Isospin-dependent cohesion |
| c_v2_mass | -0.000205 | 1/A | Mass-dependent compounding (≈0) |
| c_v4_base | 5.282364 | MeV | Baseline quartic repulsion |
| c_v4_size | -0.085018 | MeV | Size-dependent quartic |
| alpha_e_scale | 1.007419 | - | Electron cohesion scale |
| beta_e_scale | 0.504312 | - | Electron quartic scale |
| c_sym | 25.0 | MeV | Charge asymmetry coefficient |
| kappa_rho | 0.029816 | - | Density mixing parameter |

**Loss**: 0.591 (32/34 converged, physics-driven calibration set)

---

## Usage Examples

### Example 1: Magic Number Validation

```python
# Test doubly-magic He-4
python src/qfd_solver.py --A 4 --Z 2 \
  --param-file results/trial32_params.json \
  --emit-json --out-json He4.json

# Expected: E_model ≈ -24 MeV, virial_abs < 0.03, error < 0.2%
```

### Example 2: Charge Asymmetry Test

```python
# Compare Fe isotopes (different N-Z)
for A in 54 56 57 58; do
  python src/qfd_solver.py --A $A --Z 26 \
    --param-file results/trial32_params.json \
    --out-json Fe${A}.json
done

# Expected: V_sym increases with |N-Z|
```

### Example 3: Custom Calibration

```python
# Optimize for heavy isotopes only (A > 120)
python src/qfd_metaopt_ame2020.py \
  --n-calibration 20 \
  --mass-filter 120 999 \
  --max-iter 200 \
  --out-json heavy_isotope_params.json
```

---

## Key Research Findings

### 1. Light Isotope Success (A < 60)
Magic numbers and doubly-magic nuclei show excellent agreement (< 1% error):
- He-4: +0.12%
- C-12: +0.17%
- O-16: +0.46%
- Si-28: +0.16%
- Ca-40: -0.88%

### 2. Heavy Isotope Underbinding (A > 120)
Systematic negative errors (-7% to -9%) indicate missing physics:
- Pb-208: -8.39%
- Au-197: -8.75%
- U-238: -7.71%

**Hypothesis**: Surface energy, pairing effects, or regional parameter variation needed.

### 3. Compounding Law
Trial 32 found c_v2_mass ≈ 0, indicating **no mass-dependent compounding** is optimal. Phase 10 saturation experiments confirmed heavy isotopes need MORE cohesion, not less.

### 4. Symmetry Energy
QFD charge asymmetry term (c_sym = 25 MeV) successfully captures N-Z asymmetry effects without SEMF assumptions.

---

## Limitations and Future Work

### Current Limitations

1. **Heavy isotope underbinding**: 7-9% systematic errors for A > 120
2. **No deformation**: Spherical symmetry assumed (no quadrupole moments)
3. **No explicit pairing**: Mean-field only (no BCS-like correlations)
4. **Single parameter set**: Universal parameters (not regionally optimized)

### Proposed Extensions

1. **Regional calibration**: Separate parameter sets for light/medium/heavy
2. **Surface term enhancement**: Explicit A^(2/3) surface energy
3. **Pairing energy**: Even-odd staggering term
4. **Deformation**: Non-spherical ansatz with quadrupole moments
5. **Shell model interface**: Extract single-particle levels from field solutions

---

## Citation

If you use NuclideModel in your research, please cite:

```bibtex
@software{nuclidemodel2025,
  title={NuclideModel: QFD Nuclear Structure Calculator},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/NuclideModel},
  version={1.0}
}
```

---

## Contributing

Contributions welcome! Areas of interest:
- Regional parameter optimization
- Additional physics terms (pairing, deformation)
- Performance optimization (GPU, JAX)
- Extended validation (beta decay, radii, excitation energies)

See `CONTRIBUTING.md` for guidelines.

---

## License

MIT License - see `LICENSE` file.

---

## Contact

- Issues: https://github.com/yourusername/NuclideModel/issues
- Discussions: https://github.com/yourusername/NuclideModel/discussions

---

## Acknowledgments

- **AME2020 Database**: M. Wang et al., Chinese Physics C 45 (2021)
- **IAEA Nuclear Data Services**: https://www-nds.iaea.org/amdc/
- Physics validation and calibration methodology developed October 2025
