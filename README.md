# Quantum Field Dynamics (QFD)

**A Unified Physical Framework: One measured input (alpha = 1/137) derives the rest.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lean 4](https://img.shields.io/badge/Verified-Lean%204-1100%2B_Theorems-green)](projects/Lean4)
[![Python 3.10+](https://img.shields.io/badge/Code-Python%203.10-blue)](projects/)

---

## Quick Start

For a streamlined, researcher-friendly experience with runnable validation scripts:

**[QFD-Universe](https://github.com/tracyphasespace/QFD-Universe)** — Curated public repo with sector-organized validation, 3-command replication, and clean documentation.

```bash
git clone https://github.com/tracyphasespace/QFD-Universe.git
cd QFD-Universe
pip install -r requirements.txt
python run_all.py
```

This repository (`Quantum-Field-Dynamics`) is the full working codebase — formal proofs, simulations, modeling scripts, visualizations, and active research.

---

## Interactive Visualizations

Explore QFD concepts in your browser — no installation required:

### Core
- **[QFD Engine](https://tracyphasespace.github.io/QFD-Universe/visualizations/qfd_engine.html)** — 21 interdependent parameters from vacuum stiffness beta
- **[Nuclear Resonance](https://tracyphasespace.github.io/QFD-Universe/visualizations/nucleus_resonance.html)** — 3D nuclear shell structure

### Field Theory
- [QFD Primer](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/qfd-primer.html) — Start here
- [QFD Advanced](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/qfd-advanced.html) — Complex field interactions
- [Vortex Dynamics](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/vortex-dynamics.html) — Topological structures
- [Soliton Canon](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/SolitonCanonPhoton.html) — 3D photon soliton
- [Neutrino](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/Neutrino.html) — Neutrino field dynamics
- [Stability Vortex](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/stability-vortex-sphere.html) — Vortex stability sphere
- [Poynting Vector](https://tracyphasespace.github.io/QFD-Universe/visualizations/field-theory/PoyntingVector4.html) — Energy flow

### Particle Physics
- [Composite Solitons](https://tracyphasespace.github.io/QFD-Universe/visualizations/particle-physics/QFD_Composite_Solitons.html) — Particle structure
- [Lepton Analysis](https://tracyphasespace.github.io/QFD-Universe/visualizations/particle-physics/leptons-analysis.html) — Mass spectrum
- [Nuclide Table](https://tracyphasespace.github.io/QFD-Universe/visualizations/particle-physics/nuclide-table.html) — Nuclear chart

### Astrophysics
- [Black Hole Dynamics](https://tracyphasespace.github.io/QFD-Universe/visualizations/astrophysics/blackhole-dynamics.html) — Escape mechanism simulation
- [Supernova](https://tracyphasespace.github.io/QFD-Universe/visualizations/astrophysics/SuperNova.html) — Type Ia SNe analysis
- [Redshift Analysis](https://tracyphasespace.github.io/QFD-Universe/visualizations/astrophysics/redshift-analysis.html) — Cosmological redshift
- [Black Hole 3D](https://tracyphasespace.github.io/QFD-Universe/visualizations/astrophysics/BlackHole_3D_Star_Generator.html) — Star generator

### Nuclear Physics
- [Nuclear Scaling v4](https://tracyphasespace.github.io/QFD-Universe/visualizations/nuclear-physics/nuclear-scaling-mixture-v4.html) — Binding energy systematics

---

## Key Results

### Lepton Sector
| Result | Method | Accuracy |
|--------|--------|----------|
| Electron, muon, tau masses | Golden Loop: alpha -> beta -> Hill vortex geometry | chi^2 < 10^-11 |
| Electron g-2 anomaly | Vacuum polarization from vortex surface/bulk ratio | 0.001% error |
| Koide relation Q = 2/3 | Geometric phase angle projection | Exact |

**Scripts**: [`V22_Lepton_Analysis/`](V22_Lepton_Analysis/), [`Photon/`](Photon/)

### Nuclear Sector
| Result | Method | Accuracy |
|--------|--------|----------|
| 3,558 nuclear masses | Coherent soliton solver (AME2020) | <1% light nuclei |
| 5,842 isotope scaling | Q(A) = c1*A^(2/3) + c2*A | R^2 = 0.98 |

**Scripts**: [`projects/particle-physics/nuclear-soliton-solver/`](projects/particle-physics/nuclear-soliton-solver/), [`projects/particle-physics/nuclide-prediction/`](projects/particle-physics/nuclide-prediction/)

### Cosmology Sector
| Result | Method | Accuracy |
|--------|--------|----------|
| SNe Ia distance-redshift | QFD photon scattering (1,829 SNe) | chi^2/nu = 0.939 |
| Full transparency pipeline | Raw photometry -> cosmology fit | Reproducible |
| Black hole escape | 4-mechanism chain simulation | ~1% escape rate |

**Scripts**: [`V22_Supernova_Analysis/`](V22_Supernova_Analysis/), [`projects/astrophysics/`](projects/astrophysics/)

### Cross-Scale Validation
| Result | Method | Accuracy |
|--------|--------|----------|
| Golden Loop pipeline | alpha -> beta -> (e, mu, tau) across 10 realms | chi^2 < 10^-9 |

**Scripts**: [`projects/astrophysics/qfd_10_realms_pipeline/`](projects/astrophysics/qfd_10_realms_pipeline/)

---

## Simulations and Modeling Scripts

### Particle Physics — [`projects/particle-physics/`](projects/particle-physics/)
- **Nuclear Soliton Solver** — Models nuclei as coherent soliton configurations. 9-parameter physics-driven calibration against AME2020.
- **Nuclide Prediction** — Universal 2-parameter scaling law across 5,842 isotopes. Includes decay mode analysis and superheavy element predictions (Z=119-126).
- **Lepton Mass Spectrum** — MCMC Bayesian parameter estimation for the 3-parameter Hill vortex lepton model.
- **Lepton Isomers** — Investigation of lepton internal structure through isomer concepts.
- **LaGrangian Solitons** — Advanced soliton construction via Lagrangian formalism.

### Astrophysics — [`projects/astrophysics/`](projects/astrophysics/)
- **Black Hole Dynamics** — Binary black hole simulation with 4-mechanism escape chain (L1 Gatekeeper, Rotation Elevator, Thermal Discriminator, Coulomb Ejector). Optional GPU acceleration.
- **QFD Supernova V22** — Complete transparency pipeline: raw DES-SN5YR photometry -> light curves -> distances -> cosmology. Two paths: quick (30 min) and full (3-4 hours).
- **10 Realms Pipeline** — Cross-sector validation from CMB through isotopes. Validates beta universality across all physical scales.
- **SNe Model Comparison** — Head-to-head comparison of 4 distance-redshift models against Pantheon+ data.
- **Redshift Analysis** — Hubble constant validation and redshift calibration tools.

### Field Theory — [`projects/field-theory/`](projects/field-theory/)
- **Photons & Solitons** — 19-script validation suite covering g-2 prediction, CMB polarization, core compression, and beta tension analysis.
- **Poynting Vectors** — Energy flow analysis in the QFD framework.
- **Trilemma Toolkit** — Resonant atom model and lepton isomer solver.

### Photon Sector — [`Photon/`](Photon/)
- **g-2 Prediction Suite** — 27 analysis scripts validating the anomalous magnetic moment from vortex geometry. Predicts g-2 to 0.45% accuracy without free parameters.
- Entry point: `python Photon/run_all.py`

### Simulation Scripts — [`simulation/`](simulation/)
- **hbar Derivations** — Deriving Planck's constant from Clifford algebra topology and parallel computation.

---

## Formal Proofs (Lean 4)

**Location**: [`projects/Lean4/`](projects/Lean4/)

1,100+ formally verified theorems proving:
- Spacetime emergence from Clifford algebra Cl(3,3)
- Charge quantization from topological boundary conditions
- CMB axis alignment (manuscript-ready cosmology proofs)
- Lepton mass spectrum geometric derivation
- Nuclear binding from vacuum stiffness
- Koide relation Q = 2/3

```bash
cd projects/Lean4
lake build QFD.SpectralGap
lake build QFD.EmergentAlgebra
```

**Build sequentially** — never run parallel `lake build` commands.

---

## Repository Structure

```
Quantum-Field-Dynamics/
├── README.md                          # This file
├── CITATION.cff                       # How to cite
├── LICENSE                            # MIT
│
├── Photon/                            # Photon sector: g-2 prediction + validation
│   ├── run_all.py                     # Run all 27 validation scripts
│   └── analysis/                      # Individual validation scripts
│
├── V22_Lepton_Analysis/               # Lepton masses via Golden Loop (alpha -> beta)
├── V22_Supernova_Analysis/            # Cosmology: SNe Ia fit (chi^2/nu = 0.939)
├── V22_Nuclear_Analysis/              # Nuclear sector analysis
│
├── projects/
│   ├── Lean4/                         # 1,100+ formal proofs (Lean 4)
│   ├── particle-physics/
│   │   ├── nuclear-soliton-solver/    # 3,558 nuclear masses from soliton model
│   │   ├── nuclide-prediction/        # 2-param universal scaling (R^2=0.98)
│   │   ├── lepton-mass-spectrum/      # MCMC Bayesian estimation
│   │   ├── lepton-isomers/            # Internal structure hypothesis
│   │   └── LaGrangianSolitons/        # Lagrangian soliton formalism
│   ├── astrophysics/
│   │   ├── blackhole-dynamics/        # Binary BH escape simulation
│   │   ├── qfd-sn-v22/               # Full transparency SN pipeline
│   │   ├── qfd_10_realms_pipeline/    # Cross-scale golden loop
│   │   ├── sne-model-comparison/      # 4-model head-to-head comparison
│   │   └── redshift-analysis/         # Hubble constant validation
│   └── field-theory/
│       ├── Photons_&_Solitons/        # CMB, g-2, beta tension analysis
│       ├── poynting-vectors/          # Energy flow
│       └── trilemma-toolkit/          # Resonant atom model
│
├── simulation/                        # hbar derivation scripts
├── visualizations/                    # 22 interactive HTML demos
├── data/                              # Reference datasets
├── schema/                            # QFD schema definitions
└── complete_energy_functional/        # MCMC beta-convergence investigation
```

---

## Epistemic Honesty

This framework is mathematically rigorous (1,100 Lean proofs) but empirically unproven at the level required for acceptance.

**What the proofs establish**: Internal consistency — IF the framework is correct, THEN it is self-consistent.

**What they don't establish**: That the framework describes nature. Fitting 3 masses with 3 parameters is necessary but not sufficient.

**The acid test**: The g-2 prediction (0.001% accuracy on an independent observable) is the strongest evidence. Cross-sector beta consistency across lepton, nuclear, and cosmological scales is the second.

**What's needed**: Independent predictions tested against new data — charge radius, form factors, beta derived from first principles rather than fitted.

---

## For AI/LLM Tools

Machine-readable indexes for automated discovery ([llmstxt.org](https://llmstxt.org/) convention):

| Resource | Raw URL |
|----------|---------|
| llms.txt | https://raw.githubusercontent.com/tracyphasespace/Quantum-Field-Dynamics/main/llms.txt |
| Lean proof index | https://raw.githubusercontent.com/tracyphasespace/Quantum-Field-Dynamics/main/LEAN_PROOF_INDEX.txt |
| Solver index | https://raw.githubusercontent.com/tracyphasespace/Quantum-Field-Dynamics/main/SOLVER_INDEX.txt |
| robots.txt | https://raw.githubusercontent.com/tracyphasespace/Quantum-Field-Dynamics/main/robots.txt |

**Curated repo**: https://raw.githubusercontent.com/tracyphasespace/QFD-Universe/main/llms.txt

---

## Related Repositories

- **[QFD-Universe](https://github.com/tracyphasespace/QFD-Universe)** — Curated researcher-facing repo with sector-organized validation scripts, interactive visualizations, and formal proofs. Start here for replication.

---

## Contributing

- **Mathematicians**: Check the Lean 4 proofs for topological robustness
- **Physicists**: Run the solvers and stress-test the Hamiltonian parameters
- **Data Scientists**: Audit the Bayesian models in the supernova analysis

---

## License

MIT License. Please cite:

McSheery, T. (2025). *Quantum Field Dynamics: A Dynamic View of a Steady State Universe.* GitHub.

See [CITATION.cff](CITATION.cff) for structured citation data.
