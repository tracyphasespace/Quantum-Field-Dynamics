# Harmonic Halflife Predictor

**A unified geometric framework for predicting exotic nuclear decay using 18 parameters.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 Quick Start (60 seconds)

```bash
# Clone and navigate
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics/projects/nuclear-physics/harmonic_halflife_predictor

# Install
pip install -r requirements.txt

# Run everything
python run_all.py
```

**Done!** Check `results/` and `figures/` directories.

📖 **Full guide:** [QUICKSTART.md](QUICKSTART.md)

---

## What This Solves

Four previously independent nuclear phenomena, now predicted by one 18-parameter model:

| Discovery | Problem Solved | Status |
|-----------|----------------|--------|
| **Engine C** | Cluster decay energy conservation | ✅ Pythagorean N² law |
| **Engine A** | Neutron drip line location | ✅ 100% accuracy (20/20 cases) |
| **Engine B** | Fission asymmetry (80-year mystery) | ✅ Integer partitioning |
| **Engine D** | Proton drip line + topology | ✅ 96% conservation |

**Traditional approaches:** 250-400 parameters, separate models for each phenomenon
**This approach:** 18 parameters, unified geometric framework

---

## Key Results

### 1. Cluster Decay: Pythagorean Law
```
N²_parent ≈ N²_daughter + N²_cluster
```
**Example:** Ba-114 → Sn-100 + C-14 obeys 1 = 0 + 1 (perfect)

### 2. Neutron Drip Line: Geometric Failure
```
Critical tension: (c₂/c₁) × A^(1/3) > 1.701
```
**Accuracy:** 100% on experimental drip line nuclei

### 3. Fission Asymmetry: Integer Partitioning
```
If N_eff is ODD → Symmetric fission is FORBIDDEN
```
**Solves:** 80-year mystery of why U-235 fissions asymmetrically

### 4. Proton Drip Line: Coulomb-Assisted Failure
```
Critical tension: (c₂/c₁) × A^(1/3) > 0.539 (3.2× lower than neutron drip)
```
**Topology:** 96% of proton emissions preserve harmonic mode (|ΔN| ≤ 1)

---

## Generated Output

### Results (CSV files in `results/`)
- Cluster decay Pythagorean tests
- Neutron/proton drip line analyses
- Fission elongation parameters
- Nuclear geometry for 3,558 nuclei
- Halflife predictions

### Figures (PNG files in `figures/`)
- Yrast spectroscopy diagrams
- N-conservation plots
- Drip line tension analysis
- Fission neck correlation
- Proton emission validation

---

## Model Overview

**Core idea:** Nuclei are topological solitons with quantized harmonic modes

```python
BE = c₁·A^(2/3) + c₂·A + c₃·Z²/A^(1/3)

where:
  c₁(N) = c₁₀ + N·dc₁  # Surface tension
  c₂(N) = c₂₀ + N·dc₂  # Volume pressure
  c₃(N) = c₃₀ + N·dc₃  # Coulomb term
  N = harmonic quantum number
```

**Three families:**
- Family A: Balanced (c₂/c₁ = 0.26), most stable nuclei
- Family B: Surface-dominated (c₂/c₁ = 0.12), fission resistant
- Family C: Neutron-rich (c₂/c₁ = 0.20), high harmonic modes

**Total: 18 parameters** (3 families × 6 coefficients)

---

## Documentation

| File | Purpose |
|------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Complete guide with examples |
| **[NUCLEAR_UNIFICATION_MASTER.md](NUCLEAR_UNIFICATION_MASTER.md)** | Full scientific paper |
| **[SETUP.md](SETUP.md)** | Installation and environment setup |
| **[PROCESS_AND_METHODS.md](PROCESS_AND_METHODS.md)** | Technical methodology |
| **docs/ENGINE_*.md** | Individual engine discoveries |

---

## Usage Examples

### Run complete pipeline
```bash
python run_all.py
```

### Run specific engines
```bash
python scripts/cluster_decay_scanner.py     # Engine C
python scripts/neutron_drip_scanner.py      # Engine A
python scripts/fission_neck_scan.py         # Engine B
python scripts/validate_proton_engine.py    # Engine D
```

### Generate figures only
```bash
python run_all.py --figures-only
```

### Classify a nucleus
```python
from scripts.nucleus_classifier import classify_nucleus

N, family = classify_nucleus(A=238, Z=92)  # U-238
print(f"N={N}, Family={family}")  # N=2, Family=A
```

---

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
harmonic_halflife_predictor/
├── run_all.py              # One-stop pipeline runner
├── README.md               # This file
├── QUICKSTART.md           # Detailed quick start guide
├── requirements.txt        # Python dependencies
│
├── scripts/                # Validation and analysis scripts
│   ├── nucleus_classifier.py
│   ├── cluster_decay_scanner.py
│   ├── neutron_drip_scanner.py
│   ├── fission_neck_scan.py
│   ├── validate_proton_engine.py
│   └── generate_yrast_plots.py
│
├── data/                   # Nuclear data (AME2020)
├── results/                # Generated CSV files
├── figures/                # Generated plots
├── docs/                   # Detailed documentation
└── logs/                   # Execution logs
```

---

## Citation

```bibtex
@software{mcsheery2026harmonic,
  author = {McSheery, Tracy},
  title = {Harmonic Halflife Predictor: Unified Geometric Framework for Exotic Nuclear Decay},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/tracyphasespace/Quantum-Field-Dynamics/tree/main/projects/nuclear-physics/harmonic_halflife_predictor},
  note = {4-engine unified framework validated on 3558 nuclei}
}
```

See [CITATION.cff](CITATION.cff) for additional formats.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

This is a research package. For questions or collaboration:
- Open an issue: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- Read methodology: [PROCESS_AND_METHODS.md](PROCESS_AND_METHODS.md)

---

## Validation Summary

- **Nuclei analyzed:** 3,558 (AME2020 complete dataset)
- **Cluster decay:** 100% magic mode selection (N = 1 or 2)
- **Neutron drip:** 100% accuracy on highest-ratio nuclei
- **Fission asymmetry:** 4/4 cases correctly predicted
- **Proton drip:** 96.3% harmonic mode conservation
- **Parameter count:** 18 (vs. 250-400 in traditional models)

---

**🌟 Start here:** Run `python run_all.py` and check the output!

**📚 Learn more:** Read [QUICKSTART.md](QUICKSTART.md) for complete details.
