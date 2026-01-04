# Harmonic Halflife Predictor - Quick Start Guide

**One unified framework predicts all exotic nuclear decay modes using just 18 parameters.**

---

## What This Does

Predicts nuclear stability boundaries and exotic decay using geometric quantization:

| Engine | Predicts | Key Result |
|--------|----------|------------|
| **Engine A** | Neutron drip line | Critical tension ratio > 1.701 |
| **Engine B** | Fission asymmetry | Integer harmonic partitioning |
| **Engine C** | Cluster decay | Pythagorean N² conservation |
| **Engine D** | Proton drip line | Coulomb-assisted failure < 0.539 |

**Validation:** 3,558 nuclei (AME2020), 100% drip line accuracy, 96% conservation accuracy

---

## Run Everything (60 seconds)

```bash
# Clone repository
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics/projects/nuclear-physics/harmonic_halflife_predictor

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_all.py
```

**Output:**
- `results/` - All validation data (CSV files)
- `figures/` - Publication-quality plots
- `logs/` - Execution logs

---

## What Gets Generated

### Results (8 files)

1. **cluster_decay_pythagorean_test.csv** - Pythagorean conservation test
2. **neutron_drip_line_analysis.csv** - Tension ratio analysis
3. **fission_elongation_analysis.csv** - Rayleigh-Plateau instability
4. **proton_drip_line_analysis.csv** - Coulomb-assisted failure
5. **proton_emission_conservation.csv** - Harmonic mode preservation
6. **nuclear_geometry_full.csv** - Complete geometric parameters
7. **predicted_halflives_all_isotopes.csv** - All 3,558 nuclei
8. **interesting_predictions.md** - Notable predictions

### Figures (9 plots)

1. **yrast_spectral_analysis.png** - Nuclear spectroscopy (4 panels)
2. **yrast_comparison.png** - Traditional vs. harmonic
3. **halflife_prediction_validation.png** - Prediction accuracy
4. **harmonic_halflife_analysis.png** - Experimental correlations
5. **n_conservation_fission.png** - Fission conservation law
6. **neutron_drip_tension_analysis.png** - Drip line mechanics
7. **fission_neck_snap_correlation.png** - Elongation threshold
8. **proton_drip_engine_validation.png** - Proton emission topology
9. **nuclear_spectroscopy_complete.png** - 6-panel comprehensive

---

## Understanding the Model

### Core Concept

Nuclei are **topological solitons** with quantized harmonic modes:

```
BE = c₁·A^(2/3) + c₂·A + c₃·Z²/A^(1/3)
```

where coefficients vary with harmonic quantum number **N**:

```
c₁(N) = c₁₀ + N·dc₁    (surface tension)
c₂(N) = c₂₀ + N·dc₂    (volume pressure)
c₃(N) = c₃₀ + N·dc₃    (Coulomb term)
```

### Three Families

| Family | c₂/c₁ Ratio | N Range | Physics |
|--------|-------------|---------|---------|
| **A** | 0.26 | -3 to +3 | Balanced (most stable) |
| **B** | 0.12 | -3 to +3 | Surface-dominated (fission resistant) |
| **C** | 0.20 | +4 to +10 | Neutron-rich (high modes) |

**Total parameters:** 3 families × 6 coefficients = **18 parameters**

Traditional models: 250-400 parameters

---

## Key Discoveries

### 1. Cluster Decay: Pythagorean Energy Conservation

**Law:** `N²_parent ≈ N²_daughter + N²_cluster`

**Example:** Ba-114 → Sn-100 + C-14
```
N²(-1) = N²(0) + N²(+1)
   1   =   0   +   1      ✓ Perfect Pythagorean
```

**Result:** 100% of observed clusters have N = 1 or 2 (magic modes)

### 2. Neutron Drip Line: Geometric Tension Failure

**Critical condition:** `(c₂/c₁) × A^(1/3) > 1.701`

When volume pressure exceeds surface tension, neutrons "leak out."

**Accuracy:** 20/20 highest-ratio nuclei at experimental drip line (100%)

### 3. Fission Asymmetry: Integer Partitioning (80-Year Mystery Solved!)

**Breakthrough:** Fission proceeds from **excited state** with harmonic mode N_eff ≈ 7

**If N_eff is ODD → Asymmetry is MANDATORY**

```
N_eff = 7:
  Symmetric:  7 = 3.5 + 3.5  ✗ Non-integer (forbidden)
  Asymmetric: 7 = 3 + 4      ✓ Integers (allowed)
```

**Validation:** 4/4 fission cases correctly predicted (U-236, Pu-240, Cf-252, Fm-258)

### 4. Proton Drip Line: Coulomb-Assisted Failure

**Critical condition:** `(c₂/c₁) × A^(1/3) > 0.539`

**Factor 3.2× lower** than neutron drip (electrostatic repulsion assists pressure)

**Topology:** Proton emission preserves harmonic mode (|ΔN| ≤ 1 in 96% of cases)

---

## Advanced Usage

### Run Specific Engines Only

```bash
# Just cluster decay validation
python scripts/cluster_decay_scanner.py

# Just neutron drip analysis
python scripts/neutron_drip_scanner.py

# Just fission analysis
python scripts/fission_neck_scan.py

# Just proton drip validation
python scripts/validate_proton_engine.py
```

### Custom Analysis

```python
from scripts.nucleus_classifier import classify_nucleus

# Classify a nucleus
A, Z = 238, 92  # U-238
N, family = classify_nucleus(A, Z)
print(f"U-238: N={N}, Family={family}")  # Output: N=2, Family=A
```

### Generate Only Figures

```bash
python run_all.py --figures-only
```

### Skip Data Download

```bash
python run_all.py --skip-download
```

---

## File Structure (Simplified)

```
harmonic_halflife_predictor/
├── run_all.py                    ← START HERE (one-stop runner)
├── QUICKSTART.md                 ← This file
├── requirements.txt              ← pip install -r requirements.txt
│
├── scripts/
│   ├── download_ame2020.py       ← Get nuclear data
│   ├── nucleus_classifier.py     ← Core 3-family model
│   ├── cluster_decay_scanner.py  ← Engine C
│   ├── neutron_drip_scanner.py   ← Engine A
│   ├── fission_neck_scan.py      ← Engine B
│   ├── validate_proton_engine.py ← Engine D
│   └── generate_yrast_plots.py   ← Spectroscopy figures
│
├── data/
│   └── ame2020_system_energies.csv  ← Nuclear mass table
│
├── results/                      ← Generated CSV files
└── figures/                      ← Generated plots
```

**Other files:** Detailed documentation (NUCLEAR_UNIFICATION_MASTER.md), discovery reports (docs/), and technical guides (SETUP.md, PROCESS_AND_METHODS.md)

---

## Troubleshooting

### Missing dependencies
```bash
pip install pandas numpy matplotlib requests
```

### Data download fails
Manually download from: https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
Place in: `data/ame2020_system_energies.csv`

### Permission error on scripts
```bash
chmod +x scripts/*.py
```

### Plots don't display
Scripts save to `figures/` directory automatically. Check there.

---

## Citation

```bibtex
@software{mcsheery2026harmonic,
  author = {McSheery, Tracy},
  title = {Harmonic Halflife Predictor v1.0},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/tracyphasespace/Quantum-Field-Dynamics},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

---

## Next Steps

1. **Read the master paper:** `NUCLEAR_UNIFICATION_MASTER.md`
2. **Explore discoveries:** `docs/ENGINE_*.md` files
3. **Understand methods:** `PROCESS_AND_METHODS.md`
4. **Make predictions:** Modify `scripts/nucleus_classifier.py`

---

**Questions?** Open an issue at: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues

**License:** MIT (see LICENSE file)

**Authors:** Tracy McSheery, Quantum Field Dynamics Project
