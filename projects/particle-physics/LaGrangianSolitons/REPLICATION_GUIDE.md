# QFD Harmonic Model - Replication Guide

**Purpose**: Enable external scientists to replicate and extend Chapter 14 results
**Date**: 2026-01-09
**Status**: Release-ready

---

## Quick Start

```bash
# Clone and setup
cd projects/particle-physics/LaGrangianSolitons
pip install numpy pandas scipy matplotlib pyarrow

# Run the key validation (Integer Ladder χ² = 873)
python src/experiments/integer_ladder_test.py \
  --scores data/derived/harmonic_scores.parquet \
  --out reports/integer_ladder
```

Expected output: `χ² = 873.47` for β⁻ decay mode (n=1424)

---

## Repository Structure

```
LaGrangianSolitons/
├── data/
│   ├── raw/                          # Source data (NUBASE2020, AME2020)
│   └── derived/
│       ├── nuclides_all.parquet      # 3,558 ground-state nuclides
│       ├── harmonic_scores.parquet   # Scored with N_hat, epsilon, family
│       └── candidates_by_A.parquet   # Null universe for exp1
├── src/
│   ├── parse_nubase.py               # NUBASE2020 parser
│   ├── parse_ame.py                  # AME2020 parser
│   ├── harmonic_model.py             # Core model: Z(A,N), epsilon, N_hat
│   ├── fit_families.py               # 3-family parameter fitting
│   ├── score_harmonics.py            # Score all nuclides
│   └── experiments/
│       ├── integer_ladder_test.py    # χ² = 873 validation
│       ├── tacoma_narrows_test.py    # Stability selector
│       └── exp1_existence.py         # Existence test (FAILS - honest)
├── reports/
│   ├── fits/
│   │   └── family_params_stable.json # Fitted parameters
│   ├── integer_ladder/
│   │   └── integer_ladder_results.json
│   └── tacoma_narrows/
└── VALIDATION_MATRIX.md              # Honest assessment of what works
```

---

## Validated Results

### 1. Integer Ladder (χ² = 873)

**Claim**: N_hat values cluster at integers, rejecting uniform distribution

**Replication**:
```bash
python src/experiments/integer_ladder_test.py --scores data/derived/harmonic_scores.parquet
```

**Expected Results**:
| Dataset | n | χ² | p-value |
|---------|---|-----|---------|
| All nuclides | 3558 | 2285.16 | 0.0 |
| β⁻ parents | 1424 | **873.47** | 3.2×10⁻¹⁸² |
| β⁺ parents | 1008 | 676.23 | 8.9×10⁻¹⁴⁰ |

**Key Finding**: Chapter 14's χ² = 873 is specifically for β⁻ decay parents.

### 2. Beta Asymmetry

**Claim**: β⁻ and β⁺ parents occupy different harmonic regimes

**Results**:
| Mode | Mean N_hat | Interpretation |
|------|------------|----------------|
| β⁻ | 11.4 | High harmonics (neutron-rich) |
| β⁺ | 5.5 | Low harmonics (proton-rich) |

### 3. Directional Selection Rules

**Claim**: Decay direction follows harmonic mode

| Mode | Prediction | Accuracy | Source |
|------|------------|----------|--------|
| β⁻ | ΔN < 0 | 99.7% | 1494/1498 |
| β⁺ | ΔN > 0 | 83.6% | 1331/1592 |

**Replication**: See `harmonic_halflife_predictor/scripts/analyze_all_decay_transitions.py`

### 4. Threshold Engines (All Pass)

| Engine | Test | Result |
|--------|------|--------|
| A: Neutron Drip | c₂/c₁ ratio threshold | 100% |
| B: Fission | Elongation ζ > 2.0 | Consistent |
| C: Cluster Decay | N² conservation | 100% |
| D: Proton Drip | c₂/c₁ ratio threshold | 96.3% |

**Replication**: See `harmonic_halflife_predictor/scripts/`

---

## What FAILS (Honest Disclosure)

### exp1_existence: AUC = 0.481

**Claim tested**: Harmonic ε predicts which (A,Z) exist
**Result**: Worse than random (0.50)
**Baseline**: Simple valley (Z ≈ 0.4A) gives AUC = 0.976

**Replication**:
```bash
python src/experiments/exp1_existence.py \
  --candidates data/derived/candidates_by_A.parquet \
  --params reports/fits/family_params_stable.json
```

**Interpretation**: The model predicts BOUNDARIES (drip lines, fission), not INTERIORS (which nuclides exist).

---

## Data Sources

### NUBASE2020
- **Citation**: F.G. Kondev et al., Chin. Phys. C45, 030001 (2021)
- **DOI**: 10.1088/1674-1137/abddae
- **Content**: 3,558 ground-state nuclides with half-lives and decay modes

### AME2020
- **Citation**: W.J. Huang et al., Chin. Phys. C45, 030002 (2021)
- **DOI**: 10.1088/1674-1137/abddb0
- **Content**: Atomic masses, binding energies, Q-values

---

## Model Parameters

### 3-Family Model (from `family_params_stable.json`)

```
Family A (N = -3 to +3):
  c1_0 = 0.9618,  dc1 = -0.0295
  c2_0 = 0.2475,  dc2 = +0.0064
  c3_0 = -2.4107, dc3 = -0.8653

Family B (N = -3 to +3):
  c1_0 = 1.4739,  dc1 = -0.0259
  c2_0 = 0.1727,  dc2 = +0.0042
  c3_0 = 0.5027,  dc3 = -0.8655

Family C (N = 4 to 10):
  c1_0 = 1.1696,  dc1 = -0.0434
  c2_0 = 0.2326,  dc2 = +0.0050
  c3_0 = -4.4672, dc3 = -0.5130
```

### Model Equation

```
Z_pred(A, N) = c1(N)·A^(2/3) + c2(N)·A + c3(N)

where:
  c1(N) = c1_0 + N·dc1
  c2(N) = c2_0 + N·dc2
  c3(N) = c3_0 + N·dc3

N_hat(A, Z) = (Z - Z_baseline) / ΔZ
epsilon = |N_hat - round(N_hat)|  ∈ [0, 0.5]
```

---

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
pyarrow>=6.0  # For parquet files
```

---

## Contact

For questions or issues with replication:
- Open an issue on the repository
- Include your exact command, Python version, and error message

---

## License

Data: NUBASE2020/AME2020 are publicly available from IAEA
Code: MIT License (or as specified in repository)

---

## What Scientists Can Extend

1. **Different parameter fits**: Try fitting to subsets (by A range, decay mode)
2. **Additional decay modes**: EC, proton emission, neutron emission
3. **Half-life predictions**: Improve the regression models
4. **Fission asymmetry**: Test N² conservation with more cases
5. **Superheavy elements**: Predict stability islands

We welcome critical review and improvements.
