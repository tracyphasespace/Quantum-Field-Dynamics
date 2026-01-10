# QFD Harmonic Nuclear Model - Validation Package

**Purpose**: Replicate Chapter 14 "The Geometry of Existence" results
**Date**: 2026-01-09

---

## Quick Validation

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib pyarrow

# Run Integer Ladder test (χ² = 873)
python scripts/integer_ladder_test.py --scores data/harmonic_scores.parquet --out results/

# Run decay transition analysis (99.7% β⁻ accuracy)
python scripts/analyze_all_decay_transitions.py
```

---

## Validated Claims (Chapter 14)

| Claim | Expected Result | Script |
|-------|-----------------|--------|
| Integer Ladder | χ² = 873.47 (β⁻, n=1424) | `integer_ladder_test.py` |
| Beta ΔN < 0 | 99.7% accuracy | `analyze_all_decay_transitions.py` |
| Fission channels | 65 verified | `validate_fission_pythagorean.py` |
| Cluster decay | 10 channels tested | `results/cluster_decay_pythagorean_test.csv` |

---

## Files

```
harmonic_model/
├── src/
│   └── harmonic_model.py        # Core model: Z(A,N), epsilon, N_hat
├── scripts/
│   ├── integer_ladder_test.py   # χ² = 873 validation
│   ├── analyze_all_decay_transitions.py  # β± selection rules
│   ├── nucleus_classifier.py    # 3-family classifier
│   └── validate_fission_pythagorean.py   # Fission N conservation
├── data/
│   ├── harmonic_scores.parquet  # 3,558 nuclides scored
│   ├── ame2020_system_energies.csv
│   └── family_params_stable.json # Model parameters
└── results/
    ├── integer_ladder_results.json
    └── cluster_decay_pythagorean_test.csv
```

---

## Model Parameters (3-Family)

```
Family A (N = -3 to +3): c1_0=0.9618, c2_0=0.2475, c3_0=-2.4107
Family B (N = -3 to +3): c1_0=1.4739, c2_0=0.1727, c3_0=0.5027
Family C (N = 4 to 10):  c1_0=1.1696, c2_0=0.2326, c3_0=-4.4672

Z_pred(A,N) = c1(N)·A^(2/3) + c2(N)·A + c3(N)
```

---

## Data Sources

- NUBASE2020: F.G. Kondev et al., Chin. Phys. C45, 030001 (2021)
- AME2020: W.J. Huang et al., Chin. Phys. C45, 030002 (2021)
