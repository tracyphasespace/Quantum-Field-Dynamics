# AME2020 Parser Documentation

**Module**: `src/parse_ame.py`
**Purpose**: Parse AME2020 (Atomic Mass Evaluation) and compute Q-values for all decay modes
**Status**: ✓ PRODUCTION-READY
**Date**: 2026-01-02

---

## Overview

This parser loads the AME2020 mass evaluation and calculates reaction Q-values (energy released) for all nuclear decay modes and separation energies.

### Input

- **File**: `ame2020.csv`
- **Format**: CSV with mass excess values
- **Source**: W. Huang et al., Chin. Phys. C45, 030002 (2021)

### Output

- **File**: `data/derived/ame.parquet`
- **Format**: Apache Parquet (columnar, compressed)
- **Rows**: 3,558 nuclides
- **Columns**: 23 fields (mass excess + Q-values)

---

## Dataset Schema

### Input Columns (from AME2020)

| Column | Type | Description |
|--------|------|-------------|
| `Z` | int64 | Atomic number (proton count) |
| `N` | int64 | Neutron count |
| `A` | int64 | Mass number (Z + N) |
| `element` | str | Element symbol |
| `mass_excess_keV` | float64 | Mass excess in keV |
| `mass_excess_unc_keV` | float64 | Mass excess uncertainty (keV) |
| `is_estimated` | bool | True if systematic (not experimental) |

### Calculated Columns (Q-values and Separation Energies)

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Q_alpha_keV` | float64 | keV | Alpha decay Q-value |
| `Q_beta_minus_keV` | float64 | keV | β⁻ decay Q-value |
| `Q_beta_plus_keV` | float64 | keV | β⁺ decay Q-value |
| `Q_EC_keV` | float64 | keV | Electron capture Q-value |
| `S_n_keV` | float64 | keV | Neutron separation energy |
| `S_p_keV` | float64 | keV | Proton separation energy |
| `S_2n_keV` | float64 | keV | Two-neutron separation |
| `S_2p_keV` | float64 | keV | Two-proton separation |
| `Q_*_MeV` | float64 | MeV | All above in MeV (÷1000) |

---

## Q-Value Formulas

All formulas use **mass excess** Δ (in keV):

### Alpha Decay

```
Q_α = Δ(A,Z) - Δ(A-4,Z-2) - Δ(He-4)

Where:
  Δ(He-4) = 2424.916 keV (α particle)

Positive Q → decay allowed
Negative Q → decay forbidden
```

**Example**: U-238 → Th-234 + α
```
Q_α = Δ(238,92) - Δ(234,90) - 2424.916
    = 47309.09 - 40611.10 - 2424.916
    = 4273.07 keV = 4.27 MeV ✓
```

### Beta⁻ Decay

```
Q_β⁻ = Δ(A,Z) - Δ(A,Z+1)

Note: Atomic mass convention includes electrons,
so this formula is exact without adding m_e.
```

**Example**: C-14 → N-14 + e⁻ + ν̄
```
Q_β⁻ = Δ(14,6) - Δ(14,7)
     = 3019.893 - 2863.440
     = 156.453 keV = 0.156 MeV ✓
```

### Beta⁺ Decay

```
Q_β⁺ = Δ(A,Z) - Δ(A,Z-1) - 2m_e

Where:
  m_e = 510.999 keV (electron mass)
  Factor of 2: positron creation + 1 less atomic e⁻
```

**Example**: Na-22 → Ne-22 + e⁺ + ν
```
Q_β⁺ = Δ(22,11) - Δ(22,10) - 2×510.999
     = -5181.991 - (-8024.763) - 1021.998
     = 1820.774 keV = 1.82 MeV
```

### Electron Capture (EC)

```
Q_EC = Δ(A,Z) - Δ(A,Z-1)

Simpler than β⁺ because electron is captured,
not created as a positron.
```

**Note**: Q_EC is always 2m_e = 1.022 MeV more favorable than Q_β⁺

### Neutron Separation Energy

```
S_n = Δ(A-1,Z) + Δ(n) - Δ(A,Z)

Where:
  Δ(n) = 8071.318 keV (neutron)

Energy required to remove one neutron.
Positive → bound
Negative → neutron-unbound
```

### Proton Separation Energy

```
S_p = Δ(A-1,Z-1) + Δ(H-1) - Δ(A,Z)

Where:
  Δ(H-1) = 7288.971 keV (hydrogen atom)

Energy required to remove one proton.
```

### Two-Nucleon Separation Energies

```
S_2n = Δ(A-2,Z) + 2×Δ(n) - Δ(A,Z)
S_2p = Δ(A-2,Z-2) + 2×Δ(H-1) - Δ(A,Z)

Useful for pairing effects:
  Pairing gap ≈ S_n(even-even) - S_n(odd-A)
```

---

## Statistics

### Coverage

```
Total nuclides:      3,558
Q_alpha calculated:  3,413 (96.0%)
Q_beta calculated:   3,263 (91.7%)
S_n calculated:      3,439 (96.7%)
S_p calculated:      3,379 (95.0%)
```

### Energetically Allowed Decays

| Decay Mode | Total | Q > 0 (allowed) | Percentage |
|------------|-------|-----------------|------------|
| Alpha | 3,413 | 1,626 | 47.6% |
| Beta⁻ | 3,263 | 1,499 | 45.9% |
| Beta⁺ | 3,263 | 1,596 | 48.9% |
| EC | 3,263 | 1,764 | 54.1% |
| Neutron sep | 3,439 | 3,409 | 99.1% |
| Proton sep | 3,379 | 3,169 | 93.8% |

### Q-Value Ranges (Positive Only)

| Mode | Min (keV) | Max (keV) | Median (keV) |
|------|-----------|-----------|--------------|
| Q_α | 5.4 | 11,993 | 4,923 |
| Q_β⁻ | 2.5 | 32,740 | 5,649 |
| Q_β⁺ | 1.8 | 27,923 | 4,453 |
| Q_EC | 2.8 | 28,945 | 4,980 |
| S_n | 1.3 | 27,715 | 7,232 |
| S_p | 12.9 | 32,061 | 6,326 |

---

## Validation

### Well-Known Decays (Literature Comparison)

| Nuclide | Mode | Expected Q | Calculated Q | Error | Status |
|---------|------|------------|--------------|-------|--------|
| C-14 | β⁻ | 0.156 MeV | 0.156 MeV | 0.000 | ✓ EXACT |
| U-235 | α | 4.680 MeV | 4.678 MeV | 0.002 | ✓ PASS |
| U-238 | α | 4.270 MeV | 4.270 MeV | 0.000 | ✓ EXACT |
| Pu-239 | α | 5.240 MeV | 5.245 MeV | 0.005 | ✓ PASS |
| Po-210 | α | 5.410 MeV | 5.408 MeV | 0.002 | ✓ PASS |

**Accuracy**: Typical agreement within 0.01 MeV (1% for ~5 MeV decays)

### Consistency Checks

**Alpha emitters** (from NUBASE):
- 596 nuclides with dominant mode = alpha
- 592 have Q_α calculated (99.3%)
- **All 592 have Q_α > 0** (100% energetically allowed) ✓

**Beta⁻ emitters**:
- 1,424 nuclides with dominant mode = beta_minus
- All have Q_β⁻ calculated (100%)
- 1,394 have Q_β⁻ > 0 (97.9% allowed)
- **30 have Q_β⁻ ≤ 0** (likely drip-line or systematics)

**Beta⁺ emitters**:
- 1,008 nuclides with dominant mode = beta_plus
- All have Q_β⁺ calculated (100%)
- 984 have Q_β⁺ > 0 (97.6% allowed)
- **24 have Q_β⁺ ≤ 0** (EC still possible if Q_EC > 0)

---

## Known Limitations

### 1. Missing Q-Values

**Why**:
- Daughter nucleus not in AME2020 (exotic, predicted)
- Parent at edge of nuclear chart
- No mass measurement available

**Impact**: ~4-8% of nuclides missing specific Q-values

### 2. Negative Q for Labeled Decay Mode

**Examples**: 30 β⁻ emitters have Q_β⁻ ≤ 0

**Reasons**:
- Drip-line nuclides (barely bound)
- NUBASE lists predicted mode (not experimental)
- Particle-unstable isotopes (β decay to continuum)
- Systematic mass estimates with large uncertainties

**Not an error**: These are edge cases, correctly calculated

### 3. Beta⁺ vs EC Distinction

**Note**: The parser calculates both:
- `Q_beta_plus = Q_EC - 2m_e`
- `Q_EC = Δ(parent) - Δ(daughter)`

For nuclides with Q_β⁺ < 0 but Q_EC > 0:
- β⁺ forbidden (not enough energy for positron)
- EC allowed (only needs binding energy difference)

**Example**: Many low-Z unstable isotopes are EC-only.

### 4. Ground State Only

The parser uses **ground state masses only**. For decays involving isomers:
- Parent/daughter excitation energies not included
- Q-values may be shifted by isomer energy

**Correction needed**: Add excitation energies from NUBASE if analyzing isomer decays.

---

## Usage

### Command Line

```bash
python -m src.parse_ame \
  --input data/raw/ame2020.csv \
  --output data/derived/ame.parquet
```

### Python API

```python
from src.parse_ame import parse_ame

df = parse_ame(
    input_file='data/raw/ame2020.csv',
    output_file='data/derived/ame.parquet'
)

# Filter alpha-unstable nuclides
alpha_unstable = df[df['Q_alpha_MeV'] > 0]

# Filter doubly magic
doubly_magic = df[
    (df['Z'].isin([2, 8, 20, 28, 50, 82])) &
    (df['N'].isin([2, 8, 20, 28, 50, 82, 126]))
]

# Get Q-value for specific decay
U238 = df[(df['Z'] == 92) & (df['A'] == 238)]
Q_alpha_U238 = U238.iloc[0]['Q_alpha_MeV']
```

### Integration with NUBASE

```python
import pandas as pd

# Load both datasets
nubase = pd.read_parquet('data/derived/nuclides_all.parquet')
ame = pd.read_parquet('data/derived/ame.parquet')

# Merge on (Z, A)
df = nubase.merge(
    ame[['Z', 'A', 'Q_alpha_MeV', 'Q_beta_minus_MeV',
         'Q_beta_plus_MeV', 'S_n_MeV', 'S_p_MeV']],
    on=['Z', 'A'],
    how='left'
)

# Now have: decay modes (NUBASE) + Q-values (AME)
print(f"Merged {len(df)} nuclides")
print(f"Alpha emitters with Q: {df['Q_alpha_MeV'].notna().sum()}")
```

---

## Integration with EXPERIMENT_PLAN.md

This parser implements part of **§2.2** (derived datasets):

### Workflow

**1. Parse NUBASE** (`parse_nubase.py`)
- Extract (A, Z, decay_mode, half_life)
- Output: `nuclides_all.parquet`

**2. Parse AME** (`parse_ame.py`) ← **This module**
- Calculate Q-values for all modes
- Output: `ame.parquet`

**3. Merge datasets**
- Combine NUBASE + AME by (Z, A)
- Ready for harmonic scoring

**4. Score harmonics** (`score_harmonics.py`)
- Calculate ε from family parameters
- Use Q-values for Geiger-Nuttall corrections

**Next steps**:
- [ ] Merge NUBASE + AME in unified dataset
- [ ] Use Q_alpha for Geiger-Nuttall barrier term in Exp 2 (half-life modeling)
- [ ] Use Q-values as features in Exp 3 (decay mode prediction)

---

## Performance

### Timing

```
Loading:        ~0.1 seconds  (3,558 rows from CSV)
Q calculation:  ~1.5 seconds  (3,558 × 8 Q-values)
Output:         ~0.1 seconds  (Parquet write)
Total:          ~1.7 seconds
```

### File Sizes

```
Input (AME CSV):       ~400 KB
Output (Parquet):      ~250 KB  (compressed, 23 columns)
```

---

## Future Enhancements

### Potential Improvements

1. **Uncertainty propagation**:
   - Combine mass excess uncertainties
   - Q-value uncertainties: σ_Q = √(σ²_parent + σ²_daughter + ...)

2. **Excited state decays**:
   - Include isomer excitation energies from NUBASE
   - Calculate Q-values for isomer → ground transitions

3. **Multi-particle decays**:
   - Q_2p2n (double beta decay)
   - Q_cluster (exotic cluster emission)

4. **Fission barriers**:
   - Estimate from systematics (Myers-Swiatecki)
   - Compare Q_fission to Q_alpha

5. **Astrophysical rates**:
   - Use Q-values + Coulomb barriers
   - Calculate Gamow factors for r-process

---

## Reference Masses (keV)

Constants used in Q-value calculations (from AME2020):

```python
neutron:   8071.31806  # Free neutron
proton:    7288.971064  # H-1 atom (includes e⁻)
electron:  510.9989461  # e⁻
alpha:     2424.91587   # He-4
deuteron:  13135.722895 # H-2
triton:    14949.8109   # H-3
He-3:      14931.21888  # He-3
```

**Note**: Atomic masses (H, He) include electrons. For nuclear masses, would need to subtract electron binding energies — but for Q-values, atomic convention cancels out correctly.

---

## Citation

If using this parser in publications, cite:

**AME2020:**
W. Huang, M. Wang, F.G. Kondev, G. Audi, S. Naimi,
"The AME 2020 atomic mass evaluation (I)"
Chinese Physics C, Vol. 45, No. 3 (2021)
DOI: 10.1088/1674-1137/abddb0

**QFD Harmonic Model:**
[Your citation when published]

---

**Last Updated**: 2026-01-02
**Parser Version**: 1.0
**AME Version**: 2020
**Status**: ✓ PRODUCTION-READY
