# Setup and Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

### 2. Verify Data

```bash
cd scripts
python download_ame2020.py
```

This checks that the AME2020 nuclear database is present and valid.

Expected output:
```
✓ Found: ../data/ame2020_system_energies.csv
✓ Valid AME2020 data file
  - 3558 nuclei
  - Z range: 0 to 118
  - A range: 1 to 295
```

### 3. Test Nucleus Classification

```bash
python nucleus_classifier.py
```

Expected output:
```
Testing nucleus classification:
Nucleus    A     Z     N mode     Family
--------------------------------------------------
H-1        1     1     -3         A
He-4       4     2     -1         A
C-12       12    6     0          A
...
```

### 4. Run Predictions

```bash
python predict_all_halflives.py
```

This generates:
- `../results/predicted_halflives_all_isotopes.csv` (3530 nuclei)
- `../results/predicted_halflives_summary.md`

Runtime: ~30 seconds

### 5. Validate Against Experimental Data

```bash
python test_harmonic_vs_halflife.py
```

This generates:
- `../data/harmonic_halflife_results.csv` (47 experimental isotopes)
- `../data/harmonic_halflife_summary.md`
- `../figures/harmonic_halflife_analysis.png`

### 6. Analyze All Transitions

```bash
python analyze_all_decay_transitions.py
```

This analyzes 4,878 decay transitions to validate selection rules.

---

## Running from Different Directories

All scripts assume you run them from the `scripts/` directory:

```bash
cd harmonic_halflife_predictor/scripts
python predict_all_halflives.py
```

If you need to run from a different location, update the relative paths:
- Data: `../data/ame2020_system_energies.csv`
- Results: `../results/`
- Figures: `../figures/`

---

## Output Files

### Data Directory

| File | Description | Size |
|------|-------------|------|
| `ame2020_system_energies.csv` | AME2020 nuclear database (3558 nuclei) | ~320 KB |
| `harmonic_halflife_results.csv` | Experimental calibration data (47 isotopes) | ~4 KB |

### Results Directory

| File | Description | Size |
|------|-------------|------|
| `predicted_halflives_all_isotopes.csv` | Full predictions (3530 nuclei) | ~400 KB |
| `predicted_halflives_summary.md` | Statistical summary | ~2 KB |
| `interesting_predictions.md` | Extreme cases and examples | ~5 KB |

### Figures Directory

| File | Description |
|------|-------------|
| `halflife_prediction_validation.png` | 4-panel validation plot |
| `harmonic_halflife_analysis.png` | Experimental correlation plots |

---

## Troubleshooting

### ImportError: No module named 'nucleus_classifier'

**Solution**: Make sure you're running from the `scripts/` directory:
```bash
cd scripts
python predict_all_halflives.py
```

### FileNotFoundError: data/ame2020_system_energies.csv

**Solution**: Run from `scripts/` directory or check data file exists:
```bash
python download_ame2020.py
```

### ValueError: arange: cannot compute length

This occurs if there are NaN values in the data. The scripts handle this automatically by filtering invalid entries.

---

## Customization

### Modify Regression Parameters

Edit `predict_all_halflives.py` around lines 60-125:

```python
# Alpha decay model
alpha_params = [-24.14, 67.05, 2.56]  # [a, b, c]

# Beta- decay model
beta_minus_params = [9.35, -0.63, -0.61]

# Beta+ decay model
beta_plus_params = [11.39, -23.12, 0.0]
```

### Change Output Formats

To save to different formats:

```python
# In predict_all_halflives.py, line ~280
df_predictions.to_json('predictions.json', orient='records')
df_predictions.to_excel('predictions.xlsx', index=False)
```

### Filter by Element or Mass Range

```python
# In predict_all_halflives.py, after loading df_ame
df_ame = df_ame[(df_ame['Z'] >= 82) & (df_ame['Z'] <= 92)]  # Pb to U only
```

---

## Performance Notes

- **Prediction runtime**: ~30 seconds for 3530 nuclei
- **Memory usage**: ~200 MB
- **Bottleneck**: Regression model fitting (one-time, ~0.5 sec)

For very large datasets (>10,000 nuclei), consider:
1. Vectorizing the Q-value calculations
2. Pre-computing nucleus classifications
3. Using multiprocessing for independent predictions

---

## Data Format

### Input: AME2020 Data

Required columns in `ame2020_system_energies.csv`:

| Column | Type | Description |
|--------|------|-------------|
| A | int | Mass number |
| Z | int | Atomic number |
| element | str | Element symbol |
| mass_excess_MeV | float | Mass excess in MeV |
| BE_per_A_MeV | float | Binding energy per nucleon in MeV |

### Output: Predictions

Columns in `predicted_halflives_all_isotopes.csv`:

| Column | Type | Description |
|--------|------|-------------|
| A | int | Mass number |
| Z | int | Atomic number |
| element | str | Element symbol |
| N_mode | int | Harmonic mode quantum number |
| family | str | Nuclear family (A, B, or C) |
| BE_per_A | float | Binding energy per nucleon (MeV) |
| primary_decay | str | Predicted decay mode (alpha, beta-, beta+, stable) |
| Q_MeV | float | Q-value for primary decay (MeV) |
| delta_N | int | Change in harmonic mode (signed) |
| abs_delta_N | int | Absolute change in harmonic mode |
| daughter_A | int | Daughter mass number |
| daughter_Z | int | Daughter atomic number |
| daughter_N | int | Daughter harmonic mode |
| predicted_log_halflife | float | log₁₀(t₁/₂) in seconds |
| predicted_halflife_sec | float | Half-life in seconds |
| predicted_halflife_years | float | Half-life in years |
| num_decay_modes | int | Number of energetically allowed decays |

---

## Advanced Usage

### Batch Processing

Process specific isotopes:

```python
from nucleus_classifier import classify_nucleus
import pandas as pd

# Load data
df = pd.read_csv('../data/ame2020_system_energies.csv')

# Select isotopes of interest
uranium = df[df['element'] == 'U']

# Classify each
for _, row in uranium.iterrows():
    N, fam = classify_nucleus(row['A'], row['Z'])
    print(f"{row['element']}-{row['A']}: N={N}, Family={fam}")
```

### Add Custom Decay Modes

To add spontaneous fission or other modes, modify `predict_all_halflives.py`:

```python
# Around line 230, after beta+ decay section
# --- SPONTANEOUS FISSION ---
if A_p > 230:  # Heavy nuclei only
    # Add your SF model here
    pass
```

---

## Testing

Verify installation:

```bash
cd scripts

# Test 1: Classification
python -c "from nucleus_classifier import classify_nucleus; print(classify_nucleus(238, 92))"
# Expected: (2, 'A')

# Test 2: Data loading
python -c "import pandas as pd; df=pd.read_csv('../data/ame2020_system_energies.csv'); print(len(df))"
# Expected: 3558

# Test 3: Full prediction (small sample)
python -c "from predict_all_halflives import *; print('Success')"
```

---

## Next Steps

After running the predictions:

1. **Visualize Results**
   - Open `figures/halflife_prediction_validation.png`
   - Review statistical summary in `results/predicted_halflives_summary.md`

2. **Compare with Experimental Data**
   - See `data/harmonic_halflife_results.csv` for calibration isotopes
   - Check accuracy in `docs/HALFLIFE_PREDICTION_REPORT.md`

3. **Explore Extreme Cases**
   - Review `results/interesting_predictions.md` for unusual predictions
   - Longest-lived alpha emitters, fastest beta decays, etc.

4. **Read Technical Documentation**
   - `docs/HALFLIFE_PREDICTION_REPORT.md` - Full methodology
   - `docs/BETA_PLUS_MODEL_FIX.md` - Beta+ model details
   - `docs/harmonic_halflife_summary.md` - Experimental validation

---

## Support

For issues:
- Check this guide first
- Review error messages carefully
- Verify Python version (3.8+)
- Check file paths and permissions

Report bugs: https://github.com/YOUR_USERNAME/harmonic_halflife_predictor/issues

---

**Last Updated**: 2026-01-02
