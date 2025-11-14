# QFD CMB Module - Sample Data Documentation

This directory contains sample datasets and utilities for the QFD CMB Module. The data is designed to support testing, validation, and demonstration of the module's capabilities.

## Overview

The sample data system provides:
- **Minimal test datasets** for quick validation and unit testing
- **Planck-like datasets** for realistic demonstrations and parameter fitting
- **Parameter variation studies** for sensitivity analysis
- **Reference data** for regression testing
- **Data loading utilities** for easy access and manipulation

## Quick Start

### Generating Sample Data

```bash
# Generate all sample datasets
python data/generate_sample_data.py

# Generate only minimal test data (faster)
python data/generate_sample_data.py --minimal-only

# Custom output directory
python data/generate_sample_data.py --output-dir /path/to/output
```

### Loading Sample Data

```python
from data.data_utils import QFDDataLoader, load_sample_data

# Quick loading
minimal_data = load_sample_data("minimal_test")
planck_data = load_sample_data("planck_like")

# Using the full loader
loader = QFDDataLoader()
datasets = loader.list_available_datasets()
data = loader.load_minimal_test_data()
```

## Dataset Descriptions

### 1. Minimal Test Spectra (`minimal_test_spectra.csv`)

**Purpose**: Fast validation and unit testing

**Specifications**:
- Multipole range: ℓ = 2 to 100
- Data points: 99
- Computation time: ~1 second

**Columns**:
- `ell`: Multipole moment (integer)
- `C_TT`: TT angular power spectrum (dimensionless)
- `C_EE`: EE angular power spectrum (dimensionless)
- `C_TE`: TE angular power spectrum (dimensionless)
- `rho_TE`: TE correlation coefficient (dimensionless)

**Use Cases**:
- Unit testing of analysis functions
- Quick validation of code changes
- Development and debugging

### 2. Planck-like Spectra (`planck_like_spectra.csv`)

**Purpose**: Realistic demonstrations and parameter fitting

**Specifications**:
- Multipole range: ℓ = 2 to 2500
- Data points: 2,499
- Includes realistic error estimates
- Computation time: ~30 seconds

**Columns**:
- `ell`: Multipole moment (integer)
- `C_TT`: TT angular power spectrum (dimensionless)
- `C_EE`: EE angular power spectrum (dimensionless)
- `C_TE`: TE angular power spectrum (dimensionless)
- `error_TT`: TT spectrum uncertainty (dimensionless)
- `error_EE`: EE spectrum uncertainty (dimensionless)
- `error_TE`: TE spectrum uncertainty (dimensionless)
- `rho_TE`: TE correlation coefficient (dimensionless)

**Error Levels**:
- TT: 2% relative error (approximate Planck sensitivity)
- EE: 5% relative error
- TE: 3% relative error

**Use Cases**:
- Parameter fitting demonstrations
- Likelihood analysis examples
- Realistic data analysis workflows

### 3. Reference Spectra (`reference_spectra.json`)

**Purpose**: Regression testing and validation

**Specifications**:
- Selected multipole points: [2, 10, 50, 100, 200, 500, 1000, 1500, 2000]
- High-precision computation
- Includes metadata and tolerances

**Structure**:
```json
{
  "metadata": {
    "description": "Reference data for regression testing",
    "parameters": { ... },
    "tolerances": {
      "relative": 1e-10,
      "absolute": 1e-12
    }
  },
  "data": {
    "ell": [...],
    "C_TT": [...],
    "C_EE": [...],
    "C_TE": [...],
    "rho_TE": [...]
  }
}
```

**Use Cases**:
- Automated regression testing
- Validation of numerical accuracy
- Cross-platform consistency checks

### 4. Parameter Variations

**Purpose**: Parameter sensitivity analysis

**Files**:
- `rpsi_variations.csv`: Oscillation scale variations (130, 147, 165 Mpc)
- `aosc_variations.csv`: Oscillation amplitude variations (0.0, 0.3, 0.55, 0.8)
- `ns_variations.csv`: Spectral index variations (0.94, 0.96, 0.98)

**Columns**:
- `ell`: Multipole moment
- `C_TT`: TT spectrum for this parameter value
- `{parameter}`: Parameter value for this row
- `parameter_set`: Identifier string for this parameter combination

**Use Cases**:
- Parameter sensitivity studies
- Model comparison analysis
- Understanding parameter degeneracies

## Data Format Specifications

### Physical Units and Normalization

All angular power spectra are **dimensionless** and follow the standard CMB convention:

```
Cₗ = (2π)² ∫ dk k² Pₖ |Θₗ(k)|² / k³
```

Where:
- `Cₗ` is the angular power spectrum coefficient
- `Pₖ` is the primordial power spectrum
- `Θₗ(k)` is the transfer function

### Conversion to Physical Units

To convert to temperature units (μK²):
```python
# For TT spectrum
C_TT_microK2 = C_TT * (T_CMB * 1e6)**2  # T_CMB = 2.725 K

# For plotting Dₗ = ℓ(ℓ+1)Cₗ/(2π)
D_l = ell * (ell + 1) * C_TT / (2 * np.pi)
```

### Data Quality and Precision

- **Numerical precision**: All values stored with 6 significant digits
- **Computation precision**: Internal calculations use double precision
- **Grid resolution**: Optimized for balance between accuracy and speed
- **Validation**: All datasets validated against analytical limits

## Data Loading API

### QFDDataLoader Class

```python
from data.data_utils import QFDDataLoader

loader = QFDDataLoader(data_dir="data/sample")

# List available datasets
datasets = loader.list_available_datasets()

# Load specific datasets
minimal_data = loader.load_minimal_test_data()
planck_data = loader.load_planck_like_data()
reference_data = loader.load_reference_data()

# Load parameter variations
rpsi_data = loader.load_parameter_variations("rpsi")

# Get dataset information
info = loader.get_dataset_info("minimal_test_spectra")

# Validate computed results
validation = loader.validate_reference_data(computed_data)
```

### Convenience Functions

```python
from data.data_utils import (
    load_sample_data,
    create_mock_observational_data,
    extract_multipole_range,
    compute_spectrum_statistics,
    save_dataset
)

# Quick data loading
data = load_sample_data("planck_like")

# Create mock observational data with noise
mock_data = create_mock_observational_data(data, noise_level=0.03)

# Extract specific multipole range
low_ell_data = extract_multipole_range(data, ell_min=2, ell_max=100)

# Compute statistics
stats = compute_spectrum_statistics(data)

# Save custom dataset
save_dataset(custom_data, "my_analysis_results.csv")
```

## Standard Parameters

All sample data uses Planck-anchored parameters:

```python
planck_params = {
    'lA': 301.0,              # Acoustic scale parameter
    'rpsi': 147.0,            # Oscillation scale (Mpc)
    'chi_star': 14065.0,      # Distance to last scattering (Mpc)
    'sigma_chi': 250.0,       # Last scattering width (Mpc)
    'ns': 0.96,               # Spectral index
    'Aosc': 0.55,             # Oscillation amplitude
    'sigma_osc': 0.025        # Oscillation damping
}
```

These parameters are chosen to be consistent with Planck 2018 results while showcasing the QFD oscillatory features.

## Usage Examples

### Basic Data Loading and Plotting

```python
import matplotlib.pyplot as plt
from data.data_utils import load_sample_data

# Load data
data = load_sample_data("planck_like")

# Extract columns
ells = data['ell'].values
C_TT = data['C_TT'].values
C_EE = data['C_EE'].values
C_TE = data['C_TE'].values

# Plot TT spectrum
plt.figure(figsize=(10, 6))
plt.loglog(ells, ells*(ells+1)*C_TT, 'r-', linewidth=2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
plt.title('QFD CMB TT Spectrum')
plt.grid(True, alpha=0.3)
plt.show()
```

### Parameter Sensitivity Analysis

```python
from data.data_utils import QFDDataLoader
import matplotlib.pyplot as plt

loader = QFDDataLoader()

# Load parameter variations
rpsi_data = loader.load_parameter_variations("rpsi")

# Plot different rpsi values
plt.figure(figsize=(12, 8))
for rpsi_val in rpsi_data['rpsi'].unique():
    subset = rpsi_data[rpsi_data['rpsi'] == rpsi_val]
    ells = subset['ell'].values
    C_TT = subset['C_TT'].values
    plt.loglog(ells, ells*(ells+1)*C_TT, linewidth=2, 
               label=f'rpsi = {rpsi_val} Mpc')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$')
plt.title('Oscillation Scale Sensitivity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Mock Data Generation for Fitting

```python
from data.data_utils import load_sample_data, create_mock_observational_data

# Load theoretical data
theory_data = load_sample_data("planck_like")

# Create mock observations with 3% noise
mock_obs = create_mock_observational_data(theory_data, noise_level=0.03)

# Now mock_obs contains noisy data with error bars
# Perfect for parameter fitting examples
```

### Regression Testing

```python
from data.data_utils import QFDDataLoader

loader = QFDDataLoader()

# Your computed results
computed_results = {
    'ell': [...],
    'C_TT': [...],
    'C_EE': [...],
    'C_TE': [...]
}

# Validate against reference
validation = loader.validate_reference_data(computed_results)

if validation['passed']:
    print("All tests passed!")
else:
    print("Validation failed:")
    for spectrum, details in validation['details'].items():
        if not details.get('passed', True):
            print(f"  {spectrum}: {details}")
```

## File Organization

```
data/
├── README.md                    # This documentation
├── generate_sample_data.py      # Data generation script
├── data_utils.py               # Data loading utilities
└── sample/                     # Generated sample data
    ├── dataset_metadata.json   # Dataset metadata
    ├── minimal_test_spectra.csv # Minimal test data
    ├── planck_like_spectra.csv  # Planck-like data
    ├── reference_spectra.json   # Reference data
    ├── rpsi_variations.csv      # rpsi parameter study
    ├── aosc_variations.csv      # Aosc parameter study
    └── ns_variations.csv        # ns parameter study
```

## Performance Considerations

### Dataset Sizes

| Dataset | File Size | Load Time | Memory Usage |
|---------|-----------|-----------|--------------|
| Minimal test | ~15 KB | <0.01s | ~0.1 MB |
| Planck-like | ~500 KB | ~0.05s | ~2 MB |
| Parameter variations | ~200 KB each | ~0.02s | ~1 MB |
| Reference data | ~5 KB | <0.01s | ~0.05 MB |

### Optimization Tips

1. **Use minimal data for development**: Start with `minimal_test_spectra.csv` for code development
2. **Cache loaded data**: Store frequently used datasets in memory
3. **Extract relevant ranges**: Use `extract_multipole_range()` to work with subsets
4. **Batch processing**: Load parameter variations once and process all values

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Run `python data/generate_sample_data.py` to create sample data
2. **Import errors**: Ensure the QFD CMB module is in your Python path
3. **Memory issues**: Use smaller datasets or extract multipole ranges
4. **Precision issues**: Check numerical tolerances in validation functions

### Data Validation

```python
# Check data integrity
from data.data_utils import QFDDataLoader

loader = QFDDataLoader()
try:
    data = loader.load_minimal_test_data()
    print(f"Data loaded successfully: {len(data)} points")
    
    # Basic sanity checks
    assert data['ell'].min() >= 2, "Minimum multipole should be >= 2"
    assert data['C_TT'].min() > 0, "TT spectrum should be positive"
    assert not data.isnull().any().any(), "No NaN values allowed"
    
    print("Data validation passed!")
    
except Exception as e:
    print(f"Data validation failed: {e}")
```

## Contributing

When adding new sample datasets:

1. Follow the naming convention: `{purpose}_{type}_spectra.{ext}`
2. Include appropriate metadata in `dataset_metadata.json`
3. Add loading functions to `data_utils.py`
4. Update this documentation
5. Include validation tests

## References

- Planck Collaboration 2018: [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- CMB data analysis: Hu & Dodelson 2002, [arXiv:astro-ph/0110414](https://arxiv.org/abs/astro-ph/0110414)
- QFD formalism: [Project-specific references]