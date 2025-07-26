# QFD Redshift Analysis Demo

## Overview

This repository demonstrates how Quantum Field Dynamics (QFD) can explain cosmological redshift observations without requiring dark energy or cosmological acceleration. The analysis shows wavelength-independent effects that provide an alternative to the standard ΛCDM model.

## Key Features

- **Pure Physics Approach**: No dark matter, dark energy, or exotic physics required
- **E144 Validation**: Based on experimentally proven nonlinear photon interactions
- **Wavelength Independent**: Focuses on redshift-dependent dimming effects
- **Alternative Cosmology**: Provides testable alternative to dark energy paradigm

## Scientific Results

- **RMS Error**: 0.14 magnitudes agreement with observations
- **Redshift Scaling**: z^0.6 phenomenological law fitted to data
- **No Acceleration**: Standard Hubble expansion sufficient
- **Testable Predictions**: Specific signatures for observational validation

## Installation

```bash
git clone [your-qfd-redshift-repo]
cd qfd-redshift-demo
pip install -r requirements.txt
```

## Quick Start

```python
from qfd_redshift import RedshiftAnalyzer

# Create analyzer
analyzer = RedshiftAnalyzer()

# Generate redshift-distance relationship
hubble_data = analyzer.generate_hubble_diagram()

# Compare with ΛCDM predictions
comparison = analyzer.compare_with_lambda_cdm()

# Create analysis plots
analyzer.create_analysis_plots()
```

## File Structure

```
QFD_Redshift_Demo/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── qfd_redshift/               # Main package
│   ├── __init__.py
│   ├── redshift_analyzer.py    # Core redshift analysis
│   ├── cosmology.py           # Cosmological calculations
│   ├── physics.py             # QFD physics implementation
│   └── visualization.py       # Plotting utilities
├── examples/                   # Example scripts
│   ├── basic_analysis.py      # Simple redshift analysis
│   ├── hubble_diagram.py      # Generate Hubble diagram
│   └── lambda_cdm_comparison.py # Compare with standard model
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── results/                   # Output directory
```

## License

MIT License - Copyright © 2025 PhaseSpace