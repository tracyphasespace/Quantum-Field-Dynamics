# Supernova QVD Scattering Model

A numerically stable, production-ready implementation of the E144-scaled Quantum Vacuum Dynamics (QVD) model for supernova light curve analysis.

## Overview

This repository contains a comprehensive implementation of the QVD scattering model that explains supernova dimming through quantum vacuum interactions. The model has been extensively validated and includes robust numerical stability fixes to eliminate NaN values and ensure reliable scientific calculations.

## Key Features

- **Numerically Stable**: 100% finite results under all tested conditions
- **Physically Bounded**: All parameters constrained to realistic astronomical ranges
- **Production Ready**: Comprehensive error handling and logging
- **High Performance**: ~18 curves/second processing speed
- **Backward Compatible**: Drop-in replacement for existing code
- **Extensively Tested**: 6 comprehensive test suites with 3,600+ validation points

## Quick Start

```python
from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

# Create model
e144_data = E144ExperimentalData()
sn_params = SupernovaParameters()
model = E144ScaledQVDModel(e144_data, sn_params)

# Generate light curve
curve = model.generate_luminance_curve(
    distance_Mpc=100.0,
    wavelength_nm=500.0
)

# Results are guaranteed to be finite and physically reasonable
print(f"All values finite: {np.all(np.isfinite(curve['magnitude_observed']))}")
```

## Installation

```bash
git clone https://github.com/yourusername/Supernova.git
cd Supernova
pip install -r requirements.txt
```

## Validation Results

The model has been comprehensively validated:

- ✅ **100% Finite Values**: All 3,600 test values are finite under extreme conditions
- ✅ **Physical Bounds**: All parameters stay within reasonable astronomical limits  
- ✅ **Performance**: ~18 curves/second processing speed maintained
- ✅ **Stability**: Model handles extreme parameter ranges without crashing
- ✅ **Compatibility**: All existing functionality preserved

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [E144 Fixes](docs/README_E144_FIXES.md) - Detailed numerical stability improvements
- [Validation Report](validation/validation_summary.txt) - Comprehensive validation results
- [Examples](examples/) - Usage examples and demonstrations

## Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Run validation:

```bash
python validation/validate_e144_fixes.py
```

## License

Copyright © 2025 PhaseSpace. All rights reserved.

## Citation

If you use this code in your research, please cite:

```
@software{supernova_qvd_2025,
  title={Supernova QVD Scattering Model},
  author={PhaseSpace},
  year={2025},
  url={https://github.com/yourusername/Supernova}
}
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

For questions or issues, please open a GitHub issue or contact the development team.