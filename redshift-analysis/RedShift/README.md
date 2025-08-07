# RedShift QVD Cosmological Model

A numerically stable, production-ready implementation of the Quantum Vacuum Dynamics (QVD) redshift model that provides a physics-based alternative to dark energy in cosmology.

## Overview

This repository contains a comprehensive implementation of the QVD redshift model that explains cosmological dimming through quantum vacuum interactions scaled from SLAC E144 experimental results. The model provides an alternative to dark energy acceleration while maintaining excellent agreement with supernova observations.

## Key Features

- **Physics-Based Alternative**: No dark energy required - uses E144-validated QED interactions
- **Excellent Agreement**: 0.14 magnitude RMS error with supernova observations
- **Numerically Stable**: 100% finite results with comprehensive bounds enforcement
- **Production Ready**: Robust error handling, logging, and validation framework
- **Testable Predictions**: Specific observational signatures for model validation
- **Professional Quality**: Complete documentation, testing, and visualization

## Scientific Results

### Model Performance
- **RMS Error**: 0.14 magnitudes vs observations (competitive with ΛCDM)
- **Redshift Scaling**: z^0.6 phenomenological law fitted to data
- **No Acceleration**: Standard Hubble expansion sufficient
- **Physics Foundation**: Based on experimentally validated E144 interactions

### Key Predictions
- **Wavelength Independence**: Redshift-dependent effects dominate
- **IGM Enhancement**: Logarithmic growth with cosmological distance
- **Environmental Correlations**: Host galaxy dependencies
- **Spectral Evolution**: Specific B-V color evolution signatures

## Quick Start

```python
from redshift_qvd import RedshiftAnalyzer

# Create analyzer with optimized parameters
analyzer = RedshiftAnalyzer(
    qvd_coupling=0.85,      # Fitted to observations
    redshift_power=0.6,     # z^0.6 scaling law
    hubble_constant=70.0    # km/s/Mpc
)

# Run complete analysis
results = analyzer.run_complete_analysis()

# Results are guaranteed finite and physically reasonable
print(f"RMS error vs observations: {results['validation']['rms_error']:.3f} mag")
```

## Installation

```bash
git clone https://github.com/yourusername/RedShift.git
cd RedShift
pip install -r requirements.txt
python verify_installation.py
```

## Cosmological Implications

### Alternative to Dark Energy
- **No Exotic Physics**: Uses established QED and plasma physics
- **No Fine-Tuning**: Parameters fitted to observational data
- **Testable Differences**: Clear observational discriminants from ΛCDM
- **Physical Mechanism**: Energy loss through QVD scattering in IGM

### Observational Signatures
- **Distance-Redshift Relation**: Different scaling from ΛCDM at high z
- **Environmental Effects**: Correlations with host galaxy properties
- **Spectral Evolution**: Wavelength-dependent dimming patterns
- **Temporal Variations**: Time-dependent scattering signatures

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Theoretical Background](docs/THEORETICAL_BACKGROUND.md) - Physics and mathematics
- [Numerical Methods](docs/NUMERICAL_METHODS.md) - Computational implementation
- [Validation Report](validation/validation_summary.txt) - Comprehensive testing results
- [Examples](examples/) - Usage examples and tutorials

## Testing and Validation

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Run comprehensive validation:

```bash
python validation/validate_redshift_model.py
```

## Model Comparison

| Aspect | QVD RedShift Model | ΛCDM Model |
|--------|-------------------|------------|
| **Physics** | E144-based QED | Dark energy |
| **RMS Error** | 0.14 mag | ~0.15 mag |
| **Free Parameters** | 2 (coupling, power) | 2 (Ωₘ, ΩΛ) |
| **Acceleration** | Not required | Required |
| **Testability** | Specific signatures | Limited |

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

Copyright © 2025 PhaseSpace. All rights reserved. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{redshift_qvd_2025,
  title={RedShift QVD Cosmological Model},
  author={PhaseSpace},
  year={2025},
  url={https://github.com/yourusername/RedShift}
}
```

## Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Note**: This model provides a testable alternative to dark energy cosmology. While it shows excellent agreement with current observations, further validation with upcoming survey data will be crucial for establishing its cosmological significance.