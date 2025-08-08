# Redshift QVD Cosmological Model

A numerically stable implementation of the Quantum Vacuum Dynamics (QVD) redshift model. This model provides a physically-motivated, phenomenological explanation for cosmological redshift and the Cosmic Microwave Background (CMB).

**Note:** This repository implements the **wavelength-independent** QVD redshift model. It is a synergistic but distinct effect from the wavelength-dependent scattering model used for supernova analysis.

## Overview

This repository contains an implementation of the QVD redshift model. This model proposes that cosmological redshift arises from momentum exchange between photons over vast distances, an effect motivated by the physics validated at the SLAC E144 experiment.

The model provides an alternative to the standard Big Bang expansion model for explaining the origin of cosmological redshift and the CMB.

## Key Features

- **Physics-Based Motivation**: The model is motivated by the principles of Quantum Vacuum Dynamics, which have been experimentally validated.
- **Wavelength-Independent Effect**: The core redshift mechanism in this model does not depend on the wavelength of the light.
- **Numerically Stable**: The implementation includes a robust framework for bounds enforcement and error handling to ensure reliable calculations.
- **Testable Predictions**: The model makes specific predictions about the CMB and other cosmological observations.

## Scientific Context

This model is part of a larger research program to explore alternatives to the standard ΛCDM cosmological model. It specifically tackles the origin of cosmological redshift and the CMB.

The core of the model is a phenomenological power law that describes the amount of dimming as a function of redshift. This is enhanced by a model for the contribution of the Intergalactic Medium (IGM).

**Developer's Note:** The documentation currently mentions both "Wavelength Independence" and "Spectral Evolution" as predictions. These seem contradictory. Could you please clarify if "Spectral Evolution" is a feature to be added, or if my understanding of the wavelength-independent nature of this model is incorrect?

## Quick Start

```python
from redshift_qvd_package.analyzer import EnhancedRedshiftAnalyzer

# Create analyzer with optimized parameters
analyzer = EnhancedRedshiftAnalyzer(
    qvd_coupling=0.85,
    redshift_power=0.6,
    hubble_constant=70.0
)

# Run a complete analysis
# Note: The analysis needs an observational data file to run.
# results = analyzer.run_complete_analysis()

# print(f"RMS error vs observations: {results['validation']['rms_error']:.3f} mag")
```

## Installation

```bash
# After cloning the repository:
pip install -r requirements.txt
```

## Testing and Validation

Run the complete test suite:

```bash
python -m pytest redshift-analysis/RedShift/tests/ -v
```
