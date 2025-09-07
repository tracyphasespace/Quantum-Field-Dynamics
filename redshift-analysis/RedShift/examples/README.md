# QFD CMB Module Examples

This directory contains usage examples and tutorials for the QFD CMB Module.

## Files

### Python Scripts

- **`basic_usage.py`** - Demonstrates fundamental usage of the QFD CMB Module
  - Basic power spectrum calculation
  - CMB angular power spectra computation
  - Parameter sensitivity studies
  - Model comparisons

- **`advanced_fitting.py`** - Shows advanced parameter fitting techniques
  - Mock data generation
  - Maximum likelihood estimation
  - MCMC parameter sampling with emcee
  - Corner plots and uncertainty analysis
  - Model comparison using Bayesian information criteria

### Interactive Tutorial

- **`tutorial_notebook.ipynb`** - Jupyter notebook with interactive examples
  - Step-by-step tutorial with explanations
  - Interactive parameter exploration
  - Visualization of key concepts
  - Comprehensive coverage of module features

## Running the Examples

### Prerequisites

Make sure you have the QFD CMB Module installed along with the required dependencies:

```bash
pip install -r requirements.txt
```

For the advanced fitting example, you'll also need:
```bash
pip install emcee corner
```

### Basic Usage

```bash
cd examples
python basic_usage.py
```

This will generate several plots in the `outputs/` directory showing:
- Basic power spectrum comparisons
- Individual TT, EE, and TE spectra
- Combined spectra plots
- Parameter sensitivity analysis

### Advanced Fitting

```bash
cd examples
python advanced_fitting.py
```

This will perform a complete parameter fitting analysis including:
- Mock data generation
- Maximum likelihood optimization
- MCMC sampling (may take several minutes)
- Corner plot generation
- Model comparison analysis

### Interactive Tutorial

```bash
cd examples
jupyter notebook tutorial_notebook.ipynb
```

Or use JupyterLab:
```bash
jupyter lab tutorial_notebook.ipynb
```

## Output Files

All examples create output files in the `outputs/` directory:

### Basic Usage Outputs
- `basic_power_spectrum.png` - Power spectrum comparison
- `basic_tt_spectrum.png` - TT angular power spectrum
- `basic_ee_spectrum.png` - EE angular power spectrum  
- `basic_te_spectrum.png` - TE angular power spectrum
- `basic_combined_spectra.png` - All three spectra together
- `basic_parameter_sensitivity.png` - Parameter sensitivity study

### Advanced Fitting Outputs
- `advanced_fitting_comparison.png` - Data vs model comparison
- `advanced_corner_plot.png` - MCMC parameter constraints
- Console output with fitting results and statistics

## Understanding the Results

### Physical Parameters

The examples use Planck-anchored parameters:
- `lA = 301.0` - Acoustic scale parameter
- `rpsi = 147.0` Mpc - Oscillation scale
- `chi_star ≈ 14065` Mpc - Distance to last scattering
- `sigma_chi = 250.0` Mpc - Width of last scattering surface

### Key Features

1. **Oscillatory Power Spectrum**: The QFD model introduces oscillations in the primordial power spectrum with characteristic scale `rpsi`.

2. **CMB Signatures**: These oscillations propagate to the CMB angular power spectra, creating distinctive features that differ from standard ΛCDM predictions.

3. **Parameter Sensitivity**: The amplitude (`Aosc`), scale (`rpsi`), and damping (`sigma_osc`) of oscillations all affect the final CMB spectra in different ways.

4. **Statistical Analysis**: The advanced fitting example shows how to constrain QFD parameters using mock CMB data and assess model preference using information criteria.

## Customization

You can modify the examples to:
- Use different parameter values
- Change the multipole range or resolution
- Add additional physics (e.g., lensing, foregrounds)
- Implement different fitting algorithms
- Compare with real observational data

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure the QFD CMB Module is properly installed
2. **Missing dependencies**: Install required packages with `pip install -r requirements-dev.txt`
3. **Slow computation**: Reduce the number of multipoles or grid points for faster execution
4. **Memory issues**: Use smaller arrays or increase system memory
5. **Plot display issues**: Make sure you have a proper display backend for matplotlib

For more help, see the main project documentation or open an issue on GitHub.