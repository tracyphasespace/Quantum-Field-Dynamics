# API Reference - E144 Supernova QVD Model

## Core Model Classes

### `E144ScaledQVDModel`

Main class for the E144-scaled supernova QVD model with numerical stability enhancements.

#### Constructor
```python
E144ScaledQVDModel(e144_data: E144ExperimentalData, sn_params: SupernovaParameters)
```

**Parameters:**
- `e144_data`: SLAC E144 experimental parameters
- `sn_params`: Supernova physical parameters

#### Methods

##### `generate_luminance_curve(distance_Mpc, wavelength_nm, time_range_days=(-20, 100), time_resolution_days=1.0)`

Generate synthetic supernova luminance curve with QVD scattering effects.

**Parameters:**
- `distance_Mpc` (float): Distance to supernova in megaparsecs
- `wavelength_nm` (float): Observation wavelength in nanometers
- `time_range_days` (tuple): (start, end) time range in days
- `time_resolution_days` (float): Time step size in days

**Returns:**
- `dict`: Dictionary containing:
  - `time_days`: Time array
  - `magnitude_observed`: Apparent magnitudes with QVD effects
  - `magnitude_intrinsic`: Intrinsic magnitudes without QVD
  - `dimming_magnitudes`: QVD-induced dimming
  - `optical_depths`: QVD optical depths
  - `luminosity_observed_erg_s`: Observed luminosity
  - `luminosity_intrinsic_erg_s`: Intrinsic luminosity

**Example:**
```python
curve = model.generate_luminance_curve(
    distance_Mpc=100.0,
    wavelength_nm=500.0,
    time_range_days=(-20, 100),
    time_resolution_days=1.0
)
```

##### `generate_multi_wavelength_curves(distance_Mpc, wavelengths_nm=[400, 500, 600, 700, 800], time_range_days=(-20, 100))`

Generate multi-wavelength supernova curves.

**Parameters:**
- `distance_Mpc` (float): Distance to supernova
- `wavelengths_nm` (list): List of observation wavelengths
- `time_range_days` (tuple): Time range for observation

**Returns:**
- `dict`: Dictionary with wavelength-keyed curves and color evolution

##### `calculate_plasma_evolution(time_days)`

Calculate plasma properties as function of time after explosion.

**Parameters:**
- `time_days` (float): Days since supernova explosion

**Returns:**
- `dict`: Plasma properties with guaranteed finite values

##### `calculate_qvd_cross_section(wavelength_nm, plasma_density_cm3, intensity_erg_cm2_s, time_days)`

Calculate QVD scattering cross-section with numerical safety.

**Parameters:**
- `wavelength_nm` (float): Photon wavelength
- `plasma_density_cm3` (float): Electron density in plasma
- `intensity_erg_cm2_s` (float): Local photon intensity
- `time_days` (float): Time since explosion

**Returns:**
- `float`: QVD cross-section in cm² (guaranteed finite and positive)

##### `calculate_spectral_scattering(wavelength_nm, time_days)`

Calculate wavelength-dependent scattering with bounds enforcement.

**Parameters:**
- `wavelength_nm` (float): Photon wavelength
- `time_days` (float): Time since explosion

**Returns:**
- `dict`: Scattering properties with guaranteed finite values

## Numerical Safety Functions

### Module: `numerical_safety.py`

#### `safe_power(base, exponent, min_base=1e-30, max_exponent=700.0)`

Safely compute power operations avoiding negative/zero bases and overflow.

**Parameters:**
- `base`: Base value(s) for power operation
- `exponent`: Exponent value(s)
- `min_base`: Minimum allowed base value
- `max_exponent`: Maximum allowed exponent

**Returns:**
- Safe power result, always finite and positive

#### `safe_log10(value, min_value=1e-30)`

Safely compute log10 avoiding zero/negative arguments.

**Parameters:**
- `value`: Input value(s) for logarithm
- `min_value`: Minimum allowed value

**Returns:**
- Safe log10 result, always finite

#### `safe_exp(exponent, max_exponent=700.0, min_exponent=-700.0)`

Safely compute exponential avoiding overflow/underflow.

**Parameters:**
- `exponent`: Exponent value(s)
- `max_exponent`: Maximum allowed exponent
- `min_exponent`: Minimum allowed exponent

**Returns:**
- Safe exponential result, always finite and positive

#### `safe_divide(numerator, denominator, min_denominator=1e-30)`

Safely perform division avoiding divide-by-zero.

**Parameters:**
- `numerator`: Numerator value(s)
- `denominator`: Denominator value(s)
- `min_denominator`: Minimum allowed denominator

**Returns:**
- Safe division result, always finite

#### `safe_sqrt(value, min_value=0.0)`

Safely compute square root avoiding negative arguments.

**Parameters:**
- `value`: Input value(s) for square root
- `min_value`: Minimum allowed value

**Returns:**
- Safe square root result, always finite and non-negative

#### `validate_finite(value, name="value", replace_with=None)`

Validate that all values are finite, optionally replacing non-finite values.

**Parameters:**
- `value`: Value(s) to validate
- `name`: Name for logging purposes
- `replace_with`: Value to replace non-finite entries with

**Returns:**
- Validated value with finite entries

## Physical Bounds System

### Module: `physical_bounds.py`

#### `PhysicalBounds`

Dataclass containing physical bounds for QVD calculations.

**Key Bounds:**
- `MAX_OPTICAL_DEPTH`: 50.0
- `MAX_DIMMING_MAG`: 10.0 mag
- `MIN_PLASMA_DENSITY`: 1e10 cm⁻³
- `MAX_PLASMA_DENSITY`: 1e30 cm⁻³
- `MIN_TEMPERATURE`: 100.0 K
- `MAX_TEMPERATURE`: 1e10 K

#### `BoundsEnforcer`

Class for enforcing physical bounds with logging.

##### Constructor
```python
BoundsEnforcer(bounds: PhysicalBounds = None)
```

##### Methods

###### `enforce_plasma_density(density, name="plasma_density")`
###### `enforce_temperature(temperature, name="temperature")`
###### `enforce_cross_section(cross_section, name="cross_section")`
###### `enforce_optical_depth(optical_depth, name="optical_depth")`
###### `enforce_transmission(transmission, name="transmission")`
###### `enforce_dimming_magnitude(dimming, name="dimming_magnitude")`

All enforcement methods:
- Apply appropriate bounds to input values
- Log violations when bounds are applied
- Return bounded values guaranteed to be finite

#### `SafePlasmaState`

Dataclass for plasma state with automatic bounds enforcement.

```python
@dataclass
class SafePlasmaState:
    radius_cm: float
    electron_density_cm3: float
    temperature_K: float
    luminosity_erg_s: float
    intensity_erg_cm2_s: float
```

**Methods:**
- `update_radius(new_radius_cm)`: Safely update radius
- `update_density(new_density_cm3)`: Safely update density
- `update_temperature(new_temperature_K)`: Safely update temperature
- `is_physically_reasonable()`: Check if state is reasonable
- `to_dict()`: Convert to dictionary format

#### `SafeScatteringResults`

Dataclass for scattering results with automatic bounds enforcement.

```python
@dataclass
class SafeScatteringResults:
    qvd_cross_section_cm2: float
    optical_depth: float
    transmission: float
    dimming_magnitudes: float
```

**Methods:**
- `update_cross_section(new_cross_section_cm2)`: Safely update cross-section
- `update_optical_depth(new_optical_depth)`: Safely update optical depth
- `update_transmission(new_transmission)`: Safely update transmission
- `update_dimming(new_dimming_magnitudes)`: Safely update dimming
- `is_physically_reasonable()`: Check if results are reasonable
- `to_dict()`: Convert to dictionary format

## Error Handling System

### Module: `error_handling.py`

#### `QVDModelLogger`

Specialized logger for QVD model operations with context tracking.

##### Constructor
```python
QVDModelLogger(name="QVDModel", level=logging.INFO)
```

##### Methods
- `set_context(**kwargs)`: Set context information for logging
- `debug(message, **kwargs)`: Log debug message with context
- `info(message, **kwargs)`: Log info message with context
- `warning(message, **kwargs)`: Log warning with context and counting
- `error(message, **kwargs)`: Log error with context and counting
- `get_error_summary()`: Get summary of error counts
- `get_warning_summary()`: Get summary of warning counts

#### `ErrorReporter`

Collects and reports errors and warnings from QVD model calculations.

##### Methods
- `add_error(error_type, message, context=None)`: Add error to report
- `add_warning(warning_type, message, context=None)`: Add warning to report
- `add_bounds_violation(parameter, original, bounded, min_bound, max_bound)`: Add bounds violation
- `add_numerical_issue(function, issue_type, details)`: Add numerical issue
- `generate_report()`: Generate comprehensive error report
- `save_report(filepath)`: Save error report to JSON file
- `print_summary()`: Print summary of errors and warnings

#### Decorators

##### `@handle_numerical_errors`

Decorator to handle numerical errors gracefully.

```python
@handle_numerical_errors
def my_calculation_function(x, y):
    return x / y  # Will handle division by zero gracefully
```

##### `@validate_input_parameters`

Decorator to validate input parameters for NaN/Inf values.

```python
@validate_input_parameters
def my_function(wavelength, density):
    # Will warn about NaN/Inf inputs and continue safely
    return calculation(wavelength, density)
```

#### Setup Function

##### `setup_qvd_logging(level=logging.INFO, log_file=None, enable_warnings=True)`

Set up comprehensive logging for QVD model.

**Parameters:**
- `level`: Logging level
- `log_file`: Optional log file path
- `enable_warnings`: Whether to enable warning filters

**Returns:**
- Configured QVDModelLogger instance

## Warning Types

### Custom Warning Classes

- `NumericalIssueWarning`: Warning for numerical stability issues
- `PhysicalBoundsWarning`: Warning for physical bounds violations
- `ModelConsistencyWarning`: Warning for model consistency issues

## Utility Functions

### `create_safe_plasma_state(radius_cm, electron_density_cm3, temperature_K, luminosity_erg_s, bounds_enforcer=None)`

Create plasma state with enforced physical bounds.

### `create_safe_scattering_results(qvd_cross_section_cm2, optical_depth, transmission, dimming_magnitudes, bounds_enforcer=None)`

Create scattering results with enforced physical bounds.

### `is_physically_reasonable_plasma(density_cm3, temperature_K, radius_cm)`

Check if plasma parameters are physically reasonable.

### `is_physically_reasonable_scattering(cross_section_cm2, optical_depth, transmission)`

Check if scattering parameters are physically reasonable.

## Example Usage

### Basic Usage
```python
from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel

# Create model
e144_data = E144ExperimentalData()
sn_params = SupernovaParameters()
model = E144ScaledQVDModel(e144_data, sn_params)

# Generate curve (guaranteed finite results)
curve = model.generate_luminance_curve(100.0, 500.0)
```

### Advanced Usage with Error Handling
```python
from error_handling import setup_qvd_logging, ErrorReporter
from physical_bounds import SafePlasmaState
from numerical_safety import validate_finite

# Set up logging
logger = setup_qvd_logging(level=logging.INFO)
error_reporter = ErrorReporter()

# Create model with enhanced safety
model = E144ScaledQVDModel(e144_data, sn_params)

# Generate results with validation
curve = model.generate_luminance_curve(100.0, 500.0)
validate_finite(curve['magnitude_observed'], "magnitudes")

# Check for issues
if error_reporter.warnings:
    print(f"Warnings: {len(error_reporter.warnings)}")
```

### Using Safe Data Structures
```python
from physical_bounds import SafePlasmaState, SafeScatteringResults

# Create safe plasma state (auto-validates)
plasma = SafePlasmaState(
    radius_cm=1e12,
    electron_density_cm3=1e20,
    temperature_K=1e6,
    luminosity_erg_s=1e42,
    intensity_erg_cm2_s=1e15
)

# Check if reasonable
if plasma.is_physically_reasonable():
    print("Plasma state is valid")

# Convert to dictionary
plasma_dict = plasma.to_dict()
```

## Error Handling Best Practices

1. **Enable Logging**: Always set up logging for production use
2. **Validate Results**: Use `validate_finite()` on critical outputs
3. **Check Bounds**: Monitor bounds violations through logging
4. **Use Safe Structures**: Consider `SafePlasmaState` and `SafeScatteringResults` for critical applications
5. **Monitor Performance**: Use `ErrorReporter` to track computational efficiency

## Migration from Previous Versions

The API is fully backward compatible. Existing code will work without changes, but consider adding:

```python
# Add logging
from error_handling import setup_qvd_logging
logger = setup_qvd_logging()

# Add result validation
from numerical_safety import validate_finite
validate_finite(results['magnitude_observed'], "magnitudes")
```