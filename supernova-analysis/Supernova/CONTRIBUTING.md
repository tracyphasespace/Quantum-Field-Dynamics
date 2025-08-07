# Contributing to Supernova QVD

We welcome contributions to the Supernova QVD project! This document provides guidelines for contributing.

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information** including:
   - Python version and operating system
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Relevant code snippets or error messages

### Submitting Changes

1. **Fork the repository** and create a feature branch
2. **Follow the coding standards** described below
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting
6. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Supernova.git
cd Supernova

# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Run tests
python -m pytest tests/ -v

# Run validation
python validation/validate_e144_fixes.py
```

## Coding Standards

### Python Style
- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and returns
- Write **docstrings** for all public functions and classes
- Use **meaningful variable names**

### Numerical Safety
- Always use **safe mathematical operations** from `numerical_safety.py`
- **Enforce physical bounds** using `physical_bounds.py`
- **Validate finite results** before returning values
- **Handle edge cases** gracefully

### Testing Requirements
- **Write tests** for all new functionality
- **Maintain 100% finite results** in all calculations
- **Test edge cases** and extreme parameter values
- **Include performance benchmarks** for significant changes

### Documentation
- **Update API documentation** for any interface changes
- **Add examples** for new features
- **Update README** if needed
- **Include validation results** for numerical changes

## Numerical Stability Guidelines

This project prioritizes numerical stability above all else. All contributions must:

1. **Use safe operations**: Never use raw `np.power`, `np.log10`, `np.exp`, etc.
2. **Enforce bounds**: All parameters must be within physical limits
3. **Validate outputs**: All results must be finite and reasonable
4. **Handle errors gracefully**: No crashes, always return safe fallbacks
5. **Test extensively**: Validate under extreme conditions

### Example of Good Practice

```python
# Good: Uses safe operations and bounds enforcement
def calculate_cross_section(wavelength_nm, density_cm3):
    # Enforce input bounds
    safe_wavelength = enforcer.enforce_wavelength(wavelength_nm)
    safe_density = enforcer.enforce_plasma_density(density_cm3)
    
    # Use safe mathematical operations
    wavelength_ratio = safe_divide(safe_wavelength, reference_wavelength)
    scaling_factor = safe_power(wavelength_ratio, alpha)
    
    # Calculate result with bounds enforcement
    cross_section = base_value * scaling_factor * safe_density
    return enforcer.enforce_cross_section(cross_section)

# Bad: Raw operations without safety checks
def calculate_cross_section_bad(wavelength_nm, density_cm3):
    # Could produce NaN/Inf values
    ratio = wavelength_nm / reference_wavelength  # Division by zero?
    scaling = ratio ** alpha  # Negative base? Overflow?
    return base_value * scaling * density_cm3  # Unbounded result?
```

## Testing Guidelines

### Required Tests
- **Unit tests** for all new functions
- **Integration tests** for complete workflows
- **Edge case tests** with extreme parameters
- **Numerical stability tests** ensuring finite results
- **Performance tests** for significant changes

### Test Structure
```python
def test_new_function():
    """Test description"""
    # Test normal operation
    result = new_function(normal_params)
    assert np.all(np.isfinite(result))
    
    # Test edge cases
    result_edge = new_function(extreme_params)
    assert np.all(np.isfinite(result_edge))
    
    # Test bounds enforcement
    result_bounded = new_function(out_of_bounds_params)
    assert is_within_physical_bounds(result_bounded)
```

## Review Process

All contributions go through a review process:

1. **Automated checks**: Style, tests, and validation
2. **Code review**: Functionality and numerical safety
3. **Validation review**: Ensure 100% finite results maintained
4. **Documentation review**: Clarity and completeness
5. **Performance review**: No significant regressions

## Release Process

1. **Version bump** following semantic versioning
2. **Update CHANGELOG.md** with all changes
3. **Run full validation suite**
4. **Update documentation** as needed
5. **Create release notes**

## Questions?

If you have questions about contributing:

1. Check the **documentation** first
2. Search **existing issues** and discussions
3. Open a **new issue** with the "question" label
4. Contact the **development team**

Thank you for contributing to the Supernova QVD project!