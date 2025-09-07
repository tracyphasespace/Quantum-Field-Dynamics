# Contributing to QFD CMB Module

Thank you for your interest in contributing to the QFD CMB Module! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of CMB physics and numerical computing
- Familiarity with NumPy, SciPy, and scientific Python ecosystem

### Types of Contributions

We welcome several types of contributions:

- **Bug reports**: Help us identify and fix issues
- **Feature requests**: Suggest new functionality or improvements
- **Code contributions**: Implement bug fixes, new features, or optimizations
- **Documentation**: Improve existing docs or add new tutorials
- **Testing**: Add test cases or improve test coverage
- **Performance**: Optimize computational routines

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/qfd-cmb.git
cd qfd-cmb

# Add the upstream repository
git remote add upstream https://github.com/username/qfd-cmb.git
```

### 2. Create Development Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Run the demo to verify functionality
python run_demo.py --outdir test_outputs
```

## Making Changes

### 1. Create a Branch

```bash
# Create a new branch for your changes
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Branch Naming Conventions

- `feature/description` - New features
- `fix/issue-number-description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `refactor/description` - Code refactoring

### 3. Make Your Changes

- Keep changes focused and atomic
- Write clear, descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some project-specific conventions:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import organization
- **Docstrings**: NumPy-style docstrings for all public functions
- **Type hints**: Use type hints for function signatures where appropriate

### Formatting Tools

Code formatting is enforced automatically:

```bash
# Format code with Black
black qfd_cmb/ tests/ examples/

# Sort imports with isort
isort qfd_cmb/ tests/ examples/

# Check style with flake8
flake8 qfd_cmb/ tests/ examples/
```

### Scientific Computing Conventions

- Use descriptive variable names that match mathematical notation when possible
- Include units in docstrings and comments
- Prefer NumPy vectorized operations over loops
- Use `np.testing.assert_allclose` for floating-point comparisons
- Document numerical algorithms and their sources

### Example Function

```python
def oscillatory_psik(k: np.ndarray, lA: float, rpsi: float) -> np.ndarray:
    """
    Compute oscillatory power spectrum P_ψ(k) with exponential damping.
    
    Parameters
    ----------
    k : np.ndarray
        Wavenumber array in units of h/Mpc
    lA : float
        Characteristic angular scale in multipole units
    rpsi : float
        Characteristic comoving distance in Mpc/h
        
    Returns
    -------
    np.ndarray
        Power spectrum values P_ψ(k) in (Mpc/h)³
        
    Notes
    -----
    Implements the model from Equation (15) of Reference [1].
    
    References
    ----------
    [1] Author et al., "QFD CMB Analysis", Journal, Year
    """
    # Implementation here
    pass
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test complete workflows
- **Scientific validation**: Compare against known results
- **Performance tests**: Monitor computational efficiency

### Writing Tests

```python
import numpy as np
import pytest
from numpy.testing import assert_allclose

from qfd_cmb.ppsi_models import oscillatory_psik

def test_oscillatory_psik_basic():
    """Test basic functionality of oscillatory_psik."""
    k = np.logspace(-4, 1, 100)
    lA = 301.0
    rpsi = 147.0
    
    result = oscillatory_psik(k, lA, rpsi)
    
    # Check output properties
    assert result.shape == k.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)  # Power spectrum should be positive

def test_oscillatory_psik_reference():
    """Test against reference values."""
    k = np.array([0.01, 0.1, 1.0])
    lA = 301.0
    rpsi = 147.0
    
    result = oscillatory_psik(k, lA, rpsi)
    expected = np.array([1.234e-5, 2.567e-4, 3.891e-3])  # Reference values
    
    assert_allclose(result, expected, rtol=1e-10)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qfd_cmb --cov-report=html

# Run specific test file
pytest tests/test_ppsi_models.py

# Run tests matching pattern
pytest -k "test_oscillatory"

# Run tests with verbose output
pytest -v
```

## Documentation

### Docstring Requirements

All public functions must have NumPy-style docstrings:

- Brief description
- Parameters section with types and descriptions
- Returns section with types and descriptions
- Notes section for implementation details (optional)
- References section for scientific sources (optional)

### Building Documentation

```bash
# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Adding Examples

When adding new functionality:

1. Add usage examples to docstrings
2. Create example scripts in `examples/`
3. Add tutorial notebooks if appropriate
4. Update the main README if needed

## Submitting Changes

### 1. Prepare Your Changes

```bash
# Ensure all tests pass
pytest

# Check code style
pre-commit run --all-files

# Update documentation if needed
cd docs/ && make html
```

### 2. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "Add oscillatory power spectrum model

- Implement oscillatory_psik function with damping
- Add comprehensive unit tests with reference values
- Update documentation with usage examples
- Fixes #123"
```

### 3. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what changes you made and why
- **Testing**: Describe how you tested your changes
- **Documentation**: Note any documentation updates
- **Breaking changes**: Highlight any breaking changes
- **Issues**: Reference related issues with "Fixes #123"

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated as needed
- [ ] Changes generate no new warnings
```

## Review Process

### What to Expect

1. **Automated checks**: CI will run tests and style checks
2. **Maintainer review**: Core maintainers will review your code
3. **Feedback**: You may receive requests for changes
4. **Approval**: Once approved, your PR will be merged

### Review Criteria

- **Correctness**: Code works as intended
- **Testing**: Adequate test coverage
- **Style**: Follows project conventions
- **Documentation**: Clear and complete
- **Performance**: No significant performance regressions
- **Scientific accuracy**: Results are physically reasonable

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Publish to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: Contact maintainers directly for sensitive issues

### Asking Questions

When asking for help:

1. Search existing issues first
2. Provide minimal reproducible example
3. Include system information (OS, Python version, etc.)
4. Describe expected vs. actual behavior
5. Include relevant error messages

## Recognition

Contributors are recognized in:

- CHANGELOG.md for each release
- AUTHORS.md file
- GitHub contributors page
- Academic papers when appropriate

Thank you for contributing to the QFD CMB Module!