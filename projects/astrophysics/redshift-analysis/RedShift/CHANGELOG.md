# Changelog

All notable changes to the QFD CMB Module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with unit, integration, and scientific validation tests
- Continuous integration with GitHub Actions
- Code quality tools (black, flake8, isort, pre-commit)
- API documentation with Sphinx
- Usage examples and tutorials
- Sample data generation utilities
- Contribution guidelines and issue templates

### Changed
- Updated README.md with comprehensive installation and troubleshooting instructions
- Enhanced project structure for better maintainability

### Fixed
- Various bug fixes and improvements identified during testing

## [0.1.0] - 2025-09-04

### Added
- Initial release of QFD CMB Module
- Core modules for photon-photon scattering computations:
  - `ppsi_models.py` - Power spectrum models with oscillatory modulation
  - `visibility.py` - Parametric visibility functions and coordinate helpers
  - `kernels.py` - Photon-photon sin² angular/polarization kernels
  - `projector.py` - Limber and full line-of-sight projectors
  - `figures.py` - Publication-quality plotting utilities
- Demo script (`run_demo.py`) for reproducing cosmic-anchored results
- Parameter fitting scaffold (`fit_planck.py`) with emcee integration
- Basic Python packaging configuration
- Apache-2.0 license

### Features
- Support for both Limber and full line-of-sight projections
- Modular design allowing custom source functions S_T(k,η) and S_E(k,η)
- Planck-anchored parameter defaults (ℓ_A≈301, rψ≈147 Mpc, τ≈0.054)
- Clean, monochrome publication plots for TT/TE/EE spectra
- CSV output format for easy data analysis
- Compatible with Python 3.8+

### Scientific Features
- Photon-photon scattering kernel with same quadrupole geometry as Thomson
- Oscillatory power spectrum models with exponential damping
- Parametric visibility functions with Gaussian windows
- Mueller matrix formalism for polarization handling
- Support for custom cosmological parameters

---

## Release Notes Format

### Types of Changes
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Version Numbering
This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Process
1. Update version numbers in relevant files
2. Update this CHANGELOG.md with new version section
3. Create and push version tag: `git tag -a v1.0.0 -m "Release version 1.0.0"`
4. Create GitHub release with release notes
5. Publish to PyPI (if applicable)

### Contributing to Changelog
When contributing changes:
1. Add entries to the [Unreleased] section
2. Use present tense ("Add feature" not "Added feature")
3. Include issue/PR numbers when applicable
4. Group similar changes together
5. Highlight breaking changes clearly