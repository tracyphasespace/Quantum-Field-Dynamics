Contributing
============

We welcome contributions to the QFD CMB Module! This guide will help you get started 
with contributing code, documentation, or bug reports.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/yourusername/qfd-cmb.git
   cd qfd-cmb

3. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

5. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Running Tests
~~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

   pytest

Run tests with coverage:

.. code-block:: bash

   pytest --cov=qfd_cmb --cov-report=html

Run specific test files:

.. code-block:: bash

   pytest tests/test_ppsi_models.py

Code Quality
~~~~~~~~~~~~

We use several tools to maintain code quality:

* **black**: Code formatting
* **flake8**: Linting
* **isort**: Import sorting
* **pre-commit**: Git hooks for quality checks

Format your code:

.. code-block:: bash

   black qfd_cmb/ tests/
   isort qfd_cmb/ tests/

Check for linting issues:

.. code-block:: bash

   flake8 qfd_cmb/ tests/

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* A clear description of the problem
* Steps to reproduce the issue
* Expected vs actual behavior
* Your environment (OS, Python version, package versions)
* Minimal code example that demonstrates the bug

Use the GitHub issue template for bug reports.

Feature Requests
~~~~~~~~~~~~~~~~

For new features, please:

* Check if the feature already exists or is planned
* Describe the use case and motivation
* Provide examples of how the feature would be used
* Consider the scope and complexity

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation contributions are highly valued:

* Fix typos or unclear explanations
* Add examples or tutorials
* Improve API documentation
* Translate documentation

Code Contributions
~~~~~~~~~~~~~~~~~~

For code contributions:

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature-name

2. Make your changes following our coding standards
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass
6. Submit a pull request

Coding Standards
----------------

Style Guidelines
~~~~~~~~~~~~~~~~

* Follow PEP 8 for Python code style
* Use black for automatic code formatting
* Maximum line length: 88 characters (black default)
* Use descriptive variable and function names
* Add type hints where appropriate

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

* Use NumPy-style docstrings for all public functions
* Include parameter types and descriptions
* Provide usage examples in docstrings
* Document return values and exceptions

Example docstring:

.. code-block:: python

   def oscillatory_psik(k, A=1.0, ns=0.96, rpsi=147.0, Aosc=0.55, sigma_osc=0.025):
       """
       Compute oscillatory power spectrum P_ψ(k).
       
       Parameters
       ----------
       k : array_like
           Wavenumber values in units of 1/Mpc.
       A : float, optional
           Overall amplitude normalization, by default 1.0
       ns : float, optional
           Spectral index for power-law component, by default 0.96
       rpsi : float, optional
           Characteristic scale for oscillations in Mpc, by default 147.0
       Aosc : float, optional
           Amplitude of oscillatory modulation, by default 0.55
       sigma_osc : float, optional
           Damping scale for oscillations, by default 0.025
           
       Returns
       -------
       ndarray
           Power spectrum values P(k) at input wavenumbers.
           
       Examples
       --------
       >>> import numpy as np
       >>> k = np.logspace(-4, 1, 100)
       >>> Pk = oscillatory_psik(k, rpsi=150.0)
       """

Testing Guidelines
~~~~~~~~~~~~~~~~~~

* Write tests for all new functions
* Use pytest for testing framework
* Include both unit tests and integration tests
* Test edge cases and error conditions
* Aim for >90% code coverage

Example test:

.. code-block:: python

   import numpy as np
   import pytest
   from qfd_cmb.ppsi_models import oscillatory_psik

   def test_oscillatory_psik_basic():
       """Test basic functionality of oscillatory_psik."""
       k = np.array([0.1, 1.0, 10.0])
       result = oscillatory_psik(k)
       
       # Check output shape
       assert result.shape == k.shape
       
       # Check positivity
       assert np.all(result > 0)
       
       # Check reasonable values
       assert np.all(np.isfinite(result))

   def test_oscillatory_psik_parameters():
       """Test parameter variations."""
       k = np.logspace(-2, 1, 50)
       
       # Test different amplitudes
       result1 = oscillatory_psik(k, A=1.0)
       result2 = oscillatory_psik(k, A=2.0)
       np.testing.assert_allclose(result2, 2.0 * result1, rtol=1e-10)

Scientific Validation
~~~~~~~~~~~~~~~~~~~~~

For scientific code, additional validation is required:

* Compare against analytical solutions where possible
* Validate against published results
* Check physical consistency (e.g., positive definite spectra)
* Test numerical stability and convergence

Pull Request Process
--------------------

1. **Create a descriptive PR title** that summarizes the changes
2. **Fill out the PR template** with details about your changes
3. **Link to relevant issues** using "Fixes #123" or "Addresses #123"
4. **Ensure all checks pass** (tests, linting, coverage)
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally
7. **Keep your branch updated** with the main branch

PR Checklist
~~~~~~~~~~~~

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (black, flake8, isort)
- [ ] All tests pass locally
- [ ] New functionality has tests
- [ ] Documentation is updated
- [ ] Changelog is updated (if applicable)
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains the changes

Review Process
~~~~~~~~~~~~~~

All pull requests require review from at least one maintainer. The review process includes:

* Code quality and style review
* Scientific accuracy validation
* Test coverage assessment
* Documentation completeness check
* Performance impact evaluation

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment. Please:

* Be respectful and constructive in all interactions
* Focus on the technical aspects of contributions
* Provide helpful feedback and suggestions
* Be patient with new contributors
* Report any inappropriate behavior to the maintainers

Communication
~~~~~~~~~~~~~

* Use GitHub issues for bug reports and feature requests
* Use GitHub discussions for general questions and ideas
* Join our community chat for real-time discussions
* Follow our social media for updates and announcements

Recognition
~~~~~~~~~~~

Contributors are recognized in several ways:

* Listed in the AUTHORS file
* Mentioned in release notes for significant contributions
* Invited to join the development team for sustained contributions
* Co-authorship consideration for major scientific contributions

Getting Help
------------

If you need help with contributing:

* Check the documentation and examples
* Search existing issues and discussions
* Ask questions in GitHub discussions
* Contact the maintainers directly for sensitive issues

Resources
~~~~~~~~~

* `GitHub Flow Guide <https://guides.github.com/introduction/flow/>`_
* `NumPy Docstring Guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* `pytest Documentation <https://docs.pytest.org/>`_
* `Black Code Formatter <https://black.readthedocs.io/>`_

Thank you for contributing to the QFD CMB Module!