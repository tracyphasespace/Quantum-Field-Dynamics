"""
QFD Phoenix - Quantum Field Dynamics Simulations
===============================================

Setup configuration for the refactored QFD Phoenix codebase.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "readme.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="qfd-phoenix-refactored",
    version="0.1.0",
    author="QFD Research Team",
    author_email="qfd@research.org",
    description="Quantum Field Dynamics simulations for lepton g-2 calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qfd-research/phoenix-refactored",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
    ],
    
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0", "torch>=1.12.0"],
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0", "mypy>=0.950"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
    
    entry_points={
        "console_scripts": [
            "qfd-electron=orchestration.run_free_electron:main",
            "qfd-ladder=orchestration.ladder_solver:main",
            "qfd-solver=solvers.phoenix_solver:main",
            "qfd-g2-batch=orchestration.g2_predictor_batch:main",
            "qfd-g2-workflow=orchestration.g2_workflow:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "constants": ["*.json"],
        "": ["*.yaml", "*.yml"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/qfd-research/phoenix-refactored/issues",
        "Source": "https://github.com/qfd-research/phoenix-refactored",
        "Documentation": "https://qfd-phoenix.readthedocs.io/",
    },
)
