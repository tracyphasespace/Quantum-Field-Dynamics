#!/usr/bin/env python3
"""
Setup script for RedShift QVD Cosmological Model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="redshift-qvd",
    version="1.0.0",
    author="PhaseSpace",
    author_email="contact@phasespace.com",
    description="Physics-based alternative to dark energy using QVD redshift model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RedShift",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "redshift-validate=validation.validate_redshift_model:main",
            "redshift-analyze=examples.comprehensive_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "redshift_qvd": ["data/*.json", "validation/*.txt"],
    },
)