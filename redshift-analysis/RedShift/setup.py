#!/usr/bin/env python3
"""
Setup script for QFD CMB Module
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read version from __init__.py
def get_version():
    version_file = os.path.join('qfd_cmb', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

setup(
    name="qfd-cmb",
    version=get_version(),
    author="QFD Project",
    author_email="contact@qfd-project.org",
    description="Photon-Photon Scattering Projection for CMB TT/TE/EE Spectra",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/qfd-project/qfd-cmb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "fitting": ["emcee>=3.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "numpydoc>=1.1.0",
            "jupyter>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qfd-demo=run_demo:main",
            "qfd-fit=fit_planck:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cosmology cmb photon-photon-scattering quantum-field-theory",
)