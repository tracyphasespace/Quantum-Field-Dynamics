#!/usr/bin/env python3
"""
QFD Phoenix Lepton Isomers - Setup Script
==========================================

Ultra-high precision lepton mass calculations using quantum field dynamics.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="qfd-phoenix-leptons",
    version="1.0.0",
    author="QFD Phoenix Research Team",
    author_email="research@qfd-phoenix.org",
    description="Ultra-high precision lepton mass calculations using quantum field dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lepton-isomers",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/lepton-isomers/issues",
        "Documentation": "https://github.com/your-org/lepton-isomers/blob/main/README.md",
        "Source Code": "https://github.com/your-org/lepton-isomers",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "constants": ["*.json"],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "progress": [
            "tqdm>=4.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "validate-leptons=validate_all_particles:main",
        ],
    },
    keywords="physics quantum field dynamics lepton mass precision",
    zip_safe=False,
)