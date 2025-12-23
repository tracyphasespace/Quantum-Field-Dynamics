"""
QFD Supernova Cosmology Analysis Package

This package provides a complete pipeline for analyzing Type Ia supernova
data using the Quantum Field Dynamics (QFD) cosmological model with
Lean 4 formally verified parameter constraints.

Modules:
    - datasets: Data loaders for DES, Pantheon+, and custom datasets
    - stage1_fit: Per-supernova amplitude and stretch fitting
    - stage2_mcmc: Global parameter inference via MCMC
    - stage3_hubble: Hubble diagram analysis and model comparison
    - qc: Quality control gates and diagnostics
    - lean_validation: Formal constraint validation interface
    - plotting: Visualization tools
    - pipeline: End-to-end orchestrator

Example:
    >>> from qfd_sn import pipeline
    >>> pipeline.run("configs/des1499.yaml")
"""

__version__ = "22.0.0"
__author__ = "QFD Cosmology Team"
__license__ = "MIT"

from . import qc
from . import lean_validation

__all__ = [
    "qc",
    "lean_validation",
]
