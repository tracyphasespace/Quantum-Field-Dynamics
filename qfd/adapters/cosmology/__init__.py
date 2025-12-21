"""
QFD Cosmology Observable Adapters

Predictions for cosmological observables:
- Distance modulus (Type Ia SNe)
- CMB spectrum (COBE FIRAS)
- CMB power spectra (TT, TE, EE)
- BAO acoustic scale
- H0 measurements
"""

from .distance_modulus import predict_distance_modulus
from .radiative_transfer import (
    predict_distance_modulus_rt,
    predict_cmb_spectrum,
    predict_energy_balance
)

__all__ = [
    "predict_distance_modulus",
    "predict_distance_modulus_rt",
    "predict_cmb_spectrum",
    "predict_energy_balance"
]
