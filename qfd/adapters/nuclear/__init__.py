"""
QFD Nuclear Physics Adapters

Observable predictions for nuclear domain:
- Binding energy (total and per nucleon)
- Separation energies (Sn, Sp, S2n, S2p)
- Mass excess
- Charge radii
"""

from .binding_energy import predict_binding_energy

__all__ = ["predict_binding_energy"]
