# src/probes/zeeman_probe.py
from dataclasses import dataclass
from typing import Protocol

class HasZeeman(Protocol):
    species: str
    def energy_at_B(self, B: float) -> float: ...
    @property
    def B0(self) -> float: ...            # reference field
    def dE_dB_to_anomaly(self, dE_dB: float) -> float: ...  # maps slope -> a=(g-2)/2

@dataclass
class ZeemanResult:
    species: str
    dE_dB: float
    a_anom: float

def zeeman_probe(bundle: HasZeeman, dB: float = 1e-6) -> ZeemanResult:
    """
    Central finite difference for ∂E/∂B at B0. Uses bundle's mapping to a.
    """
    B0 = bundle.B0
    e_plus  = bundle.energy_at_B(B0 + dB)
    e_minus = bundle.energy_at_B(B0 - dB)
    dE_dB = (e_plus - e_minus) / (2.0 * dB)
    a = bundle.dE_dB_to_anomaly(dE_dB)
    return ZeemanResult(species=bundle.species, dE_dB=dE_dB, a_anom=a)
