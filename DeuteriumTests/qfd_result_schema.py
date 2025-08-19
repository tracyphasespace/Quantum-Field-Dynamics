#!/usr/bin/env python3
"""
qfd_result_schema.py

Standardized result schema for all QFD solvers to ensure consistent JSON output
and eliminate per-tool workarounds.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import datetime as dt


@dataclass
class QFDResultSchema:
    """
    Standard result schema for all QFD solver outputs.
    
    This ensures consistent JSON structure across all tools:
    - Deuterium.py
    - AutopilotHydrogen.py  
    - run_target_deuterium.py
    - test_genesis_constants.py
    """
    
    # === GENESIS CONSTANTS & PHYSICS ===
    alpha: float
    gamma_e_target: float
    beta: float = 3.0
    eta: float = 0.05
    kappa_time: float = 3.2
    
    # === MATTER SPECIFICATION ===
    mass_amu: float
    charge: int
    electrons: int
    
    # === NUMERICAL PARAMETERS ===
    grid_points: int
    max_radius: float
    grid_dx: float
    spectral_cutoff: float
    iters_outer: int
    tol_energy_rel: float
    
    # === RESULTS ===
    converged: bool
    E_model: float
    virial: float
    
    # === PENALTIES (standardized keys) ===
    penalty_Q: float
    penalty_B: float  
    penalty_center: float
    
    # === ENERGY BREAKDOWN ===
    T_N: float
    T_e: float
    V_coul: float
    V4_N_eff: float
    V4_e: float
    V_time_balance: float
    
    # === EVALUATION (computed) ===
    virial_ok: bool
    penalties_ok: bool
    physical_success: bool
    
    # === METADATA ===
    timestamp: str
    solver_version: str = "Genesis_v3.2"
    
    @classmethod
    def from_deuterium_result(cls, data: Dict[str, Any]) -> 'QFDResultSchema':
        """Convert Deuterium.py output to standard schema."""
        penalties = data.get("penalties", {})
        
        # Compute evaluation flags
        virial = data.get("virial", 1.0)
        pen_max = max(penalties.values()) if penalties else 1.0
        virial_ok = virial < 0.1
        penalties_ok = pen_max < 1e-5
        
        breakdown = data.get("breakdown", {})
        
        return cls(
            # Genesis Constants & Physics
            alpha=data.get("alpha", 4.0),
            gamma_e_target=data.get("gamma_e_target", 6.0),
            beta=data.get("meta", {}).get("beta", 3.0),
            eta=data.get("meta", {}).get("eta", 0.05),
            kappa_time=data.get("meta", {}).get("kappa_time", 3.2),
            
            # Matter specification
            mass_amu=data.get("mass_amu", 1.0),
            charge=data.get("charge", 1),
            electrons=data.get("electrons", 1),
            
            # Numerical parameters
            grid_points=data.get("grid_points", 96),
            max_radius=data.get("max_radius", 14.0),
            grid_dx=data.get("grid_dx", 0.29),
            spectral_cutoff=data.get("spectral_cutoff", 0.36),
            iters_outer=data.get("iters_outer", 800),
            tol_energy_rel=data.get("tol_energy_rel", 1e-7),
            
            # Results
            converged=data.get("converged", False),
            E_model=data.get("E_model", 0.0),
            virial=virial,
            
            # Penalties (standardized keys)
            penalty_Q=penalties.get("Q", 0.0),
            penalty_B=penalties.get("B", 0.0),
            penalty_center=penalties.get("center", 0.0),
            
            # Energy breakdown
            T_N=breakdown.get("T_N", 0.0),
            T_e=breakdown.get("T_e", 0.0),
            V_coul=breakdown.get("V_coul", 0.0),
            V4_N_eff=breakdown.get("V4_N_eff", 0.0),
            V4_e=breakdown.get("V4_e", 0.0),
            V_time_balance=breakdown.get("V_time_balance", 0.0),
            
            # Evaluation
            virial_ok=virial_ok,
            penalties_ok=penalties_ok,
            physical_success=virial_ok and penalties_ok,
            
            # Metadata
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def pen_max(self) -> float:
        """Maximum penalty value."""
        return max(self.penalty_Q, self.penalty_B, self.penalty_center)


# === SNAPSHOT SPECIFICATION ===

@dataclass 
class QFDSnapshotSpec:
    """
    Standard specification for .pt field snapshots.
    
    Ensures consistent field saving across all solvers for visualization.
    """
    
    # === REQUIRED FIELDS ===
    psi_N: Any  # torch.Tensor on CPU
    psi_e: Any  # torch.Tensor on CPU
    
    # === GRID METADATA ===
    grid_points: int
    max_radius: float
    grid_dx: float
    
    # === GENESIS CONSTANTS ===
    alpha: float
    gamma_e_target: float
    
    # === MATTER SPEC ===
    mass_amu: float
    charge: int
    electrons: int
    
    # === SOLVER METADATA ===
    solver_version: str = "Genesis_v3.2"
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for torch.save()."""
        if not self.timestamp:
            self.timestamp = dt.datetime.now().isoformat(timespec="seconds")
        return asdict(self)


def validate_result_schema(data: Dict[str, Any]) -> bool:
    """
    Validate that a result dictionary conforms to the standard schema.
    Returns True if valid, False otherwise.
    """
    required_keys = [
        "alpha", "gamma_e_target", "mass_amu", "charge", "electrons",
        "grid_points", "max_radius", "converged", "E_model", "virial",
        "physical_success"
    ]
    
    return all(key in data for key in required_keys)


def genesis_constants() -> Dict[str, float]:
    """
    Single source of truth for Genesis Constants.
    
    Returns the proven stable configuration:
    - alpha = 4.0
    - gamma_e_target = 6.0  
    - virial = 0.0472 (reference performance)
    """
    return {
        "alpha": 4.0,
        "gamma_e_target": 6.0,
        "reference_virial": 0.0472,
        "discovery_date": "2024-overnight-sweep",
        "status": "proven_stable"
    }