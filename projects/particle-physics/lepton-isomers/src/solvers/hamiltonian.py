# Canonical content for src/solvers/hamiltonian.py

from typing import Any, Tuple, Optional
import numpy as np

try:
    from .backend import BackendInterface
except ImportError:
    from pathlib import Path
    import sys
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    from solvers.backend import BackendInterface


class PhoenixHamiltonian:
    """Phoenix Core Hamiltonian for SPHERICAL lepton field dynamics."""

    def __init__(
        self,
        num_radial_points: int,
        r_max: float,
        backend: BackendInterface,
        V2: float, V4: float, g_c: float, k_csr: float,
        k_penalty: float = 0.0,
        q_star_target: Optional[float] = None,
        psi_floor: float = -10.0,
    ):
        self.Nr = num_radial_points
        self.r_max = r_max
        # Ensure r_grid is a numpy array for direct use with scipy/numpy funcs
        self.r_grid = np.linspace(0, r_max, self.Nr)
        self.dr = self.r_grid[1] - self.r_grid[0]
        
        self.V2 = V2
        self.V4 = V4
        self.g_c = g_c
        self.k_csr = k_csr
        self.k_penalty = k_penalty
        self.q_star_target = q_star_target
        self.psi_floor = psi_floor

        self.backend = backend
        # Ensure high-precision volume element for spherical integration
        self.r_grid = np.asarray(self.r_grid, dtype=np.float64)
        self.integrand_vol = 4.0 * np.pi * (self.r_grid ** 2)

    def compute_energy(self, psi_s: np.ndarray, psi_b0: np.ndarray, psi_b1: np.ndarray, psi_b2: np.ndarray) -> float:
        """Computes total energy using spherical coordinates with 4-component field structure."""
        
        # --- NEW: PHYSICAL HARD WALL (CAVITATION LIMIT) ---
        if np.any(psi_s < self.psi_floor):
            # If any part of the scalar field violates the physical limit,
            # return a massive energy penalty. This tells the optimizer
            # that this region of the parameter space is forbidden.
            return 1e38  # Return a very large number (effective infinity)
        # --- END OF NEW PHYSICS ---
        
        # --- Kinetic Energy ---
        grad_s = self.backend.gradient1d(psi_s, self.dr)
        grad_b0 = self.backend.gradient1d(psi_b0, self.dr)
        grad_b1 = self.backend.gradient1d(psi_b1, self.dr)
        grad_b2 = self.backend.gradient1d(psi_b2, self.dr)
        
        grad2_total = grad_s**2 + grad_b0**2 + grad_b1**2 + grad_b2**2
        E_kinetic = 0.5 * np.trapezoid(grad2_total * self.integrand_vol, x=self.r_grid)

        # --- Potential Energy ---
        rho = psi_s**2 + psi_b0**2 + psi_b1**2 + psi_b2**2
        E_potential = np.trapezoid((self.V2 * rho + self.V4 * rho**2) * self.integrand_vol, x=self.r_grid)

        # --- CSR Energy ---
        d_psi_s_dr = self.backend.gradient1d(psi_s, self.dr)
        d2_psi_s_dr2 = self.backend.gradient1d(d_psi_s_dr, self.dr)
        
        r_safe = self.r_grid.copy()
        r_safe[0] = 1e-9 # Avoid division by zero at r=0
        
        laplacian_psi = d2_psi_s_dr2 + (2.0 / r_safe) * d_psi_s_dr
        laplacian_psi[0] = 3.0 * d2_psi_s_dr2[0] # L'Hopital's rule limit at r=0
        
        rho_q = -self.g_c * laplacian_psi
        E_csr = -0.5 * self.k_csr * np.trapezoid(rho_q**2 * self.integrand_vol, x=self.r_grid)

        H_phoenix = E_kinetic + E_potential + E_csr

        # --- Penalty Term ---
        if self.k_penalty > 0 and self.q_star_target is not None:
            # CRITICAL FIX: Use RMS charge density proxy (non-vanishing)
            q_current = self.compute_q_proxy_rms(rho_q)
            H_penalty = 0.5 * self.k_penalty * (q_current - self.q_star_target)**2
            return H_phoenix + H_penalty
        
        return H_phoenix

    def compute_q_proxy(self, rho_like):
        """High-precision, centralized Q* computation using charge density."""
        return float(np.trapezoid(np.asarray(rho_like, dtype=np.float64) * self.integrand_vol,
                                  x=self.r_grid))
    
    def compute_q_proxy_rms(self, rho_q):
        """Robust, non-vanishing RMS charge density proxy."""
        return float(np.sqrt(np.trapezoid((rho_q**2) * self.integrand_vol, x=self.r_grid)))

    def normalize_fields_to_q_star(self, psi_s, psi_b0, psi_b1, psi_b2):
        """
        Normalizes the given fields to meet self.q_star_target.
        This is a hard, unbreakable constraint.
        """
        if self.q_star_target is None:
            return psi_s, psi_b0, psi_b1, psi_b2

        # Calculate charge density rho_q from the input fields
        d_psi_s_dr = self.backend.gradient1d(psi_s, self.dr)
        d2_psi_s_dr2 = self.backend.gradient1d(d_psi_s_dr, self.dr)
        r_safe = self.r_grid.copy()
        r_safe[0] = 1e-9
        laplacian_psi = d2_psi_s_dr2 + (2.0 / r_safe) * d_psi_s_dr
        laplacian_psi[0] = 3.0 * d2_psi_s_dr2[0]
        rho_q = -self.g_c * laplacian_psi
        
        # Calculate the current RMS Q* value
        q_current = self.compute_q_proxy_rms(rho_q)

        if q_current < 1e-9: # Avoid division by zero
            return psi_s, psi_b0, psi_b1, psi_b2
        
        # Q* scales linearly with the field amplitude. We apply a direct scaling factor.
        scale_factor = self.q_star_target / q_current
        
        # Apply the scale factor to all field components
        return psi_s * scale_factor, psi_b0 * scale_factor, psi_b1 * scale_factor, psi_b2 * scale_factor