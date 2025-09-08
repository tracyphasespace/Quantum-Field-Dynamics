"""
Phoenix Core Hamiltonian
=======================

Implementation of the Phoenix Core Hamiltonian for QFD simulations.

Physics Model:
H = ∫ [ ½(|∇ψ_s|² + |∇ψ_b|²) + V2·ρ + V4·ρ² - ½·k_csr·ρ_q² ] dV

Where:
- ρ = ψ_s² + |ψ_b|² (matter density)
- ρ_q = -g_c ∇² ψ_s (charge density)
- V2, V4: potential parameters
- k_csr: charge self-repulsion parameter
- g_c: charge coupling constant
"""

import numpy as np
from typing import Tuple, Any

try:
    from .backend import BackendInterface
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from solvers.backend import BackendInterface


class PhoenixHamiltonian:
    """Phoenix Core Hamiltonian for lepton field dynamics."""
    
    def __init__(
        self,
        grid_size: int,
        box_size: float,
        backend: BackendInterface,
        V2: float,
        V4: float,
        g_c: float,
        k_csr: float
    ):
        """
        Initialize Phoenix Hamiltonian.
        
        Args:
            grid_size: Grid points per dimension
            box_size: Physical box size
            backend: Computational backend
            V2: Linear potential parameter
            V4: Quadratic potential parameter  
            g_c: Charge coupling constant
            k_csr: Charge self-repulsion parameter
        """
        self.nx = self.ny = self.nz = grid_size
        self.lx = self.ly = self.lz = box_size
        self.dx = box_size / grid_size
        self.dy = box_size / grid_size
        self.dz = box_size / grid_size
        self.dV = self.dx * self.dy * self.dz
        
        self.V2 = V2
        self.V4 = V4
        self.g_c = g_c
        self.k_csr = k_csr
        
        self.backend = backend
        
        # Precompute k-space operators
        self._setup_k_space()
    
    def _setup_k_space(self):
        """Setup k-space grid and operators for FFT-based derivatives."""
        # Wave number grids
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        kz = 2 * np.pi * np.fft.fftfreq(self.nz, self.dz)
        
        # 3D k-space grid
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Laplacian operator in k-space: K² = kx² + ky² + kz²
        K2 = KX**2 + KY**2 + KZ**2
        
        # Convert to backend tensors
        self.K2 = self.backend.as_tensor(K2)
    
    def compute_densities(self, psi_s, psi_b) -> Tuple[Any, Any]:
        """
        Compute matter and charge densities.
        
        Args:
            psi_s: Scalar field (real)
            psi_b: Boson field (complex)
            
        Returns:
            (rho, rho_q): Matter density and charge density
        """
        # Matter density: ρ = ψ_s² + |ψ_b|²
        rho = psi_s**2
        if psi_b is not None:
            rho = rho + (psi_b.real**2 + psi_b.imag**2)
        
        # Charge density: ρ_q = -g_c ∇² ψ_s
        rho_q = -self.g_c * self.backend.laplacian(psi_s, self.K2)
        
        return rho, rho_q
    
    def compute_energy(self, psi_s, psi_b) -> float:
        """
        Compute total Hamiltonian energy.
        
        Args:
            psi_s: Scalar field
            psi_b: Boson field
            
        Returns:
            Total energy in eV
        """
        # Kinetic energy: ½|∇ψ|²
        grad2_s = self.backend.gradient_squared(psi_s, self.dx, self.dy, self.dz)
        E_kinetic = 0.5 * grad2_s.sum() * self.dV
        
        if psi_b is not None:
            grad2_b = self.backend.gradient_squared(psi_b.real, self.dx, self.dy, self.dz)
            grad2_b += self.backend.gradient_squared(psi_b.imag, self.dx, self.dy, self.dz)
            E_kinetic += 0.5 * grad2_b.sum() * self.dV
        
        # Compute densities
        rho, rho_q = self.compute_densities(psi_s, psi_b)
        
        # Potential energy: V2·ρ + V4·ρ²
        E_potential = (self.V2 * rho + self.V4 * rho**2).sum() * self.dV
        
        # CSR correction: -½·k_csr·ρ_q²
        E_csr = -0.5 * self.k_csr * (rho_q**2).sum() * self.dV
        
        total_energy = E_kinetic + E_potential + E_csr
        
        return float(self.backend.to_cpu(total_energy))
    
    def compute_gradient(self, psi_s, psi_b) -> Tuple[Any, Any]:
        """
        Compute field gradients for evolution.
        
        Args:
            psi_s: Scalar field
            psi_b: Boson field
            
        Returns:
            (grad_s, grad_b): Gradients for evolution
        """
        # Compute densities
        rho, rho_q = self.compute_densities(psi_s, psi_b)
        
        # Scalar field gradient: -∇²ψ_s + ∂V/∂ψ_s + CSR terms
        grad_s = -self.backend.laplacian(psi_s, self.K2)
        
        # Potential derivative: 2ψ_s(V2 + 2V4·ρ)
        potential_grad = 2 * psi_s * (self.V2 + 2 * self.V4 * rho)
        grad_s += potential_grad
        
        # CSR gradient: k_csr · g_c · ∇²(ρ_q)
        if abs(self.k_csr) > 1e-12:
            csr_grad = self.k_csr * self.g_c * self.backend.laplacian(rho_q, self.K2)
            grad_s += csr_grad
        
        # Boson field gradient (if present)
        grad_b = None
        if psi_b is not None:
            # Ensure complex field handling
            grad_b = -self.backend.laplacian(psi_b.real, self.K2) - 1j * self.backend.laplacian(psi_b.imag, self.K2)
            potential_grad_b = 2 * psi_b * (self.V2 + 2 * self.V4 * rho)
            grad_b = grad_b + potential_grad_b
        
        return grad_s, grad_b
    
    def compute_gradient_norm(self, psi_s, psi_b) -> float:
        """Compute gradient norm for adaptive time stepping."""
        grad_s, grad_b = self.compute_gradient(psi_s, psi_b)
        
        norm_s = (grad_s**2).sum()
        if grad_b is not None:
            norm_b = ((grad_b.real**2 + grad_b.imag**2)).sum()
            norm_s += norm_b
        
        return float(self.backend.to_cpu(norm_s)**0.5)
    
    def evolve(self, psi_s, psi_b, dt: float) -> Tuple[Any, Any]:
        """
        Semi-implicit time evolution step.
        
        Args:
            psi_s: Scalar field
            psi_b: Boson field
            dt: Time step
            
        Returns:
            (psi_s_new, psi_b_new): Updated fields
        """
        # Compute nonlinear gradient
        grad_s, grad_b = self.compute_gradient(psi_s, psi_b)
        
        # Semi-implicit step: linear terms in k-space, nonlinear explicit
        # ψ^(n+1) = ψ^n - dt * grad_nonlinear
        psi_s_new = psi_s - dt * grad_s
        
        psi_b_new = None
        if psi_b is not None:
            psi_b_new = psi_b - dt * grad_b
        
        # Apply stability damping if needed
        psi_s_new = self.backend.nan_to_num(psi_s_new)
        if psi_b_new is not None:
            psi_b_new = self.backend.nan_to_num(psi_b_new)
        
        return psi_s_new, psi_b_new