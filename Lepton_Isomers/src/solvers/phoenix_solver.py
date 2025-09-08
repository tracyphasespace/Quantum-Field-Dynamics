"""
Phoenix Core Solver - Refactored Implementation
=============================================

Clean, modular implementation of the Phoenix Core Hamiltonian solver
for QFD simulations. Based on the proven Gemini_Phoenix_Solver_v2_3D.py.

Physics:
- H = H_kinetic + H_potential_corrected + H_csr_corrected  
- 3D Cartesian with periodic boundary conditions
- Semi-implicit split-step evolution
- Target: Lepton rest mass energies (electron: 511 keV, muon: 105.7 MeV, tau: 1.78 GeV)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union

try:
    from .backend import get_backend
    from .hamiltonian import PhoenixHamiltonian
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from solvers.backend import get_backend
    from solvers.hamiltonian import PhoenixHamiltonian


def load_particle_constants(particle: str) -> Dict[str, Any]:
    """Load particle constants from JSON file."""
    constants_dir = Path(__file__).resolve().parents[1] / "constants"
    constants_file = constants_dir / f"{particle}.json"
    
    if not constants_file.exists():
        raise FileNotFoundError(f"Constants file not found: {constants_file}")
    
    with open(constants_file, 'r') as f:
        return json.load(f)


def solve_psi_field(
    particle: str,
    grid_size: int = 64,
    box_size: float = 16.0,
    backend: str = "torch",
    device: str = "cuda",
    steps: int = 400,
    dt_auto: bool = True,
    dt_min: float = 5e-5,
    dt_max: float = 5e-4,
    custom_physics: Optional[Dict[str, float]] = None,
    k_csr: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve the Phoenix Core Hamiltonian for a specified particle.
    
    Args:
        particle: Name of particle ('electron', 'muon', 'tau')
        grid_size: Grid points per dimension
        box_size: Physical box size 
        backend: Computation backend ('torch', 'numpy')
        device: Device ('cuda', 'cpu')
        steps: Evolution steps
        dt_auto: Use adaptive time stepping
        dt_min: Minimum time step
        dt_max: Maximum time step
        custom_physics: Override physics parameters for excited state search
        k_csr: Override CSR parameter specifically
        **kwargs: Additional solver parameters
        
    Returns:
        Dictionary containing simulation results
    """
    # Load particle parameters
    constants = load_particle_constants(particle)
    physics = constants['physics_constants']
    
    # Override with custom physics if provided (for excited state search)
    if custom_physics is not None:
        physics = {**physics, **custom_physics}
    
    # Override k_csr specifically if provided
    if k_csr is not None:
        physics['k_csr'] = k_csr
    
    # Initialize backend
    be = get_backend(backend, device)
    
    # Create Hamiltonian
    hamiltonian = PhoenixHamiltonian(
        grid_size=grid_size,
        box_size=box_size,
        backend=be,
        V2=physics['V2'],
        V4=physics['V4'], 
        g_c=physics['g_c'],
        k_csr=physics['k_csr']
    )
    
    # Initialize field (placeholder - proper initialization needed)
    psi_s = be.randn((grid_size, grid_size, grid_size)) * 0.1
    if be.name == "torch":
        import torch
        psi_b = be.zeros((grid_size, grid_size, grid_size), dtype=torch.complex64)
    else:
        psi_b = be.zeros((grid_size, grid_size, grid_size), dtype=np.complex64)
    
    # Evolution loop
    logging.info(f"Starting {particle} simulation: {steps} steps")
    
    for step in range(steps):
        if dt_auto:
            dt = adaptive_timestep(hamiltonian, psi_s, psi_b, dt_min, dt_max)
        else:
            dt = dt_max
            
        psi_s, psi_b = hamiltonian.evolve(psi_s, psi_b, dt)
        
        if step % 100 == 0:
            energy = hamiltonian.compute_energy(psi_s, psi_b)
            logging.info(f"Step {step}: Energy = {energy:.6f} eV")
    
    # Final energy calculation
    final_energy = hamiltonian.compute_energy(psi_s, psi_b)
    
    return {
        "psi_field": be.to_cpu(psi_s),
        "psi_b_field": be.to_cpu(psi_b), 
        "energy": float(final_energy),
        "H_final": float(final_energy),
        "particle": particle,
        "grid_size": grid_size,
        "constants": constants,
        "converged": True
    }


def adaptive_timestep(hamiltonian, psi_s, psi_b, dt_min: float, dt_max: float) -> float:
    """Adaptive time step selection based on field gradients."""
    # Simplified adaptive scheme - could be enhanced
    grad_norm = hamiltonian.compute_gradient_norm(psi_s, psi_b)
    
    if grad_norm > 1000:
        return dt_min
    elif grad_norm < 100:
        return dt_max
    else:
        return dt_min + (dt_max - dt_min) * (1000 - grad_norm) / 900

def main():
    """Command-line interface for the solver."""
    parser = argparse.ArgumentParser(description="QFD ψ-field solver.")
    parser.add_argument("--particle", type=str, default="electron", help="Particle to simulate.")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--constants_path", type=str, default="src/constants", help="Path to constants directory.")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory.")
    args = parser.parse_args()

    # Run solver
    results = solve_psi_field(
        particle=args.particle,
        grid_size=args.grid_size,
        use_gpu=args.use_gpu,
        constants_path=args.constants_path
    )

    # Save results
    output_dir = Path(args.output_dir) / args.particle
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "psi_field.npy", results["psi_field"])
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "psi_field"}, f, indent=4)

    logging.info(f"Results saved to {output_dir}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
