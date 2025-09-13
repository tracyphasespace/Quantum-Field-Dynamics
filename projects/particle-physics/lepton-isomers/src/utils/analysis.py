# src/utils/analysis.py
"""
Comprehensive field analysis utilities for QFD Phoenix simulations.

Provides advanced analysis capabilities for quantum field dynamics,
including field statistics, energy analysis, spatial distribution,
convergence monitoring, and physics feature extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FieldStatistics:
    """Comprehensive field statistics."""

    mean: float
    std: float
    max_val: float
    min_val: float
    rms: float
    norm_l2: float
    norm_max: float


@dataclass
class SpatialAnalysis:
    """Spatial field distribution analysis."""

    center_of_mass_r: float  # Radial center of mass for spherical coordinates
    radius_rms: float
    radius_max: float
    spherical_symmetry: float  # 0-1, closer to 1 means more spherically symmetric
    localization_factor: float  # Measure of field concentration


@dataclass
class EnergyBreakdown:
    """Energy component breakdown."""

    kinetic: float
    potential_v2: float
    potential_v4: float
    csr_energy: float
    total: float
    virial_ratio: float  # 2*T/V for virial theorem check


@dataclass
class ConvergenceMetrics:
    """Convergence and stability metrics."""

    gradient_norm: float
    energy_variance: float
    field_stability: float
    time_derivative_norm: float


def compute_field_statistics(field: np.ndarray) -> FieldStatistics:
    """Compute comprehensive field statistics."""
    flat_field = field.flatten()

    return FieldStatistics(
        mean=float(np.mean(flat_field)),
        std=float(np.std(flat_field)),
        max_val=float(np.max(flat_field)),
        min_val=float(np.min(flat_field)),
        rms=float(np.sqrt(np.mean(flat_field**2))),
        norm_l2=float(np.sqrt(np.sum(flat_field**2))),
        norm_max=float(np.max(np.abs(flat_field))),
    )


def analyze_spatial_distribution(
    field: np.ndarray, r_max: float = 1.0
) -> SpatialAnalysis:
    """Analyze spatial distribution of the field."""
    num_radial_points = field.shape[0]

    # Create coordinate grids (1D radial)
    r = np.linspace(0, r_max, num_radial_points)

    # Field density (probability)
    rho = field**2
    total_rho = np.sum(rho)

    if total_rho < 1e-15:
        # Handle zero field case
        return SpatialAnalysis(
            center_of_mass=(0.0, 0.0, 0.0),
            radius_rms=0.0,
            radius_max=0.0,
            spherical_symmetry=1.0,
            localization_factor=0.0,
        )

    # Center of mass (for 1D, only r-component is meaningful)
    cm_r = np.sum(r * rho) / total_rho

    # RMS radius
    radius_rms = float(np.sqrt(np.sum(r**2 * rho) / total_rho))

    # Maximum radius (99% containment)
    sorted_r = np.sort(r.flatten())
    sorted_rho = np.sort(rho.flatten())[::-1]  # Descending order
    cumsum = np.cumsum(sorted_rho)
    idx_99 = np.argmax(cumsum >= 0.99 * total_rho)
    radius_max = float(sorted_r[idx_99])

    # Spherical symmetry measure (simplified for 1D)
    spherical_symmetry = 1.0 # Assumed for 1D radial fields

    # Localization factor (inverse participation ratio)
    participation_ratio = (np.sum(rho) ** 2) / np.sum(rho**2)
    localization_factor = 1.0 / participation_ratio if participation_ratio > 0 else 0.0

    return SpatialAnalysis(
        center_of_mass=(float(cm_r), 0.0, 0.0), # Only r-component is meaningful
        radius_rms=radius_rms,
        radius_max=radius_max,
        spherical_symmetry=spherical_symmetry,
        localization_factor=float(localization_factor),
    )


def compute_energy_breakdown(
    psi_s: np.ndarray,
    psi_b: Optional[np.ndarray],
    hamiltonian_params: Dict[str, float],
    r_max: float = 1.0, # Changed from box_size
) -> EnergyBreakdown:
    """Compute detailed energy breakdown."""

    num_radial_points = psi_s.shape[0]

    # k-space grid for 1D
    k = np.fft.fftfreq(num_radial_points, d=r_max / num_radial_points) * 2 * np.pi
    K2 = k**2 # K2 is now 1D

    # Field densities
    rho = psi_s**2
    if psi_b is not None:
        rho += np.abs(psi_b) ** 2

    # Kinetic energy (via FFT)
    psi_s_k = np.fft.fft(psi_s) # Changed from fftn to fft
    kinetic_s = 0.5 * np.sum(K2 * np.abs(psi_s_k) ** 2).real

    kinetic = float(kinetic_s)
    if psi_b is not None:
        psi_b_k = np.fft.fft(psi_b) # Changed from fftn to fft
        kinetic_b = 0.5 * np.sum(K2 * np.abs(psi_b_k) ** 2).real
        kinetic += float(kinetic_b)

    # Potential energies
    V2 = hamiltonian_params.get("V2", 0.0)
    V4 = hamiltonian_params.get("V4", 0.0)
    g_c = hamiltonian_params.get("g_c", 1.0)
    k_csr = hamiltonian_params.get("k_csr", 0.0)

    potential_v2 = float(V2 * np.sum(rho))
    potential_v4 = float(V4 * np.sum(rho**2))

    # CSR energy
    if k_csr != 0.0:
        # Charge density ρ_q = -g_c ∇²ψ_s
        laplacian_psi_s = np.fft.ifft(-K2 * psi_s_k).real # Changed from ifftn to ifft
        rho_q = -g_c * laplacian_psi_s
        csr_energy = float(-0.5 * k_csr * np.sum(rho_q**2))  # Attractive CSR
    else:
        csr_energy = 0.0

    total = kinetic + potential_v2 + potential_v4 + csr_energy

    # Virial ratio (2T/V)
    potential_total = potential_v2 + potential_v4 + csr_energy
    virial_ratio = (
        float(2 * kinetic / potential_total) if abs(potential_total) > 1e-15 else 0.0
    )

    return EnergyBreakdown(
        kinetic=kinetic,
        potential_v2=potential_v2,
        potential_v4=potential_v4,
        csr_energy=csr_energy,
        total=total,
        virial_ratio=virial_ratio,
    )


def analyze_convergence(
    field_history: List[np.ndarray],
    energy_history: List[float],
    gradient_norms: List[float],
) -> ConvergenceMetrics:
    """Analyze convergence behavior from simulation history."""

    if len(field_history) < 2:
        return ConvergenceMetrics(0.0, 0.0, 0.0, 0.0)

    # Current gradient norm
    gradient_norm = float(gradient_norms[-1]) if gradient_norms else 0.0

    # Energy variance over recent history
    recent_energies = energy_history[-min(10, len(energy_history)) :]
    energy_variance = (
        float(np.var(recent_energies)) if len(recent_energies) > 1 else 0.0
    )

    # Field stability (compare recent field with previous)
    field_current = field_history[-1]
    field_previous = field_history[-2]
    field_diff = field_current - field_previous
    field_stability = float(np.sqrt(np.mean(field_diff**2)))

    # Time derivative norm
    if len(field_history) >= 2:
        field_current = field_history[-1]
        field_previous = field_history[-2]
        time_derivative = field_current - field_previous
        time_derivative_norm = float(np.sqrt(np.sum(time_derivative**2)))
    else:
        time_derivative_norm = 0.0

    return ConvergenceMetrics(
        gradient_norm=gradient_norm,
        energy_variance=energy_variance,
        field_stability=field_stability,
        time_derivative_norm=time_derivative_norm,
    )


def extract_physics_features(
    psi_s: np.ndarray,
    psi_b: Optional[np.ndarray],
    hamiltonian_params: Dict[str, float],
    r_max: float = 1.0,
) -> Dict[str, Any]:
    """Extract comprehensive physics features from field data."""

    # Basic field statistics
    stats_s = compute_field_statistics(psi_s)
    spatial = analyze_spatial_distribution(psi_s, r_max)
    energy = compute_energy_breakdown(psi_s, psi_b, hamiltonian_params, r_max)

    # Additional physics quantities
    rho = psi_s**2
    if psi_b is not None:
        rho += np.abs(psi_b) ** 2

    # Effective radius and moment of inertia
    num_radial_points = psi_s.shape[0]
    r = np.linspace(0, r_max, num_radial_points)

    total_rho = np.sum(rho)
    if total_rho > 1e-15:
        R_eff = float(np.sqrt(np.sum(r**2 * rho) / total_rho))
        I_moment = float(np.sum(r**2 * rho))
    else:
        R_eff = 0.0
        I_moment = 0.0

    # Q* proxy (field coupling strength)
    q_star = hamiltonian_params.get("q_star", 0.0)
    if q_star == 0.0:
        # Estimate from field characteristics
        q_star = float(np.sum(rho * r) / (total_rho + 1e-15))

    # Magnetic moment (classical approximation)
    # For 1D spherical, magnetic moment is typically zero or handled differently
    # This approximation is for 3D Cartesian fields.
    magnetic_moment = 0.0 # Placeholder for 1D spherical

    return {
        # Field statistics
        "field_mean": stats_s.mean,
        "field_std": stats_s.std,
        "field_max": stats_s.max_val,
        "field_rms": stats_s.rms,
        "field_norm_l2": stats_s.norm_l2,
        # Spatial distribution
        "center_of_mass": spatial.center_of_mass,
        "radius_rms": spatial.radius_rms,
        "radius_max": spatial.radius_max,
        "spherical_symmetry": spatial.spherical_symmetry,
        "localization_factor": spatial.localization_factor,
        # Energy breakdown
        "kinetic_energy": energy.kinetic,
        "potential_v2": energy.potential_v2,
        "potential_v4": energy.potential_v4,
        "csr_energy": energy.csr_energy,
        "total_energy": energy.total,
        "virial_ratio": energy.virial_ratio,
        # Physics features
        "R_eff": R_eff,
        "I_moment": I_moment,
        "q_star": q_star,
        "magnetic_moment": magnetic_moment,
        "total_density": float(total_rho),
    }


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced analysis of simulation results with comprehensive physics extraction."""

    psi_s = results["psi_s"]
    psi_b = results.get("psi_b")

    # Basic compatibility with old interface
    basic_analysis = {
        "mean_psi": float(np.mean(psi_s)),
        "max_psi": float(np.max(psi_s)),
        "energy": results["energy"],
        "particle": results["particle"],
        "num_radial_points": results["num_radial_points"],
    }

    # Enhanced physics analysis if parameters available
    hamiltonian_params = results.get("constants", {}).get("physics_constants", {})
    r_max = results.get("r_max", 1.0)

    if hamiltonian_params:
        physics_features = extract_physics_features(
            psi_s, psi_b, hamiltonian_params, r_max
        )
        basic_analysis.update(physics_features)

    # Add field statistics
    field_stats = compute_field_statistics(psi_s)
    basic_analysis.update(
        {
            "field_statistics": {
                "std": field_stats.std,
                "rms": field_stats.rms,
                "norm_l2": field_stats.norm_l2,
                "norm_max": field_stats.norm_max,
            }
        }
    )

    # Add spatial analysis
    spatial = analyze_spatial_distribution(psi_s, r_max)
    basic_analysis.update(
        {
            "spatial_analysis": {
                "center_of_mass": spatial.center_of_mass,
                "radius_rms": spatial.radius_rms,
                "spherical_symmetry": spatial.spherical_symmetry,
                "localization_factor": spatial.localization_factor,
            }
        }
    )

    return basic_analysis


def generate_analysis_report(
    results: Dict[str, Any], output_path: Optional[Path] = None
) -> str:
    """Generate a comprehensive analysis report."""

    analysis = analyze_results(results)

    report = []
    report.append("QFD Phoenix - Simulation Analysis Report")
    report.append("=" * 50)
    report.append(f"Particle: {analysis['particle']}")
    report.append(f"Num Radial Points: {analysis['num_radial_points']}")
    report.append(f"Total Energy: {analysis['energy']:.6f} eV")
    report.append("")

    # Field statistics
    report.append("Field Statistics:")
    report.append(f"  Mean: {analysis['mean_psi']:.6e}")
    report.append(f"  Max: {analysis['max_psi']:.6e}")
    report.append(f"  RMS: {analysis['field_statistics']['rms']:.6e}")
    report.append(f"  L2 Norm: {analysis['field_statistics']['norm_l2']:.6e}")
    report.append("")

    # Spatial analysis
    if "spatial_analysis" in analysis:
        spatial = analysis["spatial_analysis"]
        report.append("Spatial Distribution:")
        report.append(
            f"  Center of Mass: ({spatial['center_of_mass'][0]:.4f})" # Only r-component is meaningful
        )
        report.append(f"  RMS Radius: {spatial['radius_rms']:.6f}")
        report.append(f"  Spherical Symmetry: {spatial['spherical_symmetry']:.4f}")
        report.append(f"  Localization Factor: {spatial['localization_factor']:.6e}")
        report.append("")

    # Physics features
    if "R_eff" in analysis:
        report.append("Physics Features:")
        report.append(f"  Effective Radius: {analysis['R_eff']:.6e}")
        report.append(f"  Moment of Inertia: {analysis['I_moment']:.6e}")
        report.append(f"  Q* Parameter: {analysis['q_star']:.6f}")
        report.append(f"  Magnetic Moment: {analysis['magnetic_moment']:.6e}")
        report.append("")

    # Energy breakdown
    if "kinetic_energy" in analysis:
        report.append("Energy Breakdown:")
        report.append(f"  Kinetic: {analysis['kinetic_energy']:.6f} eV")
        report.append(f"  Potential V2: {analysis['potential_v2']:.6f} eV")
        report.append(f"  Potential V4: {analysis['potential_v4']:.6f} eV")
        report.append(f"  CSR Energy: {analysis['csr_energy']:.6f} eV")
        report.append(f"  Virial Ratio: {analysis['virial_ratio']:.4f}")
        report.append("")

    report_text = "\n".join(report)

    if output_path:
        output_path.write_text(report_text, encoding="utf-8")

    return report_text

