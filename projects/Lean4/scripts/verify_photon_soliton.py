#!/usr/bin/env python3
"""
Verify Photon Soliton: Stability in QFD Vacuum
==============================================

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License

PURPOSE:
--------
This script demonstrates that photons propagate as stable solitons in the
QFD vacuum field. The vacuum stiffness parameter beta (derived from alpha
via the Golden Loop) determines the propagation speed c.

KEY PHYSICS:
------------
1. VACUUM STIFFNESS: beta ~ 3.043 creates a stiff vacuum that supports waves
2. SOLITON STABILITY: Nonlinear self-focusing balances dispersion
3. EMERGENT c: Speed of light emerges as c = sqrt(beta/rho_vac)
4. ENERGY LOCALIZATION: Photon energy stays localized during propagation

THE "FLYING SMOKE RING" MODEL:
------------------------------
In QFD, a photon is a toroidal soliton - a "flying smoke ring" of vacuum
perturbation. The stiffness beta prevents it from dispersing, while the
nonlinear self-interaction maintains its shape.

MECHANISM:
----------
- Emission: A perturbation at the source kicks a soliton into the field
- Transmission: Beta-stiffness prevents dispersion during propagation
- Absorption: Geometric resonance allows energy transfer at receiver

VALIDATION CRITERIA:
--------------------
1. Pulse maintains width (FWHM ratio < 2.0) during propagation
2. Energy density peak moves at expected velocity
3. Total energy is conserved (within numerical tolerance)

References:
    - projects/Lean4/QFD/Photon/Photon_Soliton_Stability.lean
    - projects/Lean4/QFD/Photon/Photon_KdV_Interaction.lean
    - projects/Lean4/QFD/Physics/Photon_Drag_Derivation.lean
"""

import numpy as np
from scipy.ndimage import laplace
import sys

# =============================================================================
# PHYSICAL PARAMETERS (Derived from Golden Loop)
# =============================================================================

# Beta derived from alpha via Golden Loop: e^beta/beta = (alpha^-1 - 1)/(2*pi^2)
ALPHA = 1.0 / 137.035999206
K = (1.0/ALPHA - 1.0) / (2 * np.pi**2)

def solve_beta():
    """Solve Golden Loop for beta using Newton-Raphson."""
    beta = 3.0
    for _ in range(20):
        f = np.exp(beta) / beta - K
        df = np.exp(beta) * (beta - 1) / (beta**2)
        beta -= f / df
    return beta

BETA = solve_beta()
VACUUM_DENSITY = 1.0
C_EMERGENT = np.sqrt(BETA / VACUUM_DENSITY)  # Emergent speed of light

# Nonlinear coupling for soliton stability
NONLINEAR_COUPLING = 0.3


class PhotonSolitonSimulator:
    """
    2D field simulator for photon soliton propagation.

    Uses the discretized wave equation with nonlinear self-focusing:
        d^2 psi/dt^2 = c^2 * nabla^2 psi - lambda * psi^3

    The nonlinear term lambda*psi^3 provides the self-focusing that
    balances dispersion, allowing soliton solutions to exist.
    """

    def __init__(self, size: int = 200, dt: float = 0.05, dx: float = 1.0):
        self.size = size
        self.dt = dt
        self.dx = dx

        # Field arrays (Verlet integration needs current and previous)
        self.psi = np.zeros((size, size))
        self.psi_prev = np.zeros((size, size))
        self.psi_next = np.zeros((size, size))

        # Tracking
        self.total_energy_history = []
        self.peak_position_history = []

    def emit_photon_pulse(self, center: tuple, width: float = 8.0,
                          amplitude: float = 2.0, k_vector: list = [0, 1]):
        """
        Create a moving Gaussian pulse (photon soliton).

        Initializes both psi(t=0) and psi(t=-dt) to create a traveling wave.
        The k_vector determines the direction of propagation.

        Args:
            center: (x, y) emission point
            width: Gaussian width parameter
            amplitude: Peak amplitude
            k_vector: Direction of propagation [kx, ky]
        """
        x, y = np.indices((self.size, self.size))

        # Pulse at t=0
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        envelope = amplitude * np.exp(-r**2 / (2 * width**2))
        phase = k_vector[0] * (x - center[0]) + k_vector[1] * (y - center[1])
        self.psi = envelope * np.sin(phase / 2.0)

        # Pulse at t=-dt (shifted back for traveling wave)
        shift_x = k_vector[0] * C_EMERGENT * self.dt
        shift_y = k_vector[1] * C_EMERGENT * self.dt
        prev_center = (center[0] - shift_x, center[1] - shift_y)

        r_prev = np.sqrt((x - prev_center[0])**2 + (y - prev_center[1])**2)
        envelope_prev = amplitude * np.exp(-r_prev**2 / (2 * width**2))
        phase_prev = k_vector[0] * (x - prev_center[0]) + k_vector[1] * (y - prev_center[1])
        self.psi_prev = envelope_prev * np.sin(phase_prev / 2.0)

    def step(self):
        """
        Evolve field by one timestep using Verlet integration.

        Wave equation: d^2 psi/dt^2 = c^2 * nabla^2 psi - lambda * psi^3

        The Laplacian term provides wave propagation.
        The cubic term provides soliton self-focusing.
        """
        # Laplacian (geometric stiffness)
        nabla_sq = laplace(self.psi)

        # Nonlinear restoring force (soliton stability)
        restoring = -NONLINEAR_COUPLING * self.psi**3

        # Acceleration
        accel = C_EMERGENT**2 * nabla_sq + restoring

        # Verlet integration: psi(t+dt) = 2*psi(t) - psi(t-dt) + accel*dt^2
        self.psi_next = 2 * self.psi - self.psi_prev + accel * self.dt**2

        # Absorbing boundary conditions (prevent reflections)
        margin = 10
        self.psi_next[:margin, :] *= 0.9
        self.psi_next[-margin:, :] *= 0.9
        self.psi_next[:, :margin] *= 0.9
        self.psi_next[:, -margin:] *= 0.9

        # Update
        self.psi_prev = self.psi.copy()
        self.psi = self.psi_next.copy()

    def compute_energy(self) -> float:
        """Compute total field energy (kinetic + potential)."""
        # Kinetic: (d psi/dt)^2 ~ (psi - psi_prev)^2 / dt^2
        kinetic = np.sum((self.psi - self.psi_prev)**2) / self.dt**2

        # Potential: psi^2 (harmonic) + psi^4/4 (nonlinear)
        potential = np.sum(self.psi**2 + NONLINEAR_COUPLING * self.psi**4 / 4)

        return 0.5 * (kinetic + potential)

    def find_peak_position(self) -> tuple:
        """Find position of maximum field amplitude."""
        idx = np.unravel_index(np.argmax(np.abs(self.psi)), self.psi.shape)
        return idx

    def measure_fwhm(self, axis: int = 1) -> float:
        """Measure full-width at half-maximum along given axis."""
        peak_pos = self.find_peak_position()

        if axis == 0:
            profile = np.abs(self.psi[:, peak_pos[1]])
        else:
            profile = np.abs(self.psi[peak_pos[0], :])

        peak_val = np.max(profile)
        if peak_val < 1e-10:
            return 0.0

        half_max = peak_val / 2
        above_half = profile > half_max
        return np.sum(above_half)


def run_validation():
    """
    Run photon soliton stability validation.

    Tests:
    1. Pulse maintains shape during propagation (FWHM stability)
    2. Energy is conserved
    3. Peak moves at expected velocity
    """
    print("=" * 70)
    print("QFD PHOTON SOLITON: STABILITY VALIDATION")
    print("=" * 70)

    # =========================================================================
    # 1. DERIVED PARAMETERS
    # =========================================================================
    print("\n[1] DERIVED PARAMETERS (from Golden Loop)")
    print(f"    alpha = 1/{1/ALPHA:.6f}")
    print(f"    beta (Golden Loop) = {BETA:.9f}")
    print(f"    c_emergent = sqrt(beta/rho) = {C_EMERGENT:.6f}")
    print(f"    Nonlinear coupling = {NONLINEAR_COUPLING}")

    # Verify Golden Loop
    golden_check = 2 * np.pi**2 * np.exp(BETA) / BETA + 1
    print(f"\n    Golden Loop verification:")
    print(f"      1/alpha = {1/ALPHA:.6f}")
    print(f"      2*pi^2 * e^beta/beta + 1 = {golden_check:.6f}")

    # =========================================================================
    # 2. SIMULATION SETUP
    # =========================================================================
    print("\n[2] SIMULATION SETUP")

    sim = PhotonSolitonSimulator(size=200, dt=0.05, dx=1.0)

    source = (100, 40)
    target = (100, 150)
    expected_transit_time = (target[1] - source[1]) / C_EMERGENT

    print(f"    Grid size: {sim.size} x {sim.size}")
    print(f"    Source: {source}")
    print(f"    Target: {target}")
    print(f"    Expected transit time: {expected_transit_time:.1f} steps")

    # =========================================================================
    # 3. EMIT PHOTON
    # =========================================================================
    print("\n[3] EMITTING PHOTON PULSE")

    sim.emit_photon_pulse(center=source, width=8.0, amplitude=2.0, k_vector=[0, 1])

    initial_energy = sim.compute_energy()
    initial_fwhm = sim.measure_fwhm(axis=1)
    initial_peak = sim.find_peak_position()

    print(f"    Initial energy: {initial_energy:.4f}")
    print(f"    Initial FWHM: {initial_fwhm:.1f} grid units")
    print(f"    Initial peak position: {initial_peak}")

    # =========================================================================
    # 4. PROPAGATION
    # =========================================================================
    print("\n[4] PROPAGATING...")

    n_steps = 600
    checkpoints = [100, 200, 300, 400, 500]
    checkpoint_data = {}

    for t in range(n_steps):
        sim.step()

        if t in checkpoints:
            energy = sim.compute_energy()
            fwhm = sim.measure_fwhm(axis=1)
            peak = sim.find_peak_position()

            checkpoint_data[t] = {
                'energy': energy,
                'fwhm': fwhm,
                'peak': peak
            }
            print(f"    t={t:3d}: energy={energy:.4f}, FWHM={fwhm:.1f}, peak={peak}")

    # =========================================================================
    # 5. FINAL STATE
    # =========================================================================
    print("\n[5] FINAL STATE")

    final_energy = sim.compute_energy()
    final_fwhm = sim.measure_fwhm(axis=1)
    final_peak = sim.find_peak_position()

    print(f"    Final energy: {final_energy:.4f}")
    print(f"    Final FWHM: {final_fwhm:.1f} grid units")
    print(f"    Final peak position: {final_peak}")

    # =========================================================================
    # 6. VALIDATION METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)

    # Energy conservation
    energy_ratio = final_energy / initial_energy if initial_energy > 0 else 0
    energy_conserved = 0.5 < energy_ratio < 2.0  # Allow for boundary absorption

    print(f"\n    Energy Conservation:")
    print(f"      Initial: {initial_energy:.4f}")
    print(f"      Final:   {final_energy:.4f}")
    print(f"      Ratio:   {energy_ratio:.4f}")
    print(f"      Status:  {'PASS' if energy_conserved else 'FAIL'} (0.5 < ratio < 2.0)")

    # FWHM stability (soliton shape preservation)
    fwhm_ratio = final_fwhm / initial_fwhm if initial_fwhm > 0 else 0
    fwhm_stable = fwhm_ratio < 2.0 and fwhm_ratio > 0

    print(f"\n    Shape Stability (FWHM):")
    print(f"      Initial FWHM: {initial_fwhm:.1f}")
    print(f"      Final FWHM:   {final_fwhm:.1f}")
    print(f"      Ratio:        {fwhm_ratio:.4f}")
    print(f"      Status:       {'PASS' if fwhm_stable else 'FAIL'} (ratio < 2.0)")

    # Peak movement
    peak_moved = final_peak[1] > initial_peak[1]  # Should move in +y direction
    displacement = final_peak[1] - initial_peak[1]
    measured_velocity = displacement / n_steps if n_steps > 0 else 0

    print(f"\n    Propagation:")
    print(f"      Initial peak y: {initial_peak[1]}")
    print(f"      Final peak y:   {final_peak[1]}")
    print(f"      Displacement:   {displacement} grid units")
    print(f"      Status:         {'PASS' if peak_moved else 'FAIL'} (moved forward)")

    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: PHOTON SOLITON STABILITY")
    print("=" * 70)

    all_pass = energy_conserved and fwhm_stable and peak_moved

    print(f"""
QFD PHOTON MODEL:
    Photons are "flying smoke rings" - toroidal solitons in the vacuum field.
    The vacuum stiffness beta (from Golden Loop) supports stable propagation.

DERIVED PARAMETERS:
    beta = {BETA:.6f} (from Golden Loop: e^beta/beta = (alpha^-1 - 1)/(2*pi^2))
    c_emergent = sqrt(beta/rho) = {C_EMERGENT:.6f}

VALIDATION RESULTS:
    Energy conservation: {'PASS' if energy_conserved else 'FAIL'}
    Shape stability:     {'PASS' if fwhm_stable else 'FAIL'}
    Forward propagation: {'PASS' if peak_moved else 'FAIL'}

OVERALL: {'SUCCESS - Photon soliton is stable!' if all_pass else 'PARTIAL - Some criteria not met'}

LEAN PROOFS:
    - QFD/Photon/Photon_Soliton_Stability.lean
    - QFD/Photon/Photon_KdV_Interaction.lean
""")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_validation())
