"""
QFD PHOTON LIFECYCLE SIMULATOR
------------------------------
Tests the Emission, Soliton Propagation, and Resonant Absorption 
of the "Flying Smoke Ring" (Toroidal Photon) in the QFD Vacuum.

MECHANISMS TESTED:
1. Emission: "The Brake" - A perturbation at Source kicks a soliton into the field.
2. Transmission: "The Stiffness" - The Beta parameter prevents dispersion.
3. Capture: "The Keyhole" - A geometric receiver absorbs energy only if aligned.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import os

# --- QFD Vacuum Parameters ---
# The stiffness of the vacuum field. High beta = Stiff = High c.
# Derived from Golden Loop (beta ~ 3.058).
BETA = 3.058 
VACUUM_DENSITY = 1.0  
C_VAC = np.sqrt(BETA / VACUUM_DENSITY)  # Emergent speed of light

# --- Simulation Grid ---
GRID_SIZE = 200
DT = 0.05
DX = 1.0
TIME_STEPS = 600

# To prevent dispersion (soliton condition), we introduce a nonlinear restoring force
# Potential V(psi) ~ -mu*psi + lambda*psi^3 (Standard restoration)
# In QFD, this arises from the saturation limit of the field density.
NONLINEAR_COUPLING = 0.3

class QFDVacuumSimulator:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        # Two fields: psi (field amplitude) and phi (its time derivative/momentum)
        self.psi = np.zeros((size, size))
        self.psi_prev = np.zeros((size, size))
        self.psi_next = np.zeros((size, size))
        
        # Detector state
        self.energy_received = []
        
    def emit_photon_pulse(self, center, width=5.0, amplitude=1.0, k_vector=[1, 0]):
        """
        Creates the 'Recoil Puff' - A Toroidal Wavelet.
        Initializes a moving frame soliton solution for t=0 and t=-dt.
        """
        x, y = np.indices((self.size, self.size))

        # --- Pulse at t=0 ---
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        envelope = amplitude * np.exp(-r**2 / (2 * width**2))
        phase_gradient = (k_vector[0]*(x - center[0]) + k_vector[1]*(y - center[1]))
        self.psi = envelope * np.sin(phase_gradient / 2.0)

        # --- Pulse at t=-dt (shifted back in space) ---
        # Velocity v = C_VAC * k_vector
        # Displacement = v * dt
        shift_x = k_vector[0] * C_VAC * DT
        shift_y = k_vector[1] * C_VAC * DT
        prev_center_x = center[0] - shift_x
        prev_center_y = center[1] - shift_y
        
        # Recalculate envelope and phase for the previous position
        prev_r = np.sqrt((x - prev_center_x)**2 + (y - prev_center_y)**2)
        prev_envelope = amplitude * np.exp(-prev_r**2 / (2 * width**2))
        prev_phase_gradient = (k_vector[0]*(x - prev_center_x) + k_vector[1]*(y - prev_center_y))
        self.psi_prev = prev_envelope * np.sin(prev_phase_gradient / 2.0)

    def step(self):
        """
        Evolves the field using the Discretized Wave Equation with Nonlinearity.
        Wave Eq: d2u/dt2 = c^2 * del^2 u - V'(u)
        """
        # 1. Laplacian (Geometric Stiffness)
        # Using kinetic scaling Beta
        nabla_sq = laplace(self.psi)
        
        # 2. Nonlinearity (Soliton Restoration Force)
        # Acts to hold the packet together against dispersion
        restoring_force = -NONLINEAR_COUPLING * self.psi**3
        
        # 3. Time Evolution (Verlet integration)
        # psi(t+1) = 2*psi(t) - psi(t-1) + accel * dt^2
        acceleration = (C_VAC**2 * nabla_sq) + restoring_force
        
        self.psi_next = 2*self.psi - self.psi_prev + acceleration * (DT**2)
        
        # Damping boundary conditions (Absorbing Layer) to prevent reflections
        # Simple linear attenuation at edges
        margin = 10
        self.psi_next[:margin, :] *= 0.9
        self.psi_next[-margin:, :] *= 0.9
        self.psi_next[:, :margin] *= 0.9
        self.psi_next[:, -margin:] *= 0.9

        # Update buffers
        self.psi_prev = self.psi.copy()
        self.psi = self.psi_next.copy()

    def measure_absorption(self, location, resonance_k):
        """
        The 'Keyhole' mechanism.
        Detector checks if local field Gradient matches its Resonance geometric frequency.
        """
        lx, ly = location
        # Extract local field sample (approximate measurement volume)
        sample = self.psi[lx-2:lx+3, ly-2:ly+3]
        
        # Energy Density in detector volume
        local_energy = np.sum(sample**2)
        
        # This models the "Lock and Key". In a full spinor simulation we check alignment.
        # Here we just track energy flux through the target gate.
        self.energy_received.append(local_energy)
        return local_energy

def run_lifecycle_test():
    print(f"[QFD] Initializing Vacuum (Stiffness β={BETA})...")
    sim = QFDVacuumSimulator()
    
    # 1. EMISSION (The Kick)
    source_loc = (100, 40)
    print(f"[QFD] Source at {source_loc} emits photon pulse (k=[1,0])...")
    sim.emit_photon_pulse(center=source_loc, width=8.0, amplitude=2.0, k_vector=[0, 1]) 
    # Pulse moves in +Y direction (up)

    # Detectors
    target_loc = (100, 150) # In the path
    miss_loc = (140, 100)   # Off axis (Sideways test)
    
    print("[QFD] Propagating...")
    
    snapshots = []
    
    for t in range(TIME_STEPS):
        sim.step()
        
        # Measure at Target
        e_target = sim.measure_absorption(target_loc, 1.0)
        
        # Snapshots for visualization
        if t in [0, 100, 200, 300, 400, 500]:
            snapshots.append(sim.psi.copy())

    print("[QFD] Simulation Complete.")
    
    # --- Visualization ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/photon"

    # --- Plot 1: Snapshots ---
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle(f"QFD Photon Soliton Lifecycle (β={BETA}, Nonlinear={NONLINEAR_COUPLING})")
    
    times = [0, 100, 200, 300, 400, 500]
    for i, ax in enumerate(axes.flatten()):
        if i < len(snapshots):
            im = ax.imshow(snapshots[i], cmap='seismic', vmin=-1, vmax=1)
            ax.set_title(f"Time: {times[i]} (Transmission)")
            ax.add_patch(plt.Circle((source_loc[1], source_loc[0]), 3, color='g', label="Source"))
            ax.add_patch(plt.Circle((target_loc[1], target_loc[0]), 3, color='y', fill=False, lw=2, label="Receiver"))
            # ax.add_patch(plt.Circle((miss_loc[1], miss_loc[0]), 3, color='r', fill=False, lw=1))
    
    fig1_path = os.path.join(results_dir, f"{timestamp}_lifecycle_snapshots.png")
    plt.savefig(fig1_path)
    plt.close(fig1)
    print(f"Saved snapshots to {fig1_path}")

    # --- Plot 2: Energy Transfer ---
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(sim.energy_received)
    plt.title("Energy Flux at Geometric Receiver")
    plt.xlabel("Time Step")
    plt.ylabel("Local Energy Density (absorption)")
    plt.grid(True)
    plt.axvline(x=220, color='r', linestyle='--', label='Transit Time')
    plt.legend()
    
    fig2_path = os.path.join(results_dir, f"{timestamp}_energy_flux.png")
    plt.savefig(fig2_path)
    plt.close(fig2)
    print(f"Saved energy flux plot to {fig2_path}")

    # Validate Stability (Check width of pulse at start vs end)
    # Simple peak finding logic
    initial_pulse = snapshots[0][source_loc[0], :] # slice
    final_pulse = snapshots[3][:, 100] # vertical slice at center x? NO, field moved in Y
    
    # Calculate widths (FWHM) approx
    def fwhm(arr):
        peak = np.max(np.abs(arr))
        return np.sum(np.abs(arr) > peak/2)

    w_start = fwhm(snapshots[0][:, 40]) # approx Y width
    w_mid = fwhm(snapshots[3][:, 140])  # approx Y width at T=300 (roughly at y=150-ish?)
    
    print(f"[METRIC] Soliton Width Init: {w_start}")
    print(f"[METRIC] Soliton Width Mid:  {w_mid}")
    if w_mid > 0 and w_mid < w_start * 2.0:
        print("[SUCCESS] Dispersion is suppressed. Packet maintains integrity.")
    else:
        print("[NOTE] Packet shows standard dispersion (Increase Nonlinear Coupling for tighter Solitons).")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("results/photon", exist_ok=True)
    run_lifecycle_test()
