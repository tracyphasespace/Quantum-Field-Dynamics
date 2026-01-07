import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_nucleus_field():
    # --- CONFIGURATION ---
    # Resolution of the field
    phi_steps = 40    # Azimuthal resolution
    theta_steps = 20  # Polar resolution
    layers = 6        # Number of shells in the "atmosphere"
    
    # Resonance Parameters (The "Chladni" Logic)
    # These represent specific harmonic modes in the phase space
    l_mode = 4  # Polar nodes (latitude bands)
    m_mode = 3  # Azimuthal nodes (longitudinal stripes)
    amplitude = 0.3 # Intensity of the resonance deformation

    # --- GEOMETRY GENERATION ---
    # We define the space not as a Cartesian grid, but as a series of 
    # spherical manifolds (shells).
    
    phi = np.linspace(0, 2 * np.pi, phi_steps)
    theta = np.linspace(0, np.pi, theta_steps)
    phi, theta = np.meshgrid(phi, theta)

    # Initialize lists to hold our vector data
    all_x, all_y, all_z = [], [], []
    all_u, all_v, all_w = [], [], [] # Vector components
    all_colors = []

    # Iterate through spherical shells (The "Atmosphere")
    for r_base in np.linspace(1.0, 4.0, layers):
        
        # --- CL(3,3) RESONANCE SIMULATION ---
        # Instead of a static radius, the manifold is deformed by a standing wave.
        # This simulates the "pulsing" bivector field.
        # Function: Spherical Harmonic approximation Y(l,m)
        resonance_scalar = np.sin(l_mode * theta) * np.cos(m_mode * phi)
        
        # Apply deformation to the radius
        # Nodes (zero points) remain at r_base; Antinodes extend/contract
        r_deformed = r_base * (1 + amplitude * resonance_scalar)

        # Convert Spherical -> Cartesian (embedding the manifold)
        x = r_deformed * np.sin(theta) * np.cos(phi)
        y = r_deformed * np.sin(theta) * np.sin(phi)
        z = r_deformed * np.cos(theta)

        # --- 1/r FIELD DYNAMICS ---
        # The vector magnitude (flow) must decay as 1/r
        # We also orient the vectors normal to the surface (radial flow)
        
        # Normalized direction vectors (Unit Normals)
        norm_factor = np.sqrt(x**2 + y**2 + z**2)
        u_dir = x / norm_factor
        v_dir = y / norm_factor
        w_dir = z / norm_factor

        # 1/r decay magnitude
        # Note: We square r in denominator if modeling strict intensity, 
        # but 1/r is requested for "atmosphere" potential/field falloff.
        magnitude = 1.0 / r_deformed 
        
        # Apply resonance to the VECTOR MAGNITUDE as well 
        # (The field is stronger at antinodes)
        magnitude *= (1 + 0.5 * resonance_scalar)

        u = u_dir * magnitude
        v = v_dir * magnitude
        w = w_dir * magnitude

        # Color coding based on Phase (Resonance Scalar) rather than just height
        # This highlights the "Chladni" nodal structure
        color_values = resonance_scalar.flatten()

        # Append to lists
        all_x.append(x.flatten())
        all_y.append(y.flatten())
        all_z.append(z.flatten())
        all_u.append(u.flatten())
        all_v.append(v.flatten())
        all_w.append(w.flatten())
        all_colors.append(color_values)

    # Flatten logic for plotting
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    w = np.concatenate(all_w)
    c = np.concatenate(all_colors)

    # --- VISUALIZATION ---
    import os
    from datetime import datetime

    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Remove axes for clean "void" look
    ax.set_axis_off()

    # Quiver Plot
    # cmap='coolwarm' nicely separates the positive/negative phases of the wave
    q = ax.quiver(x, y, z, u, v, w, length=0.4, normalize=False, 
                  cmap='coolwarm', array=c, linewidths=1.5, arrow_length_ratio=0.3)

    # Enforce aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title("Nucleus: Resonant 1/r Atmosphere (Mode l=4, m=3)", color='white')
    
    # --- Save Figure ---
    results_dir = "results/nucleus_visualization"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/{timestamp}_nucleus_l4_m3.png"
    plt.savefig(filename, facecolor='black')
    plt.close(fig)
    print(f"Saved nucleus visualization to: {filename}")


if __name__ == "__main__":
    generate_nucleus_field()
