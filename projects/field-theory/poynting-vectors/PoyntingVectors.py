import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
frames = 100
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
omega = 2 * np.pi / frames

# --- Figure with 3 subplots ---
fig = plt.figure(figsize=(17, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.5])

# === 1. Circular Polarization Arrow (yz-plane) ===
ax1 = fig.add_subplot(gs[0])
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dotted')
ax1.add_artist(circle)
arrow, = ax1.plot([], [], 'bo-', linewidth=3, markersize=10, label='E tip')
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_xlabel('y')
ax1.set_ylabel('z')
ax1.set_title('Circular Polarization: E Tip in yz-plane')
ax1.legend()

# === 2. Static 3D "Wave Ribbons" (snapshot) ===
ax2 = fig.add_subplot(gs[1], projection='3d')
phase_static = 0
Ey_ribbon = np.cos(x + phase_static)
Bz_ribbon = np.cos(x + phase_static)
Sx = Ey_ribbon * Bz_ribbon

ax2.plot(x, Ey_ribbon, np.zeros_like(x), color='b', linewidth=2, label='E field tip ($e_2$)')
ax2.plot(x, np.zeros_like(x), Bz_ribbon, color='g', linewidth=2, label='B field tip ($e_3$)')
ax2.plot(x, 0.1 * Sx, 0.1 * Sx, color='orange', linewidth=2, label='Poynting $\\vec{S}$ (ribbon, scaled)')
ax2.set_xlim(x[0], x[-1])
ax2.set_ylim(-1.1, 1.1)
ax2.set_zlim(-1.1, 1.1)
ax2.set_xlabel('x (propagation)')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('3D E, B, and Poynting Vectors\n(static snapshot)')
ax2.legend()

# === 3. Animated 3D Wave Ribbons ===
ax3 = fig.add_subplot(gs[2], projection='3d')
lineE, = ax3.plot([], [], [], 'b', linewidth=2, label='E field tip ($e_2$)')
lineB, = ax3.plot([], [], [], 'g', linewidth=2, label='B field tip ($e_3$)')
lineS, = ax3.plot([], [], [], 'orange', linewidth=2, label='Poynting $\\vec{S}$ (scaled)')
ax3.set_xlim(x[0], x[-1])
ax3.set_ylim(-1.1, 1.1)
ax3.set_zlim(-1.1, 1.1)
ax3.set_xlabel('x (propagation)')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('Animated 3D E, B, and Poynting\n(wave ribbons)')
ax3.legend()

def update(frame):
    phase = frame * omega
    # 1. Circular polarization arrow
    Ey = np.cos(phase)
    Ez = np.sin(phase)
    arrow.set_data([0, Ey], [0, Ez])

    # 3. Animated 3D wave ribbons
    Ey_ribbon = np.cos(x + phase)
    Bz_ribbon = np.cos(x + phase)
    Sx = Ey_ribbon * Bz_ribbon
    lineE.set_data(x, Ey_ribbon)
    lineE.set_3d_properties(np.zeros_like(x))
    lineB.set_data(x, np.zeros_like(x))
    lineB.set_3d_properties(Bz_ribbon)
    lineS.set_data(x, 0.1 * Sx)
    lineS.set_3d_properties(0.1 * Sx)

    return arrow, lineE, lineB, lineS

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

plt.tight_layout()
plt.show()
