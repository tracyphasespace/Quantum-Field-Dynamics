#!/usr/bin/env python3
"""
Interactive 3D Nucleus Visualization - Resonant 1/r Atmosphere

Creates a self-contained HTML file with rotatable 3D vector field.
Uses Plotly for browser-based interactivity (no server needed).

Usage:
    python nucleus_3d_interactive.py
    # Opens nucleus_resonance.html in browser
"""

import numpy as np
import plotly.graph_objects as go

def generate_nucleus_field():
    # --- CONFIGURATION ---
    phi_steps = 60    # Azimuthal resolution (increased)
    theta_steps = 40  # Polar resolution (increased)
    layers = 12       # Number of shells in the "atmosphere" (increased)

    # Resonance Parameters (The "Chladni" Logic)
    l_mode = 4  # Polar nodes (latitude bands)
    m_mode = 3  # Azimuthal nodes (longitudinal stripes)
    amplitude = 0.35  # Intensity of the resonance deformation

    # --- GEOMETRY GENERATION ---
    phi = np.linspace(0, 2 * np.pi, phi_steps)
    theta = np.linspace(0.1, np.pi - 0.1, theta_steps)  # Avoid poles
    phi, theta = np.meshgrid(phi, theta)

    all_x, all_y, all_z = [], [], []
    all_u, all_v, all_w = [], [], []
    all_colors = []

    for r_base in np.linspace(0.8, 2.5, layers):  # Tighter shell spacing
        # Cl(3,3) Resonance - Spherical Harmonic approximation
        resonance_scalar = np.sin(l_mode * theta) * np.cos(m_mode * phi)

        # Deformed radius
        r_deformed = r_base * (1 + amplitude * resonance_scalar)

        # Spherical -> Cartesian
        x = r_deformed * np.sin(theta) * np.cos(phi)
        y = r_deformed * np.sin(theta) * np.sin(phi)
        z = r_deformed * np.cos(theta)

        # 1/r field with resonance modulation
        norm_factor = np.sqrt(x**2 + y**2 + z**2)
        magnitude = (1.0 / r_deformed) * (1 + 0.5 * resonance_scalar)

        u = (x / norm_factor) * magnitude
        v = (y / norm_factor) * magnitude
        w = (z / norm_factor) * magnitude

        all_x.append(x.flatten())
        all_y.append(y.flatten())
        all_z.append(z.flatten())
        all_u.append(u.flatten())
        all_v.append(v.flatten())
        all_w.append(w.flatten())
        all_colors.append(resonance_scalar.flatten())

    # Concatenate all layers
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    w = np.concatenate(all_w)
    c = np.concatenate(all_colors)

    return x, y, z, u, v, w, c


def create_reference_geometry():
    """Create RGB axes and reference circle for orientation."""
    traces = []

    axis_len = 3.5

    # X axis (Red)
    traces.append(go.Scatter3d(
        x=[0, axis_len], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=6),
        text=['', 'X'],
        textposition='top center',
        textfont=dict(color='red', size=16),
        name='X axis',
        showlegend=False
    ))

    # Y axis (Green)
    traces.append(go.Scatter3d(
        x=[0, 0], y=[0, axis_len], z=[0, 0],
        mode='lines+text',
        line=dict(color='lime', width=6),
        text=['', 'Y'],
        textposition='top center',
        textfont=dict(color='lime', size=16),
        name='Y axis',
        showlegend=False
    ))

    # Z axis (Blue)
    traces.append(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_len],
        mode='lines+text',
        line=dict(color='cyan', width=6),
        text=['', 'Z'],
        textposition='top center',
        textfont=dict(color='cyan', size=16),
        name='Z axis',
        showlegend=False
    ))

    # Reference circle in XY plane (equator)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    r_circle = 3.0
    traces.append(go.Scatter3d(
        x=r_circle * np.cos(theta_circle),
        y=r_circle * np.sin(theta_circle),
        z=np.zeros_like(theta_circle),
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='XY equator',
        showlegend=False
    ))

    # Reference circle in XZ plane (meridian)
    traces.append(go.Scatter3d(
        x=r_circle * np.cos(theta_circle),
        y=np.zeros_like(theta_circle),
        z=r_circle * np.sin(theta_circle),
        mode='lines',
        line=dict(color='yellow', width=2, dash='dot'),
        name='XZ meridian',
        showlegend=False
    ))

    return traces


def create_interactive_html(output_file="nucleus_resonance.html"):
    """Generate interactive 3D HTML visualization."""

    x, y, z, u, v, w, c = generate_nucleus_field()

    # Normalize colors to [0, 1] for colorscale
    c_norm = (c - c.min()) / (c.max() - c.min())

    # Start with reference geometry
    ref_traces = create_reference_geometry()

    # Create Cone plot (Plotly's 3D vector field)
    fig = go.Figure(data=ref_traces + [go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        colorscale='RdBu_r',  # Red-Blue diverging (like coolwarm)
        cmin=-1, cmax=1,
        sizemode="absolute",
        sizeref=0.08,  # Smaller cones for denser appearance
        showscale=True,
        colorbar=dict(
            title="Phase",
            tickvals=[-1, 0, 1],
            ticktext=["−", "Node", "+"]
        ),
        hovertemplate=(
            "x: %{x:.2f}<br>"
            "y: %{y:.2f}<br>"
            "z: %{z:.2f}<br>"
            "magnitude: %{u:.3f}<extra></extra>"
        )
    )])

    # Dark theme matching original
    fig.update_layout(
        title=dict(
            text="Nucleus: Resonant 1/r Atmosphere (Mode l=4, m=3)",
            font=dict(color="white", size=18),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgb(10, 10, 20)',
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(10, 10, 20)',
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Add instructions annotation
    fig.add_annotation(
        text="🖱️ Drag to rotate | Scroll to zoom | Shift+drag to pan",
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        showarrow=False,
        font=dict(color="gray", size=12),
        bgcolor="rgba(0,0,0,0.5)"
    )

    # Export to standalone HTML
    fig.write_html(
        output_file,
        include_plotlyjs=True,  # Self-contained (no CDN needed)
        full_html=True,
        config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'displaylogo': False
        }
    )

    print(f"✓ Created: {output_file}")
    print(f"  Open in browser to interact with 3D visualization")

    return fig


if __name__ == "__main__":
    import os

    # Output to visualizations directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "nucleus_resonance.html")

    fig = create_interactive_html(output_path)

    # Also show in browser if running interactively
    try:
        fig.show()
    except Exception:
        pass  # Headless environment
