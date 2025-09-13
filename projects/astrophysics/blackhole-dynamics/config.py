import torch

class SimConfig:
    # --- Physics Parameters ---
    ALPHA_1: float = 1.0
    ALPHA_2: float = 0.1
    PHI_VAC: float = 1.0
    K_M: float = -1.0
    PARTICLE_MASS: float = 1.0

    # --- Field Solution ---
    R_MIN_ODE: float = 1e-8
    R_MAX_ODE: float = 100.0
    PHI_0_VALUES: list[float] = [3.0 * PHI_VAC, 5.0 * PHI_VAC, 8.0 * PHI_VAC]

    # --- Trajectory Simulation ---
    ODE_RTOL: float = 1e-9
    ODE_ATOL: float = 1e-11
    ENERGY_CONSERVATION_RTOL: float = 1e-5
    RECAPTURE_RADIUS_FACTOR: float = 5.0
    ESCAPE_RADIUS_FACTOR: float = 20.0

    # --- Statistical Analysis ---
    NUM_PARTICLES_STATS: int = 20
    D_VALUES_STATS_NORM: list[float] = [5.0, 10.0, 20.0]
    MASS_RATIOS_STATS: list[float] = [0.5, 1.0]

    # --- Execution Settings ---
    RUN_FIELD_SOLUTION: bool = True
    RUN_SADDLE_ANALYSIS: bool = True
    RUN_TRAJECTORY_SIM: bool = True
    RUN_ESCAPE_STATS: bool = True
    GENERATE_FIGURES: bool = True
    USE_GPU_ODE: bool = torch.cuda.is_available()
    USE_GPU_SADDLE_SEARCH: bool = torch.cuda.is_available()

    # --- Plotting ---
    FIGURE_DIR: str = "figures_refactored"

    # --- Type Hints ---
    if torch.cuda.is_available():
        TORCH_DTYPE = torch.float64
        DEVICE = torch.device('cuda')
    else:
        TORCH_DTYPE = None
        DEVICE = 'cpu'