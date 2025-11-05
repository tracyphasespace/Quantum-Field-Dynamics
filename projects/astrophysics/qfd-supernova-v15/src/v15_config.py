"""V15 configuration management for the QFD supernova pipeline."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class SamplerConfig:
    """Configuration for emcee sampler (proven architecture from V1)."""

    n_walkers: int = 32
    """Number of emcee walkers (V1 used 32, worked well)"""

    n_steps: int = 5000
    """Total MCMC steps"""

    n_burn: int = 1000
    """Burn-in steps to discard"""

    moves: str = "stretch"
    """emcee move strategy ('stretch' is default, proven)"""

    backend: Optional[str] = None
    """HDF5 backend file for checkpointing (optional)"""

    n_threads: int = 1
    """Number of CPU threads for parallel SN evaluation (1=sequential, >1=parallel)"""


@dataclass
class PhysicsConfig:
    """QFD physics parameters."""

    # Fixed parameters (from V1 benchmarks)
    k_J_init: float = 70.0
    """Initial k_J (Hubble-like parameter) [km/s/Mpc]"""

    eta_prime_init: float = 0.01
    """Initial eta_prime (QFD coupling)"""

    xi_init: float = 30.0
    """Initial xi (QFD parameter)"""

    # Priors (from V1)
    k_J_min: float = 20.0
    k_J_max: float = 120.0

    eta_prime_min: float = 1e-4
    eta_prime_max: float = 0.1

    xi_min: float = 1.0
    xi_max: float = 100.0


@dataclass
class DataConfig:
    """Data loading and filtering."""

    lightcurves_path: Path
    """Path to lightcurves CSV file"""

    n_sne: Optional[int] = None
    """Number of SNe to fit (None = all)"""

    start_sne: int = 0
    """Starting index for SNe selection"""

    z_min: Optional[float] = None
    """Minimum redshift filter"""

    z_max: Optional[float] = None
    """Maximum redshift filter"""

    require_peak: bool = True
    """Require SNe to have peak detection"""


@dataclass
class OutputConfig:
    """Output configuration."""

    output_dir: Path
    """Directory for results"""

    save_chains: bool = True
    """Save full MCMC chains"""

    save_corner: bool = True
    """Generate corner plot"""

    tag: str = ""
    """Optional tag for output files"""


@dataclass
class V15Config:
    """Complete V15 configuration."""

    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(
        lightcurves_path=Path("../data/unified/lightcurves_unified_v2_clean.csv")
    ))
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        output_dir=Path("../results/v15_test/")
    ))

    # Reproducibility
    random_seed: int = 42

    # Logging
    verbose: bool = True
    log_file: Optional[Path] = None

    @classmethod
    def from_args(cls, args):
        """Create config from CLI arguments."""
        return cls(
            sampler=SamplerConfig(
                n_walkers=getattr(args, 'n_walkers', 32),
                n_steps=getattr(args, 'n_steps', 5000),
                n_burn=getattr(args, 'n_burn', 1000),
            ),
            data=DataConfig(
                lightcurves_path=Path(args.lightcurves),
                n_sne=getattr(args, 'n_sne', None),
                start_sne=getattr(args, 'start_sne', 0),
                z_min=getattr(args, 'z_min', None),
                z_max=getattr(args, 'z_max', None),
            ),
            output=OutputConfig(
                output_dir=Path(getattr(args, 'outdir', '../results/v15_test/')),
                tag=getattr(args, 'tag', ''),
            )
        )

    def validate(self):
        """Validate configuration."""
        assert self.sampler.n_walkers > 0, "Need at least 1 walker"
        assert self.sampler.n_steps > 0, "Need at least 1 step"
        assert self.data.lightcurves_path.exists(), f"Data file not found: {self.data.lightcurves_path}"

        # Ensure output directory exists
        self.output.output_dir.mkdir(parents=True, exist_ok=True)
