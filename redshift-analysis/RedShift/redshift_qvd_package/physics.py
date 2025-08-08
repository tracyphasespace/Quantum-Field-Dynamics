#!/usr/bin/env python3
"""
Enhanced QVD Redshift Physics Module
===================================

Core physics implementation for QVD redshift analysis with numerical stability.
Implements wavelength-independent redshift-dependent dimming with comprehensive
bounds enforcement and error handling.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import logging
from typing import Union, Dict

from .numerical_safety import (
    safe_power, safe_log10, safe_exp, safe_divide, safe_sqrt,
    validate_finite
)
from .physical_bounds import BoundsEnforcer
from .error_handling import setup_qvd_logging

logger = logging.getLogger(__name__)

class EnhancedQVDPhysics:
    """
    Implements the core physics of the wavelength-independent QVD redshift model.

    This model explains cosmological dimming as a combination of a baseline
    power-law redshift effect and an enhancement from the Intergalactic Medium (IGM).
    """

    def __init__(self,
                 qvd_coupling: float = 0.85,
                 redshift_power: float = 0.6,
                 igm_enhancement: float = 0.7,
                 enable_logging: bool = True):
        """
        Initialize the QVD physics model.

        Parameters:
        -----------
        qvd_coupling : float
            The primary coupling strength of the redshift-dimming effect.
        redshift_power : float
            The exponent of the power-law scaling with redshift (z^P).
        igm_enhancement : float
            A factor controlling the contribution of the IGM to the dimming.
        enable_logging : bool
            Whether to enable detailed logging.
        """
        if enable_logging:
            setup_qvd_logging(level=logging.INFO, enable_warnings=True)

        self.bounds_enforcer = BoundsEnforcer()

        # Store the core physical parameters of the model
        self.qvd_coupling = qvd_coupling
        self.redshift_power = redshift_power
        self.igm_enhancement = igm_enhancement

        # Physical constants used in secondary calculations
        self.sigma_thomson = 6.652e-25  # Thomson scattering cross-section in cm²

        logger.info(f"EnhancedQVDPhysics initialized: coupling={self.qvd_coupling:.3f}, "
                   f"power={self.redshift_power:.3f}, igm_enhancement={self.igm_enhancement:.3f}")

    def calculate_redshift_dimming(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the total QVD dimming for a given redshift.

        This is the main calculation, combining a baseline power-law redshift
        effect with an IGM enhancement.
        """
        # Ensure redshift values are within physical bounds for the calculation
        safe_redshift = self.bounds_enforcer.enforce_redshift(redshift, "dimming_input_redshift")

        # 1. Calculate the baseline dimming effect from the power-law
        base_dimming = self.qvd_coupling * safe_power(safe_redshift, self.redshift_power)

        # 2. Calculate the enhancement from the Intergalactic Medium (IGM)
        igm_contribution = self._calculate_igm_effects(safe_redshift)

        # 3. Combine the two effects for the total dimming
        total_dimming = base_dimming + igm_contribution

        # 4. Ensure the final result is finite and physically reasonable
        total_dimming = validate_finite(total_dimming, "total_dimming", replace_with=0.0)
        total_dimming = self.bounds_enforcer.enforce_dimming_magnitude(total_dimming, "final_dimming")

        return total_dimming

    def _calculate_igm_effects(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the contribution of the Intergalactic Medium (IGM) to the dimming.

        This phenomenological model assumes the IGM effect grows logarithmically
        with distance and is enhanced by the increasing density of the IGM at higher redshifts.
        """
        # IGM density is proportional to (1+z)^3. The effect scales with sqrt of this.
        igm_density_factor = safe_power(1 + redshift, 1.5)

        # The effect grows logarithmically with redshift (as a proxy for distance)
        log_factor = safe_log10(1 + redshift)

        # Combine the factors with the overall enhancement parameter
        igm_contribution = self.igm_enhancement * log_factor * igm_density_factor

        # Ensure the result is finite and within reasonable physical bounds
        igm_contribution = validate_finite(igm_contribution, "igm_contribution", replace_with=0.0)
        igm_contribution = self.bounds_enforcer.enforce_dimming_magnitude(igm_contribution, "igm_dimming")

        return igm_contribution

    def get_model_parameters(self) -> Dict[str, float]:
        """Returns a dictionary of the current model parameters."""
        return {
            'qvd_coupling': self.qvd_coupling,
            'redshift_power': self.redshift_power,
            'igm_enhancement': self.igm_enhancement,
        }

    def update_parameters(self, **kwargs):
        """Updates model parameters from a dictionary."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Attempted to update unknown parameter: {key}")
        logger.info(f"Updated model parameters: {kwargs}")
