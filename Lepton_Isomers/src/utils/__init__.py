"""
QFD Utils Module
===============

Utility functions for analysis, I/O operations, and stability prediction.

Includes:
- Standard analysis and I/O utilities  
- Integrated stability analysis for lepton lifetime modeling via geometric indices
- Legacy stability predictor (for canonical results compatibility)
"""

from .analysis import analyze_results
from .io import save_results
from .stability_analysis import StabilityPredictor
# stability_predictor (legacy) imported as needed

__all__ = ['analyze_results', 'save_results', 'StabilityPredictor']