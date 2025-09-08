"""
QFD Orchestration Module
=======================

High-level orchestration and workflow management for QFD simulations.

Provides complete workflows for:
- Phoenix solver ladder targeting (energy pinning)
- G-2 prediction batch processing  
- End-to-end lepton physics analysis
- CSR parameter sweeps and analysis

Proposed Three-Objective Paradigm:
- Isomer workflow for unified lepton theory
- Electron calibration → muon/tau prediction
- Field-fundamental g-2 calculations via Zeeman experiments
"""

from .ladder_solver import LadderSolver, run_electron_ladder, run_muon_ladder
from .g2_predictor_batch import G2PredictorBatch
from .g2_workflow import G2Workflow
from .isomer_workflow import IsomerWorkflow, run_complete_isomer_workflow

__all__ = [
    'LadderSolver', 
    'run_electron_ladder', 
    'run_muon_ladder',
    'G2PredictorBatch',
    'G2Workflow',
    'IsomerWorkflow',
    'run_complete_isomer_workflow'
]