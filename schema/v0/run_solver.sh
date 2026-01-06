#!/bin/bash
# QFD Grand Solver v0.3 - Launch Script
#
# Usage:
#   ./run_solver.sh experiments/ccl_fit_v1.json
#   ./run_solver.sh experiments/test_pipeline_v1.json

set -euo pipefail

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure qfd module can be found
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Run solver
python "$SCRIPT_DIR/solve_v03.py" "$@"
