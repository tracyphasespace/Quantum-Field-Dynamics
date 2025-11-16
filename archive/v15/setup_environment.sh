#!/bin/bash
#
# Environment Setup Script for QFD Supernova V15
#
# This script sets up a complete Python environment for reproducing the paper results.
#
# Usage:
#   ./setup_environment.sh
#
# What it does:
#   1. Checks system requirements
#   2. Creates Python virtual environment
#   3. Installs dependencies
#   4. Verifies installation
#   5. Runs basic tests
#

set -e  # Exit on error

echo "================================================================================"
echo "QFD SUPERNOVA V15 - ENVIRONMENT SETUP"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check system requirements
echo "Step 1: Checking system requirements..."
echo ""

# Check Python version
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    print_status "Python 3.12 found: $(python3.12 --version)"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$PYTHON_VERSION" == "3.12" ]] || [[ "$PYTHON_VERSION" == "3.13" ]] || [[ "$PYTHON_VERSION" > "3.12" ]]; then
        print_status "Python found: $(python3 --version)"
    else
        print_error "Python 3.12+ required, found: $(python3 --version)"
        echo "Install Python 3.12+ first:"
        echo "  Ubuntu/Debian: sudo apt install python3.12"
        echo "  Or download from: https://www.python.org/downloads/"
        exit 1
    fi
else
    print_error "Python 3 not found!"
    echo "Install Python 3.12+ first:"
    echo "  Ubuntu/Debian: sudo apt install python3.12"
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected - will use CPU (50-100x slower!)"
    GPU_AVAILABLE=false
fi

# Check disk space
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_status "Available disk space: $AVAILABLE_SPACE"

# Check RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | awk 'NR==2 {print $2}')
    print_status "Total RAM: $TOTAL_RAM"
fi

echo ""

# Step 2: Create virtual environment
echo "Step 2: Creating Python virtual environment..."
echo ""

if [[ -d ".venv" ]]; then
    print_warning "Virtual environment already exists at .venv"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        print_status "Removed old environment"
    else
        print_status "Using existing environment"
    fi
fi

if [[ ! -d ".venv" ]]; then
    $PYTHON_CMD -m venv .venv
    print_status "Created virtual environment at .venv"
fi

# Activate virtual environment
source .venv/bin/activate
print_status "Activated virtual environment"

echo ""

# Step 3: Upgrade pip
echo "Step 3: Upgrading pip..."
echo ""

python -m pip install --upgrade pip --quiet
print_status "pip upgraded to $(pip --version | cut -d' ' -f2)"

echo ""

# Step 4: Install dependencies
echo "Step 4: Installing Python packages..."
echo ""

if [[ ! -f "requirements.txt" ]]; then
    print_error "requirements.txt not found!"
    exit 1
fi

echo "This may take 5-10 minutes..."
echo ""

if [[ "$GPU_AVAILABLE" == "true" ]]; then
    print_status "Installing with GPU support (CUDA 13+)..."
    pip install -r requirements.txt --quiet
else
    print_warning "Installing CPU-only version (no GPU)..."
    # Install CPU-only JAX
    pip install numpy scipy pandas matplotlib seaborn corner arviz --quiet
    pip install "jax[cpu]" --quiet
    pip install numpyro --quiet
fi

print_status "All packages installed"

echo ""

# Step 5: Verify installation
echo "Step 5: Verifying installation..."
echo ""

# Test imports
python -c "
import sys
import jax
import numpy as np
import numpyro
import arviz as az
import matplotlib
import pandas as pd
import scipy

print('✓ All core packages imported successfully')
print()
print('Package versions:')
print(f'  JAX:      {jax.__version__}')
print(f'  NumPyro:  {numpyro.__version__}')
print(f'  NumPy:    {np.__version__}')
print(f'  SciPy:    {scipy.__version__}')
print(f'  Pandas:   {pd.__version__}')
print(f'  ArviZ:    {az.__version__}')
print()
print('JAX devices:', jax.devices())
"

if [[ $? -eq 0 ]]; then
    print_status "Installation verified"
else
    print_error "Verification failed!"
    exit 1
fi

echo ""

# Step 6: Check data
echo "Step 6: Checking data files..."
echo ""

if [[ -f "data/lightcurves_unified_v2_min3.csv" ]]; then
    LINES=$(wc -l < data/lightcurves_unified_v2_min3.csv)
    print_status "Lightcurves file found: $LINES lines"

    if [[ $LINES -gt 100000 ]]; then
        print_status "Data looks complete (~118k lines expected)"
    else
        print_warning "Data file seems small (expected ~118k lines)"
    fi
else
    print_error "Lightcurves data not found!"
    echo "Expected: data/lightcurves_unified_v2_min3.csv"
    echo "See data/README.md for instructions"
fi

echo ""

# Step 7: Run basic tests
echo "Step 7: Running basic tests..."
echo ""

if [[ -f "tests/test_regression_nov5.py" ]]; then
    # Just check the code is valid, don't require results yet
    python -c "
import sys
sys.path.insert(0, 'tests')
try:
    import test_regression_nov5
    print('✓ Regression test module loaded')
except Exception as e:
    print(f'✗ Could not load test module: {e}')
    sys.exit(1)
"
    print_status "Test modules loadable"
else
    print_warning "Regression tests not found (optional)"
fi

echo ""

# Step 8: Summary
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Environment ready at: .venv"
echo ""
echo "To activate this environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Verify data is present: ls -lh data/"
echo "  2. Run tests: python tests/test_regression_nov5.py"
echo "  3. Run pipeline: ./scripts/run_full_pipeline.sh"
echo ""

if [[ "$GPU_AVAILABLE" == "true" ]]; then
    echo "GPU detected - pipeline should take 4-6 hours"
else
    echo "⚠️  No GPU - pipeline will take 24-48 hours!"
    echo "Consider using a machine with NVIDIA GPU for reasonable runtime"
fi

echo ""
echo "For full instructions, see: REPRODUCTION_GUIDE.md"
echo "================================================================================"
