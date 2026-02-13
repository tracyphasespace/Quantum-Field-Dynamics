# QFD Reproducible Validation Environment
# ========================================
# Builds a container with Python + all dependencies for running
# QFD validation scripts. Lean 4 is NOT included (too large for CI;
# Lean builds are verified separately).
#
# Usage:
#   docker build -t qfd-validation .
#   docker run qfd-validation
#
# The default CMD runs all validation scripts.

FROM python:3.12-slim

LABEL maintainer="Tracy McSheery"
LABEL description="QFD validation environment for reproducible results"
LABEL version="1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt* ./
RUN pip install --no-cache-dir numpy scipy 2>/dev/null || true
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy project files
COPY qfd/ ./qfd/
COPY validation_scripts/ ./validation_scripts/
COPY projects/particle-physics/lepton-isomer-ladder/ ./projects/particle-physics/lepton-isomer-ladder/
COPY projects/particle-physics/soliton-fragmentation/ ./projects/particle-physics/soliton-fragmentation/

# Verify shared constants on build
RUN python -m qfd.shared_constants

# Default: run core validation scripts
CMD ["python", "validation_scripts/run_core_validations.py"]
