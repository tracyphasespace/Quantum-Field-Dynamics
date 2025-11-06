# QFD Supernova Analysis V15 - Docker Image
# Provides reproducible environment for running the full analysis pipeline
#
# Build: docker build -t qfd-supernova-v15 .
# Run: docker run --gpus all -v $(pwd)/results:/app/results qfd-supernova-v15

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY *.py ./

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# Default command: run help
CMD ["python", "-c", "print('QFD Supernova V15 Docker Container\n\nAvailable commands:\n  python src/stage1_persn_sweep.py --help\n  python src/stage2_mcmc_numpyro.py --help\n  python src/stage3_merge.py --help\n  python scripts/make_paper_figures.py --help\n  python scripts/evaluate_holdout.py --help\n  python scripts/compare_abc_variants.py --help\n\nSee README.md for full documentation.')"]
