#!/bin/bash
set -e
cd /

# System dependencies for Dedalus
apt-get update && apt-get install -y libopenmpi-dev libfftw3-dev libfftw3-mpi-dev libhdf5-mpi-dev

# Hugging Face CLI
pip install huggingface_hub[cli]

# Workspace setup
mkdir -p /workspace/models
mkdir -p /content
cd /content
git clone https://github.com/kampanaut/accelerated-data-driven-pde-discovery.git .

mkdir -p data/datasets
ln -s /workspace/models data/models

# Dedalus needs MPI/FFTW paths explicitly
export MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi
export FFTW_PATH=/usr/lib/x86_64-linux-gnu
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/openmpi:${C_INCLUDE_PATH:-}

# Python packages
uv sync

# Download Heat dataset
hf download kampanaut/ai-chaos-theory-heat --repo-type dataset --include "heat_*_00*" --local-dir data/datasets/heat_train-1 && \
hf download kampanaut/ai-chaos-theory-heat --repo-type dataset --include "heat_*_v0*" --local-dir data/datasets/heat_val-1 && \
hf download kampanaut/ai-chaos-theory-heat --repo-type dataset --include "heat_*_t0*" --local-dir data/datasets/heat_test-1
