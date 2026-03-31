#!/bin/bash
set -e

# System dependencies for Dedalus
apt-get update && apt-get install -y libopenmpi-dev libfftw3-dev libfftw3-mpi-dev libhdf5-mpi-dev

# GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt-get update && apt-get install -y gh

# Hugging Face CLI
pip install huggingface_hub[cli]

# Dedalus needs MPI/FFTW paths explicitly
export MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi
export FFTW_PATH=/usr/lib/x86_64-linux-gnu
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/openmpi:${C_INCLUDE_PATH:-}

# Python packages
uv sync

# Auth (interactive)
gh auth login
hf login
