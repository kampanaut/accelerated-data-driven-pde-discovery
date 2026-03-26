#!/bin/bash
set -e

# System dependencies for Dedalus
apt-get update && apt-get install -y libopenmpi-dev libfftw3-dev libhdf5-dev
pip install mpi4py

# Python packages
uv sync
