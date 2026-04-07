#!/bin/bash
set -e
cd /

# System dependencies for Dedalus
apt-get update && apt-get install -y libopenmpi-dev libfftw3-dev libfftw3-mpi-dev libhdf5-mpi-dev unzip

# uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Yazi file manager (musl build — no glibc version dependency)
wget -q https://github.com/sxyazi/yazi/releases/latest/download/yazi-x86_64-unknown-linux-musl.zip -O /tmp/yazi.zip
unzip -q /tmp/yazi.zip -d /tmp/yazi
mv /tmp/yazi/yazi-x86_64-unknown-linux-musl/yazi /usr/local/bin/
rm -rf /tmp/yazi /tmp/yazi.zip

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

# Make runner scripts executable
chmod u+x scripts/*.sh

# Download datasets
# Usage: ./install.sh [heat|nl_heat|br]  (no flag = all)
REPO="kampanaut/maml-pde-datasets"

download_heat() {
    hf download $REPO --repo-type dataset --include "heat_train-1/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "heat_val-1/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "heat_test-1/*" --local-dir data/datasets
}

download_nl_heat() {
    hf download $REPO --repo-type dataset --include "nl_heat_train-1/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "nl_heat_val-1/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "nl_heat_test-1/*" --local-dir data/datasets
}

download_br() {
    hf download $REPO --repo-type dataset --include "br_train-2/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "br_val-2/*" --local-dir data/datasets
    hf download $REPO --repo-type dataset --include "br_test-2/*" --local-dir data/datasets
}

case "${1:-all}" in
    heat)    download_heat ;;
    nl_heat) download_nl_heat ;;
    br)      download_br ;;
    all)     download_heat && download_nl_heat && download_br ;;
    *)       echo "Usage: $0 [heat|nl_heat|br|all]"; exit 1 ;;
esac
