#!/bin/bash
set -e
cd /

# System dependencies for Dedalus
if ! dpkg -s libopenmpi-dev &>/dev/null; then
    apt-get update && apt-get install -y libopenmpi-dev libfftw3-dev libfftw3-mpi-dev libhdf5-mpi-dev unzip
else
    echo "System deps already installed, skipping"
fi

# uv package manager
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv already installed, skipping"
fi

# Yazi file manager (musl build — no glibc version dependency)
if ! command -v yazi &>/dev/null; then
    wget -q https://github.com/sxyazi/yazi/releases/latest/download/yazi-x86_64-unknown-linux-musl.zip -O /tmp/yazi.zip
    unzip -q /tmp/yazi.zip -d /tmp/yazi
    mv /tmp/yazi/yazi-x86_64-unknown-linux-musl/yazi /usr/local/bin/
    rm -rf /tmp/yazi /tmp/yazi.zip
else
    echo "yazi already installed, skipping"
fi

# Hugging Face CLI
if ! command -v hf &>/dev/null; then
    pip install huggingface_hub[cli]
else
    echo "hf CLI already installed, skipping"
fi

# Workspace setup
mkdir -p /workspace/models
mkdir -p /content
cd /content

if [ ! -d ".git" ]; then
    git clone https://github.com/kampanaut/accelerated-data-driven-pde-discovery.git .
else
    echo "Repo already cloned, pulling latest"
    git pull
fi

mkdir -p data/datasets
[ -L data/models ] || ln -s /workspace/models data/models

# Dedalus needs MPI/FFTW paths explicitly
export MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi
export FFTW_PATH=/usr/lib/x86_64-linux-gnu
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/openmpi:${C_INCLUDE_PATH:-}

export UV_CACHE_DIR=/root/.cache/uv

# Python packages
uv sync

# Make runner scripts executable
chmod u+x scripts/*.sh

# Download datasets
# Usage: ./install.sh [heat|nl_heat|br|all]  (no flag = all)
REPO="kampanaut/maml-pde-datasets"

download_heat() {
    [ -d data/datasets/heat_train-1 ] && echo "heat_train-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "heat_train-1/*" --local-dir data/datasets
    [ -d data/datasets/heat_train-2 ] && echo "heat_train-2 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "heat_train-2/*" --local-dir data/datasets
    [ -d data/datasets/heat_val-1 ] && echo "heat_val-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "heat_val-1/*" --local-dir data/datasets
    [ -d data/datasets/heat_test-1 ] && echo "heat_test-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "heat_test-1/*" --local-dir data/datasets
}

download_nl_heat() {
    [ -d data/datasets/nl_heat_train-1 ] && echo "nl_heat_train-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "nl_heat_train-1/*" --local-dir data/datasets
    [ -d data/datasets/nl_heat_val-1 ] && echo "nl_heat_val-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "nl_heat_val-1/*" --local-dir data/datasets
    [ -d data/datasets/nl_heat_test-1 ] && echo "nl_heat_test-1 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "nl_heat_test-1/*" --local-dir data/datasets
}

download_br() {
    [ -d data/datasets/br_train-2 ] && echo "br_train-2 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "br_train-2/*" --local-dir data/datasets
    [ -d data/datasets/br_val-2 ] && echo "br_val-2 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "br_val-2/*" --local-dir data/datasets
    [ -d data/datasets/br_test-2 ] && echo "br_test-2 exists, skipping" || \
        hf download $REPO --repo-type dataset --include "br_test-2/*" --local-dir data/datasets
}

download_dir() {
    local dir="$1"
    [ -d "data/datasets/$dir" ] && echo "$dir exists, skipping" || \
        hf download $REPO --repo-type dataset --include "$dir/*" --local-dir data/datasets
}

DATASETS="${@:-all}"
for ds in $DATASETS; do
    case "$ds" in
        heat)    download_heat ;;
        nl_heat) download_nl_heat ;;
        br)      download_br ;;
        all)     download_heat && download_nl_heat && download_br ;;
        *)       download_dir "$ds" ;;
    esac
done
