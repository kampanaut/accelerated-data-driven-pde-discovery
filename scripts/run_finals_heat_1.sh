#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/23] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/23] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/23] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/23] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

# === EVALUATE ===
echo "=== EVAL [1/23] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/23] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/23] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/23] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

# === VISUALIZE ===
echo "=== VIS [1/23] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/23] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/23] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/23] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml
