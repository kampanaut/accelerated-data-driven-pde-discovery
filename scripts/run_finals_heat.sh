#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/7] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/7] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/7] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/7] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== TRAIN [5/7] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/7] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/7] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

# === EVALUATE ===
echo "=== EVAL [1/7] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/7] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/7] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/7] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== EVAL [5/7] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/7] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/7] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

# === VISUALIZE ===
echo "=== VIS [1/7] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/7] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/7] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/7] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== VIS [5/7] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/7] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/7] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml
