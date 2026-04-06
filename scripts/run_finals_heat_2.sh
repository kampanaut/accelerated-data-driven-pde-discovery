#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/8] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/8] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/8] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== TRAIN [8/8] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

# === EVALUATE ===
echo "=== EVAL [5/8] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/8] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/8] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== EVAL [8/8] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

# === VISUALIZE ===
echo "=== VIS [5/8] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/8] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/8] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== VIS [8/8] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml
