#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/11] heat-9-silu-lam01-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== TRAIN [10/11] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== TRAIN [11/11] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

# === EVALUATE ===
echo "=== EVAL [9/11] heat-9-silu-lam01-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== EVAL [10/11] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== EVAL [11/11] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

# === VISUALIZE ===
echo "=== VIS [9/11] heat-9-silu-lam01-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== VIS [10/11] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== VIS [11/11] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml
