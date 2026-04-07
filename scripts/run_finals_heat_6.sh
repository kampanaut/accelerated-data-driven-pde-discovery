#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/23] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== TRAIN [22/23] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/train_maml.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml

echo "=== TRAIN [23/23] heat-23-sin-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/train_maml.py --config configs/finals/heat-23-sin-lam005-3layers-cg10-poly3-2k-sse.yaml

# === EVALUATE ===
echo "=== EVAL [21/23] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== EVAL [22/23] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/evaluate.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml

echo "=== EVAL [23/23] heat-23-sin-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/evaluate.py --config configs/finals/heat-23-sin-lam005-3layers-cg10-poly3-2k-sse.yaml

# === VISUALIZE ===
echo "=== VIS [21/23] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== VIS [22/23] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/visualize.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml

echo "=== VIS [23/23] heat-23-sin-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/visualize.py --config configs/finals/heat-23-sin-lam005-3layers-cg10-poly3-2k-sse.yaml
