#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/23] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== TRAIN [18/23] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [19/23] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [20/23] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

# === EVALUATE ===
echo "=== EVAL [17/23] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== EVAL [18/23] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [19/23] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [20/23] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

# === VISUALIZE ===
echo "=== VIS [17/23] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== VIS [18/23] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== VIS [19/23] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== VIS [20/23] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml
