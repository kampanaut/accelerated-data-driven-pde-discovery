#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

# === EVALUATE ===
echo "=== EVAL [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

# === VISUALIZE ===
echo "=== VIS [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml
