#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/2] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/2] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

# === EVALUATE ===
echo "=== EVAL [1/2] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/2] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

# === VISUALIZE ===
echo "=== VIS [1/2] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/2] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml
