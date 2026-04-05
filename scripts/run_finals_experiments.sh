#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/1] heat-1 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1.yaml

# === EVALUATE ===
echo "=== EVAL [1/1] heat-1 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1.yaml

# === VISUALIZE ===
echo "=== VIS [1/1] heat-1 ==="
uv run python scripts/visualize.py --config configs/finals/heat-1.yaml
