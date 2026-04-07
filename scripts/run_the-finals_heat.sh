#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/6] heat-1-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/heat-1-adam.yaml

echo "=== TRAIN [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/heat-2-adam+lbfgs.yaml

# === EVALUATE ===
echo "=== EVAL [1/6] heat-1-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/heat-1-adam.yaml

echo "=== EVAL [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/heat-2-adam+lbfgs.yaml

# === VISUALIZE ===
echo "=== VIS [1/6] heat-1-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/heat-1-adam.yaml

echo "=== VIS [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/heat-2-adam+lbfgs.yaml
