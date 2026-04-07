#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/6] br-5-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/br-5-adam.yaml

echo "=== TRAIN [6/6] br-6-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/br-6-adam+lbfgs.yaml

# === EVALUATE ===
echo "=== EVAL [5/6] br-5-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/br-5-adam.yaml

echo "=== EVAL [6/6] br-6-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/br-6-adam+lbfgs.yaml

# === VISUALIZE ===
echo "=== VIS [5/6] br-5-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/br-5-adam.yaml

echo "=== VIS [6/6] br-6-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/br-6-adam+lbfgs.yaml
