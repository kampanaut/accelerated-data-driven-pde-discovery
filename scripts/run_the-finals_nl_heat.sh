#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [3/6] nl_heat-3-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== TRAIN [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml

# === EVALUATE ===
echo "=== EVAL [3/6] nl_heat-3-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== EVAL [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml

# === VISUALIZE ===
echo "=== VIS [3/6] nl_heat-3-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== VIS [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml
