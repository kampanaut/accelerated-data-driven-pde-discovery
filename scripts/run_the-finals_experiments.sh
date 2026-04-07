#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/6] heat-1-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/heat-1-adam.yaml

echo "=== TRAIN [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/heat-2-adam+lbfgs.yaml

echo "=== TRAIN [3/6] nl_heat-3-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== TRAIN [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml

echo "=== TRAIN [5/6] br-5-adam ==="
uv run python scripts/train_maml.py --config configs/the-finals/br-5-adam.yaml

echo "=== TRAIN [6/6] br-6-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals/br-6-adam+lbfgs.yaml

# === EVALUATE ===
echo "=== EVAL [1/6] heat-1-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/heat-1-adam.yaml

echo "=== EVAL [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/heat-2-adam+lbfgs.yaml

echo "=== EVAL [3/6] nl_heat-3-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== EVAL [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml

echo "=== EVAL [5/6] br-5-adam ==="
uv run python scripts/evaluate.py --config configs/the-finals/br-5-adam.yaml

echo "=== EVAL [6/6] br-6-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals/br-6-adam+lbfgs.yaml

# === VISUALIZE ===
echo "=== VIS [1/6] heat-1-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/heat-1-adam.yaml

echo "=== VIS [2/6] heat-2-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/heat-2-adam+lbfgs.yaml

echo "=== VIS [3/6] nl_heat-3-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/nl_heat-3-adam.yaml

echo "=== VIS [4/6] nl_heat-4-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/nl_heat-4-adam+lbfgs.yaml

echo "=== VIS [5/6] br-5-adam ==="
uv run python scripts/visualize.py --config configs/the-finals/br-5-adam.yaml

echo "=== VIS [6/6] br-6-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals/br-6-adam+lbfgs.yaml
