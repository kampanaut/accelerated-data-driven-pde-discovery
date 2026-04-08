#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/8] nl_heat-5-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-5-adam+lbfgs.yaml

echo "=== TRAIN [6/8] nl_heat-6-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-6-adam-mb25.yaml

echo "=== TRAIN [7/8] nl_heat-7-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-7-adam+lbfgs-5k.yaml

echo "=== TRAIN [8/8] nl_heat-8-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [5/8] nl_heat-5-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-5-adam+lbfgs.yaml

echo "=== EVAL [6/8] nl_heat-6-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-6-adam-mb25.yaml

echo "=== EVAL [7/8] nl_heat-7-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-7-adam+lbfgs-5k.yaml

echo "=== EVAL [8/8] nl_heat-8-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [5/8] nl_heat-5-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-5-adam+lbfgs.yaml

echo "=== VIS [6/8] nl_heat-6-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-6-adam-mb25.yaml

echo "=== VIS [7/8] nl_heat-7-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-7-adam+lbfgs-5k.yaml

echo "=== VIS [8/8] nl_heat-8-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs-mb25.yaml
