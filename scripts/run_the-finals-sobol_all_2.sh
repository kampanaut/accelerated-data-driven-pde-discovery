#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/6] nl_heat-5-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-5-adam-mb25.yaml

echo "=== TRAIN [6/6] nl_heat-6-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-6-adam+lbfgs-5k.yaml

# === EVALUATE ===
echo "=== EVAL [5/6] nl_heat-5-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-5-adam-mb25.yaml

echo "=== EVAL [6/6] nl_heat-6-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-6-adam+lbfgs-5k.yaml

# === VISUALIZE ===
echo "=== VIS [5/6] nl_heat-5-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-5-adam-mb25.yaml

echo "=== VIS [6/6] nl_heat-6-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-6-adam+lbfgs-5k.yaml
