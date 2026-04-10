#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== TRAIN [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== TRAIN [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== TRAIN [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== EVAL [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== EVAL [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== EVAL [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== VIS [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== VIS [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== VIS [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml
