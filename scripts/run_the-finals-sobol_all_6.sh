#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== TRAIN [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== TRAIN [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml

# === EVALUATE ===
echo "=== EVAL [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml

echo "=== EVAL [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== EVAL [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== EVAL [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml

# === VISUALIZE ===
echo "=== VIS [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml

echo "=== VIS [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== VIS [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== VIS [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml
