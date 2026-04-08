#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== TRAIN [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== TRAIN [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== TRAIN [12/14] nl_heat-12-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml

# === EVALUATE ===
echo "=== EVAL [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== EVAL [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== EVAL [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== EVAL [12/14] nl_heat-12-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml

# === VISUALIZE ===
echo "=== VIS [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== VIS [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== VIS [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== VIS [12/14] nl_heat-12-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml
