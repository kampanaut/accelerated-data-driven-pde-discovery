#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== TRAIN [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== TRAIN [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== TRAIN [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml

# === EVALUATE ===
echo "=== EVAL [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== EVAL [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== EVAL [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== EVAL [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml

# === VISUALIZE ===
echo "=== VIS [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== VIS [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== VIS [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== VIS [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml
