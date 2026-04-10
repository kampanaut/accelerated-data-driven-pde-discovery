#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== TRAIN [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== TRAIN [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== TRAIN [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== EVAL [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== EVAL [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== EVAL [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== VIS [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== VIS [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== VIS [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml
