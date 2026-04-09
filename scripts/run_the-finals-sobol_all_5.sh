#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== TRAIN [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== TRAIN [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== TRAIN [20/30] nl_heat-20-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml

# === EVALUATE ===
echo "=== EVAL [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== EVAL [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== EVAL [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== EVAL [20/30] nl_heat-20-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml

# === VISUALIZE ===
echo "=== VIS [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== VIS [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== VIS [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== VIS [20/30] nl_heat-20-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml
