#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== TRAIN [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== TRAIN [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== TRAIN [16/22] nl_heat-16-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml

# === EVALUATE ===
echo "=== EVAL [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== EVAL [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== EVAL [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== EVAL [16/22] nl_heat-16-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml

# === VISUALIZE ===
echo "=== VIS [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== VIS [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== VIS [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== VIS [16/22] nl_heat-16-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml
