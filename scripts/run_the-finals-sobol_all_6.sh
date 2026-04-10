#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== TRAIN [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== TRAIN [23/36] nl_heat-23-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== TRAIN [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml

# === EVALUATE ===
echo "=== EVAL [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== EVAL [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== EVAL [23/36] nl_heat-23-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== EVAL [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml

# === VISUALIZE ===
echo "=== VIS [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== VIS [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== VIS [23/36] nl_heat-23-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== VIS [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml
