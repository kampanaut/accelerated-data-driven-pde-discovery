#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/50] heat-25-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-25-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== TRAIN [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== TRAIN [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml

# === EVALUATE ===
echo "=== EVAL [25/50] heat-25-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-25-100x150x450-anil-2epoch.yaml

echo "=== EVAL [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== EVAL [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== EVAL [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml

# === VISUALIZE ===
echo "=== VIS [25/50] heat-25-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-25-100x150x450-anil-2epoch.yaml

echo "=== VIS [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== VIS [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== VIS [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml
