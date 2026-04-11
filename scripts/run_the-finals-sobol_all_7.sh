#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== TRAIN [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== TRAIN [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== EVAL [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== EVAL [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== EVAL [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== VIS [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== VIS [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== VIS [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml
