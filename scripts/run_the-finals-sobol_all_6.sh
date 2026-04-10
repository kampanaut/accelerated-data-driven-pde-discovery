#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== TRAIN [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== TRAIN [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== EVAL [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== EVAL [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== EVAL [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== VIS [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== VIS [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== VIS [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml
