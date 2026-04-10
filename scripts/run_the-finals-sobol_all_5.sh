#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== TRAIN [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml

# === EVALUATE ===
echo "=== EVAL [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== EVAL [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== EVAL [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml

# === VISUALIZE ===
echo "=== VIS [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== VIS [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== VIS [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml
