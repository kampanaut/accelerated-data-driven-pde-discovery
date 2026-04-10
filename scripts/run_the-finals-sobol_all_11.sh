#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== TRAIN [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== EVAL [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== VIS [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml
