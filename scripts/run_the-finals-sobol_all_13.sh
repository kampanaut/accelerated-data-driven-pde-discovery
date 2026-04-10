#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== TRAIN [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== EVAL [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== VIS [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml
