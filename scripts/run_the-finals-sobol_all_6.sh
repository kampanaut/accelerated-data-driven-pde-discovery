#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== TRAIN [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== EVAL [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== VIS [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml
