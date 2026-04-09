#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== TRAIN [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== EVAL [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== VIS [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml
