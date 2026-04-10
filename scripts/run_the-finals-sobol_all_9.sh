#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== TRAIN [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== TRAIN [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== TRAIN [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== EVAL [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== EVAL [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== EVAL [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== VIS [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== VIS [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== VIS [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml
