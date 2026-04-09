#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== TRAIN [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== TRAIN [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== TRAIN [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== EVAL [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== EVAL [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== EVAL [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== VIS [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== VIS [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== VIS [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml
