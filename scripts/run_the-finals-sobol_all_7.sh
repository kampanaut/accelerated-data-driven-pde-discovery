#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== TRAIN [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== TRAIN [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== TRAIN [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== EVAL [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== EVAL [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== EVAL [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== VIS [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== VIS [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== VIS [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml
