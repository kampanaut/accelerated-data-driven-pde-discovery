#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== TRAIN [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== TRAIN [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== TRAIN [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== EVAL [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== EVAL [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== EVAL [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== VIS [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== VIS [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== VIS [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml
