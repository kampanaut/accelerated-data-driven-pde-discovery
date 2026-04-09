#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== TRAIN [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== EVAL [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== VIS [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml
