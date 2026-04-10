#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== TRAIN [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== TRAIN [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== TRAIN [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== EVAL [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== EVAL [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== EVAL [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== VIS [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== VIS [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== VIS [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml
