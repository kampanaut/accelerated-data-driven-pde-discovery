#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== TRAIN [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== TRAIN [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== TRAIN [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== EVAL [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== EVAL [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== EVAL [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== VIS [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== VIS [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== VIS [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml
