#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== TRAIN [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== TRAIN [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== TRAIN [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== EVAL [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== EVAL [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== EVAL [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== VIS [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== VIS [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== VIS [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml
