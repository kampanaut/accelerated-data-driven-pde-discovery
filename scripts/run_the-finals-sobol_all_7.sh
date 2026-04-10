#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== TRAIN [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== TRAIN [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== TRAIN [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== EVAL [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== EVAL [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== EVAL [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== VIS [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== VIS [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== VIS [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml
