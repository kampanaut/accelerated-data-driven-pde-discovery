#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== TRAIN [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== TRAIN [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== TRAIN [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== EVAL [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== EVAL [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== EVAL [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== VIS [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== VIS [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== VIS [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml
