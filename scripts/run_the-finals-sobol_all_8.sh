#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== TRAIN [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== TRAIN [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== TRAIN [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== EVAL [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== EVAL [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== EVAL [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== VIS [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== VIS [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== VIS [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml
