#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== TRAIN [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== TRAIN [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== EVAL [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== EVAL [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== EVAL [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== VIS [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== VIS [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== VIS [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml
