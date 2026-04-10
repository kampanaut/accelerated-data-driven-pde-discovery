#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== TRAIN [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== TRAIN [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== TRAIN [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== EVAL [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== EVAL [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== EVAL [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== VIS [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== VIS [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== VIS [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml
