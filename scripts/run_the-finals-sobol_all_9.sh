#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== TRAIN [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== TRAIN [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== EVAL [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== EVAL [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== EVAL [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== VIS [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== VIS [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== VIS [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml
