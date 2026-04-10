#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== TRAIN [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== TRAIN [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== TRAIN [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== EVAL [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== EVAL [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== EVAL [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== VIS [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== VIS [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== VIS [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml
