#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== TRAIN [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== TRAIN [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== TRAIN [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== EVAL [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== EVAL [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== EVAL [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== VIS [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== VIS [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== VIS [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml
