#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/4] heat-1-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== TRAIN [2/4] heat-2-full ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml

# === EVALUATE ===
echo "=== EVAL [1/4] heat-1-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== EVAL [2/4] heat-2-full ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml

# === VISUALIZE ===
echo "=== VIS [1/4] heat-1-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== VIS [2/4] heat-2-full ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml
