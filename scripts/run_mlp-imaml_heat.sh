#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/2] heat-1-imaml-lbfgs ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-1-imaml-lbfgs.yaml

echo "=== TRAIN [2/2] heat-2-imaml-lbfgs+metal ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-2-imaml-lbfgs+metal.yaml

# === EVALUATE ===
echo "=== EVAL [1/2] heat-1-imaml-lbfgs ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-1-imaml-lbfgs.yaml

echo "=== EVAL [2/2] heat-2-imaml-lbfgs+metal ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-2-imaml-lbfgs+metal.yaml

# === VISUALIZE ===
echo "=== VIS [1/2] heat-1-imaml-lbfgs ==="
uv run python scripts/visualize.py --config configs/mlp/heat-1-imaml-lbfgs.yaml

echo "=== VIS [2/2] heat-2-imaml-lbfgs+metal ==="
uv run python scripts/visualize.py --config configs/mlp/heat-2-imaml-lbfgs+metal.yaml
