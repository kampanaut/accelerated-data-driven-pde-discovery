#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/18] heat-17-k800-imaml-lbfgs-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-17-k800-imaml-lbfgs-sin.yaml

echo "=== TRAIN [18/18] heat-18-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-18-k800-imaml-lbfgs+metal-sin.yaml

# === EVALUATE ===
echo "=== EVAL [17/18] heat-17-k800-imaml-lbfgs-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-17-k800-imaml-lbfgs-sin.yaml

echo "=== EVAL [18/18] heat-18-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-18-k800-imaml-lbfgs+metal-sin.yaml

# === VISUALIZE ===
echo "=== VIS [17/18] heat-17-k800-imaml-lbfgs-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-17-k800-imaml-lbfgs-sin.yaml

echo "=== VIS [18/18] heat-18-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-18-k800-imaml-lbfgs+metal-sin.yaml
