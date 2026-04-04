#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/24] heat-21-k800-imaml-lbfgs+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-21-k800-imaml-lbfgs+metal-silu.yaml

echo "=== TRAIN [22/24] heat-22-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-22-k800-imaml-lbfgs+metal-sin.yaml

echo "=== TRAIN [23/24] heat-23-k10-imaml-lbfgs+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-23-k10-imaml-lbfgs+metal-silu.yaml

echo "=== TRAIN [24/24] heat-24-k10-imaml-lbfgs+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-24-k10-imaml-lbfgs+metal-sin.yaml

# === EVALUATE ===
echo "=== EVAL [21/24] heat-21-k800-imaml-lbfgs+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-21-k800-imaml-lbfgs+metal-silu.yaml

echo "=== EVAL [22/24] heat-22-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-22-k800-imaml-lbfgs+metal-sin.yaml

echo "=== EVAL [23/24] heat-23-k10-imaml-lbfgs+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-23-k10-imaml-lbfgs+metal-silu.yaml

echo "=== EVAL [24/24] heat-24-k10-imaml-lbfgs+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-24-k10-imaml-lbfgs+metal-sin.yaml

# === VISUALIZE ===
echo "=== VIS [21/24] heat-21-k800-imaml-lbfgs+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-21-k800-imaml-lbfgs+metal-silu.yaml

echo "=== VIS [22/24] heat-22-k800-imaml-lbfgs+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-22-k800-imaml-lbfgs+metal-sin.yaml

echo "=== VIS [23/24] heat-23-k10-imaml-lbfgs+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-23-k10-imaml-lbfgs+metal-silu.yaml

echo "=== VIS [24/24] heat-24-k10-imaml-lbfgs+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-24-k10-imaml-lbfgs+metal-sin.yaml
