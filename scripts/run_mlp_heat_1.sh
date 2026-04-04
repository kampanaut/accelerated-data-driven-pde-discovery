#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/18] heat-1-k800-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== TRAIN [2/18] heat-2-k800-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== TRAIN [3/18] heat-3-k10-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== TRAIN [4/18] heat-4-k10-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-4-k10-baseline-sin.yaml

# === EVALUATE ===
echo "=== EVAL [1/18] heat-1-k800-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== EVAL [2/18] heat-2-k800-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== EVAL [3/18] heat-3-k10-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== EVAL [4/18] heat-4-k10-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-4-k10-baseline-sin.yaml

# === VISUALIZE ===
echo "=== VIS [1/18] heat-1-k800-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== VIS [2/18] heat-2-k800-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== VIS [3/18] heat-3-k10-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== VIS [4/18] heat-4-k10-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-4-k10-baseline-sin.yaml
