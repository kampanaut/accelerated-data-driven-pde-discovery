#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/18] heat-5-k800-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== TRAIN [6/18] heat-6-k800-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== TRAIN [7/18] heat-7-k10-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== TRAIN [8/18] heat-8-k10-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-8-k10-metal-sin.yaml

# === EVALUATE ===
echo "=== EVAL [5/18] heat-5-k800-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== EVAL [6/18] heat-6-k800-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== EVAL [7/18] heat-7-k10-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== EVAL [8/18] heat-8-k10-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-8-k10-metal-sin.yaml

# === VISUALIZE ===
echo "=== VIS [5/18] heat-5-k800-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== VIS [6/18] heat-6-k800-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== VIS [7/18] heat-7-k10-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== VIS [8/18] heat-8-k10-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-8-k10-metal-sin.yaml
