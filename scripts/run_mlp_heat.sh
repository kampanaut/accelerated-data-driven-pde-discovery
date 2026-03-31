#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/16] heat-1-k800-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== TRAIN [2/16] heat-2-k800-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== TRAIN [3/16] heat-3-k10-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== TRAIN [4/16] heat-4-k10-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-4-k10-baseline-sin.yaml

echo "=== TRAIN [5/16] heat-5-k800-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== TRAIN [6/16] heat-6-k800-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== TRAIN [7/16] heat-7-k10-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== TRAIN [8/16] heat-8-k10-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-8-k10-metal-sin.yaml

echo "=== TRAIN [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== TRAIN [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== TRAIN [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== TRAIN [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml

echo "=== TRAIN [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== TRAIN [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== TRAIN [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== TRAIN [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml

# === EVALUATE ===
echo "=== EVAL [1/16] heat-1-k800-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== EVAL [2/16] heat-2-k800-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== EVAL [3/16] heat-3-k10-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== EVAL [4/16] heat-4-k10-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-4-k10-baseline-sin.yaml

echo "=== EVAL [5/16] heat-5-k800-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== EVAL [6/16] heat-6-k800-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== EVAL [7/16] heat-7-k10-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== EVAL [8/16] heat-8-k10-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-8-k10-metal-sin.yaml

echo "=== EVAL [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== EVAL [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== EVAL [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== EVAL [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml

echo "=== EVAL [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== EVAL [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== EVAL [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== EVAL [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml

# === VISUALIZE ===
echo "=== VIS [1/16] heat-1-k800-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-1-k800-baseline-silu.yaml

echo "=== VIS [2/16] heat-2-k800-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-2-k800-baseline-sin.yaml

echo "=== VIS [3/16] heat-3-k10-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-3-k10-baseline-silu.yaml

echo "=== VIS [4/16] heat-4-k10-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-4-k10-baseline-sin.yaml

echo "=== VIS [5/16] heat-5-k800-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-5-k800-metal-silu.yaml

echo "=== VIS [6/16] heat-6-k800-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-6-k800-metal-sin.yaml

echo "=== VIS [7/16] heat-7-k10-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-7-k10-metal-silu.yaml

echo "=== VIS [8/16] heat-8-k10-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-8-k10-metal-sin.yaml

echo "=== VIS [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== VIS [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== VIS [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== VIS [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml

echo "=== VIS [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== VIS [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== VIS [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== VIS [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml
