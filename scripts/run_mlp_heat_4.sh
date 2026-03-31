#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== TRAIN [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== TRAIN [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== TRAIN [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml

# === EVALUATE ===
echo "=== EVAL [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== EVAL [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== EVAL [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== EVAL [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml

# === VISUALIZE ===
echo "=== VIS [13/16] heat-13-k800-mamlpp+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-13-k800-mamlpp+metal-silu.yaml

echo "=== VIS [14/16] heat-14-k800-mamlpp+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-14-k800-mamlpp+metal-sin.yaml

echo "=== VIS [15/16] heat-15-k10-mamlpp+metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-15-k10-mamlpp+metal-silu.yaml

echo "=== VIS [16/16] heat-16-k10-mamlpp+metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-16-k10-mamlpp+metal-sin.yaml
