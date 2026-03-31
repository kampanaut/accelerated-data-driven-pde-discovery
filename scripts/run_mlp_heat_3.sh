#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== TRAIN [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== TRAIN [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== TRAIN [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml

# === EVALUATE ===
echo "=== EVAL [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== EVAL [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== EVAL [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== EVAL [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml

# === VISUALIZE ===
echo "=== VIS [9/16] heat-9-k800-mamlpp-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-9-k800-mamlpp-silu.yaml

echo "=== VIS [10/16] heat-10-k800-mamlpp-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-10-k800-mamlpp-sin.yaml

echo "=== VIS [11/16] heat-11-k10-mamlpp-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-11-k10-mamlpp-silu.yaml

echo "=== VIS [12/16] heat-12-k10-mamlpp-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-12-k10-mamlpp-sin.yaml
