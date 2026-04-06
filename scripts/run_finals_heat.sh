#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/13] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/13] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/13] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/13] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== TRAIN [5/13] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/13] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/13] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== TRAIN [8/13] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== TRAIN [9/13] heat-9-silu-lam01-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== TRAIN [10/13] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== TRAIN [11/13] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== TRAIN [12/13] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [13/13] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

# === EVALUATE ===
echo "=== EVAL [1/13] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/13] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/13] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/13] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== EVAL [5/13] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/13] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/13] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== EVAL [8/13] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== EVAL [9/13] heat-9-silu-lam01-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== EVAL [10/13] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== EVAL [11/13] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== EVAL [12/13] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [13/13] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

# === VISUALIZE ===
echo "=== VIS [1/13] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/13] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/13] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/13] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== VIS [5/13] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/13] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/13] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== VIS [8/13] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== VIS [9/13] heat-9-silu-lam01-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== VIS [10/13] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== VIS [11/13] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== VIS [12/13] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== VIS [13/13] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml
