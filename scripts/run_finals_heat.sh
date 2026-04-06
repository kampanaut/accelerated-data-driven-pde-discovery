#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/15] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/15] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/15] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/15] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== TRAIN [5/15] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/15] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/15] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== TRAIN [8/15] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== TRAIN [9/15] heat-9-silu-lam01-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== TRAIN [10/15] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== TRAIN [11/15] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== TRAIN [12/15] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [13/15] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [14/15] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== TRAIN [15/15] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

# === EVALUATE ===
echo "=== EVAL [1/15] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/15] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/15] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/15] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== EVAL [5/15] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/15] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/15] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== EVAL [8/15] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== EVAL [9/15] heat-9-silu-lam01-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== EVAL [10/15] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== EVAL [11/15] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== EVAL [12/15] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [13/15] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [14/15] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== EVAL [15/15] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

# === VISUALIZE ===
echo "=== VIS [1/15] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/15] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/15] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/15] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== VIS [5/15] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/15] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/15] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== VIS [8/15] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== VIS [9/15] heat-9-silu-lam01-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== VIS [10/15] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== VIS [11/15] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== VIS [12/15] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== VIS [13/15] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== VIS [14/15] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== VIS [15/15] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml
