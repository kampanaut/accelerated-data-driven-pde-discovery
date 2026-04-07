#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/21] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/21] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/21] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/21] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== TRAIN [5/21] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/21] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/21] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== TRAIN [8/21] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== TRAIN [9/21] heat-9-silu-lam01-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== TRAIN [10/21] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== TRAIN [11/21] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== TRAIN [12/21] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [13/21] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [14/21] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== TRAIN [15/21] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== TRAIN [16/21] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== TRAIN [17/21] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== TRAIN [18/21] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [19/21] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [20/21] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== TRAIN [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

# === EVALUATE ===
echo "=== EVAL [1/21] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/21] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/21] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/21] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== EVAL [5/21] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/21] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/21] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== EVAL [8/21] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== EVAL [9/21] heat-9-silu-lam01-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== EVAL [10/21] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== EVAL [11/21] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== EVAL [12/21] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [13/21] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [14/21] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== EVAL [15/21] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== EVAL [16/21] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== EVAL [17/21] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== EVAL [18/21] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [19/21] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [20/21] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== EVAL [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

# === VISUALIZE ===
echo "=== VIS [1/21] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/21] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/21] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/21] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== VIS [5/21] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/21] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/21] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== VIS [8/21] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== VIS [9/21] heat-9-silu-lam01-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== VIS [10/21] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== VIS [11/21] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== VIS [12/21] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== VIS [13/21] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== VIS [14/21] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== VIS [15/21] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== VIS [16/21] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== VIS [17/21] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== VIS [18/21] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== VIS [19/21] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== VIS [20/21] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== VIS [21/21] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml
