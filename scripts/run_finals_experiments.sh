#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/22] heat-1-sin ==="
uv run python scripts/train_maml.py --config configs/finals/heat-1-sin.yaml

echo "=== TRAIN [2/22] heat-2-silu ==="
uv run python scripts/train_maml.py --config configs/finals/heat-2-silu.yaml

echo "=== TRAIN [3/22] heat-3-silu-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== TRAIN [4/22] heat-4-silu-lam05 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== TRAIN [5/22] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== TRAIN [6/22] heat-6-silu-lam05-3layers ==="
uv run python scripts/train_maml.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== TRAIN [7/22] heat-7-silu-lam05-cg10 ==="
uv run python scripts/train_maml.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== TRAIN [8/22] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== TRAIN [9/22] heat-9-silu-lam01-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== TRAIN [10/22] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== TRAIN [11/22] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== TRAIN [12/22] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [13/22] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [14/22] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== TRAIN [15/22] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== TRAIN [16/22] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== TRAIN [17/22] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== TRAIN [18/22] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [19/22] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== TRAIN [20/22] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== TRAIN [21/22] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== TRAIN [22/22] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/train_maml.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml

# === EVALUATE ===
echo "=== EVAL [1/22] heat-1-sin ==="
uv run python scripts/evaluate.py --config configs/finals/heat-1-sin.yaml

echo "=== EVAL [2/22] heat-2-silu ==="
uv run python scripts/evaluate.py --config configs/finals/heat-2-silu.yaml

echo "=== EVAL [3/22] heat-3-silu-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== EVAL [4/22] heat-4-silu-lam05 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== EVAL [5/22] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== EVAL [6/22] heat-6-silu-lam05-3layers ==="
uv run python scripts/evaluate.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== EVAL [7/22] heat-7-silu-lam05-cg10 ==="
uv run python scripts/evaluate.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== EVAL [8/22] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== EVAL [9/22] heat-9-silu-lam01-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== EVAL [10/22] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== EVAL [11/22] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== EVAL [12/22] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [13/22] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [14/22] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== EVAL [15/22] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== EVAL [16/22] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== EVAL [17/22] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== EVAL [18/22] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [19/22] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== EVAL [20/22] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== EVAL [21/22] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== EVAL [22/22] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/evaluate.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml

# === VISUALIZE ===
echo "=== VIS [1/22] heat-1-sin ==="
uv run python scripts/visualize.py --config configs/finals/heat-1-sin.yaml

echo "=== VIS [2/22] heat-2-silu ==="
uv run python scripts/visualize.py --config configs/finals/heat-2-silu.yaml

echo "=== VIS [3/22] heat-3-silu-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-3-silu-cosine.yaml

echo "=== VIS [4/22] heat-4-silu-lam05 ==="
uv run python scripts/visualize.py --config configs/finals/heat-4-silu-lam05.yaml

echo "=== VIS [5/22] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

echo "=== VIS [6/22] heat-6-silu-lam05-3layers ==="
uv run python scripts/visualize.py --config configs/finals/heat-6-silu-lam05-3layers.yaml

echo "=== VIS [7/22] heat-7-silu-lam05-cg10 ==="
uv run python scripts/visualize.py --config configs/finals/heat-7-silu-lam05-cg10.yaml

echo "=== VIS [8/22] heat-8-silu-lam05-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-8-silu-lam05-3layers-cg10-cosine.yaml

echo "=== VIS [9/22] heat-9-silu-lam01-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-9-silu-lam01-cosine.yaml

echo "=== VIS [10/22] heat-10-silu-lam01-3layers-cg10-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-10-silu-lam01-3layers-cg10-cosine.yaml

echo "=== VIS [11/22] heat-11-silu-lam05-3layers-cg5-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-11-silu-lam05-3layers-cg5-cosine.yaml

echo "=== VIS [12/22] heat-12-silu-lam005-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-12-silu-lam005-3layers-cg10-cosine-2k.yaml

echo "=== VIS [13/22] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== VIS [14/22] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== VIS [15/22] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== VIS [16/22] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

echo "=== VIS [17/22] heat-17-silu-lam01-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-17-silu-lam01-3layers-cg10-exp-2k.yaml

echo "=== VIS [18/22] heat-18-silu-lam005-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-18-silu-lam005-3layers-cg10-poly3-2k.yaml

echo "=== VIS [19/22] heat-19-silu-lam01-3layers-cg10-poly3-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-19-silu-lam01-3layers-cg10-poly3-2k.yaml

echo "=== VIS [20/22] heat-20-silu-lam005-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-20-silu-lam005-3layers-cg10-plateau-2k.yaml

echo "=== VIS [21/22] heat-21-silu-lam01-3layers-cg10-plateau-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-21-silu-lam01-3layers-cg10-plateau-2k.yaml

echo "=== VIS [22/22] heat-22-silu-lam005-3layers-cg10-poly3-2k-sse ==="
uv run python scripts/visualize.py --config configs/finals/heat-22-silu-lam005-3layers-cg10-poly3-2k-sse.yaml
