#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/19] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== TRAIN [14/19] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== TRAIN [15/19] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/train_maml.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== TRAIN [16/19] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/train_maml.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

# === EVALUATE ===
echo "=== EVAL [13/19] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== EVAL [14/19] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== EVAL [15/19] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/evaluate.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== EVAL [16/19] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/evaluate.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml

# === VISUALIZE ===
echo "=== VIS [13/19] heat-13-silu-lam01-3layers-cg10-cosine-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-13-silu-lam01-3layers-cg10-cosine-2k.yaml

echo "=== VIS [14/19] heat-14-silu-lam005-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-14-silu-lam005-3layers-cg10-exp.yaml

echo "=== VIS [15/19] heat-15-silu-lam01-3layers-cg10-exp ==="
uv run python scripts/visualize.py --config configs/finals/heat-15-silu-lam01-3layers-cg10-exp.yaml

echo "=== VIS [16/19] heat-16-silu-lam005-3layers-cg10-exp-2k ==="
uv run python scripts/visualize.py --config configs/finals/heat-16-silu-lam005-3layers-cg10-exp-2k.yaml
