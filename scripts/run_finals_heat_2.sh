#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/5] heat-5-silu-lam05-cosine ==="
uv run python scripts/train_maml.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

# === EVALUATE ===
echo "=== EVAL [5/5] heat-5-silu-lam05-cosine ==="
uv run python scripts/evaluate.py --config configs/finals/heat-5-silu-lam05-cosine.yaml

# === VISUALIZE ===
echo "=== VIS [5/5] heat-5-silu-lam05-cosine ==="
uv run python scripts/visualize.py --config configs/finals/heat-5-silu-lam05-cosine.yaml
