#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [3/3] br-3 ==="
uv run python scripts/train_maml.py --config configs/the-finals/br-3.yaml

# === EVALUATE ===
echo "=== EVAL [3/3] br-3 ==="
uv run python scripts/evaluate.py --config configs/the-finals/br-3.yaml

# === VISUALIZE ===
echo "=== VIS [3/3] br-3 ==="
uv run python scripts/visualize.py --config configs/the-finals/br-3.yaml
