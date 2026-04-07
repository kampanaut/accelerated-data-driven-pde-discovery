#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [2/2] nl_heat-2 ==="
uv run python scripts/train_maml.py --config configs/the-finals-5k/nl_heat-2.yaml

# === EVALUATE ===
echo "=== EVAL [2/2] nl_heat-2 ==="
uv run python scripts/evaluate.py --config configs/the-finals-5k/nl_heat-2.yaml

# === VISUALIZE ===
echo "=== VIS [2/2] nl_heat-2 ==="
uv run python scripts/visualize.py --config configs/the-finals-5k/nl_heat-2.yaml
