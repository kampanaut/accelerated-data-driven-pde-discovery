#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/16] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/16] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/16] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/16] heat-8-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [5/16] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/16] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/16] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/16] heat-8-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [5/16] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/16] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/16] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/16] heat-8-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-300x300x300-anil.yaml
