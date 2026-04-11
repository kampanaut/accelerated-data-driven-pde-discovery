#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== TRAIN [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== EVAL [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== VIS [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== VIS [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml
