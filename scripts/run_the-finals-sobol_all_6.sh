#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== TRAIN [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== TRAIN [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== TRAIN [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml

# === EVALUATE ===
echo "=== EVAL [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== EVAL [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== EVAL [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== EVAL [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml

# === VISUALIZE ===
echo "=== VIS [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== VIS [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== VIS [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== VIS [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml
