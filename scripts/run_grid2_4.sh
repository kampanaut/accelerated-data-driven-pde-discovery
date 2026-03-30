#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [19/24] heat-19-5step-k10-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-19-5step-k10-metal-sin.yaml

echo "=== TRAIN [20/24] heat-20-5step-k800-metal-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-20-5step-k800-metal-sin.yaml

echo "=== TRAIN [21/24] heat-21-5step-k10-maml++-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-21-5step-k10-maml++-sin.yaml

echo "=== TRAIN [22/24] heat-22-5step-k800-maml++-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-22-5step-k800-maml++-sin.yaml

echo "=== TRAIN [23/24] heat-23-5step-k10-metal+maml++-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-23-5step-k10-metal+maml++-sin.yaml

echo "=== TRAIN [24/24] heat-24-5step-k800-metal+maml++-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-24-5step-k800-metal+maml++-sin.yaml

# === EVALUATE ===
echo "=== EVAL [19/24] heat-19-5step-k10-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-19-5step-k10-metal-sin.yaml

echo "=== EVAL [20/24] heat-20-5step-k800-metal-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-20-5step-k800-metal-sin.yaml

echo "=== EVAL [21/24] heat-21-5step-k10-maml++-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-21-5step-k10-maml++-sin.yaml

echo "=== EVAL [22/24] heat-22-5step-k800-maml++-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-22-5step-k800-maml++-sin.yaml

echo "=== EVAL [23/24] heat-23-5step-k10-metal+maml++-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-23-5step-k10-metal+maml++-sin.yaml

echo "=== EVAL [24/24] heat-24-5step-k800-metal+maml++-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-24-5step-k800-metal+maml++-sin.yaml

# === VISUALIZE ===
echo "=== VIS [19/24] heat-19-5step-k10-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-19-5step-k10-metal-sin.yaml

echo "=== VIS [20/24] heat-20-5step-k800-metal-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-20-5step-k800-metal-sin.yaml

echo "=== VIS [21/24] heat-21-5step-k10-maml++-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-21-5step-k10-maml++-sin.yaml

echo "=== VIS [22/24] heat-22-5step-k800-maml++-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-22-5step-k800-maml++-sin.yaml

echo "=== VIS [23/24] heat-23-5step-k10-metal+maml++-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-23-5step-k10-metal+maml++-sin.yaml

echo "=== VIS [24/24] heat-24-5step-k800-metal+maml++-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-24-5step-k800-metal+maml++-sin.yaml
