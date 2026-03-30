#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/24] heat-13-5step-k10-maml++-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-13-5step-k10-maml++-silu.yaml

echo "=== TRAIN [14/24] heat-14-5step-k800-maml++-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-14-5step-k800-maml++-silu.yaml

echo "=== TRAIN [15/24] heat-15-5step-k10-metal+maml++-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-15-5step-k10-metal+maml++-silu.yaml

echo "=== TRAIN [16/24] heat-16-5step-k800-metal+maml++-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-16-5step-k800-metal+maml++-silu.yaml

echo "=== TRAIN [17/24] heat-17-5step-k10-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-17-5step-k10-baseline-sin.yaml

echo "=== TRAIN [18/24] heat-18-5step-k800-baseline-sin ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-18-5step-k800-baseline-sin.yaml

# === EVALUATE ===
echo "=== EVAL [13/24] heat-13-5step-k10-maml++-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-13-5step-k10-maml++-silu.yaml

echo "=== EVAL [14/24] heat-14-5step-k800-maml++-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-14-5step-k800-maml++-silu.yaml

echo "=== EVAL [15/24] heat-15-5step-k10-metal+maml++-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-15-5step-k10-metal+maml++-silu.yaml

echo "=== EVAL [16/24] heat-16-5step-k800-metal+maml++-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-16-5step-k800-metal+maml++-silu.yaml

echo "=== EVAL [17/24] heat-17-5step-k10-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-17-5step-k10-baseline-sin.yaml

echo "=== EVAL [18/24] heat-18-5step-k800-baseline-sin ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-18-5step-k800-baseline-sin.yaml

# === VISUALIZE ===
echo "=== VIS [13/24] heat-13-5step-k10-maml++-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-13-5step-k10-maml++-silu.yaml

echo "=== VIS [14/24] heat-14-5step-k800-maml++-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-14-5step-k800-maml++-silu.yaml

echo "=== VIS [15/24] heat-15-5step-k10-metal+maml++-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-15-5step-k10-metal+maml++-silu.yaml

echo "=== VIS [16/24] heat-16-5step-k800-metal+maml++-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-16-5step-k800-metal+maml++-silu.yaml

echo "=== VIS [17/24] heat-17-5step-k10-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-17-5step-k10-baseline-sin.yaml

echo "=== VIS [18/24] heat-18-5step-k800-baseline-sin ==="
uv run python scripts/visualize.py --config configs/mlp/heat-18-5step-k800-baseline-sin.yaml
