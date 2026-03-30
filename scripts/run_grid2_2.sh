#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [7/24] cheat-7-5step-k10-metal+maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-7-5step-k10-metal+maml++.yaml

echo "=== TRAIN [8/24] cheat-8-5step-k800-metal+maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-8-5step-k800-metal+maml++.yaml

echo "=== TRAIN [9/24] heat-9-5step-k10-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-9-5step-k10-baseline-silu.yaml

echo "=== TRAIN [10/24] heat-10-5step-k800-baseline-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-10-5step-k800-baseline-silu.yaml

echo "=== TRAIN [11/24] heat-11-5step-k10-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-11-5step-k10-metal-silu.yaml

echo "=== TRAIN [12/24] heat-12-5step-k800-metal-silu ==="
uv run python scripts/train_maml.py --config configs/mlp/heat-12-5step-k800-metal-silu.yaml

# === EVALUATE ===
echo "=== EVAL [7/24] cheat-7-5step-k10-metal+maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-7-5step-k10-metal+maml++.yaml

echo "=== EVAL [8/24] cheat-8-5step-k800-metal+maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-8-5step-k800-metal+maml++.yaml

echo "=== EVAL [9/24] heat-9-5step-k10-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-9-5step-k10-baseline-silu.yaml

echo "=== EVAL [10/24] heat-10-5step-k800-baseline-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-10-5step-k800-baseline-silu.yaml

echo "=== EVAL [11/24] heat-11-5step-k10-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-11-5step-k10-metal-silu.yaml

echo "=== EVAL [12/24] heat-12-5step-k800-metal-silu ==="
uv run python scripts/evaluate.py --config configs/mlp/heat-12-5step-k800-metal-silu.yaml

# === VISUALIZE ===
echo "=== VIS [7/24] cheat-7-5step-k10-metal+maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-7-5step-k10-metal+maml++.yaml

echo "=== VIS [8/24] cheat-8-5step-k800-metal+maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-8-5step-k800-metal+maml++.yaml

echo "=== VIS [9/24] heat-9-5step-k10-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-9-5step-k10-baseline-silu.yaml

echo "=== VIS [10/24] heat-10-5step-k800-baseline-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-10-5step-k800-baseline-silu.yaml

echo "=== VIS [11/24] heat-11-5step-k10-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-11-5step-k10-metal-silu.yaml

echo "=== VIS [12/24] heat-12-5step-k800-metal-silu ==="
uv run python scripts/visualize.py --config configs/mlp/heat-12-5step-k800-metal-silu.yaml
