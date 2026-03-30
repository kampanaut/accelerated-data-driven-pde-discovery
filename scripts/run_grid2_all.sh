#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== TRAIN [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== TRAIN [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== TRAIN [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== TRAIN [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== TRAIN [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml

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
echo "=== EVAL [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== EVAL [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== EVAL [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== EVAL [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== EVAL [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== EVAL [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml

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
echo "=== VIS [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== VIS [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== VIS [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== VIS [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== VIS [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== VIS [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml

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
