#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/8] heat-1-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== TRAIN [2/8] heat-2-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== TRAIN [3/8] heat-3-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== TRAIN [4/8] heat-4-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-4-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [1/8] heat-1-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== EVAL [2/8] heat-2-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== EVAL [3/8] heat-3-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== EVAL [4/8] heat-4-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-4-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [1/8] heat-1-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== VIS [2/8] heat-2-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== VIS [3/8] heat-3-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== VIS [4/8] heat-4-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-4-k10-metal.yaml
