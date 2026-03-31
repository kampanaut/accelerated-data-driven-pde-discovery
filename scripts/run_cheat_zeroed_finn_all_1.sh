#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== TRAIN [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== TRAIN [3/48] br-3-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== TRAIN [4/48] br-4-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== EVAL [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== EVAL [3/48] br-3-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== EVAL [4/48] br-4-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== VIS [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== VIS [3/48] br-3-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== VIS [4/48] br-4-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml
