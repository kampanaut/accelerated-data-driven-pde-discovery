#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== TRAIN [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== TRAIN [11/48] br-11-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== TRAIN [12/48] br-12-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== EVAL [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== EVAL [11/48] br-11-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== EVAL [12/48] br-12-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== VIS [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== VIS [11/48] br-11-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== VIS [12/48] br-12-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml
