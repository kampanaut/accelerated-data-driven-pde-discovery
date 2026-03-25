#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-1-1step-k800-baseline.yaml

echo "=== TRAIN [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-2-1step-k10-baseline.yaml

echo "=== TRAIN [3/48] br-3-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-3-1step-k800-metal.yaml

echo "=== TRAIN [4/48] br-4-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-4-1step-k10-metal.yaml

echo "=== TRAIN [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-5-1step-k800-spectral.yaml

echo "=== TRAIN [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-6-1step-k10-spectral.yaml

echo "=== TRAIN [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-7-1step-k800-metal-spectral.yaml

echo "=== TRAIN [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-8-1step-k10-metal-spectral.yaml

echo "=== TRAIN [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== TRAIN [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== TRAIN [11/48] br-11-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== TRAIN [12/48] br-12-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml

echo "=== TRAIN [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-13-5step-k800-spectral.yaml

echo "=== TRAIN [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-14-5step-k10-spectral.yaml

echo "=== TRAIN [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-15-5step-k800-metal-spectral.yaml

echo "=== TRAIN [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/br-16-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-1-1step-k800-baseline.yaml

echo "=== EVAL [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-2-1step-k10-baseline.yaml

echo "=== EVAL [3/48] br-3-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-3-1step-k800-metal.yaml

echo "=== EVAL [4/48] br-4-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-4-1step-k10-metal.yaml

echo "=== EVAL [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-5-1step-k800-spectral.yaml

echo "=== EVAL [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-6-1step-k10-spectral.yaml

echo "=== EVAL [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-7-1step-k800-metal-spectral.yaml

echo "=== EVAL [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-8-1step-k10-metal-spectral.yaml

echo "=== EVAL [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== EVAL [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== EVAL [11/48] br-11-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== EVAL [12/48] br-12-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml

echo "=== EVAL [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-13-5step-k800-spectral.yaml

echo "=== EVAL [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-14-5step-k10-spectral.yaml

echo "=== EVAL [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-15-5step-k800-metal-spectral.yaml

echo "=== EVAL [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/br-16-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-1-1step-k800-baseline.yaml

echo "=== VIS [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-2-1step-k10-baseline.yaml

echo "=== VIS [3/48] br-3-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-3-1step-k800-metal.yaml

echo "=== VIS [4/48] br-4-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-4-1step-k10-metal.yaml

echo "=== VIS [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-5-1step-k800-spectral.yaml

echo "=== VIS [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-6-1step-k10-spectral.yaml

echo "=== VIS [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-7-1step-k800-metal-spectral.yaml

echo "=== VIS [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-8-1step-k10-metal-spectral.yaml

echo "=== VIS [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-9-5step-k800-baseline.yaml

echo "=== VIS [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-10-5step-k10-baseline.yaml

echo "=== VIS [11/48] br-11-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-11-5step-k800-metal.yaml

echo "=== VIS [12/48] br-12-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-12-5step-k10-metal.yaml

echo "=== VIS [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-13-5step-k800-spectral.yaml

echo "=== VIS [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-14-5step-k10-spectral.yaml

echo "=== VIS [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-15-5step-k800-metal-spectral.yaml

echo "=== VIS [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/br-16-5step-k10-metal-spectral.yaml
