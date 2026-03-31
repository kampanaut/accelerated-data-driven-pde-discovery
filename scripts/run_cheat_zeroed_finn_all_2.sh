#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== TRAIN [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== TRAIN [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== TRAIN [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== EVAL [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== EVAL [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== EVAL [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== VIS [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== VIS [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== VIS [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml
