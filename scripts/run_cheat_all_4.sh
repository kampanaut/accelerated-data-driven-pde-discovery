#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat/br-13-5step-k800-spectral.yaml

echo "=== TRAIN [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat/br-14-5step-k10-spectral.yaml

echo "=== TRAIN [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat/br-15-5step-k800-metal-spectral.yaml

echo "=== TRAIN [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat/br-16-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat/br-13-5step-k800-spectral.yaml

echo "=== EVAL [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat/br-14-5step-k10-spectral.yaml

echo "=== EVAL [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat/br-15-5step-k800-metal-spectral.yaml

echo "=== EVAL [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat/br-16-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat/br-13-5step-k800-spectral.yaml

echo "=== VIS [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat/br-14-5step-k10-spectral.yaml

echo "=== VIS [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat/br-15-5step-k800-metal-spectral.yaml

echo "=== VIS [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat/br-16-5step-k10-metal-spectral.yaml
