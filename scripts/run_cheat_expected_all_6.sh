#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_expected/heat-21-1step-k800-spectral.yaml

echo "=== TRAIN [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_expected/heat-22-1step-k10-spectral.yaml

echo "=== TRAIN [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_expected/heat-23-1step-k800-metal-spectral.yaml

echo "=== TRAIN [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_expected/heat-24-1step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_expected/heat-21-1step-k800-spectral.yaml

echo "=== EVAL [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_expected/heat-22-1step-k10-spectral.yaml

echo "=== EVAL [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_expected/heat-23-1step-k800-metal-spectral.yaml

echo "=== EVAL [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_expected/heat-24-1step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_expected/heat-21-1step-k800-spectral.yaml

echo "=== VIS [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_expected/heat-22-1step-k10-spectral.yaml

echo "=== VIS [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_expected/heat-23-1step-k800-metal-spectral.yaml

echo "=== VIS [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_expected/heat-24-1step-k10-metal-spectral.yaml
