#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/heat-29-5step-k800-spectral.yaml

echo "=== TRAIN [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/heat-30-5step-k10-spectral.yaml

echo "=== TRAIN [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/heat-31-5step-k800-metal-spectral.yaml

echo "=== TRAIN [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/heat-32-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/heat-29-5step-k800-spectral.yaml

echo "=== EVAL [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/heat-30-5step-k10-spectral.yaml

echo "=== EVAL [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/heat-31-5step-k800-metal-spectral.yaml

echo "=== EVAL [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/heat-32-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/heat-29-5step-k800-spectral.yaml

echo "=== VIS [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/heat-30-5step-k10-spectral.yaml

echo "=== VIS [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/heat-31-5step-k800-metal-spectral.yaml

echo "=== VIS [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/heat-32-5step-k10-metal-spectral.yaml
