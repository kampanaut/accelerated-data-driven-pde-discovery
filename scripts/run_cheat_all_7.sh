#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat/heat-25-5step-k800-baseline.yaml

echo "=== TRAIN [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat/heat-26-5step-k10-baseline.yaml

echo "=== TRAIN [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat/heat-27-5step-k800-metal.yaml

echo "=== TRAIN [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat/heat-28-5step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat/heat-25-5step-k800-baseline.yaml

echo "=== EVAL [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat/heat-26-5step-k10-baseline.yaml

echo "=== EVAL [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat/heat-27-5step-k800-metal.yaml

echo "=== EVAL [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat/heat-28-5step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat/heat-25-5step-k800-baseline.yaml

echo "=== VIS [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat/heat-26-5step-k10-baseline.yaml

echo "=== VIS [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat/heat-27-5step-k800-metal.yaml

echo "=== VIS [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat/heat-28-5step-k10-metal.yaml
