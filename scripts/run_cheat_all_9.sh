#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat/nl_heat-33-1step-k800-baseline.yaml

echo "=== TRAIN [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat/nl_heat-34-1step-k10-baseline.yaml

echo "=== TRAIN [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat/nl_heat-35-1step-k800-metal.yaml

echo "=== TRAIN [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat/nl_heat-36-1step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat/nl_heat-33-1step-k800-baseline.yaml

echo "=== EVAL [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat/nl_heat-34-1step-k10-baseline.yaml

echo "=== EVAL [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat/nl_heat-35-1step-k800-metal.yaml

echo "=== EVAL [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat/nl_heat-36-1step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat/nl_heat-33-1step-k800-baseline.yaml

echo "=== VIS [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat/nl_heat-34-1step-k10-baseline.yaml

echo "=== VIS [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat/nl_heat-35-1step-k800-metal.yaml

echo "=== VIS [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat/nl_heat-36-1step-k10-metal.yaml
