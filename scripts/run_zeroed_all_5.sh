#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/zeroed/heat-17-1step-k800-baseline.yaml

echo "=== TRAIN [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/zeroed/heat-18-1step-k10-baseline.yaml

echo "=== TRAIN [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/zeroed/heat-19-1step-k800-metal.yaml

echo "=== TRAIN [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/zeroed/heat-20-1step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/zeroed/heat-17-1step-k800-baseline.yaml

echo "=== EVAL [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/zeroed/heat-18-1step-k10-baseline.yaml

echo "=== EVAL [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/zeroed/heat-19-1step-k800-metal.yaml

echo "=== EVAL [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/zeroed/heat-20-1step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/zeroed/heat-17-1step-k800-baseline.yaml

echo "=== VIS [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/zeroed/heat-18-1step-k10-baseline.yaml

echo "=== VIS [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/zeroed/heat-19-1step-k800-metal.yaml

echo "=== VIS [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/zeroed/heat-20-1step-k10-metal.yaml
