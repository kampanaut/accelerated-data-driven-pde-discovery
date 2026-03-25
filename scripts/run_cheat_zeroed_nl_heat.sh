#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-33-1step-k800-baseline.yaml

echo "=== TRAIN [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-34-1step-k10-baseline.yaml

echo "=== TRAIN [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-35-1step-k800-metal.yaml

echo "=== TRAIN [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-36-1step-k10-metal.yaml

echo "=== TRAIN [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-37-1step-k800-spectral.yaml

echo "=== TRAIN [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-38-1step-k10-spectral.yaml

echo "=== TRAIN [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== TRAIN [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== TRAIN [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-41-5step-k800-baseline.yaml

echo "=== TRAIN [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-42-5step-k10-baseline.yaml

echo "=== TRAIN [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-43-5step-k800-metal.yaml

echo "=== TRAIN [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-44-5step-k10-metal.yaml

echo "=== TRAIN [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-45-5step-k800-spectral.yaml

echo "=== TRAIN [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-46-5step-k10-spectral.yaml

echo "=== TRAIN [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== TRAIN [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed/nl_heat-48-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-33-1step-k800-baseline.yaml

echo "=== EVAL [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-34-1step-k10-baseline.yaml

echo "=== EVAL [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-35-1step-k800-metal.yaml

echo "=== EVAL [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-36-1step-k10-metal.yaml

echo "=== EVAL [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-37-1step-k800-spectral.yaml

echo "=== EVAL [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-38-1step-k10-spectral.yaml

echo "=== EVAL [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== EVAL [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== EVAL [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-41-5step-k800-baseline.yaml

echo "=== EVAL [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-42-5step-k10-baseline.yaml

echo "=== EVAL [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-43-5step-k800-metal.yaml

echo "=== EVAL [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-44-5step-k10-metal.yaml

echo "=== EVAL [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-45-5step-k800-spectral.yaml

echo "=== EVAL [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-46-5step-k10-spectral.yaml

echo "=== EVAL [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== EVAL [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed/nl_heat-48-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-33-1step-k800-baseline.yaml

echo "=== VIS [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-34-1step-k10-baseline.yaml

echo "=== VIS [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-35-1step-k800-metal.yaml

echo "=== VIS [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-36-1step-k10-metal.yaml

echo "=== VIS [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-37-1step-k800-spectral.yaml

echo "=== VIS [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-38-1step-k10-spectral.yaml

echo "=== VIS [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== VIS [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== VIS [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-41-5step-k800-baseline.yaml

echo "=== VIS [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-42-5step-k10-baseline.yaml

echo "=== VIS [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-43-5step-k800-metal.yaml

echo "=== VIS [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-44-5step-k10-metal.yaml

echo "=== VIS [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-45-5step-k800-spectral.yaml

echo "=== VIS [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-46-5step-k10-spectral.yaml

echo "=== VIS [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== VIS [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed/nl_heat-48-5step-k10-metal-spectral.yaml
