#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== TRAIN [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== TRAIN [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== TRAIN [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_finn/nl_heat-40-1step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== EVAL [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== EVAL [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== EVAL [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_finn/nl_heat-40-1step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== VIS [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== VIS [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== VIS [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_finn/nl_heat-40-1step-k10-metal-spectral.yaml
