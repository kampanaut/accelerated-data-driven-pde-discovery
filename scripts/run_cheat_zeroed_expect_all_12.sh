#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/nl_heat-45-5step-k800-spectral.yaml

echo "=== TRAIN [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/nl_heat-46-5step-k10-spectral.yaml

echo "=== TRAIN [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== TRAIN [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_expect/nl_heat-48-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/nl_heat-45-5step-k800-spectral.yaml

echo "=== EVAL [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/nl_heat-46-5step-k10-spectral.yaml

echo "=== EVAL [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== EVAL [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_expect/nl_heat-48-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/nl_heat-45-5step-k800-spectral.yaml

echo "=== VIS [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/nl_heat-46-5step-k10-spectral.yaml

echo "=== VIS [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== VIS [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_expect/nl_heat-48-5step-k10-metal-spectral.yaml
