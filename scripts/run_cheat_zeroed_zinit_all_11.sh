#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/nl_heat-41-5step-k800-baseline.yaml

echo "=== TRAIN [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/nl_heat-42-5step-k10-baseline.yaml

echo "=== TRAIN [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/nl_heat-43-5step-k800-metal.yaml

echo "=== TRAIN [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_zinit/nl_heat-44-5step-k10-metal.yaml

# === EVALUATE ===
echo "=== EVAL [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/nl_heat-41-5step-k800-baseline.yaml

echo "=== EVAL [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/nl_heat-42-5step-k10-baseline.yaml

echo "=== EVAL [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/nl_heat-43-5step-k800-metal.yaml

echo "=== EVAL [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_zinit/nl_heat-44-5step-k10-metal.yaml

# === VISUALIZE ===
echo "=== VIS [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/nl_heat-41-5step-k800-baseline.yaml

echo "=== VIS [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/nl_heat-42-5step-k10-baseline.yaml

echo "=== VIS [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/nl_heat-43-5step-k800-metal.yaml

echo "=== VIS [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_zinit/nl_heat-44-5step-k10-metal.yaml
