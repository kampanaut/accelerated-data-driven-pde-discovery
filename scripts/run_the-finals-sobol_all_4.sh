#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== TRAIN [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml

# === EVALUATE ===
echo "=== EVAL [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== EVAL [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml

# === VISUALIZE ===
echo "=== VIS [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== VIS [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml
