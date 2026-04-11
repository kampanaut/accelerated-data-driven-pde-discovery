#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== TRAIN [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml

# === EVALUATE ===
echo "=== EVAL [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== EVAL [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml

# === VISUALIZE ===
echo "=== VIS [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== VIS [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== VIS [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml
