#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== TRAIN [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== TRAIN [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml

# === EVALUATE ===
echo "=== EVAL [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== EVAL [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== EVAL [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml

# === VISUALIZE ===
echo "=== VIS [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== VIS [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== VIS [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml
