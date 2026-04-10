#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== TRAIN [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== EVAL [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== VIS [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== VIS [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml
