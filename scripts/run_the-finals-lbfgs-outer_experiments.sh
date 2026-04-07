#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/4] heat-1-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== TRAIN [2/4] heat-2-full ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml

echo "=== TRAIN [3/4] nl_heat-3-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/nl_heat-3-mb25.yaml

echo "=== TRAIN [4/4] nl_heat-4-full ==="
uv run python scripts/train_maml.py --config configs/the-finals-lbfgs-outer/nl_heat-4-full.yaml

# === EVALUATE ===
echo "=== EVAL [1/4] heat-1-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== EVAL [2/4] heat-2-full ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml

echo "=== EVAL [3/4] nl_heat-3-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/nl_heat-3-mb25.yaml

echo "=== EVAL [4/4] nl_heat-4-full ==="
uv run python scripts/evaluate.py --config configs/the-finals-lbfgs-outer/nl_heat-4-full.yaml

# === VISUALIZE ===
echo "=== VIS [1/4] heat-1-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/heat-1-mb25.yaml

echo "=== VIS [2/4] heat-2-full ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/heat-2-full.yaml

echo "=== VIS [3/4] nl_heat-3-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/nl_heat-3-mb25.yaml

echo "=== VIS [4/4] nl_heat-4-full ==="
uv run python scripts/visualize.py --config configs/the-finals-lbfgs-outer/nl_heat-4-full.yaml
