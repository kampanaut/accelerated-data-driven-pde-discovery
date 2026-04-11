#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== TRAIN [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== TRAIN [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== EVAL [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== EVAL [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== EVAL [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== VIS [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== VIS [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== VIS [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml
