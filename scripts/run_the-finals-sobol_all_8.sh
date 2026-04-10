#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== TRAIN [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== TRAIN [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== TRAIN [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== EVAL [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== EVAL [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== EVAL [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== VIS [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== VIS [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== VIS [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml
