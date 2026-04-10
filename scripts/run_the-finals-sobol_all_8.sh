#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== TRAIN [30/50] nl_heat-30-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== TRAIN [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== TRAIN [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml

# === EVALUATE ===
echo "=== EVAL [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== EVAL [30/50] nl_heat-30-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== EVAL [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== EVAL [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml

# === VISUALIZE ===
echo "=== VIS [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== VIS [30/50] nl_heat-30-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== VIS [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== VIS [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml
