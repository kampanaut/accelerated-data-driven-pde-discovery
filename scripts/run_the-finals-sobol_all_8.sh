#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== TRAIN [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== TRAIN [31/52] nl_heat-31-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== TRAIN [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml

# === EVALUATE ===
echo "=== EVAL [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== EVAL [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== EVAL [31/52] nl_heat-31-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== EVAL [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml

# === VISUALIZE ===
echo "=== VIS [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== VIS [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== VIS [31/52] nl_heat-31-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== VIS [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml
