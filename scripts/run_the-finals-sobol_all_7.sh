#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== TRAIN [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== TRAIN [27/44] nl_heat-27-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== TRAIN [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml

# === EVALUATE ===
echo "=== EVAL [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== EVAL [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== EVAL [27/44] nl_heat-27-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== EVAL [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml

# === VISUALIZE ===
echo "=== VIS [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== VIS [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== VIS [27/44] nl_heat-27-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== VIS [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml
