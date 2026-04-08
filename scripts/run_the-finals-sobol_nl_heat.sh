#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/16] nl_heat-9-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-9-adam+lbfgs.yaml

echo "=== TRAIN [10/16] nl_heat-10-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-10-adam-mb25.yaml

echo "=== TRAIN [11/16] nl_heat-11-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-5k.yaml

echo "=== TRAIN [12/16] nl_heat-12-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-mb25.yaml

echo "=== TRAIN [13/16] nl_heat-13-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-250x250.yaml

echo "=== TRAIN [14/16] nl_heat-14-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-250x250-anil.yaml

echo "=== TRAIN [15/16] nl_heat-15-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-15-100x100x100-anil.yaml

echo "=== TRAIN [16/16] nl_heat-16-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-16-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [9/16] nl_heat-9-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-9-adam+lbfgs.yaml

echo "=== EVAL [10/16] nl_heat-10-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-10-adam-mb25.yaml

echo "=== EVAL [11/16] nl_heat-11-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-5k.yaml

echo "=== EVAL [12/16] nl_heat-12-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-mb25.yaml

echo "=== EVAL [13/16] nl_heat-13-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-250x250.yaml

echo "=== EVAL [14/16] nl_heat-14-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-250x250-anil.yaml

echo "=== EVAL [15/16] nl_heat-15-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-15-100x100x100-anil.yaml

echo "=== EVAL [16/16] nl_heat-16-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-16-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [9/16] nl_heat-9-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-9-adam+lbfgs.yaml

echo "=== VIS [10/16] nl_heat-10-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-10-adam-mb25.yaml

echo "=== VIS [11/16] nl_heat-11-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-5k.yaml

echo "=== VIS [12/16] nl_heat-12-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-mb25.yaml

echo "=== VIS [13/16] nl_heat-13-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-250x250.yaml

echo "=== VIS [14/16] nl_heat-14-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-250x250-anil.yaml

echo "=== VIS [15/16] nl_heat-15-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-15-100x100x100-anil.yaml

echo "=== VIS [16/16] nl_heat-16-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-16-300x300x300-anil.yaml
