#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/14] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/14] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/14] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/14] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/14] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/14] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/14] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/14] nl_heat-8-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs.yaml

echo "=== TRAIN [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== TRAIN [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== TRAIN [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== TRAIN [12/14] nl_heat-12-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml

echo "=== TRAIN [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== TRAIN [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml

# === EVALUATE ===
echo "=== EVAL [1/14] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/14] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/14] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/14] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/14] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/14] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/14] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/14] nl_heat-8-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs.yaml

echo "=== EVAL [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== EVAL [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== EVAL [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== EVAL [12/14] nl_heat-12-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml

echo "=== EVAL [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== EVAL [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml

# === VISUALIZE ===
echo "=== VIS [1/14] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/14] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/14] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/14] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/14] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/14] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/14] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/14] nl_heat-8-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-8-adam+lbfgs.yaml

echo "=== VIS [9/14] nl_heat-9-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-9-adam-mb25.yaml

echo "=== VIS [10/14] nl_heat-10-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs-5k.yaml

echo "=== VIS [11/14] nl_heat-11-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-11-adam+lbfgs-mb25.yaml

echo "=== VIS [12/14] nl_heat-12-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-250x250.yaml

echo "=== VIS [13/14] nl_heat-13-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-250x250-anil.yaml

echo "=== VIS [14/14] nl_heat-14-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-100x100x100-anil.yaml
