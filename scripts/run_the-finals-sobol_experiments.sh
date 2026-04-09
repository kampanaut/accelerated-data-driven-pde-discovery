#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/18] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/18] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/18] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/18] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/18] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/18] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/18] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/18] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== TRAIN [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== TRAIN [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== TRAIN [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml

echo "=== TRAIN [13/18] nl_heat-13-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-adam+lbfgs-mb25.yaml

echo "=== TRAIN [14/18] nl_heat-14-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-250x250.yaml

echo "=== TRAIN [15/18] nl_heat-15-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-15-250x250-anil.yaml

echo "=== TRAIN [16/18] nl_heat-16-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-16-100x100x100-anil.yaml

echo "=== TRAIN [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== TRAIN [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [1/18] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/18] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/18] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/18] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/18] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/18] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/18] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/18] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== EVAL [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== EVAL [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== EVAL [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml

echo "=== EVAL [13/18] nl_heat-13-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-adam+lbfgs-mb25.yaml

echo "=== EVAL [14/18] nl_heat-14-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-250x250.yaml

echo "=== EVAL [15/18] nl_heat-15-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-15-250x250-anil.yaml

echo "=== EVAL [16/18] nl_heat-16-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-16-100x100x100-anil.yaml

echo "=== EVAL [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== EVAL [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [1/18] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/18] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/18] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/18] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/18] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/18] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/18] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/18] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/18] heat-9-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-300x300x300-anil.yaml

echo "=== VIS [10/18] nl_heat-10-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-10-adam+lbfgs.yaml

echo "=== VIS [11/18] nl_heat-11-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-11-adam-mb25.yaml

echo "=== VIS [12/18] nl_heat-12-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs-5k.yaml

echo "=== VIS [13/18] nl_heat-13-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-adam+lbfgs-mb25.yaml

echo "=== VIS [14/18] nl_heat-14-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-250x250.yaml

echo "=== VIS [15/18] nl_heat-15-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-15-250x250-anil.yaml

echo "=== VIS [16/18] nl_heat-16-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-16-100x100x100-anil.yaml

echo "=== VIS [17/18] nl_heat-17-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-350x350-anil.yaml

echo "=== VIS [18/18] nl_heat-18-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-300x300x300-anil.yaml
