#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/22] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/22] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/22] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/22] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/22] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/22] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/22] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/22] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/22] heat-9-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== TRAIN [10/22] heat-10-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== TRAIN [11/22] heat-11-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== TRAIN [12/22] nl_heat-12-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs.yaml

echo "=== TRAIN [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== TRAIN [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== TRAIN [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== TRAIN [16/22] nl_heat-16-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml

echo "=== TRAIN [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== TRAIN [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== TRAIN [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== TRAIN [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml

echo "=== TRAIN [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== TRAIN [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [1/22] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/22] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/22] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/22] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/22] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/22] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/22] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/22] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/22] heat-9-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== EVAL [10/22] heat-10-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== EVAL [11/22] heat-11-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== EVAL [12/22] nl_heat-12-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs.yaml

echo "=== EVAL [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== EVAL [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== EVAL [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== EVAL [16/22] nl_heat-16-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml

echo "=== EVAL [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== EVAL [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== EVAL [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== EVAL [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml

echo "=== EVAL [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== EVAL [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [1/22] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/22] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/22] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/22] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/22] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/22] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/22] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/22] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/22] heat-9-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== VIS [10/22] heat-10-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== VIS [11/22] heat-11-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== VIS [12/22] nl_heat-12-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-12-adam+lbfgs.yaml

echo "=== VIS [13/22] nl_heat-13-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-13-adam-mb25.yaml

echo "=== VIS [14/22] nl_heat-14-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-14-adam+lbfgs-5k.yaml

echo "=== VIS [15/22] nl_heat-15-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-15-adam+lbfgs-mb25.yaml

echo "=== VIS [16/22] nl_heat-16-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-16-250x250.yaml

echo "=== VIS [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== VIS [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== VIS [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== VIS [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml

echo "=== VIS [21/22] nl_heat-21-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-100x100x300-anil.yaml

echo "=== VIS [22/22] nl_heat-22-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-300x300x300-anil.yaml
