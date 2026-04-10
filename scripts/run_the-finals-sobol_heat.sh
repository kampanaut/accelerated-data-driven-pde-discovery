#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/42] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/42] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/42] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/42] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/42] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/42] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/42] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/42] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/42] heat-9-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== TRAIN [10/42] heat-10-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== TRAIN [11/42] heat-11-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== TRAIN [12/42] heat-12-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== TRAIN [13/42] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== TRAIN [14/42] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== TRAIN [15/42] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== TRAIN [16/42] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [17/42] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/42] heat-18-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== TRAIN [19/42] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [20/42] heat-20-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-20-100x120x150x450-anil.yaml

echo "=== TRAIN [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [1/42] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/42] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/42] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/42] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/42] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/42] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/42] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/42] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/42] heat-9-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== EVAL [10/42] heat-10-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== EVAL [11/42] heat-11-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== EVAL [12/42] heat-12-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== EVAL [13/42] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== EVAL [14/42] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== EVAL [15/42] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== EVAL [16/42] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== EVAL [17/42] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/42] heat-18-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== EVAL [19/42] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [20/42] heat-20-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-20-100x120x150x450-anil.yaml

echo "=== EVAL [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [1/42] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/42] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/42] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/42] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/42] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/42] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/42] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/42] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/42] heat-9-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== VIS [10/42] heat-10-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== VIS [11/42] heat-11-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== VIS [12/42] heat-12-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== VIS [13/42] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== VIS [14/42] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== VIS [15/42] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== VIS [16/42] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== VIS [17/42] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/42] heat-18-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== VIS [19/42] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== VIS [20/42] heat-20-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-20-100x120x150x450-anil.yaml

echo "=== VIS [21/42] heat-21-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-100x150x450-anil-2epoch.yaml
