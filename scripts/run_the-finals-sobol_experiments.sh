#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/36] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/36] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/36] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/36] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/36] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/36] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/36] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/36] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/36] heat-9-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== TRAIN [10/36] heat-10-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== TRAIN [11/36] heat-11-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== TRAIN [12/36] heat-12-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== TRAIN [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== TRAIN [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== TRAIN [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== TRAIN [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== TRAIN [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml

echo "=== TRAIN [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== TRAIN [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== TRAIN [23/36] nl_heat-23-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== TRAIN [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml

echo "=== TRAIN [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== TRAIN [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== TRAIN [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== TRAIN [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml

echo "=== TRAIN [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== TRAIN [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== TRAIN [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== TRAIN [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml

echo "=== TRAIN [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== TRAIN [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== TRAIN [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [1/36] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/36] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/36] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/36] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/36] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/36] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/36] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/36] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/36] heat-9-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== EVAL [10/36] heat-10-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== EVAL [11/36] heat-11-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== EVAL [12/36] heat-12-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== EVAL [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== EVAL [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== EVAL [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== EVAL [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== EVAL [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== EVAL [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== EVAL [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml

echo "=== EVAL [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== EVAL [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== EVAL [23/36] nl_heat-23-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== EVAL [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml

echo "=== EVAL [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== EVAL [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== EVAL [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== EVAL [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml

echo "=== EVAL [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== EVAL [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== EVAL [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== EVAL [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml

echo "=== EVAL [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== EVAL [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== EVAL [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== EVAL [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [1/36] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/36] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/36] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/36] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/36] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/36] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/36] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/36] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/36] heat-9-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== VIS [10/36] heat-10-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== VIS [11/36] heat-11-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== VIS [12/36] heat-12-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== VIS [13/36] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== VIS [14/36] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== VIS [15/36] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== VIS [16/36] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== VIS [17/36] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/36] heat-18-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x150x450-anil-2epoch.yaml

echo "=== VIS [19/36] nl_heat-19-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs.yaml

echo "=== VIS [20/36] nl_heat-20-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-adam-mb25.yaml

echo "=== VIS [21/36] nl_heat-21-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs-5k.yaml

echo "=== VIS [22/36] nl_heat-22-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs-mb25.yaml

echo "=== VIS [23/36] nl_heat-23-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-250x250.yaml

echo "=== VIS [24/36] nl_heat-24-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-250x250-anil.yaml

echo "=== VIS [25/36] nl_heat-25-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-100x100x100-anil.yaml

echo "=== VIS [26/36] nl_heat-26-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-350x350-anil.yaml

echo "=== VIS [27/36] nl_heat-27-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-100x150x300-anil.yaml

echo "=== VIS [28/36] nl_heat-28-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-100x100x300-anil.yaml

echo "=== VIS [29/36] nl_heat-29-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-300x300x300-anil.yaml

echo "=== VIS [30/36] nl_heat-30-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil-bypass.yaml

echo "=== VIS [31/36] nl_heat-31-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x100-anil-bypass.yaml

echo "=== VIS [32/36] nl_heat-32-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x120x300-anil-bypass.yaml

echo "=== VIS [33/36] nl_heat-33-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-100x100x300-anil-bypass.yaml

echo "=== VIS [34/36] nl_heat-34-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil-2epoch.yaml

echo "=== VIS [35/36] nl_heat-35-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x150x450-anil.yaml

echo "=== VIS [36/36] nl_heat-36-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x150x450-anil-2epoch.yaml
