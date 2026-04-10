#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/44] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/44] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/44] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/44] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/44] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/44] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/44] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/44] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/44] heat-9-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== TRAIN [10/44] heat-10-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== TRAIN [11/44] heat-11-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== TRAIN [12/44] heat-12-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== TRAIN [13/44] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== TRAIN [14/44] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== TRAIN [15/44] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== TRAIN [16/44] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [17/44] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/44] heat-18-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== TRAIN [19/44] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [20/44] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== TRAIN [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== TRAIN [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml

echo "=== TRAIN [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== TRAIN [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== TRAIN [27/44] nl_heat-27-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== TRAIN [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml

echo "=== TRAIN [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== TRAIN [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== TRAIN [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== TRAIN [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml

echo "=== TRAIN [33/44] nl_heat-33-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-300x300x300-anil.yaml

echo "=== TRAIN [34/44] nl_heat-34-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil-bypass.yaml

echo "=== TRAIN [35/44] nl_heat-35-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x100x100-anil-bypass.yaml

echo "=== TRAIN [36/44] nl_heat-36-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x120x300-anil-bypass.yaml

echo "=== TRAIN [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== TRAIN [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== TRAIN [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml

echo "=== TRAIN [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== TRAIN [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [1/44] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/44] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/44] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/44] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/44] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/44] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/44] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/44] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/44] heat-9-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== EVAL [10/44] heat-10-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== EVAL [11/44] heat-11-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== EVAL [12/44] heat-12-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== EVAL [13/44] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== EVAL [14/44] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== EVAL [15/44] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== EVAL [16/44] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== EVAL [17/44] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/44] heat-18-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== EVAL [19/44] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [20/44] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== EVAL [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== EVAL [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== EVAL [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml

echo "=== EVAL [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== EVAL [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== EVAL [27/44] nl_heat-27-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== EVAL [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml

echo "=== EVAL [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== EVAL [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== EVAL [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== EVAL [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml

echo "=== EVAL [33/44] nl_heat-33-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-300x300x300-anil.yaml

echo "=== EVAL [34/44] nl_heat-34-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil-bypass.yaml

echo "=== EVAL [35/44] nl_heat-35-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x100x100-anil-bypass.yaml

echo "=== EVAL [36/44] nl_heat-36-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x120x300-anil-bypass.yaml

echo "=== EVAL [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== EVAL [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== EVAL [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== EVAL [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml

echo "=== EVAL [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== EVAL [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [1/44] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/44] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/44] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/44] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/44] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/44] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/44] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/44] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/44] heat-9-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== VIS [10/44] heat-10-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== VIS [11/44] heat-11-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== VIS [12/44] heat-12-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== VIS [13/44] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== VIS [14/44] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== VIS [15/44] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== VIS [16/44] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== VIS [17/44] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/44] heat-18-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== VIS [19/44] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== VIS [20/44] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [21/44] heat-21-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-100x120x150x450-anil.yaml

echo "=== VIS [22/44] heat-22-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-22-100x150x450-anil-2epoch.yaml

echo "=== VIS [23/44] nl_heat-23-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs.yaml

echo "=== VIS [24/44] nl_heat-24-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam-mb25.yaml

echo "=== VIS [25/44] nl_heat-25-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-5k.yaml

echo "=== VIS [26/44] nl_heat-26-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs-mb25.yaml

echo "=== VIS [27/44] nl_heat-27-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-250x250.yaml

echo "=== VIS [28/44] nl_heat-28-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-250x250-anil.yaml

echo "=== VIS [29/44] nl_heat-29-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil.yaml

echo "=== VIS [30/44] nl_heat-30-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-350x350-anil.yaml

echo "=== VIS [31/44] nl_heat-31-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x150x300-anil.yaml

echo "=== VIS [32/44] nl_heat-32-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x100x300-anil.yaml

echo "=== VIS [33/44] nl_heat-33-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-300x300x300-anil.yaml

echo "=== VIS [34/44] nl_heat-34-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil-bypass.yaml

echo "=== VIS [35/44] nl_heat-35-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x100x100-anil-bypass.yaml

echo "=== VIS [36/44] nl_heat-36-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x120x300-anil-bypass.yaml

echo "=== VIS [37/44] nl_heat-37-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-100x100x300-anil-bypass.yaml

echo "=== VIS [38/44] nl_heat-38-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x150x300-anil-2epoch.yaml

echo "=== VIS [39/44] nl_heat-39-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x150x450-anil.yaml

echo "=== VIS [40/44] nl_heat-40-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x120x150x300-anil.yaml

echo "=== VIS [41/44] nl_heat-41-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-32x32x32x32x32x300-anil.yaml

echo "=== VIS [42/44] nl_heat-42-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [43/44] nl_heat-43-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x450-anil.yaml

echo "=== VIS [44/44] nl_heat-44-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-44-100x150x450-anil-2epoch.yaml
