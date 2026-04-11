#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/52] heat-1-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== TRAIN [2/52] heat-2-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== TRAIN [3/52] heat-3-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== TRAIN [4/52] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== TRAIN [5/52] heat-5-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== TRAIN [6/52] heat-6-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== TRAIN [7/52] heat-7-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== TRAIN [8/52] heat-8-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== TRAIN [9/52] heat-9-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== TRAIN [10/52] heat-10-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== TRAIN [11/52] heat-11-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== TRAIN [12/52] heat-12-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== TRAIN [13/52] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== TRAIN [14/52] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== TRAIN [15/52] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== TRAIN [16/52] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== TRAIN [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== TRAIN [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== TRAIN [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== TRAIN [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== TRAIN [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml

echo "=== TRAIN [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== TRAIN [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== TRAIN [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== TRAIN [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml

echo "=== TRAIN [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== TRAIN [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== TRAIN [31/52] nl_heat-31-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== TRAIN [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml

echo "=== TRAIN [33/52] nl_heat-33-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil.yaml

echo "=== TRAIN [34/52] nl_heat-34-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil.yaml

echo "=== TRAIN [35/52] nl_heat-35-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x150x300-anil.yaml

echo "=== TRAIN [36/52] nl_heat-36-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil.yaml

echo "=== TRAIN [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== TRAIN [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== TRAIN [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== TRAIN [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml

echo "=== TRAIN [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== TRAIN [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== TRAIN [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml

echo "=== TRAIN [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== TRAIN [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml

echo "=== TRAIN [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== TRAIN [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== TRAIN [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== TRAIN [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [1/52] heat-1-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== EVAL [2/52] heat-2-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== EVAL [3/52] heat-3-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== EVAL [4/52] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== EVAL [5/52] heat-5-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== EVAL [6/52] heat-6-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== EVAL [7/52] heat-7-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== EVAL [8/52] heat-8-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== EVAL [9/52] heat-9-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== EVAL [10/52] heat-10-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== EVAL [11/52] heat-11-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== EVAL [12/52] heat-12-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== EVAL [13/52] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== EVAL [14/52] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== EVAL [15/52] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== EVAL [16/52] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== EVAL [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== EVAL [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== EVAL [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== EVAL [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== EVAL [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== EVAL [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml

echo "=== EVAL [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== EVAL [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== EVAL [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== EVAL [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml

echo "=== EVAL [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== EVAL [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== EVAL [31/52] nl_heat-31-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== EVAL [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml

echo "=== EVAL [33/52] nl_heat-33-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil.yaml

echo "=== EVAL [34/52] nl_heat-34-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil.yaml

echo "=== EVAL [35/52] nl_heat-35-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x150x300-anil.yaml

echo "=== EVAL [36/52] nl_heat-36-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil.yaml

echo "=== EVAL [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== EVAL [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== EVAL [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== EVAL [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml

echo "=== EVAL [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== EVAL [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== EVAL [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== EVAL [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml

echo "=== EVAL [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== EVAL [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml

echo "=== EVAL [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== EVAL [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== EVAL [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== EVAL [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [1/52] heat-1-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-1-adam+lbfgs.yaml

echo "=== VIS [2/52] heat-2-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-2-adam-mb25.yaml

echo "=== VIS [3/52] heat-3-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-3-adam+lbfgs-5k.yaml

echo "=== VIS [4/52] heat-4-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-4-adam+lbfgs-mb25.yaml

echo "=== VIS [5/52] heat-5-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-5-250x250.yaml

echo "=== VIS [6/52] heat-6-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-6-250x250-anil.yaml

echo "=== VIS [7/52] heat-7-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-7-100x100x100-anil.yaml

echo "=== VIS [8/52] heat-8-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-8-350x350-anil.yaml

echo "=== VIS [9/52] heat-9-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-9-100x150x300-anil.yaml

echo "=== VIS [10/52] heat-10-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-10-100x100x300-anil.yaml

echo "=== VIS [11/52] heat-11-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-11-300x300x300-anil.yaml

echo "=== VIS [12/52] heat-12-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-12-350x350-anil-bypass.yaml

echo "=== VIS [13/52] heat-13-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-13-100x100x100-anil-bypass.yaml

echo "=== VIS [14/52] heat-14-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-14-100x120x300-anil-bypass.yaml

echo "=== VIS [15/52] heat-15-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-15-100x100x300-anil-bypass.yaml

echo "=== VIS [16/52] heat-16-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-16-100x150x300-anil-2epoch.yaml

echo "=== VIS [17/52] heat-17-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-17-100x150x450-anil.yaml

echo "=== VIS [18/52] heat-18-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-18-100x120x150x300-anil.yaml

echo "=== VIS [19/52] heat-19-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-19-32x32x32x32x32x300-anil.yaml

echo "=== VIS [20/52] heat-20-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-20-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [21/52] heat-21-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-21-153x301x450-anil.yaml

echo "=== VIS [22/52] heat-22-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-22-116x228x339x450-anil.yaml

echo "=== VIS [23/52] heat-23-100x150x300x600-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-23-100x150x300x600-anil.yaml

echo "=== VIS [24/52] heat-24-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-24-94x183x272x361x450-anil.yaml

echo "=== VIS [25/52] heat-25-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-25-100x120x150x450-anil.yaml

echo "=== VIS [26/52] heat-26-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/heat-26-100x150x450-anil-2epoch.yaml

echo "=== VIS [27/52] nl_heat-27-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-adam+lbfgs.yaml

echo "=== VIS [28/52] nl_heat-28-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-adam-mb25.yaml

echo "=== VIS [29/52] nl_heat-29-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-5k.yaml

echo "=== VIS [30/52] nl_heat-30-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-adam+lbfgs-mb25.yaml

echo "=== VIS [31/52] nl_heat-31-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-250x250.yaml

echo "=== VIS [32/52] nl_heat-32-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-250x250-anil.yaml

echo "=== VIS [33/52] nl_heat-33-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil.yaml

echo "=== VIS [34/52] nl_heat-34-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-350x350-anil.yaml

echo "=== VIS [35/52] nl_heat-35-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x150x300-anil.yaml

echo "=== VIS [36/52] nl_heat-36-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil.yaml

echo "=== VIS [37/52] nl_heat-37-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-300x300x300-anil.yaml

echo "=== VIS [38/52] nl_heat-38-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-350x350-anil-bypass.yaml

echo "=== VIS [39/52] nl_heat-39-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x100x100-anil-bypass.yaml

echo "=== VIS [40/52] nl_heat-40-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x120x300-anil-bypass.yaml

echo "=== VIS [41/52] nl_heat-41-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-100x100x300-anil-bypass.yaml

echo "=== VIS [42/52] nl_heat-42-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-100x150x300-anil-2epoch.yaml

echo "=== VIS [43/52] nl_heat-43-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-43-100x150x450-anil.yaml

echo "=== VIS [44/52] nl_heat-44-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-44-100x120x150x300-anil.yaml

echo "=== VIS [45/52] nl_heat-45-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-45-32x32x32x32x32x300-anil.yaml

echo "=== VIS [46/52] nl_heat-46-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-46-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [47/52] nl_heat-47-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-47-153x301x450-anil.yaml

echo "=== VIS [48/52] nl_heat-48-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-48-116x228x339x450-anil.yaml

echo "=== VIS [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== VIS [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== VIS [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== VIS [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml
