#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== TRAIN [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== TRAIN [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml

echo "=== TRAIN [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== TRAIN [30/50] nl_heat-30-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== TRAIN [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== TRAIN [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml

echo "=== TRAIN [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== TRAIN [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== TRAIN [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== TRAIN [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml

echo "=== TRAIN [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== TRAIN [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== TRAIN [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== TRAIN [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml

echo "=== TRAIN [41/50] nl_heat-41-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [42/50] nl_heat-42-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil.yaml

echo "=== TRAIN [43/50] nl_heat-43-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x300-anil.yaml

echo "=== TRAIN [44/50] nl_heat-44-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-44-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== TRAIN [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== TRAIN [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== TRAIN [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml

echo "=== TRAIN [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== TRAIN [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== EVAL [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== EVAL [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml

echo "=== EVAL [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== EVAL [30/50] nl_heat-30-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== EVAL [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== EVAL [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml

echo "=== EVAL [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== EVAL [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== EVAL [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== EVAL [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml

echo "=== EVAL [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== EVAL [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== EVAL [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== EVAL [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml

echo "=== EVAL [41/50] nl_heat-41-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-100x150x300-anil-2epoch.yaml

echo "=== EVAL [42/50] nl_heat-42-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil.yaml

echo "=== EVAL [43/50] nl_heat-43-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x300-anil.yaml

echo "=== EVAL [44/50] nl_heat-44-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-44-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== EVAL [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== EVAL [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== EVAL [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml

echo "=== EVAL [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== EVAL [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [26/50] nl_heat-26-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-adam+lbfgs.yaml

echo "=== VIS [27/50] nl_heat-27-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-adam-mb25.yaml

echo "=== VIS [28/50] nl_heat-28-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-adam+lbfgs-5k.yaml

echo "=== VIS [29/50] nl_heat-29-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-adam+lbfgs-mb25.yaml

echo "=== VIS [30/50] nl_heat-30-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-250x250.yaml

echo "=== VIS [31/50] nl_heat-31-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-250x250-anil.yaml

echo "=== VIS [32/50] nl_heat-32-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x100x100-anil.yaml

echo "=== VIS [33/50] nl_heat-33-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil.yaml

echo "=== VIS [34/50] nl_heat-34-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x150x300-anil.yaml

echo "=== VIS [35/50] nl_heat-35-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil.yaml

echo "=== VIS [36/50] nl_heat-36-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-300x300x300-anil.yaml

echo "=== VIS [37/50] nl_heat-37-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-350x350-anil-bypass.yaml

echo "=== VIS [38/50] nl_heat-38-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x100x100-anil-bypass.yaml

echo "=== VIS [39/50] nl_heat-39-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x120x300-anil-bypass.yaml

echo "=== VIS [40/50] nl_heat-40-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x100x300-anil-bypass.yaml

echo "=== VIS [41/50] nl_heat-41-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-100x150x300-anil-2epoch.yaml

echo "=== VIS [42/50] nl_heat-42-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil.yaml

echo "=== VIS [43/50] nl_heat-43-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-43-100x120x150x300-anil.yaml

echo "=== VIS [44/50] nl_heat-44-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-44-32x32x32x32x32x300-anil.yaml

echo "=== VIS [45/50] nl_heat-45-32x64x72x128x136x245x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-45-32x64x72x128x136x245x300-anil.yaml

echo "=== VIS [46/50] nl_heat-46-153x301x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-46-153x301x450-anil.yaml

echo "=== VIS [47/50] nl_heat-47-116x228x339x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-47-116x228x339x450-anil.yaml

echo "=== VIS [48/50] nl_heat-48-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-48-94x183x272x361x450-anil.yaml

echo "=== VIS [49/50] nl_heat-49-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-49-100x120x150x450-anil.yaml

echo "=== VIS [50/50] nl_heat-50-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-50-100x150x450-anil-2epoch.yaml
