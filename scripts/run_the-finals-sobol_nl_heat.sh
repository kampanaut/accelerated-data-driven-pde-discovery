#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== TRAIN [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== TRAIN [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml

echo "=== TRAIN [25/42] nl_heat-25-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-mb25.yaml

echo "=== TRAIN [26/42] nl_heat-26-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-250x250.yaml

echo "=== TRAIN [27/42] nl_heat-27-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-250x250-anil.yaml

echo "=== TRAIN [28/42] nl_heat-28-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil.yaml

echo "=== TRAIN [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== TRAIN [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== TRAIN [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== TRAIN [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml

echo "=== TRAIN [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== TRAIN [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== TRAIN [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== TRAIN [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml

echo "=== TRAIN [37/42] nl_heat-37-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [38/42] nl_heat-38-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x150x450-anil.yaml

echo "=== TRAIN [39/42] nl_heat-39-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x300-anil.yaml

echo "=== TRAIN [40/42] nl_heat-40-32x32x32x32x32x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-32x32x32x32x32x300-anil.yaml

echo "=== TRAIN [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== TRAIN [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== EVAL [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== EVAL [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml

echo "=== EVAL [25/42] nl_heat-25-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-mb25.yaml

echo "=== EVAL [26/42] nl_heat-26-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-250x250.yaml

echo "=== EVAL [27/42] nl_heat-27-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-250x250-anil.yaml

echo "=== EVAL [28/42] nl_heat-28-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil.yaml

echo "=== EVAL [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== EVAL [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== EVAL [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== EVAL [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml

echo "=== EVAL [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== EVAL [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== EVAL [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== EVAL [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml

echo "=== EVAL [37/42] nl_heat-37-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-100x150x300-anil-2epoch.yaml

echo "=== EVAL [38/42] nl_heat-38-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x150x450-anil.yaml

echo "=== EVAL [39/42] nl_heat-39-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x300-anil.yaml

echo "=== EVAL [40/42] nl_heat-40-32x32x32x32x32x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-32x32x32x32x32x300-anil.yaml

echo "=== EVAL [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== EVAL [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [22/42] nl_heat-22-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam+lbfgs.yaml

echo "=== VIS [23/42] nl_heat-23-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam-mb25.yaml

echo "=== VIS [24/42] nl_heat-24-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-5k.yaml

echo "=== VIS [25/42] nl_heat-25-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-adam+lbfgs-mb25.yaml

echo "=== VIS [26/42] nl_heat-26-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-250x250.yaml

echo "=== VIS [27/42] nl_heat-27-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-250x250-anil.yaml

echo "=== VIS [28/42] nl_heat-28-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil.yaml

echo "=== VIS [29/42] nl_heat-29-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-350x350-anil.yaml

echo "=== VIS [30/42] nl_heat-30-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x150x300-anil.yaml

echo "=== VIS [31/42] nl_heat-31-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil.yaml

echo "=== VIS [32/42] nl_heat-32-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-300x300x300-anil.yaml

echo "=== VIS [33/42] nl_heat-33-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-350x350-anil-bypass.yaml

echo "=== VIS [34/42] nl_heat-34-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x100x100-anil-bypass.yaml

echo "=== VIS [35/42] nl_heat-35-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x120x300-anil-bypass.yaml

echo "=== VIS [36/42] nl_heat-36-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x100x300-anil-bypass.yaml

echo "=== VIS [37/42] nl_heat-37-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-100x150x300-anil-2epoch.yaml

echo "=== VIS [38/42] nl_heat-38-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x150x450-anil.yaml

echo "=== VIS [39/42] nl_heat-39-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x300-anil.yaml

echo "=== VIS [40/42] nl_heat-40-32x32x32x32x32x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-32x32x32x32x32x300-anil.yaml

echo "=== VIS [41/42] nl_heat-41-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-41-100x120x150x450-anil.yaml

echo "=== VIS [42/42] nl_heat-42-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-42-100x150x450-anil-2epoch.yaml
