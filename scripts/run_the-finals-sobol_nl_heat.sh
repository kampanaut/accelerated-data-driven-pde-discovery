#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== TRAIN [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== TRAIN [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== TRAIN [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml

echo "=== TRAIN [25/40] nl_heat-25-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== TRAIN [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== TRAIN [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== TRAIN [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml

echo "=== TRAIN [29/40] nl_heat-29-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x150x300-anil.yaml

echo "=== TRAIN [30/40] nl_heat-30-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil.yaml

echo "=== TRAIN [31/40] nl_heat-31-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-300x300x300-anil.yaml

echo "=== TRAIN [32/40] nl_heat-32-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-350x350-anil-bypass.yaml

echo "=== TRAIN [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== TRAIN [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== TRAIN [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== TRAIN [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml

echo "=== TRAIN [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== TRAIN [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== TRAIN [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== TRAIN [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== EVAL [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== EVAL [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== EVAL [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml

echo "=== EVAL [25/40] nl_heat-25-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== EVAL [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== EVAL [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== EVAL [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml

echo "=== EVAL [29/40] nl_heat-29-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x150x300-anil.yaml

echo "=== EVAL [30/40] nl_heat-30-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil.yaml

echo "=== EVAL [31/40] nl_heat-31-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-300x300x300-anil.yaml

echo "=== EVAL [32/40] nl_heat-32-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-350x350-anil-bypass.yaml

echo "=== EVAL [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== EVAL [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== EVAL [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== EVAL [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml

echo "=== EVAL [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== EVAL [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== EVAL [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== EVAL [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [21/40] nl_heat-21-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-adam+lbfgs.yaml

echo "=== VIS [22/40] nl_heat-22-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-adam-mb25.yaml

echo "=== VIS [23/40] nl_heat-23-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-adam+lbfgs-5k.yaml

echo "=== VIS [24/40] nl_heat-24-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-adam+lbfgs-mb25.yaml

echo "=== VIS [25/40] nl_heat-25-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== VIS [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== VIS [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== VIS [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml

echo "=== VIS [29/40] nl_heat-29-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x150x300-anil.yaml

echo "=== VIS [30/40] nl_heat-30-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil.yaml

echo "=== VIS [31/40] nl_heat-31-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-300x300x300-anil.yaml

echo "=== VIS [32/40] nl_heat-32-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-350x350-anil-bypass.yaml

echo "=== VIS [33/40] nl_heat-33-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-33-100x100x100-anil-bypass.yaml

echo "=== VIS [34/40] nl_heat-34-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-34-100x120x300-anil-bypass.yaml

echo "=== VIS [35/40] nl_heat-35-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-35-100x100x300-anil-bypass.yaml

echo "=== VIS [36/40] nl_heat-36-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-36-100x150x300-anil-2epoch.yaml

echo "=== VIS [37/40] nl_heat-37-100x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-37-100x150x450-anil.yaml

echo "=== VIS [38/40] nl_heat-38-100x120x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-38-100x120x150x300-anil.yaml

echo "=== VIS [39/40] nl_heat-39-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-39-100x120x150x450-anil.yaml

echo "=== VIS [40/40] nl_heat-40-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-40-100x150x450-anil-2epoch.yaml
