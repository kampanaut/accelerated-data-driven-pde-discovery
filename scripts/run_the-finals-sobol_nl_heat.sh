#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [16/30] nl_heat-16-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-16-adam+lbfgs.yaml

echo "=== TRAIN [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== TRAIN [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== TRAIN [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== TRAIN [20/30] nl_heat-20-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml

echo "=== TRAIN [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== TRAIN [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== TRAIN [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== TRAIN [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml

echo "=== TRAIN [25/30] nl_heat-25-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-100x100x300-anil.yaml

echo "=== TRAIN [26/30] nl_heat-26-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-300x300x300-anil.yaml

echo "=== TRAIN [27/30] nl_heat-27-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-350x350-anil-bypass.yaml

echo "=== TRAIN [28/30] nl_heat-28-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil-bypass.yaml

echo "=== TRAIN [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== TRAIN [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml

# === EVALUATE ===
echo "=== EVAL [16/30] nl_heat-16-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-16-adam+lbfgs.yaml

echo "=== EVAL [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== EVAL [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== EVAL [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== EVAL [20/30] nl_heat-20-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml

echo "=== EVAL [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== EVAL [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== EVAL [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== EVAL [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml

echo "=== EVAL [25/30] nl_heat-25-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-100x100x300-anil.yaml

echo "=== EVAL [26/30] nl_heat-26-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-300x300x300-anil.yaml

echo "=== EVAL [27/30] nl_heat-27-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-350x350-anil-bypass.yaml

echo "=== EVAL [28/30] nl_heat-28-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil-bypass.yaml

echo "=== EVAL [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== EVAL [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml

# === VISUALIZE ===
echo "=== VIS [16/30] nl_heat-16-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-16-adam+lbfgs.yaml

echo "=== VIS [17/30] nl_heat-17-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-adam-mb25.yaml

echo "=== VIS [18/30] nl_heat-18-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-adam+lbfgs-5k.yaml

echo "=== VIS [19/30] nl_heat-19-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-mb25.yaml

echo "=== VIS [20/30] nl_heat-20-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-250x250.yaml

echo "=== VIS [21/30] nl_heat-21-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-250x250-anil.yaml

echo "=== VIS [22/30] nl_heat-22-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-100x100x100-anil.yaml

echo "=== VIS [23/30] nl_heat-23-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-350x350-anil.yaml

echo "=== VIS [24/30] nl_heat-24-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-100x150x300-anil.yaml

echo "=== VIS [25/30] nl_heat-25-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-100x100x300-anil.yaml

echo "=== VIS [26/30] nl_heat-26-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-300x300x300-anil.yaml

echo "=== VIS [27/30] nl_heat-27-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-350x350-anil-bypass.yaml

echo "=== VIS [28/30] nl_heat-28-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-100x100x100-anil-bypass.yaml

echo "=== VIS [29/30] nl_heat-29-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x120x300-anil-bypass.yaml

echo "=== VIS [30/30] nl_heat-30-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x100x300-anil-bypass.yaml
