#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== TRAIN [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== TRAIN [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== TRAIN [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml

echo "=== TRAIN [21/32] nl_heat-21-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-21-250x250.yaml

echo "=== TRAIN [22/32] nl_heat-22-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-22-250x250-anil.yaml

echo "=== TRAIN [23/32] nl_heat-23-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-23-100x100x100-anil.yaml

echo "=== TRAIN [24/32] nl_heat-24-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-24-350x350-anil.yaml

echo "=== TRAIN [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== TRAIN [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== TRAIN [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== TRAIN [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml

echo "=== TRAIN [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== TRAIN [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== TRAIN [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== TRAIN [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== EVAL [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== EVAL [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== EVAL [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml

echo "=== EVAL [21/32] nl_heat-21-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-21-250x250.yaml

echo "=== EVAL [22/32] nl_heat-22-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-22-250x250-anil.yaml

echo "=== EVAL [23/32] nl_heat-23-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-23-100x100x100-anil.yaml

echo "=== EVAL [24/32] nl_heat-24-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-24-350x350-anil.yaml

echo "=== EVAL [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== EVAL [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== EVAL [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== EVAL [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml

echo "=== EVAL [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== EVAL [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== EVAL [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== EVAL [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [17/32] nl_heat-17-adam+lbfgs ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-adam+lbfgs.yaml

echo "=== VIS [18/32] nl_heat-18-adam-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-adam-mb25.yaml

echo "=== VIS [19/32] nl_heat-19-adam+lbfgs-5k ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-adam+lbfgs-5k.yaml

echo "=== VIS [20/32] nl_heat-20-adam+lbfgs-mb25 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-adam+lbfgs-mb25.yaml

echo "=== VIS [21/32] nl_heat-21-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-21-250x250.yaml

echo "=== VIS [22/32] nl_heat-22-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-22-250x250-anil.yaml

echo "=== VIS [23/32] nl_heat-23-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-23-100x100x100-anil.yaml

echo "=== VIS [24/32] nl_heat-24-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-24-350x350-anil.yaml

echo "=== VIS [25/32] nl_heat-25-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-100x150x300-anil.yaml

echo "=== VIS [26/32] nl_heat-26-100x100x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-100x100x300-anil.yaml

echo "=== VIS [27/32] nl_heat-27-300x300x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-300x300x300-anil.yaml

echo "=== VIS [28/32] nl_heat-28-350x350-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil-bypass.yaml

echo "=== VIS [29/32] nl_heat-29-100x100x100-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-29-100x100x100-anil-bypass.yaml

echo "=== VIS [30/32] nl_heat-30-100x120x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-30-100x120x300-anil-bypass.yaml

echo "=== VIS [31/32] nl_heat-31-100x100x300-anil-bypass ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-31-100x100x300-anil-bypass.yaml

echo "=== VIS [32/32] nl_heat-32-100x150x300-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-32-100x150x300-anil-2epoch.yaml
