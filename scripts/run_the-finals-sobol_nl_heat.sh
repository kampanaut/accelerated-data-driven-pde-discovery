#!/usr/bin/env bash
set -e

# === TRAIN ===
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
