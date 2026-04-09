#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== TRAIN [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== TRAIN [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== TRAIN [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml

# === EVALUATE ===
echo "=== EVAL [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== EVAL [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== EVAL [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== EVAL [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml

# === VISUALIZE ===
echo "=== VIS [17/22] nl_heat-17-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-17-250x250-anil.yaml

echo "=== VIS [18/22] nl_heat-18-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-18-100x100x100-anil.yaml

echo "=== VIS [19/22] nl_heat-19-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-19-350x350-anil.yaml

echo "=== VIS [20/22] nl_heat-20-100x150x300-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-20-100x150x300-anil.yaml
