#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [25/40] nl_heat-25-250x250 ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== TRAIN [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== TRAIN [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== TRAIN [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml

# === EVALUATE ===
echo "=== EVAL [25/40] nl_heat-25-250x250 ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== EVAL [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== EVAL [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== EVAL [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml

# === VISUALIZE ===
echo "=== VIS [25/40] nl_heat-25-250x250 ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-25-250x250.yaml

echo "=== VIS [26/40] nl_heat-26-250x250-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-26-250x250-anil.yaml

echo "=== VIS [27/40] nl_heat-27-100x100x100-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-27-100x100x100-anil.yaml

echo "=== VIS [28/40] nl_heat-28-350x350-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-28-350x350-anil.yaml
