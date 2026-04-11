#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== TRAIN [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== TRAIN [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== TRAIN [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/train_maml.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml

# === EVALUATE ===
echo "=== EVAL [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== EVAL [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== EVAL [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== EVAL [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/evaluate.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml

# === VISUALIZE ===
echo "=== VIS [49/52] nl_heat-49-100x150x300x600-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-49-100x150x300x600-anil.yaml

echo "=== VIS [50/52] nl_heat-50-94x183x272x361x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-50-94x183x272x361x450-anil.yaml

echo "=== VIS [51/52] nl_heat-51-100x120x150x450-anil ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-51-100x120x150x450-anil.yaml

echo "=== VIS [52/52] nl_heat-52-100x150x450-anil-2epoch ==="
uv run python scripts/visualize.py --config configs/the-finals-sobol/nl_heat-52-100x150x450-anil-2epoch.yaml
