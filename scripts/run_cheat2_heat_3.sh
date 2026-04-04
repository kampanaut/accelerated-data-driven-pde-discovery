#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== TRAIN [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml

# === EVALUATE ===
echo "=== EVAL [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== EVAL [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml

# === VISUALIZE ===
echo "=== VIS [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== VIS [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml
