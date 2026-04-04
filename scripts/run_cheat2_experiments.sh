#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/10] heat-1-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== TRAIN [2/10] heat-2-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== TRAIN [3/10] heat-3-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== TRAIN [4/10] heat-4-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-4-k10-metal.yaml

echo "=== TRAIN [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== TRAIN [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== TRAIN [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== TRAIN [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml

echo "=== TRAIN [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== TRAIN [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml

# === EVALUATE ===
echo "=== EVAL [1/10] heat-1-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== EVAL [2/10] heat-2-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== EVAL [3/10] heat-3-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== EVAL [4/10] heat-4-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-4-k10-metal.yaml

echo "=== EVAL [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== EVAL [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== EVAL [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== EVAL [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml

echo "=== EVAL [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== EVAL [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml

# === VISUALIZE ===
echo "=== VIS [1/10] heat-1-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-1-k800-baseline.yaml

echo "=== VIS [2/10] heat-2-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-2-k10-baseline.yaml

echo "=== VIS [3/10] heat-3-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-3-k800-metal.yaml

echo "=== VIS [4/10] heat-4-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-4-k10-metal.yaml

echo "=== VIS [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== VIS [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== VIS [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== VIS [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml

echo "=== VIS [9/10] heat-9-k800-imaml-lbfgs ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-9-k800-imaml-lbfgs.yaml

echo "=== VIS [10/10] heat-10-k800-imaml-lbfgs+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-10-k800-imaml-lbfgs+metal.yaml
