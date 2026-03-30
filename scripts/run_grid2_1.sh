#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== TRAIN [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== TRAIN [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== TRAIN [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== TRAIN [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== TRAIN [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/train_maml.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml

# === EVALUATE ===
echo "=== EVAL [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== EVAL [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== EVAL [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== EVAL [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== EVAL [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== EVAL [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/evaluate.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml

# === VISUALIZE ===
echo "=== VIS [1/24] cheat-1-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-1-5step-k10-baseline.yaml

echo "=== VIS [2/24] cheat-2-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-2-5step-k800-baseline.yaml

echo "=== VIS [3/24] cheat-3-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-3-5step-k10-metal.yaml

echo "=== VIS [4/24] cheat-4-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-4-5step-k800-metal.yaml

echo "=== VIS [5/24] cheat-5-5step-k10-maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-5-5step-k10-maml++.yaml

echo "=== VIS [6/24] cheat-6-5step-k800-maml++ ==="
uv run python scripts/visualize.py --config configs/cheat2/cheat-6-5step-k800-maml++.yaml
