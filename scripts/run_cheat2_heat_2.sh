#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== TRAIN [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== TRAIN [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== TRAIN [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/train_maml.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml

# === EVALUATE ===
echo "=== EVAL [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== EVAL [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== EVAL [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== EVAL [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/evaluate.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml

# === VISUALIZE ===
echo "=== VIS [5/10] heat-5-k800-mamlpp ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-5-k800-mamlpp.yaml

echo "=== VIS [6/10] heat-6-k10-mamlpp ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-6-k10-mamlpp.yaml

echo "=== VIS [7/10] heat-7-k800-mamlpp+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-7-k800-mamlpp+metal.yaml

echo "=== VIS [8/10] heat-8-k10-mamlpp+metal ==="
uv run python scripts/visualize.py --config configs/cheat2/heat-8-k10-mamlpp+metal.yaml
