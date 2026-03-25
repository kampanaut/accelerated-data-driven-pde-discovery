#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-17-1step-k800-baseline.yaml

echo "=== TRAIN [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-18-1step-k10-baseline.yaml

echo "=== TRAIN [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-19-1step-k800-metal.yaml

echo "=== TRAIN [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-20-1step-k10-metal.yaml

echo "=== TRAIN [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-21-1step-k800-spectral.yaml

echo "=== TRAIN [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-22-1step-k10-spectral.yaml

echo "=== TRAIN [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-23-1step-k800-metal-spectral.yaml

echo "=== TRAIN [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-24-1step-k10-metal-spectral.yaml

echo "=== TRAIN [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-25-5step-k800-baseline.yaml

echo "=== TRAIN [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-26-5step-k10-baseline.yaml

echo "=== TRAIN [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-27-5step-k800-metal.yaml

echo "=== TRAIN [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-28-5step-k10-metal.yaml

echo "=== TRAIN [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-29-5step-k800-spectral.yaml

echo "=== TRAIN [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-30-5step-k10-spectral.yaml

echo "=== TRAIN [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-31-5step-k800-metal-spectral.yaml

echo "=== TRAIN [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zero_init/heat-32-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-17-1step-k800-baseline.yaml

echo "=== EVAL [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-18-1step-k10-baseline.yaml

echo "=== EVAL [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-19-1step-k800-metal.yaml

echo "=== EVAL [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-20-1step-k10-metal.yaml

echo "=== EVAL [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-21-1step-k800-spectral.yaml

echo "=== EVAL [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-22-1step-k10-spectral.yaml

echo "=== EVAL [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-23-1step-k800-metal-spectral.yaml

echo "=== EVAL [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-24-1step-k10-metal-spectral.yaml

echo "=== EVAL [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-25-5step-k800-baseline.yaml

echo "=== EVAL [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-26-5step-k10-baseline.yaml

echo "=== EVAL [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-27-5step-k800-metal.yaml

echo "=== EVAL [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-28-5step-k10-metal.yaml

echo "=== EVAL [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-29-5step-k800-spectral.yaml

echo "=== EVAL [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-30-5step-k10-spectral.yaml

echo "=== EVAL [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-31-5step-k800-metal-spectral.yaml

echo "=== EVAL [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zero_init/heat-32-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
#echo "=== VIS [17/48] heat-17-1step-k800-baseline ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-17-1step-k800-baseline.yaml

#echo "=== VIS [18/48] heat-18-1step-k10-baseline ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-18-1step-k10-baseline.yaml

#echo "=== VIS [19/48] heat-19-1step-k800-metal ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-19-1step-k800-metal.yaml

#echo "=== VIS [20/48] heat-20-1step-k10-metal ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-20-1step-k10-metal.yaml

#echo "=== VIS [21/48] heat-21-1step-k800-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-21-1step-k800-spectral.yaml

#echo "=== VIS [22/48] heat-22-1step-k10-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-22-1step-k10-spectral.yaml

#echo "=== VIS [23/48] heat-23-1step-k800-metal-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-23-1step-k800-metal-spectral.yaml

#echo "=== VIS [24/48] heat-24-1step-k10-metal-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-24-1step-k10-metal-spectral.yaml

#echo "=== VIS [25/48] heat-25-5step-k800-baseline ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-25-5step-k800-baseline.yaml

#echo "=== VIS [26/48] heat-26-5step-k10-baseline ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-26-5step-k10-baseline.yaml

#echo "=== VIS [27/48] heat-27-5step-k800-metal ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-27-5step-k800-metal.yaml

#echo "=== VIS [28/48] heat-28-5step-k10-metal ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-28-5step-k10-metal.yaml

#echo "=== VIS [29/48] heat-29-5step-k800-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-29-5step-k800-spectral.yaml

#echo "=== VIS [30/48] heat-30-5step-k10-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-30-5step-k10-spectral.yaml

#echo "=== VIS [31/48] heat-31-5step-k800-metal-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-31-5step-k800-metal-spectral.yaml

#echo "=== VIS [32/48] heat-32-5step-k10-metal-spectral ==="
#uv run python scripts/visualize.py --config configs/cheat_zero_init/heat-32-5step-k10-metal-spectral.yaml
