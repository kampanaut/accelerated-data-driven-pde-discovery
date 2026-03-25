#!/usr/bin/env bash
set -e

# === TRAIN ===
echo "=== TRAIN [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== TRAIN [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== TRAIN [3/48] br-3-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== TRAIN [4/48] br-4-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml

echo "=== TRAIN [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== TRAIN [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== TRAIN [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== TRAIN [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml

echo "=== TRAIN [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-9-5step-k800-baseline.yaml

echo "=== TRAIN [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-10-5step-k10-baseline.yaml

echo "=== TRAIN [11/48] br-11-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-11-5step-k800-metal.yaml

echo "=== TRAIN [12/48] br-12-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-12-5step-k10-metal.yaml

echo "=== TRAIN [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-13-5step-k800-spectral.yaml

echo "=== TRAIN [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-14-5step-k10-spectral.yaml

echo "=== TRAIN [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-15-5step-k800-metal-spectral.yaml

echo "=== TRAIN [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/br-16-5step-k10-metal-spectral.yaml

echo "=== TRAIN [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-17-1step-k800-baseline.yaml

echo "=== TRAIN [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-18-1step-k10-baseline.yaml

echo "=== TRAIN [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-19-1step-k800-metal.yaml

echo "=== TRAIN [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-20-1step-k10-metal.yaml

echo "=== TRAIN [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-21-1step-k800-spectral.yaml

echo "=== TRAIN [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-22-1step-k10-spectral.yaml

echo "=== TRAIN [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-23-1step-k800-metal-spectral.yaml

echo "=== TRAIN [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-24-1step-k10-metal-spectral.yaml

echo "=== TRAIN [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-25-5step-k800-baseline.yaml

echo "=== TRAIN [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-26-5step-k10-baseline.yaml

echo "=== TRAIN [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-27-5step-k800-metal.yaml

echo "=== TRAIN [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-28-5step-k10-metal.yaml

echo "=== TRAIN [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-29-5step-k800-spectral.yaml

echo "=== TRAIN [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-30-5step-k10-spectral.yaml

echo "=== TRAIN [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-31-5step-k800-metal-spectral.yaml

echo "=== TRAIN [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/heat-32-5step-k10-metal-spectral.yaml

echo "=== TRAIN [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-33-1step-k800-baseline.yaml

echo "=== TRAIN [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-34-1step-k10-baseline.yaml

echo "=== TRAIN [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-35-1step-k800-metal.yaml

echo "=== TRAIN [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-36-1step-k10-metal.yaml

echo "=== TRAIN [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== TRAIN [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== TRAIN [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== TRAIN [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== TRAIN [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-41-5step-k800-baseline.yaml

echo "=== TRAIN [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-42-5step-k10-baseline.yaml

echo "=== TRAIN [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-43-5step-k800-metal.yaml

echo "=== TRAIN [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-44-5step-k10-metal.yaml

echo "=== TRAIN [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-45-5step-k800-spectral.yaml

echo "=== TRAIN [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-46-5step-k10-spectral.yaml

echo "=== TRAIN [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== TRAIN [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/train_maml.py --config configs/cheat_zeroed_finn/nl_heat-48-5step-k10-metal-spectral.yaml

# === EVALUATE ===
echo "=== EVAL [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== EVAL [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== EVAL [3/48] br-3-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== EVAL [4/48] br-4-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml

echo "=== EVAL [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== EVAL [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== EVAL [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== EVAL [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml

echo "=== EVAL [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-9-5step-k800-baseline.yaml

echo "=== EVAL [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-10-5step-k10-baseline.yaml

echo "=== EVAL [11/48] br-11-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-11-5step-k800-metal.yaml

echo "=== EVAL [12/48] br-12-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-12-5step-k10-metal.yaml

echo "=== EVAL [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-13-5step-k800-spectral.yaml

echo "=== EVAL [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-14-5step-k10-spectral.yaml

echo "=== EVAL [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-15-5step-k800-metal-spectral.yaml

echo "=== EVAL [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/br-16-5step-k10-metal-spectral.yaml

echo "=== EVAL [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-17-1step-k800-baseline.yaml

echo "=== EVAL [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-18-1step-k10-baseline.yaml

echo "=== EVAL [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-19-1step-k800-metal.yaml

echo "=== EVAL [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-20-1step-k10-metal.yaml

echo "=== EVAL [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-21-1step-k800-spectral.yaml

echo "=== EVAL [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-22-1step-k10-spectral.yaml

echo "=== EVAL [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-23-1step-k800-metal-spectral.yaml

echo "=== EVAL [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-24-1step-k10-metal-spectral.yaml

echo "=== EVAL [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-25-5step-k800-baseline.yaml

echo "=== EVAL [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-26-5step-k10-baseline.yaml

echo "=== EVAL [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-27-5step-k800-metal.yaml

echo "=== EVAL [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-28-5step-k10-metal.yaml

echo "=== EVAL [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-29-5step-k800-spectral.yaml

echo "=== EVAL [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-30-5step-k10-spectral.yaml

echo "=== EVAL [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-31-5step-k800-metal-spectral.yaml

echo "=== EVAL [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/heat-32-5step-k10-metal-spectral.yaml

echo "=== EVAL [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-33-1step-k800-baseline.yaml

echo "=== EVAL [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-34-1step-k10-baseline.yaml

echo "=== EVAL [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-35-1step-k800-metal.yaml

echo "=== EVAL [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-36-1step-k10-metal.yaml

echo "=== EVAL [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== EVAL [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== EVAL [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== EVAL [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== EVAL [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-41-5step-k800-baseline.yaml

echo "=== EVAL [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-42-5step-k10-baseline.yaml

echo "=== EVAL [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-43-5step-k800-metal.yaml

echo "=== EVAL [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-44-5step-k10-metal.yaml

echo "=== EVAL [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-45-5step-k800-spectral.yaml

echo "=== EVAL [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-46-5step-k10-spectral.yaml

echo "=== EVAL [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== EVAL [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/evaluate.py --config configs/cheat_zeroed_finn/nl_heat-48-5step-k10-metal-spectral.yaml

# === VISUALIZE ===
echo "=== VIS [1/48] br-1-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-1-1step-k800-baseline.yaml

echo "=== VIS [2/48] br-2-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-2-1step-k10-baseline.yaml

echo "=== VIS [3/48] br-3-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-3-1step-k800-metal.yaml

echo "=== VIS [4/48] br-4-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-4-1step-k10-metal.yaml

echo "=== VIS [5/48] br-5-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-5-1step-k800-spectral.yaml

echo "=== VIS [6/48] br-6-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-6-1step-k10-spectral.yaml

echo "=== VIS [7/48] br-7-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-7-1step-k800-metal-spectral.yaml

echo "=== VIS [8/48] br-8-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-8-1step-k10-metal-spectral.yaml

echo "=== VIS [9/48] br-9-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-9-5step-k800-baseline.yaml

echo "=== VIS [10/48] br-10-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-10-5step-k10-baseline.yaml

echo "=== VIS [11/48] br-11-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-11-5step-k800-metal.yaml

echo "=== VIS [12/48] br-12-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-12-5step-k10-metal.yaml

echo "=== VIS [13/48] br-13-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-13-5step-k800-spectral.yaml

echo "=== VIS [14/48] br-14-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-14-5step-k10-spectral.yaml

echo "=== VIS [15/48] br-15-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-15-5step-k800-metal-spectral.yaml

echo "=== VIS [16/48] br-16-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/br-16-5step-k10-metal-spectral.yaml

echo "=== VIS [17/48] heat-17-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-17-1step-k800-baseline.yaml

echo "=== VIS [18/48] heat-18-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-18-1step-k10-baseline.yaml

echo "=== VIS [19/48] heat-19-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-19-1step-k800-metal.yaml

echo "=== VIS [20/48] heat-20-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-20-1step-k10-metal.yaml

echo "=== VIS [21/48] heat-21-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-21-1step-k800-spectral.yaml

echo "=== VIS [22/48] heat-22-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-22-1step-k10-spectral.yaml

echo "=== VIS [23/48] heat-23-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-23-1step-k800-metal-spectral.yaml

echo "=== VIS [24/48] heat-24-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-24-1step-k10-metal-spectral.yaml

echo "=== VIS [25/48] heat-25-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-25-5step-k800-baseline.yaml

echo "=== VIS [26/48] heat-26-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-26-5step-k10-baseline.yaml

echo "=== VIS [27/48] heat-27-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-27-5step-k800-metal.yaml

echo "=== VIS [28/48] heat-28-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-28-5step-k10-metal.yaml

echo "=== VIS [29/48] heat-29-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-29-5step-k800-spectral.yaml

echo "=== VIS [30/48] heat-30-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-30-5step-k10-spectral.yaml

echo "=== VIS [31/48] heat-31-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-31-5step-k800-metal-spectral.yaml

echo "=== VIS [32/48] heat-32-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/heat-32-5step-k10-metal-spectral.yaml

echo "=== VIS [33/48] nl_heat-33-1step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-33-1step-k800-baseline.yaml

echo "=== VIS [34/48] nl_heat-34-1step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-34-1step-k10-baseline.yaml

echo "=== VIS [35/48] nl_heat-35-1step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-35-1step-k800-metal.yaml

echo "=== VIS [36/48] nl_heat-36-1step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-36-1step-k10-metal.yaml

echo "=== VIS [37/48] nl_heat-37-1step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-37-1step-k800-spectral.yaml

echo "=== VIS [38/48] nl_heat-38-1step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-38-1step-k10-spectral.yaml

echo "=== VIS [39/48] nl_heat-39-1step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-39-1step-k800-metal-spectral.yaml

echo "=== VIS [40/48] nl_heat-40-1step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-40-1step-k10-metal-spectral.yaml

echo "=== VIS [41/48] nl_heat-41-5step-k800-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-41-5step-k800-baseline.yaml

echo "=== VIS [42/48] nl_heat-42-5step-k10-baseline ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-42-5step-k10-baseline.yaml

echo "=== VIS [43/48] nl_heat-43-5step-k800-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-43-5step-k800-metal.yaml

echo "=== VIS [44/48] nl_heat-44-5step-k10-metal ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-44-5step-k10-metal.yaml

echo "=== VIS [45/48] nl_heat-45-5step-k800-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-45-5step-k800-spectral.yaml

echo "=== VIS [46/48] nl_heat-46-5step-k10-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-46-5step-k10-spectral.yaml

echo "=== VIS [47/48] nl_heat-47-5step-k800-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-47-5step-k800-metal-spectral.yaml

echo "=== VIS [48/48] nl_heat-48-5step-k10-metal-spectral ==="
uv run python scripts/visualize.py --config configs/cheat_zeroed_finn/nl_heat-48-5step-k10-metal-spectral.yaml
