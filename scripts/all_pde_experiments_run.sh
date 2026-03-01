#! /usr/bin/bash

uv run python scripts/train_maml.py --config configs/br-eksperimentasyon-2.yaml && \
uv run python scripts/evaluate.py --config configs/br-eksperimentasyon-2.yaml && \
uv run python scripts/visualize.py --config configs/br-eksperimentasyon-2.yaml && \
uv run python scripts/train_maml.py --config configs/ns-eksperimentasyon-1.yaml && \
uv run python scripts/evaluate.py --config configs/ns-eksperimentasyon-1.yaml && \
uv run python scripts/visualize.py --config configs/ns-eksperimentasyon-1.yaml && \
uv run python scripts/train_maml.py --config configs/heat-eksperimentasyon-1.yaml && \
uv run python scripts/evaluate.py --config configs/heat-eksperimentasyon-1.yaml && \
uv run python scripts/visualize.py --config configs/heat-eksperimentasyon-1.yaml && \
uv run python scripts/train_maml.py --config configs/nl_heat-eksperimentasyon-1.yaml && \
uv run python scripts/evaluate.py --config configs/nl_heat-eksperimentasyon-1.yaml && \
uv run python scripts/visualize.py --config configs/nl_heat-eksperimentasyon-1.yaml && \
uv run python scripts/train_maml.py --config configs/lo-eksperimentasyon-1.yaml && \
uv run python scripts/evaluate.py --config configs/lo-eksperimentasyon-1.yaml && \
uv run python scripts/visualize.py --config configs/lo-eksperimentasyon-1.yaml && \
uv run python scripts/train_maml.py --config configs/fhn-eksperimentasyon-1.yaml && \
uv run python scripts/evaluate.py --config configs/fhn-eksperimentasyon-1.yaml && \
uv run python scripts/visualize.py --config configs/fhn-eksperimentasyon-1.yaml
