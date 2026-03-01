#! /usr/bin/bash

uv run python scripts/dataset_analyser.py data/datasets/br_train-2/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/br_test-2/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/ns_train-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/ns_test-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/heat_train-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/heat_test-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/nl_heat_train-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/nl_heat_test-1/ --n_points 50000 --k_snapshots 10
uv run python scripts/dataset_analyser.py data/datasets/lo_train-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/lo_test-1/ --n_points 50000 --k_snapshots 10
uv run python scripts/dataset_analyser.py data/datasets/br_val-2/ --n_points 50000 --k_snapshots 10
uv run python scripts/dataset_analyser.py data/datasets/ns_val-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/heat_val-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/nl_heat_val-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/lo_val-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/fhn_train-1/ --n_points 50000 --k_snapshots 10 && \
uv run python scripts/dataset_analyser.py data/datasets/fhn_test-1/ --n_points 50000 --k_snapshots 10
uv run python scripts/dataset_analyser.py data/datasets/fhn_val-1/ --n_points 50000 --k_snapshots 10
