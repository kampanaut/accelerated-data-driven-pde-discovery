#!/usr/bin/env python3
"""
Test script for Week 1 components.

Validates that PDEOperatorNetwork and TaskDataLoader work correctly with dummy
and real data.

Usage:
    python scripts/test_week1_components.py --data-dir data/datasets/gaussian_all_types
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader


def test_network_creation():
    """Test that we can create the network with different configurations."""
    print("=" * 60)
    print("Test 1: Network Creation")
    print("=" * 60)

    # Default configuration
    print("\n1.1 Default configuration...")
    net_default = PDEOperatorNetwork()
    print(f"✓ Created: {net_default}")

    # Custom configuration
    print("\n1.2 Custom configuration (ReLU, pyramid 128→64→32)...")
    net_custom = PDEOperatorNetwork(
        hidden_dims=[128, 64, 32],
        activation='relu'
    )
    print(f"✓ Created: {net_custom}")

    print("\n✅ Network creation tests passed!\n")


def test_network_forward_pass():
    """Test network forward pass with dummy data."""
    print("=" * 60)
    print("Test 2: Network Forward Pass")
    print("=" * 60)

    net = PDEOperatorNetwork()

    # Create dummy input (batch_size=32, features=10)
    print("\n2.1 Forward pass with batch_size=32...")
    x = torch.randn(32, 10)
    y = net(x)

    assert y.shape == (32, 2), f"Expected shape (32, 2), got {y.shape}"
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")

    # Single sample
    print("\n2.2 Forward pass with single sample...")
    x_single = torch.randn(1, 10)
    y_single = net(x_single)

    assert y_single.shape == (1, 2), f"Expected shape (1, 2), got {y_single.shape}"
    print(f"✓ Input shape: {x_single.shape}")
    print(f"✓ Output shape: {y_single.shape}")

    print("\n✅ Forward pass tests passed!\n")


def test_data_loader(data_dir: str):
    """Test task data loader with real .npz files."""
    print("=" * 60)
    print("Test 3: Task Data Loader")
    print("=" * 60)

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("   Skipping data loader tests.")
        print("   Generate data first with: python scripts/generate_ns_data.py")
        return False

    # Load tasks
    print(f"\n3.1 Loading tasks from {data_dir}...")
    loader = MetaLearningDataLoader(data_path)

    if len(loader) == 0:
        print(f"\n⚠️  No tasks found in {data_dir}")
        return False

    print(f"\n✓ Loaded {len(loader)} tasks")

    # Test support/query split
    print("\n3.2 Testing support/query split...")
    task = loader.tasks[0]
    print(f"Using task: {task.task_name}")
    print(f"  Total samples: {task.n_samples:,}")

    K_shot = min(100, task.n_samples // 2)
    query_size = min(1000, task.n_samples - K_shot)

    support, query = task.get_support_query_split(
        K_shot=K_shot,
        query_size=query_size,
        seed=42
    )

    print(f"\n✓ Support set:")
    print(f"    Features shape: {support[0].shape}")
    print(f"    Targets shape: {support[1].shape}")
    print(f"✓ Query set:")
    print(f"    Features shape: {query[0].shape}")
    print(f"    Targets shape: {query[1].shape}")

    assert support[0].shape == (K_shot, 10)
    assert support[1].shape == (K_shot, 2)
    assert query[0].shape == (query_size, 10)
    assert query[1].shape == (query_size, 2)

    # Test network on real data
    print("\n3.3 Testing network on real task data...")
    net = PDEOperatorNetwork()
    features_tensor = torch.tensor(support[0], dtype=torch.float32)
    predictions = net(features_tensor)

    print(f"✓ Network prediction shape: {predictions.shape}")
    print(f"  Input range: [{features_tensor.min():.3f}, {features_tensor.max():.3f}]")
    print(f"  Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Test task batch sampling
    print("\n3.4 Testing meta-batch sampling...")
    n_tasks = min(4, len(loader))
    batch = loader.sample_batch(n_tasks=n_tasks, seed=42)

    print(f"✓ Sampled {len(batch)} tasks for meta-batch")
    for i, task in enumerate(batch):
        print(f"  Task {i+1}: {task.task_name} ({task.n_samples:,} samples)")

    # Test train/test split
    print("\n3.5 Testing train/test split...")
    if len(loader) >= 5:  # Only test if we have enough tasks
        train_tasks, test_tasks = loader.train_test_split(test_ratio=0.2, seed=42)
        print(f"✓ Split {len(loader)} tasks into:")
        print(f"    Train: {len(train_tasks)} tasks")
        print(f"    Test: {len(test_tasks)} tasks")
    else:
        print(f"  Skipped (need at least 5 tasks, have {len(loader)})")

    print("\n✅ Data loader tests passed!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Week 1 components")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/gaussian_all_types",
        help="Directory containing .npz task files"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Week 1 Component Tests")
    print("=" * 60 + "\n")

    # Run tests
    test_network_creation()
    test_network_forward_pass()
    data_loaded = test_data_loader(args.data_dir)

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("✅ Network creation: PASSED")
    print("✅ Forward pass: PASSED")
    if data_loaded:
        print("✅ Data loader: PASSED")
        print("\n🎉 All Week 1 components working!")
    else:
        print("⚠️  Data loader: SKIPPED (no data)")
        print("\nNetwork components ready. Generate data to test full pipeline.")

    print("\nNext steps:")
    if not data_loaded:
        print("  1. Generate data: python scripts/generate_ns_data.py --config configs/gaussian_all_types_test.yaml")
        print("  2. Re-run tests: python scripts/test_week1_components.py")
    else:
        print("  Week 1 complete! Ready for Week 2 (MAML implementation)")

    print()


if __name__ == "__main__":
    main()
