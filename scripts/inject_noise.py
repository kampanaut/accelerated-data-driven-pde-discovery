"""
Inject Gaussian noise into task .npz files for robustness experiments.

This script:
1. Iterates through all task .npz files in a directory
2. Adds Gaussian noise scaled by array standard deviation
3. Saves noisy versions to separate *_noisy.npz files

Usage:
    python scripts/inject_noise.py --data-dir data/datasets/meta_train
    python scripts/inject_noise.py --data-dir data/datasets/meta_train --noise-levels 0.01 0.05 0.10 --seed 42
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List


# Arrays to add noise to (all field arrays in the .npz)
NOISY_ARRAYS = [
    'u', 'v',
    'u_x', 'u_y', 'u_xx', 'u_yy',
    'v_x', 'v_y', 'v_xx', 'v_yy',
    'u_t', 'v_t'
]


def add_gaussian_noise(
    array: np.ndarray,
    noise_level: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Add Gaussian noise scaled by array standard deviation.

    Args:
        array: Input array to add noise to
        noise_level: Fraction of std to use as noise sigma (e.g., 0.01 = 1%)
        rng: Random state for reproducibility

    Returns:
        Noisy array with same shape and dtype
    """
    sigma = noise_level * np.std(array)
    noise = rng.normal(0, sigma, array.shape)
    return (array + noise).astype(array.dtype)


def check_metadata_status(npz_path: Path) -> tuple:
    """
    Check metadata .txt file for task status.

    The metadata file is authoritative — generate_ns_data.py validates
    data quality before marking status as SUCCESS.

    Args:
        npz_path: Path to .npz file

    Returns:
        (is_success, error_msg): Tuple of success status and error message if failed
    """
    txt_path = npz_path.with_suffix('.txt')

    if not txt_path.exists():
        return True, None  # No metadata file, assume valid

    with open(txt_path, 'r') as f:
        content = f.read()

    # Check for FAILED status
    if 'Status: FAILED' in content:
        # Extract error message if present
        for line in content.split('\n'):
            if line.startswith('Error:'):
                return False, line.strip()
        return False, "Status: FAILED"

    return True, None


def inject_noise_into_task(
    npz_path: Path,
    noise_levels: List[float],
    seed: int
) -> Path:
    """
    Create task_noisy.npz with noisy versions at multiple noise levels.

    Output keys: 'noise_0.01/u', 'noise_0.01/v', etc.

    Args:
        npz_path: Path to clean task.npz file
        noise_levels: List of noise levels (e.g., [0.01, 0.05, 0.10])
        seed: Random seed for reproducibility

    Returns:
        Path to created noisy file, or None if data is invalid

    Raises:
        ValueError: If data contains NaN/Inf (diverged simulation)
    """
    # Check metadata file for status (authoritative source)
    is_valid, reason = check_metadata_status(npz_path)
    if not is_valid:
        raise ValueError(f"Task marked as failed: {reason}")

    data = np.load(npz_path, allow_pickle=True)

    rng = np.random.RandomState(seed)

    noisy_arrays = {}

    for level in noise_levels:
        level_key = f"noise_{level:.2f}"

        for array_name in NOISY_ARRAYS:
            if array_name not in data:
                continue

            clean_array = data[array_name]
            noisy_array = add_gaussian_noise(clean_array, level, rng)
            noisy_arrays[f"{level_key}/{array_name}"] = noisy_array

    # Save to task_noisy.npz
    output_path = npz_path.parent / f"{npz_path.stem}_noisy.npz"
    np.savez_compressed(output_path, **noisy_arrays)

    return output_path


def process_directory(
    data_dir: Path,
    noise_levels: List[float],
    seed: int,
    overwrite: bool = False
) -> int:
    """
    Process all task .npz files in directory.

    Args:
        data_dir: Directory containing task .npz files
        noise_levels: List of noise levels to apply
        seed: Random seed (incremented per file for variety)
        overwrite: If True, overwrite existing noisy files

    Returns:
        Number of files processed
    """
    data_dir = Path(data_dir)

    # Find all .npz files that aren't already noisy files
    npz_files = [
        f for f in sorted(data_dir.glob("*.npz"))
        if not f.stem.endswith("_noisy")
    ]

    if not npz_files:
        print(f"No task .npz files found in {data_dir}")
        return 0

    print(f"Found {len(npz_files)} task files in {data_dir}")
    print(f"Noise levels: {noise_levels}")
    print(f"Base seed: {seed}")
    print()

    processed = 0
    skipped_invalid = 0

    for i, npz_path in enumerate(npz_files):
        output_path = npz_path.parent / f"{npz_path.stem}_noisy.npz"

        if output_path.exists() and not overwrite:
            print(f"  Skipping {npz_path.name} (noisy file exists)")
            continue

        # Use different seed for each file for variety
        file_seed = seed + i

        try:
            inject_noise_into_task(npz_path, noise_levels, file_seed)
            print(f"  Created {output_path.name}")
            processed += 1
        except ValueError as e:
            print(f"  Skipping {npz_path.name}: {e}")
            skipped_invalid += 1

    print()
    print(f"Processed {processed} files")
    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} invalid/diverged files")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Inject Gaussian noise into task .npz files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing task .npz files"
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Noise levels as fraction of std (default: 0.01 0.05 0.10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing noisy files"
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return 1

    process_directory(
        args.data_dir,
        args.noise_levels,
        args.seed,
        args.overwrite
    )
    return 0


if __name__ == "__main__":
    exit(main())
