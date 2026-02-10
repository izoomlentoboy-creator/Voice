#!/usr/bin/env python3
"""Generate healthy population reference statistics for the interpreter.

Run this after training the model. It loads the dataset, filters
healthy-only samples, and saves per-feature mean/std to an NPZ file
that the interpreter uses for z-score computation.

Usage:
    python server/generate_ref_stats.py [--max-samples N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from voice_disorder_detection.data_loader import VoiceDataLoader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate healthy reference stats")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    loader = VoiceDataLoader()
    X, y, _, _, _ = loader.extract_dataset(
        mode="binary",
        max_samples=args.max_samples,
        use_cache=True,
    )

    # Filter healthy samples (label 0)
    healthy_mask = y == 0
    X_healthy = X[healthy_mask]

    print(f"Total samples: {len(X)}")
    print(f"Healthy samples: {len(X_healthy)}")
    print(f"Features: {X_healthy.shape[1]}")

    mean = X_healthy.mean(axis=0)
    std = X_healthy.std(axis=0)

    # Avoid division by zero
    std[std < 1e-10] = 1.0

    out_path = Path(project_root) / "models" / "healthy_ref_stats.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean, std=std)

    print(f"Saved to {out_path}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    main()
