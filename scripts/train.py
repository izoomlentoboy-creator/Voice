#!/usr/bin/env python3
"""Train the voice disorder detection model.

Usage:
    python scripts/train.py                     # default: binary + ensemble
    python scripts/train.py --backend logreg    # logistic regression baseline
    python scripts/train.py --augment           # with data augmentation
    python scripts/train.py --max-samples 200   # quick test run
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice_disorder_detection.pipeline import VoiceDisorderPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Train voice disorder model")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--backend", choices=["ensemble", "logreg", "cnn"], default="ensemble")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dbdir", default=None)
    parser.add_argument(
        "--extra-data", type=str, default=None,
        help="Path to a zip archive with additional audio files for training",
    )
    parser.add_argument(
        "--extra-data-label", type=int, default=None, choices=[0, 1],
        help="Default label for extra data (0=healthy, 1=pathological)",
    )
    args = parser.parse_args()

    pipeline = VoiceDisorderPipeline(
        mode=args.mode, backend=args.backend, dbdir=args.dbdir,
    )

    result = pipeline.train(
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        augment=args.augment,
        extra_data=args.extra_data,
        extra_data_label=args.extra_data_label,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
