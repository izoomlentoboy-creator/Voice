#!/usr/bin/env python3
"""Train the voice disorder detection model.

Usage:
    python scripts/train.py                     # default: binary + ensemble
    python scripts/train.py --backend logreg    # logistic regression baseline
    python scripts/train.py --augment           # with data augmentation
    python scripts/train.py --max-samples 200   # quick test run
    python scripts/train.py --synthetic         # use synthetic data (no audio download needed)
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
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no audio download needed)")
    parser.add_argument("--dbdir", default=None)
    args = parser.parse_args()

    pipeline = VoiceDisorderPipeline(
        mode=args.mode, backend=args.backend, dbdir=args.dbdir,
    )

    result = pipeline.train(
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        augment=args.augment,
        synthetic=args.synthetic,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
