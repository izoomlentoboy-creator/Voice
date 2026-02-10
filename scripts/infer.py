#!/usr/bin/env python3
"""Run inference on an audio file.

Usage:
    python scripts/infer.py path/to/audio.wav
    python scripts/infer.py --session 42        # from sbvoicedb session
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice_disorder_detection import config
from voice_disorder_detection.pipeline import VoiceDisorderPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Voice disorder inference")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV/FLAC/MP3)")
    parser.add_argument("--session", type=int, help="sbvoicedb session ID")
    parser.add_argument("--mode", default="binary")
    parser.add_argument("--backend", default="ensemble")
    parser.add_argument("--dbdir", default=None)
    args = parser.parse_args()

    if not args.audio_file and args.session is None:
        parser.error("Provide an audio file path or --session ID")

    pipeline = VoiceDisorderPipeline(
        mode=args.mode, backend=args.backend, dbdir=args.dbdir,
    )

    if args.audio_file:
        result = pipeline.predict_from_file(args.audio_file)
    else:
        result = pipeline.predict_from_session(args.session)

    label = result["label"]
    if args.mode == config.MODE_BINARY:
        diagnosis = "PATHOLOGICAL" if label == 1 else "HEALTHY"
    else:
        diagnosis = f"Class {label}"

    print(f"Diagnosis:  {diagnosis}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Abstain:    {result['abstain']}")
    if result.get("abstain_reason"):
        print(f"Reason:     {result['abstain_reason']}")
    print(f"Details:    {json.dumps(result['probabilities'])}")
    print("\nNOTE: This is a screening tool, not a medical diagnosis.")


if __name__ == "__main__":
    main()
