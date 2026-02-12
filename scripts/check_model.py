#!/usr/bin/env python3
"""Check whether the voice disorder detection model is trained.

Verifies the presence and validity of all required model artifacts:
  - Model file (.joblib)
  - Scaler file (.joblib)
  - Label encoder (.joblib)
  - Training metadata (.json)

Usage:
    python scripts/check_model.py
    python scripts/check_model.py --backend logreg
    python scripts/check_model.py --mode multiclass
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_disorder_detection import config


def check_model_trained(mode: str = "binary", backend: str = "ensemble") -> dict:
    """Check if the model and all supporting artifacts exist and are loadable.

    Returns a dict with detailed status information.
    """
    results = {
        "mode": mode,
        "backend": backend,
        "trained": False,
        "artifacts": {},
        "errors": [],
    }

    # Define expected artifact paths
    artifacts = {
        "model": config.model_path(mode, backend),
        "scaler": config.scaler_path(mode, backend),
        "label_encoder": config.label_encoder_path(mode),
        "metadata": config.metadata_path(),
    }

    all_present = True
    for name, path in artifacts.items():
        exists = path.exists()
        info = {"path": str(path), "exists": exists}
        if exists:
            info["size_bytes"] = path.stat().st_size
        else:
            all_present = False
        results["artifacts"][name] = info

    # Try to load the model if the file exists
    if artifacts["model"].exists():
        try:
            import joblib
            model = joblib.load(artifacts["model"])
            results["artifacts"]["model"]["loadable"] = True
        except Exception as e:
            results["artifacts"]["model"]["loadable"] = False
            results["errors"].append(f"Model file exists but cannot be loaded: {e}")
            all_present = False

    # Check metadata contents
    if artifacts["metadata"].exists():
        try:
            with open(artifacts["metadata"]) as f:
                meta = json.load(f)
            results["training_metadata"] = meta
        except Exception as e:
            results["errors"].append(f"Metadata file exists but cannot be read: {e}")

    results["trained"] = all_present and artifacts["model"].exists()
    return results


def main():
    parser = argparse.ArgumentParser(description="Check if the model is trained")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument(
        "--backend", choices=["ensemble", "logreg", "cnn"], default="ensemble"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = check_model_trained(args.mode, args.backend)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print("=" * 55)
    print("  MODEL TRAINING STATUS CHECK")
    print("=" * 55)
    print(f"  Mode:    {result['mode']}")
    print(f"  Backend: {result['backend']}")
    print()

    if result["trained"]:
        print("  Status:  TRAINED")
    else:
        print("  Status:  NOT TRAINED")
    print()

    print("  Artifacts:")
    for name, info in result["artifacts"].items():
        mark = "+" if info["exists"] else "-"
        line = f"    [{mark}] {name}: {info['path']}"
        if info["exists"]:
            size_kb = info["size_bytes"] / 1024
            line += f" ({size_kb:.1f} KB)"
        print(line)

    if result.get("training_metadata"):
        meta = result["training_metadata"]
        print()
        print("  Training info:")
        for key in ("trained_at", "n_samples", "n_features", "backend", "mode"):
            if key in meta:
                print(f"    {key}: {meta[key]}")

    if result["errors"]:
        print()
        print("  Errors:")
        for err in result["errors"]:
            print(f"    ! {err}")

    if not result["trained"]:
        print()
        print("  To train the model run:")
        print(f"    python main.py --mode {result['mode']}"
              f" --backend {result['backend']} train")

    print()
    sys.exit(0 if result["trained"] else 1)


if __name__ == "__main__":
    main()
