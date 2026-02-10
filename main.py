#!/usr/bin/env python3
"""Voice Disorder Detection System — CLI entry point.

Usage:
    python main.py train [--mode binary|multiclass] [--max-samples N] [--no-cache]
    python main.py predict --file <audio_path>
    python main.py predict --session <session_id>
    python main.py feedback --session <session_id> --label <correct_label> [--note "..."]
    python main.py apply-feedback [--full-retrain]
    python main.py self-test [--type full|cv|quick] [--max-samples N]
    python main.py optimize [--max-samples N] [--iterations N]
    python main.py status
    python main.py db-info
"""

import argparse
import json
import logging
import sys

from voice_disorder_detection import config
from voice_disorder_detection.pipeline import VoiceDisorderPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train the model."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
        download_mode="lazy",
    )

    result = pipeline.train(
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        run_evaluation=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict voice disorder."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
    )

    if args.file:
        result = pipeline.predict_from_file(args.file)
    elif args.session is not None:
        result = pipeline.predict_from_session(args.session)
    else:
        print("Error: specify --file or --session", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    label = result["label"]
    confidence = result["confidence"]

    if args.mode == config.MODE_BINARY:
        diagnosis = "PATHOLOGICAL" if label == 1 else "HEALTHY"
    else:
        diagnosis = f"Class {label}"

    print(f"  Diagnosis:  {diagnosis}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Probabilities: {json.dumps(result['probabilities'], indent=4)}")

    if "actual_type" in result:
        actual = "PATHOLOGICAL" if result["actual_type"] == "p" else "HEALTHY"
        print(f"  Actual:     {actual}")
        if "actual_pathologies" in result:
            print(f"  Pathologies: {', '.join(result['actual_pathologies'])}")


def cmd_feedback(args: argparse.Namespace) -> None:
    """Submit a correction."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
    )

    # Get the audio from the session
    session = pipeline.loader.db.get_session(
        args.session, query_recordings=True,
    )
    if session is None:
        print(f"Error: session {args.session} not found", file=sys.stderr)
        sys.exit(1)

    from voice_disorder_detection.feature_extractor import extract_all_features
    import numpy as np

    features_list = []
    for rec in session.recordings:
        if rec.utterance not in pipeline.loader.utterances:
            continue
        rec_full = pipeline.loader.db.get_recording(rec.id, full_file_paths=True)
        if rec_full is None:
            continue
        try:
            audio = rec_full.nspdata
        except Exception:
            audio = None
        if audio is None or len(audio) == 0:
            continue
        feats = extract_all_features(audio, rec_full.rate)
        features_list.append(feats)

    if not features_list:
        print("Error: no usable recordings in this session", file=sys.stderr)
        sys.exit(1)

    combined = np.mean(features_list, axis=0)
    predicted = pipeline.model.predict(combined.reshape(1, -1))[0]

    result = pipeline.feedback.add_correction(
        features=combined,
        predicted_label=int(predicted),
        correct_label=args.label,
        session_id=args.session,
        note=args.note or "",
    )

    print("\nCorrection recorded:")
    print(f"  Predicted: {predicted}")
    print(f"  Correct:   {args.label}")
    print(f"  Stats:     {json.dumps(pipeline.feedback.get_correction_stats(), indent=2)}")


def cmd_apply_feedback(args: argparse.Namespace) -> None:
    """Apply accumulated feedback."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
    )

    result = pipeline.apply_feedback(full_retrain=args.full_retrain)
    print("\nFeedback applied:")
    print(json.dumps(result, indent=2))


def cmd_self_test(args: argparse.Namespace) -> None:
    """Run self-tests."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
    )

    result = pipeline.self_test(
        max_samples=args.max_samples,
        test_type=args.type,
    )

    print("\n" + "=" * 60)
    print(f"SELF-TEST RESULTS ({args.type})")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    # Regression check
    regression = pipeline.tester.check_regression()
    if regression.get("status") == "regression_detected":
        print("\n⚠ WARNING: Performance regression detected!")
        print(json.dumps(regression, indent=2))


def cmd_optimize(args: argparse.Namespace) -> None:
    """Optimize hyperparameters."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
    )

    result = pipeline.optimize(
        max_samples=args.max_samples,
        n_iter=args.iterations,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
        download_mode="off",
    )

    result = pipeline.status()
    print(json.dumps(result, indent=2, default=str))


def cmd_db_info(args: argparse.Namespace) -> None:
    """Show database information."""
    pipeline = VoiceDisorderPipeline(
        mode=args.mode,
        dbdir=args.dbdir,
        download_mode="off",
    )

    try:
        stats = pipeline.loader.get_database_stats()
        print("\nDatabase Information:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Could not get database info: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice Disorder Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["binary", "multiclass"],
        default="binary",
        help="Classification mode (default: binary)",
    )
    parser.add_argument(
        "--dbdir", default=None,
        help="Database directory (default: system data dir)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--max-samples", type=int, default=None)
    p_train.add_argument("--no-cache", action="store_true")

    # predict
    p_predict = subparsers.add_parser("predict", help="Predict voice disorder")
    p_predict.add_argument("--file", type=str, help="Path to audio file")
    p_predict.add_argument("--session", type=int, help="Database session ID")

    # feedback
    p_fb = subparsers.add_parser("feedback", help="Submit a correction")
    p_fb.add_argument("--session", type=int, required=True)
    p_fb.add_argument("--label", type=int, required=True)
    p_fb.add_argument("--note", type=str, default="")

    # apply-feedback
    p_afb = subparsers.add_parser("apply-feedback", help="Apply corrections")
    p_afb.add_argument("--full-retrain", action="store_true")

    # self-test
    p_st = subparsers.add_parser("self-test", help="Run self-tests")
    p_st.add_argument(
        "--type", choices=["full", "cv", "quick"], default="full",
    )
    p_st.add_argument("--max-samples", type=int, default=None)

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    p_opt.add_argument("--max-samples", type=int, default=None)
    p_opt.add_argument("--iterations", type=int, default=20)

    # status
    subparsers.add_parser("status", help="Show system status")

    # db-info
    subparsers.add_parser("db-info", help="Show database info")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "feedback": cmd_feedback,
        "apply-feedback": cmd_apply_feedback,
        "self-test": cmd_self_test,
        "optimize": cmd_optimize,
        "status": cmd_status,
        "db-info": cmd_db_info,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
