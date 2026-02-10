#!/usr/bin/env python3
"""Voice Disorder Detection System — CLI entry point.

Usage:
    python main.py train [--backend ensemble|logreg] [--augment] [--max-samples N]
    python main.py predict --file <audio_path>
    python main.py predict --session <session_id>
    python main.py compare-baselines [--max-samples N]
    python main.py feedback --session <session_id> --label <correct_label>
    python main.py apply-feedback [--full-retrain]
    python main.py self-test [--type full|cv|quick|subgroups] [--max-samples N]
    python main.py optimize [--max-samples N] [--iterations N]
    python main.py explain [--max-samples N]
    python main.py calibrate [--method isotonic|sigmoid] [--max-samples N]
    python main.py fit-monitor [--max-samples N]
    python main.py check-drift [--max-samples N]
    python main.py report [--max-samples N] [--output-dir DIR]
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


def cmd_train(args):
    pipeline = VoiceDisorderPipeline(
        mode=args.mode, backend=args.backend, dbdir=args.dbdir,
    )
    result = pipeline.train(
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        augment=args.augment,
    )
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


def cmd_predict(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
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
    diagnosis = "PATHOLOGICAL" if label == 1 else "HEALTHY" if args.mode == config.MODE_BINARY else f"Class {label}"

    print(f"  Diagnosis:  {diagnosis}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Abstain:    {result['abstain']}")
    if result.get("abstain_reason"):
        print(f"  Reason:     {result['abstain_reason']}")
    print(f"  Probabilities: {json.dumps(result['probabilities'], indent=4)}")
    if "actual_type" in result:
        actual = "PATHOLOGICAL" if result["actual_type"] == "p" else "HEALTHY"
        print(f"  Actual:     {actual}")
    print("\n  NOTE: This is a screening tool, not a medical diagnosis.")


def cmd_compare(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, dbdir=args.dbdir)
    result = pipeline.compare_baselines(max_samples=args.max_samples)
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON (patient-level CV)")
    print("=" * 60)
    for backend, metrics in result.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"\n  [{backend}]")
            print(f"    Accuracy:    {metrics['accuracy']:.4f} (+/- {metrics.get('accuracy_std', 0):.4f})")
            print(f"    F1:          {metrics['f1']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
            print(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")
            print(f"    PR-AUC:      {metrics['pr_auc']:.4f}")
            print(f"    Brier:       {metrics['brier']:.4f}")


def cmd_feedback(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    session = pipeline.loader.db.get_session(args.session, query_recordings=True)
    if session is None:
        print(f"Error: session {args.session} not found", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    from voice_disorder_detection.feature_extractor import extract_all_features

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
    pipeline.feedback.add_correction(
        features=combined, predicted_label=int(predicted),
        correct_label=args.label, session_id=args.session, note=args.note or "",
    )
    print(f"\nCorrection recorded: predicted={predicted}, correct={args.label}")
    print(json.dumps(pipeline.feedback.get_correction_stats(), indent=2))


def cmd_apply_feedback(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.apply_feedback(full_retrain=args.full_retrain)
    print(json.dumps(result, indent=2))


def cmd_self_test(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.self_test(max_samples=args.max_samples, test_type=args.type)
    print("\n" + "=" * 60)
    print(f"SELF-TEST RESULTS ({args.type})")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    regression = pipeline.tester.check_regression()
    if regression.get("status") == "regression_detected":
        print("\nWARNING: Performance regression detected!")
        print(json.dumps(regression, indent=2))


def cmd_optimize(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.optimize(max_samples=args.max_samples, n_iter=args.iterations)
    print(json.dumps(result, indent=2, default=str))


def cmd_explain(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.explain(max_samples=args.max_samples)
    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE")
    print("=" * 60)
    print(json.dumps(result, indent=2))


def cmd_calibrate(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.calibrate(max_samples=args.max_samples, method=args.method)
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Method:           {result['method']}")
    print(f"  ECE before:       {result['ece_before']:.4f}")
    print(f"  ECE after:        {result['ece_after']:.4f}")
    print(f"  ECE improvement:  {result['ece_improvement']:.4f}")
    print(f"  MCE before:       {result['mce_before']:.4f}")
    print(f"  MCE after:        {result['mce_after']:.4f}")
    thresholds = result.get("threshold_optimization", {})
    if thresholds:
        print("\n  Optimal Thresholds:")
        for criterion, vals in thresholds.items():
            if isinstance(vals, dict) and "threshold" in vals:
                print(f"    {criterion}: threshold={vals['threshold']:.3f}"
                      f"  sens={vals.get('sensitivity', 0):.4f}"
                      f"  spec={vals.get('specificity', 0):.4f}")


def cmd_fit_monitor(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.fit_domain_monitor(max_samples=args.max_samples)
    print("\n" + "=" * 60)
    print("DOMAIN MONITOR FITTED")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


def cmd_check_drift(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.check_drift(max_samples=args.max_samples)
    print("\n" + "=" * 60)
    print("DRIFT CHECK RESULTS")
    print("=" * 60)
    print(f"  Drift detected: {result.get('drift_detected', 'n/a')}")
    print(f"  Summary: {result.get('summary', 'n/a')}")
    if result.get("per_feature"):
        print(f"\n  Drifted features ({len(result['per_feature'])}):")
        for f in result["per_feature"][:10]:
            print(f"    feature[{f['feature_index']}]: KS p={f['ks_p_value']:.6f}, PSI={f['psi']:.4f}")


def cmd_report(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir)
    result = pipeline.generate_report(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
    print("\n" + "=" * 60)
    print("EVALUATION REPORT GENERATED")
    print("=" * 60)
    files = result.get("_files", {})
    print(f"  JSON:     {files.get('json', 'n/a')}")
    print(f"  Markdown: {files.get('markdown', 'n/a')}")
    ev = result.get("evaluation", {})
    print(f"\n  Hold-out accuracy:     {ev.get('accuracy', 'n/a')}")
    print(f"  Hold-out sensitivity:  {ev.get('sensitivity', 'n/a')}")
    print(f"  Hold-out specificity:  {ev.get('specificity', 'n/a')}")
    cv = result.get("cross_validation", {})
    print(f"  CV accuracy (mean):    {cv.get('accuracy_mean', 'n/a')}")
    cal = result.get("calibration", {})
    if cal:
        print(f"  ECE before/after:      {cal.get('ece_before', '?')} -> {cal.get('ece_after', '?')}")


def cmd_status(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, backend=args.backend, dbdir=args.dbdir, download_mode="off")
    print(json.dumps(pipeline.status(), indent=2, default=str))


def cmd_db_info(args):
    pipeline = VoiceDisorderPipeline(mode=args.mode, dbdir=args.dbdir, download_mode="off")
    try:
        print(json.dumps(pipeline.loader.get_database_stats(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Voice Disorder Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--backend", choices=["ensemble", "logreg", "cnn"], default="ensemble")
    parser.add_argument("--dbdir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation")

    p = sub.add_parser("predict")
    p.add_argument("--file", type=str)
    p.add_argument("--session", type=int)

    p = sub.add_parser("compare-baselines")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("feedback")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--label", type=int, required=True)
    p.add_argument("--note", type=str, default="")

    p = sub.add_parser("apply-feedback")
    p.add_argument("--full-retrain", action="store_true")

    p = sub.add_parser("self-test")
    p.add_argument("--type", choices=["full", "cv", "quick", "subgroups"], default="full")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("optimize")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--iterations", type=int, default=20)

    p = sub.add_parser("explain")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("calibrate")
    p.add_argument("--method", choices=["isotonic", "sigmoid"], default="isotonic")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("fit-monitor")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("check-drift")
    p.add_argument("--max-samples", type=int)

    p = sub.add_parser("report")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--output-dir", type=str, default=None)

    sub.add_parser("status")
    sub.add_parser("db-info")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "train": cmd_train, "predict": cmd_predict,
        "compare-baselines": cmd_compare, "feedback": cmd_feedback,
        "apply-feedback": cmd_apply_feedback, "self-test": cmd_self_test,
        "optimize": cmd_optimize, "explain": cmd_explain,
        "calibrate": cmd_calibrate, "fit-monitor": cmd_fit_monitor,
        "check-drift": cmd_check_drift, "report": cmd_report,
        "status": cmd_status, "db-info": cmd_db_info,
    }
    cmds.get(args.command, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
