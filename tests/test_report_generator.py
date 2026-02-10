"""Tests for reproducible report generation."""

import json

import numpy as np

from voice_disorder_detection.model import VoiceDisorderModel
from voice_disorder_detection.report_generator import (
    _config_snapshot,
    generate_report,
)


def _make_data(n=100, d=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, 2, n)
    speaker_ids = list(range(n))  # one speaker per sample for simplicity
    metadata = [
        {"gender": rng.choice(["m", "w"]), "age": int(rng.randint(20, 80))}
        for _ in range(n)
    ]
    return X, y, speaker_ids, metadata


class TestConfigSnapshot:
    def test_has_key_fields(self):
        snap = _config_snapshot()
        assert "sample_rate" in snap
        assert "n_mfcc" in snap
        assert "abstain_threshold" in snap
        assert "cv_folds" in snap
        assert "utterances" in snap


class TestReportGeneration:
    def test_report_creates_files(self, tmp_path):
        X, y, speaker_ids, metadata = _make_data()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)

        report = generate_report(
            model, X, y,
            speaker_ids=speaker_ids,
            metadata=metadata,
            output_dir=tmp_path,
        )

        # Check files created
        assert "_files" in report
        json_path = report["_files"]["json"]
        md_path = report["_files"]["markdown"]
        assert json_path.endswith(".json")
        assert md_path.endswith(".md")

        # Verify JSON is valid
        with open(json_path) as f:
            loaded = json.load(f)
        assert "meta" in loaded
        assert "evaluation" in loaded

    def test_report_has_all_sections(self, tmp_path):
        X, y, speaker_ids, metadata = _make_data()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)

        report = generate_report(
            model, X, y,
            speaker_ids=speaker_ids,
            metadata=metadata,
            output_dir=tmp_path,
        )

        assert "meta" in report
        assert "config" in report
        assert "dataset" in report
        assert "evaluation" in report
        assert "cross_validation" in report
        assert "subgroup_analysis" in report

    def test_report_meta_fields(self, tmp_path):
        X, y, speaker_ids, _ = _make_data()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)

        report = generate_report(model, X, y, speaker_ids=speaker_ids, output_dir=tmp_path)
        meta = report["meta"]
        assert "timestamp" in meta
        assert "git_revision" in meta
        assert "mode" in meta
        assert "backend" in meta

    def test_dataset_section(self, tmp_path):
        X, y, speaker_ids, _ = _make_data(n=80)
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)

        report = generate_report(model, X, y, speaker_ids=speaker_ids, output_dir=tmp_path)
        ds = report["dataset"]
        assert ds["n_samples"] == 80
        assert ds["n_features"] == 50
        assert ds["n_unique_speakers"] == 80


class TestMarkdownRendering:
    def test_renders_valid_markdown(self, tmp_path):
        X, y, speaker_ids, metadata = _make_data()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)

        report = generate_report(
            model, X, y,
            speaker_ids=speaker_ids,
            metadata=metadata,
            output_dir=tmp_path,
        )

        md_path = report["_files"]["markdown"]
        with open(md_path) as f:
            md = f.read()

        assert "# Voice Disorder Detection" in md
        assert "## Dataset" in md
        assert "## Hold-out Evaluation" in md
        assert "## Cross-Validation" in md
        assert "| Metric | Value |" in md
