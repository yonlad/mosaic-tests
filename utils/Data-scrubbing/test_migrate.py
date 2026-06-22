#!/usr/bin/env python3
"""Unit tests for migrate.py helper functions."""

import json
import tempfile
from datetime import date
from pathlib import Path

from migrate import (
    is_participant_session,
    parse_capture_date,
    extract_dedup_key,
    load_flagged_keys,
)


class TestIsParticipantSession:
    def test_valid_uuid(self):
        assert is_participant_session("31dc9a6b-04c0-4a95-8450-56db0cd90d34") is True

    def test_zero_uuid(self):
        assert is_participant_session("00000000-0000-0000-0000-000000000000") is True

    def test_resized_white_bg(self):
        assert is_participant_session("resized_white_bg") is False

    def test_resized_black_bg(self):
        assert is_participant_session("resized_black_bg") is False

    def test_flora_screenshot(self):
        assert is_participant_session("flora-screenshot") is False

    def test_empty_string(self):
        assert is_participant_session("") is False

    def test_uppercase_uuid_rejected(self):
        assert is_participant_session("31DC9A6B-04C0-4A95-8450-56DB0CD90D34") is False


class TestParseCaptureDate:
    def test_standard_filename(self):
        assert parse_capture_date("capture_20260607_071909.jpg") == date(2026, 6, 7)

    def test_new_year(self):
        assert parse_capture_date("capture_20260101_000000.jpg") == date(2026, 1, 1)

    def test_no_capture_prefix(self):
        assert parse_capture_date("random_file.jpg") is None

    def test_invalid_date_digits(self):
        assert parse_capture_date("capture_99991399_000000.jpg") is None

    def test_bg_removed_suffix(self):
        assert parse_capture_date("capture_20250628_133311_bg_removed_fit.jpg") == date(2025, 6, 28)


class TestExtractDedupKey:
    def test_standard_path(self):
        key = "selected-images/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_system_path(self):
        key = "selected-images/system-1/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_reviewed_path(self):
        key = "selected-images/reviewed-images/system-1/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_no_uuid_returns_none(self):
        key = "selected-images/resized_white_bg/capture_20250628_133311_bg_removed_fit.jpg"
        assert extract_dedup_key(key) is None


class TestLoadFlaggedKeys:
    def test_loads_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "bucket": "pistoletto.moe",
                "items": [
                    {"s3_key": "selected-images/abc/capture_1.jpg", "blend_ids": []},
                    {"s3_key": "selected-images/def/capture_2.jpg", "blend_ids": ["b1"]},
                ],
            }
            Path(tmpdir, "manifest1.json").write_text(json.dumps(manifest))
            result = load_flagged_keys(Path(tmpdir))
            assert "pistoletto.moe" in result
            assert len(result["pistoletto.moe"]) == 2
            assert "selected-images/abc/capture_1.jpg" in result["pistoletto.moe"]

    def test_multiple_buckets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "a.jpg", "blend_ids": []}]}
            m2 = {"bucket": "pistoletto.moe2", "items": [{"s3_key": "b.jpg", "blend_ids": []}]}
            Path(tmpdir, "m1.json").write_text(json.dumps(m1))
            Path(tmpdir, "m2.json").write_text(json.dumps(m2))
            result = load_flagged_keys(Path(tmpdir))
            assert len(result) == 2
            assert "a.jpg" in result["pistoletto.moe"]
            assert "b.jpg" in result["pistoletto.moe2"]

    def test_same_bucket_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "a.jpg", "blend_ids": []}]}
            m2 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "b.jpg", "blend_ids": []}]}
            Path(tmpdir, "m1.json").write_text(json.dumps(m1))
            Path(tmpdir, "m2.json").write_text(json.dumps(m2))
            result = load_flagged_keys(Path(tmpdir))
            assert len(result["pistoletto.moe"]) == 2

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_flagged_keys(Path(tmpdir)) == {}

    def test_nonexistent_dir(self):
        assert load_flagged_keys(Path("/nonexistent/path")) == {}
