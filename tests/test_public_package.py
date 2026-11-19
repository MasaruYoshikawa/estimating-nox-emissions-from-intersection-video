"""Publication inputs distinguish recorded geometry from legacy examples."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_emission.calculate_speeds import _effective_scale_for_video


def test_recorded_scalar_parameters_match_width_adjustment() -> None:
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "data/calibration/recorded_scalar_calibration.json").read_text())
    assert record["method"] == "single_chord_pixel_to_metre_scale"
    expected = {
        "Sukiyabashi": (762.77, 36.18, 21.0828, 2560, 1440),
        "Shinjuku": (528.47, 42.29, 12.4963, 1440, 900),
    }
    for site, (pixels, metres, scale, width, height) in expected.items():
        item = record["sites"][site]
        assert item["chord_length_pixels"] == pixels
        assert item["chord_length_metres"] == metres
        assert item["recorded_pixels_per_metre"] == scale
        assert item["reference_width_pixels"] == width
        assert item["reference_height_pixels"] == height
        assert "image_points" not in item
        assert "world_points_metres" not in item
        for video_width in (1440, 2560):
            assert _effective_scale_for_video(scale, video_width, height) == pytest.approx(
                scale * video_width / width
            )


def test_legacy_examples_are_not_published_as_measurements() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "data/calibration/site_calibration_v2.json": "site_calibration_v2.json",
        "data/manifests/speed_validation_v2.csv": "speed_validation_v2.csv",
        "data/manifests/two_site_coverage_v2.csv": "two_site_coverage_v2.csv",
    }
    for old_path, name in paths.items():
        assert not (root / old_path).exists()
        assert (root / "tests/fixtures/legacy" / name).is_file()
    assert not (root / "src/vehicle_emission/inference/run_v2_analysis.py").exists()
    assert (root / "archive/legacy/run_v2_analysis.py").is_file()
