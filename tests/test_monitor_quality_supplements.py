from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_emission.inference.monitor_quality_supplements import (
    SOURCE_NAMES,
    _quality_pool,
    _verified_source,
    correct_monitor_series,
    run_supplements,
)


@pytest.fixture
def monitoring() -> tuple[pd.DataFrame, pd.DataFrame]:
    ambient = pd.DataFrame(
        {
            "monitoring_place": ["Sukiyabashi", "Shinjuku", "Setagaya", "Setagaya"],
            "timestamp_utc": ["2026-05-01T00:00:00Z"] * 3 + ["2026-05-01T01:00:00Z"],
            "nox_ppm": [0.049, 0, 0, 0.007],
            "no2_ppm": [0.040, 0, 0, 0.006],
        }
    )
    quality = pd.DataFrame(
        {
            "monitoring_place": ["Sukiyabashi", "Shinjuku", "Setagaya", "Setagaya"],
            "monitor_start_utc": ["2026-05-01T00:00:00Z"] * 3 + ["2026-05-01T01:00:00Z"],
            "provider_station_id": ["13101520", "13104540", "0110", "0110"],
            "provider_value": ["-", "0", "欠測", "7"],
            "nox_ppm": [np.nan, 0, np.nan, 0.007],
            "quality_status": ["missing", "valid", "uncollected", "valid"],
        }
    )
    return ambient, quality


def test_provider_flags_correct_nonzero_missing_and_retain_real_zero(monitoring) -> None:
    ambient, quality = monitoring
    original = ambient.copy(deep=True)
    result = correct_monitor_series(ambient, quality)
    assert pd.isna(result.loc[0, "nox_ppm"])
    assert result.loc[1, "nox_ppm"] == 0
    assert pd.isna(result.loc[2, "nox_ppm"])
    assert result.loc[3, "nox_ppm"] == 0.007
    pd.testing.assert_frame_equal(ambient, original)
    pd.testing.assert_series_equal(result.no2_ppm, original.no2_ppm)


def test_unreviewed_station_or_hour_is_not_recoded_as_missing(monitoring) -> None:
    ambient, quality = monitoring
    result = correct_monitor_series(ambient, quality.iloc[[0]])
    assert result.loc[1, "nox_ppm"] == 0
    assert result.loc[2, "nox_ppm"] == 0
    assert result.loc[3, "nox_ppm"] == 0.007
    assert result.quality_status.isna().sum() == 3


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "unknown_status",
        "missing_numeric",
        "valid_nan",
        "wrong_value",
        "wrong_time",
        "missing_key",
    ],
)
def test_bad_quality_evidence_is_rejected(monitoring, kind) -> None:
    ambient, quality = monitoring
    if kind == "duplicate":
        quality = pd.concat([quality, quality.iloc[:1]], ignore_index=True)
    elif kind == "unknown_status":
        quality.loc[0, "quality_status"] = "probably_down"
    elif kind == "missing_numeric":
        quality.loc[0, "nox_ppm"] = 0
    elif kind == "valid_nan":
        quality.loc[1, "nox_ppm"] = np.nan
    elif kind == "wrong_value":
        quality.loc[1, "nox_ppm"] = 0.005
    elif kind == "wrong_time":
        quality.loc[0, "monitor_start_utc"] = "2026-05-01T00:30:00Z"
    else:
        quality.loc[0, "monitoring_place"] = np.nan
    with pytest.raises(ValueError):
        correct_monitor_series(ambient, quality)


def test_duplicate_source_observation_is_rejected(monitoring) -> None:
    ambient, quality = monitoring
    with pytest.raises(ValueError, match="duplicate source"):
        correct_monitor_series(pd.concat([ambient, ambient.iloc[:1]], ignore_index=True), quality)


def test_all_weather_pool_excludes_only_its_receptor_flags(monitoring) -> None:
    _, quality = monitoring
    rows = pd.DataFrame(
        {
            "site": ["Sukiyabashi", "Shinjuku", "Sukiyabashi", "Shinjuku"],
            "monitor_start_utc": ["2026-05-01T00:00:00Z"] * 2 + ["2026-05-01T01:00:00Z"] * 2,
            "ambient_nox_ppm": [0.049, 0, 0.005, 0.006],
        }
    )
    retained, excluded = _quality_pool(rows, quality)
    assert len(retained) == 3
    assert retained.iloc[0].site == "Shinjuku"
    assert retained.iloc[0].ambient_nox_ppm == 0
    assert excluded.site.tolist() == ["Sukiyabashi"]
    assert excluded.quality_status.tolist() == ["missing"]


def test_only_used_sources_are_verified_against_export_manifest(tmp_path: Path) -> None:
    for name in SOURCE_NAMES:
        (tmp_path / f"{name}.csv").write_text("id,value\n1,2\n")
    outputs = {
        f"{name}.csv": hashlib.sha256((tmp_path / f"{name}.csv").read_bytes()).hexdigest()
        for name in SOURCE_NAMES
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"outputs": outputs}))
    paths = _verified_source(tmp_path)
    assert set(paths) == {*SOURCE_NAMES, "source_manifest"}
    (tmp_path / "weather_history.csv").write_text("id,value\n1,999\n")
    with pytest.raises(ValueError, match="weather_history.csv"):
        _verified_source(tmp_path)


def test_existing_output_is_not_modified(tmp_path: Path) -> None:
    output = tmp_path / "preserved"
    output.mkdir()
    marker = output / "unchanged.csv"
    marker.write_text("value\n42\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_supplements(
            panel=tmp_path / "panel.csv",
            quality=tmp_path / "quality.csv",
            source_directory=tmp_path / "source",
            all_weather_panel=tmp_path / "all.csv",
            output_directory=output,
        )
    assert marker.read_text() == "value\n42\n"


def test_output_cannot_be_nested_in_input_bundle(tmp_path: Path) -> None:
    inputs = tmp_path / "published"
    inputs.mkdir()
    with pytest.raises(ValueError, match="separate"):
        run_supplements(
            panel=inputs / "panel.csv",
            quality=inputs / "quality.csv",
            source_directory=tmp_path / "source",
            all_weather_panel=tmp_path / "all" / "all.csv",
            output_directory=inputs / "new",
        )
