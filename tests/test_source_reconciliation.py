"""Received bundles remain raw evidence after deterministic reconciliation."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from vehicle_emission import source_reconciliation
from vehicle_emission.source_export import export_source
from vehicle_emission.source_reconciliation import main, reconcile_source


@pytest.fixture
def raw_bundle(tmp_path: Path) -> Path:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE video_master (
                video_id INTEGER, intersection_id TEXT, video_path TEXT,
                recording_datetime_utc TEXT, duration_seconds REAL, video_name TEXT);
            INSERT INTO video_master VALUES
                (1, 'Sukiyabashi', 'data/raw_video/counting_test1/count1.mp4', '2026-05-01T00:55:00Z', 600, 'count1'),
                (2, 'Sukiyabashi', 'two.mp4', '2026-05-01T01:00:00Z', 600, 'data_20260501_0900_001'),
                (3, 'Shinjuku', 'three.mp4', '2026-06-01T09:00:00+09:00', 600, 'data_0900_001'),
                (4, 'Shinjuku', 'four.mp4', '2026-06-01 00:00:00', 600, 'data_20260601_0900_002'),
                (5, 'Shinjuku', 'five.mp4', '2026-06-01T01:00:00Z', NULL, 'unknown');
            CREATE TABLE video_emissions (
                video_id INTEGER, estimation_method TEXT, nox_grams REAL,
                nox_hybrid_adjusted_grams REAL, vehicle_count INTEGER);
            INSERT INTO video_emissions VALUES
                (1, 'MOVEStar', 1.5, 0.2, 999), (1, 'VT-CPFM', 2, 0.3, 999),
                (3, 'MOVEStar', 0, NULL, 50);
            CREATE TABLE entering_counts (
                video_id INTEGER, direction TEXT, class_id INTEGER, vehicle_count INTEGER);
            INSERT INTO entering_counts VALUES
                (1, 'left', NULL, 10), (1, 'left', 1, 7), (1, 'left', 2, 3),
                (2, 'left', NULL, 7), (2, 'right', NULL, 68), (2, 'right', 1, 69),
                (3, 'right', NULL, 0);
            CREATE TABLE crossing_events (event_id INTEGER, video_id INTEGER);
            INSERT INTO crossing_events VALUES (1, 1), (2, 3);
            CREATE TABLE air_pollution_history (
                timestamp_utc TEXT, monitoring_place TEXT, nox_ppm REAL,
                no_ppm REAL, no2_ppm REAL, co_ppm REAL, ox_ppm REAL);
            INSERT INTO air_pollution_history VALUES
                ('2026-05-01T00:00:00Z', 'recorded_label', NULL, NULL, NULL, NULL, NULL),
                ('2026-05-01T01:00:00Z', 'recorded_label', 0, NULL, NULL, NULL, NULL);
            CREATE TABLE weather_history (
                timestamp_utc TEXT, precipitation REAL, temperature REAL, wind_speed REAL,
                wind_direction TEXT, cloud_thickness REAL);
            INSERT INTO weather_history VALUES ('2026-05-01T00:00:00Z', NULL, 20, 0, 'North', NULL);
        """
        )
    output = tmp_path / "raw"
    export_source(database=database, output_directory=output)
    return output


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rehash(raw: Path, name: str) -> None:
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][name] = hashlib.sha256((raw / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_raw_reconciliation_is_read_only_deterministic_and_not_analysis_ready(
    raw_bundle: Path, tmp_path: Path
) -> None:
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_bundle.iterdir()
    }
    first, second = tmp_path / "first", tmp_path / "second"
    verification = reconcile_source(source_directory=raw_bundle, output_directory=first)
    assert main(["--source-directory", str(raw_bundle), "--output-directory", str(second)]) == 0
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_bundle.iterdir()
    }
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert set(path.name for path in first.iterdir()) == {
        "verification.json",
        "summary.json",
        "video_reconciliation.csv",
        "stored_media_hourly.csv",
    }
    assert verification["status"] == "raw_source_reconciliation_not_analysis_ready"
    assert verification["count_audit_replayed"] is True
    assert verification["exporter_matches_current_code"] is True
    assert verification["verified_table_rows"]["crossing_events"] == 2
    assert verification["verified_table_rows"]["vehicle_classification_master"] == 0
    for name, expected in verification["outputs"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == expected


def test_stored_media_union_splits_cross_hour_intervals_without_splitting_emissions(
    raw_bundle: Path, tmp_path: Path
) -> None:
    output = tmp_path / "audit"
    reconcile_source(source_directory=raw_bundle, output_directory=output)
    videos = {row["video_id"]: row for row in _csv(output / "video_reconciliation.csv")}
    assert videos["1"]["stored_interval_crosses_hour"] == "True"
    assert videos["1"]["source_record_role"] == "count_test"
    assert videos["2"]["source_record_role"] == "dated_filename"
    assert videos["3"]["source_record_role"] == "clock_only_filename"
    assert videos["1"]["stored_media_end_utc"] == "2026-05-01T01:05:00Z"
    assert videos["1"]["movestar_nox_grams"] == "1.5"
    assert videos["1"]["movestar_nox_hybrid_adjusted_grams"] == "0.2"
    assert videos["1"]["movestar_track_count"] == "999"
    assert videos["1"]["recorded_entering_count"] == "10"
    assert videos["3"]["recorded_entering_count"] == "0"
    assert videos["3"]["movestar_nox_grams"] == "0.0"
    assert videos["3"]["movestar_nox_hybrid_adjusted_grams"] == ""
    assert videos["2"]["recorded_entering_count"] == ""
    assert videos["2"]["count_audit_status"] == "unavailable_or_inconsistent"
    assert videos["4"]["recorded_entering_count"] == ""
    assert videos["4"]["stored_interval_status"] == "missing_or_invalid_aware_start"
    assert videos["5"]["stored_interval_status"] == "missing_or_invalid_positive_duration"
    hours = _csv(output / "stored_media_hourly.csv")
    suki = [row for row in hours if row["source_site"] == "Sukiyabashi"]
    assert [row["stored_media_union_seconds"] for row in suki] == ["300.0", "600.0"]
    assert [row["stored_media_overlap_seconds"] for row in suki] == ["0.0", "300.0"]
    assert not any("nox" in key or "valid_seconds" in key for key in hours[0])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["monitor_stored_labels"] == {
        "recorded_label": {"rows": 2, "missing_nox_rows": 1}
    }
    assert summary["sites"]["Shinjuku"]["stored_media_touched_hours"] == 1


@pytest.mark.parametrize("damage", ["hash", "row_count", "header", "audit", "orphan"])
def test_corrupt_or_inconsistent_bundle_never_publishes(
    raw_bundle: Path, tmp_path: Path, damage: str
) -> None:
    output = tmp_path / "audit"
    if damage == "row_count":
        manifest_path = raw_bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tables"]["video_master"]["rows"] += 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        name = "entering_count_audit.csv" if damage == "audit" else "crossing_events.csv"
        path = raw_bundle / name
        text = path.read_text()
        if damage == "header":
            text = text.replace("source_site", "wrong_header", 1)
        elif damage == "orphan":
            text = text.replace("Sukiyabashi,1,1", "Sukiyabashi,1,999")
        elif damage == "audit":
            text = text.replace("total_and_classes_agree", "direction_total_only")
        else:
            text += "tampered\n"
        path.write_text(text)
        if damage != "hash":
            _rehash(raw_bundle, name)
    with pytest.raises(ValueError):
        reconcile_source(source_directory=raw_bundle, output_directory=output)
    assert not output.exists()


def test_source_change_during_reconciliation_never_publishes(
    raw_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_csv = source_reconciliation._write_csv

    def mutate_source(*args: object) -> None:
        write_csv(*args)
        (raw_bundle / "weather_history.csv").write_text("changed\n")

    monkeypatch.setattr(source_reconciliation, "_write_csv", mutate_source)
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="changed during reconciliation"):
        reconcile_source(source_directory=raw_bundle, output_directory=output)
    assert not output.exists()


def test_existing_output_and_outputs_inside_source_are_rejected(
    raw_bundle: Path, tmp_path: Path
) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        reconcile_source(source_directory=raw_bundle, output_directory=output)
    assert sentinel.read_text() == "keep"
    with pytest.raises(ValueError, match="outside the source bundle"):
        reconcile_source(source_directory=raw_bundle, output_directory=raw_bundle / "audit")


def test_original_environment_replay_keeps_monitor_intervals_and_weather_instants_distinct(
    raw_bundle: Path, tmp_path: Path
) -> None:
    monitor, weather = tmp_path / "monitor.csv", tmp_path / "weather.csv"
    monitor.write_text(
        "Date,Time,NO(ppm),NO2(ppm),NOx(ppm),CO(ppm),Ox(ppm)\n01/05/2026,10,-,,NaN,,\n01/05/2026,11,,,0,,\n"
    )
    weather.write_text(
        "Date,Time,Tempareture,precipitation,wind speed,wind direction,cloud thinckness\n01/05/2026,09:00:00,20,,0,North,\n"
    )
    output = tmp_path / "environment"
    verification = reconcile_source(
        source_directory=raw_bundle,
        output_directory=output,
        monitor_sources={"recorded_label": monitor},
        weather_source=weather,
        hour_convention="end-labelled-hour",
    )
    summary = json.loads((output / "summary.json").read_text())
    comparison = summary["original_environment_comparison"]
    assert comparison["monitor_comparisons"]["recorded_label"]["status"] == "matches_original_rows"
    assert comparison["monitor_comparisons"]["recorded_label"]["compared_cells"] == 10
    assert comparison["weather_comparison"]["status"] == "matches_original_rows"
    assert comparison["weather_comparison"]["compared_cells"] == 5
    assert len(verification["original_environment_sources"]) == 2
    assert "weather label t+1h" in comparison["precipitation_support"]
    assert not (output / "video_seconds.csv").exists()


def test_original_environment_mismatch_is_reported_not_promoted_or_a_transfer_error(
    raw_bundle: Path, tmp_path: Path
) -> None:
    monitor = tmp_path / "monitor.csv"
    monitor.write_text(
        "Date,Time,NO(ppm),NO2(ppm),NOx(ppm),CO(ppm),Ox(ppm)\n01/05/2026,10,,,0.5,,\n"
    )
    with pytest.raises(ValueError, match="explicit hour convention"):
        reconcile_source(
            source_directory=raw_bundle,
            output_directory=tmp_path / "no_convention",
            monitor_sources={"recorded_label": monitor},
        )
    output = tmp_path / "mismatch"
    reconcile_source(
        source_directory=raw_bundle,
        output_directory=output,
        monitor_sources={"recorded_label": monitor},
        hour_convention="end-labelled-hour",
    )
    comparison = json.loads((output / "summary.json").read_text())[
        "original_environment_comparison"
    ]["monitor_comparisons"]["recorded_label"]
    assert comparison["status"] == "mismatch"
    assert comparison["extra_stored_rows"] == 1
    assert comparison["mismatched_cells_by_field"] == {"nox_ppm": 1}


@pytest.mark.parametrize("duration", ["0", "-1", "NaN"])
def test_invalid_media_duration_stays_unknown(
    raw_bundle: Path, tmp_path: Path, duration: str
) -> None:
    path = raw_bundle / "video_master.csv"
    path.write_text(path.read_text().replace("00:55:00Z,600.0", f"00:55:00Z,{duration}"))
    _rehash(raw_bundle, "video_master.csv")
    output = tmp_path / "audit"
    reconcile_source(source_directory=raw_bundle, output_directory=output)
    video = _csv(output / "video_reconciliation.csv")[0]
    assert video["stored_interval_status"] == "missing_or_invalid_positive_duration"
    assert video["stored_media_end_utc"] == ""
