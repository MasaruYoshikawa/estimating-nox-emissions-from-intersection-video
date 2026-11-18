"""The source-host command copies recorded facts, not inferred valid coverage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from vehicle_emission import source_export
from vehicle_emission.source_export import _count_audit, export_source, main


@pytest.fixture
def source_db(tmp_path: Path) -> Path:
    database = tmp_path / "results.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE video_master (
                video_id INTEGER, intersection_id TEXT, video_path TEXT,
                recording_datetime_utc TEXT, duration_seconds REAL, fps REAL);
            INSERT INTO video_master VALUES
                (1, 'Sukiyabashi', '20260501/data_0900.mp4', '2026-05-01T00:00:00Z', 600, 30),
                (2, 'Shinjuku', '20260601/data_0900.mp4', '2026-06-01 00:00:00', NULL, 30),
                (3, 'Other', 'other.mp4', '2026-06-01T00:00:00Z', 600, 30);
            CREATE TABLE video_emissions (
                video_id INTEGER, estimation_method TEXT, nox_grams REAL,
                nox_hybrid_adjusted_grams REAL, vehicle_count INTEGER);
            INSERT INTO video_emissions VALUES
                (1, 'MOVEStar', 1.25, 1.1, 50), (2, 'MOVEStar', 2, 1.7, 60),
                (3, 'MOVEStar', 3, 2.6, 70);
            CREATE TABLE entering_counts (
                video_id INTEGER, direction TEXT, class_id INTEGER, vehicle_count INTEGER);
            INSERT INTO entering_counts VALUES
                (1, 'left', NULL, 10), (1, 'left', 1, 7), (1, 'left', 2, 3),
                (2, 'right', 1, 8), (2, 'right', 2, 2), (3, 'left', NULL, 99);
            CREATE TABLE crossing_events (
                event_id INTEGER, video_id INTEGER, vehicle_id INTEGER,
                line_name TEXT, direction TEXT, frame_id INTEGER);
            INSERT INTO crossing_events VALUES
                (1, 1, 7, 'exit1', 'positive_to_negative', 100),
                (2, 1, 7, 'exit2', 'negative_to_positive', 200),
                (3, 2, 8, 'exit1', 'positive_to_negative', 300);
            CREATE TABLE air_pollution_history (
                timestamp_utc TEXT, monitoring_place TEXT, nox_ppm REAL);
            INSERT INTO air_pollution_history VALUES
                ('2026-05-01T00:00:00Z', 'Kasumigaseki', 0.01),
                ('2026-06-01T00:00:00Z', 'Shinjuku', NULL);
            CREATE TABLE weather_history (timestamp_utc TEXT, precipitation REAL);
            INSERT INTO weather_history VALUES ('2026-05-01T00:00:00Z', NULL);
            """
        )
    return database


def _csv(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_two_site_raw_export_preserves_timing_missingness_and_emission_definitions(
    source_db: Path, tmp_path: Path
) -> None:
    output = tmp_path / "source"
    manifest = export_source(database=source_db, output_directory=output)
    videos = _csv(output, "video_master")
    assert [row["source_site"] for row in videos] == ["Sukiyabashi", "Shinjuku"]
    assert videos[0]["recording_datetime_utc"] == "2026-05-01T00:00:00Z"
    assert videos[1]["recording_datetime_utc"] == "2026-06-01 00:00:00"
    assert videos[1]["duration_seconds"] == ""
    emissions = _csv(output, "video_emissions")
    assert emissions[0]["nox_grams"] == "1.25"
    assert emissions[0]["nox_hybrid_adjusted_grams"] == "1.1"
    assert emissions[0]["vehicle_count"] == "50"
    assert _csv(output, "air_pollution_history")[1]["nox_ppm"] == ""
    assert _csv(output, "weather_history")[0]["precipitation"] == ""
    assert len(_csv(output, "crossing_events")) == 3
    assert manifest["status"] == "raw_source_export_not_analysis_ready"
    assert manifest["tables"]["vehicle_classification_master"]["available"] is False
    assert manifest["database"]["content_hash"] is None
    assert manifest["tables"]["video_master"]["parameters"] == ["Shinjuku", "Sukiyabashi"]
    assert "intersection_id IN (?,?)" in manifest["tables"]["video_master"]["query"]
    assert len(manifest["exporter_sha256"]) == 64
    assert not (output / "video_seconds.csv").exists()
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_count_totals_and_classes_are_never_added_together(source_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "source"
    export_source(database=source_db, output_directory=output)
    audit = _csv(output, "entering_count_audit")
    assert [row["recorded_entering_count"] for row in audit] == ["10", "10"]
    assert [row["count_status"] for row in audit] == ["class_sum_only", "total_and_classes_agree"]


@pytest.mark.parametrize(
    ("classes", "values", "status"),
    [
        ([None, None], [10, 10], "duplicate_logical_key"),
        ([None, 1], [10, 8], "direction_total_class_mismatch"),
        ([4], [10], "noncanonical_class"),
        ([1], [-1], "invalid_count"),
        ([1], [0.5], "invalid_count"),
        ([1], [None], "invalid_count"),
    ],
)
def test_ambiguous_count_rows_remain_unavailable(
    classes: list[int | None], values: list[float | None], status: str
) -> None:
    rows = [
        {
            "source_site": "Sukiyabashi",
            "video_id": 1,
            "direction": "left",
            "class_id": class_id,
            "vehicle_count": count,
        }
        for class_id, count in zip(classes, values, strict=True)
    ]
    audit = _count_audit([("Sukiyabashi", 1)], rows)
    assert audit[0]["count_status"] == status
    assert audit[0]["recorded_entering_count"] is None


def test_missing_counts_are_not_zero_and_actual_zero_is_preserved() -> None:
    audit = _count_audit([("Sukiyabashi", 1)], [])
    assert audit[0]["count_status"] == "not_recorded"
    assert audit[0]["recorded_entering_count"] is None
    rows = [
        {
            "source_site": "Shinjuku",
            "video_id": 2,
            "direction": "left",
            "class_id": None,
            "vehicle_count": 0,
        }
    ]
    assert _count_audit([("Shinjuku", 2)], rows)[0]["recorded_entering_count"] == 0


def test_read_only_replay_is_identical_and_site_filter_is_explicit(
    source_db: Path, tmp_path: Path
) -> None:
    before = source_db.read_bytes()
    before_mtime = source_db.stat().st_mtime_ns
    first, second = tmp_path / "first", tmp_path / "second"
    for destination in (first, second):
        assert (
            main(
                [
                    "--database",
                    str(source_db),
                    "--output-directory",
                    str(destination),
                    "--sites",
                    "Shinjuku",
                ]
            )
            == 0
        )
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert source_db.read_bytes() == before
    assert source_db.stat().st_mtime_ns == before_mtime
    assert not list(source_db.parent.glob("results.db-*"))
    assert len(_csv(first, "video_master")) == 1
    assert _csv(first, "video_master")[0]["source_site"] == "Shinjuku"


def test_existing_output_and_source_sidecars_are_preserved(source_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "source"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_source(database=source_db, output_directory=output)
    assert sentinel.read_text(encoding="utf-8") == "existing"
    sidecar = Path(str(source_db) + "-wal")
    sidecar.write_bytes(b"existing source sidecar")
    with pytest.raises(ValueError, match="closed source snapshot"):
        export_source(database=source_db, output_directory=tmp_path / "fresh")
    assert sidecar.read_bytes() == b"existing source sidecar"
    assert not (tmp_path / "fresh").exists()


def test_schema_error_does_not_publish_partial_bundle(source_db: Path, tmp_path: Path) -> None:
    with sqlite3.connect(source_db) as connection:
        connection.execute("ALTER TABLE video_emissions RENAME COLUMN estimation_method TO unknown")
    with pytest.raises(ValueError, match="lacks required fields"):
        export_source(database=source_db, output_directory=tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_manifest_matches_published_json(source_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "source"
    manifest = export_source(database=source_db, output_directory=output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_closed_wal_mode_database_remains_unchanged(source_db: Path, tmp_path: Path) -> None:
    with sqlite3.connect(source_db) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    connection.close()
    before = source_db.read_bytes()
    manifest = export_source(database=source_db, output_directory=tmp_path / "source")
    assert source_db.read_bytes() == before
    sidecars = manifest["database"]["reader_sidecars_after"]
    assert sidecars.get("-wal", 0) == 0
    if "-shm" in sidecars:
        assert Path(str(source_db) + "-shm").stat().st_size == sidecars["-shm"]
    export_source(database=source_db, output_directory=tmp_path / "replay")


def test_sqlite_connection_rejects_writes(
    source_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_connect = sqlite3.connect

    def checked_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        connection = original_connect(*args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM video_master")
        connection.rollback()
        return connection

    monkeypatch.setattr(sqlite3, "connect", checked_connect)
    export_source(database=source_db, output_directory=tmp_path / "source")


def test_detected_source_change_prevents_publication(
    source_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_hash = source_export._sha256
    before = source_db.stat()

    def changed_source(path: Path) -> str:
        os.utime(source_db, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        return original_hash(path)

    monkeypatch.setattr(source_export, "_sha256", changed_source)
    with pytest.raises(ValueError, match="changed during export"):
        export_source(database=source_db, output_directory=tmp_path / "source")
    assert not (tmp_path / "source").exists()
