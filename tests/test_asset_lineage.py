from __future__ import annotations

import json
import sqlite3

import pytest

from vehicle_emission.asset_lineage import load_video_master, main, reconcile_assets


def _drive(*files: tuple[str, str, int]) -> dict:
    return {
        "schema_version": 1,
        "site": "Sukiyabashi",
        "files": [
            {"folder_date": folder_date, "name": name, "size_bytes": size}
            for folder_date, name, size in files
        ],
    }


def _database(*names: str) -> dict:
    return {
        "schema_version": 1,
        "site": "Sukiyabashi",
        "videos": [{"video_name": name} for name in names],
    }


def _features(*names: str) -> dict:
    return {
        "schema_version": 1,
        "site": "Sukiyabashi",
        "features": [{"video_name": name} for name in names],
    }


def test_repeated_raw_basename_is_ambiguous_even_with_one_database_row() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(
            ("20260501", "data_118.mp4", 182_622_162),
            ("20260503", "data_118.mp4", 441_188_400),
        ),
        database_index=_database("data_118"),
        feature_index=_features("data_118"),
    )

    assert report["counts"]["repeated_drive_names"] == 1
    assert report["counts"]["extra_drive_name_occurrences"] == 1
    assert report["counts"]["drive_names"] == 1
    assert report["counts"]["drive_names_without_database_row"] == 0
    assert report["records"][0]["raw_database_status"] == "ambiguous_name"
    assert report["records"][0]["feature_status"] == "one_row"
    assert [item["folder_date"] for item in report["records"][0]["drive_files"]] == [
        "20260501",
        "20260503",
    ]


def test_same_folder_can_hold_distinct_drive_files_with_same_name() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(
            ("20260501", "data_118.mp4", 100),
            ("20260501", "data_118.mp4", 200),
        ),
    )
    assert report["counts"]["drive_files"] == 2
    assert report["counts"]["repeated_drive_names"] == 1
    assert report["records"][0]["raw_database_status"] == "ambiguous_name"


def test_unique_match_remains_name_only_and_absent_indexes_are_not_missing() -> None:
    drive = _drive(("20260501", "data_20260501_0600_001.mp4", 100))
    without_database = reconcile_assets(site="Sukiyabashi", drive_inventory=drive)
    assert without_database["records"][0]["raw_database_status"] == "database_not_checked"
    assert without_database["records"][0]["feature_status"] == "not_checked"
    assert without_database["records"][0]["database_rows"] is None
    assert without_database["records"][0]["feature_rows"] is None
    assert without_database["counts"]["drive_names_without_database_row"] is None

    with_database = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=drive,
        database_index=_database("data_20260501_0600_001"),
        feature_index=_features(),
    )
    assert with_database["records"][0]["raw_database_status"] == "name_match_only"
    assert with_database["records"][0]["feature_status"] == "missing"
    assert with_database["records"][0]["feature_rows"] == 0
    assert with_database["comparison_level"] == "name_only"


def test_orphans_and_duplicate_features_remain_visible() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(("20260501", "data_20260501_0600_001.mp4", 100)),
        database_index=_database("data_20260501_0600_001", "db_only"),
        feature_index=_features("db_only", "feature_only", "feature_only"),
    )
    rows = {row["video_name"]: row for row in report["records"]}
    assert rows["db_only"]["raw_database_status"] == "database_only"
    assert rows["db_only"]["feature_status"] == "one_row"
    assert rows["feature_only"]["raw_database_status"] == "feature_only"
    assert rows["feature_only"]["feature_status"] == "multiple_rows"
    assert rows["feature_only"]["feature_rows"] == 2
    assert report["counts"]["raw_database_statuses"] == {
        "feature_only": 1,
        "name_match_only": 1,
        "database_only": 1,
    }
    assert report["counts"]["database_names_without_drive_file"] == 1
    assert report["counts"]["feature_names_without_database_row"] == 1
    assert report["counts"]["database_names_without_feature_row"] == 1


def test_duplicate_features_do_not_hide_missing_raw_video() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(),
        database_index=_database("db_only"),
        feature_index=_features("db_only", "db_only"),
    )
    assert report["records"][0]["raw_database_status"] == "database_only"
    assert report["records"][0]["feature_status"] == "multiple_rows"
    assert report["counts"]["raw_database_statuses"] == {"database_only": 1}
    assert report["counts"]["feature_statuses"] == {"multiple_rows": 1}
    assert report["counts"]["database_names_without_drive_file"] == 1


def test_ambiguous_raw_name_does_not_hide_missing_database_row() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(
            ("20260501", "data_118.mp4", 100),
            ("20260503", "data_118.mp4", 200),
        ),
        database_index=_database(),
    )
    assert report["records"][0]["raw_database_status"] == "ambiguous_name"
    assert report["counts"]["drive_names_without_database_row"] == 1


def test_duplicate_database_names_are_reported_without_hiding_feature_status() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(("20260501", "data_118.mp4", 100)),
        database_index=_database("data_118", "data_118"),
        feature_index=_features(),
    )
    assert report["records"][0]["raw_database_status"] == "ambiguous_name"
    assert report["records"][0]["feature_status"] == "missing"
    assert report["counts"]["database_rows"] == 2
    assert report["counts"]["database_names"] == 1


def test_filename_patterns_and_folder_date_are_diagnostics_only() -> None:
    report = reconcile_assets(
        site="Sukiyabashi",
        drive_inventory=_drive(
            ("20260501", "data_170.mp4", 100),
            ("20260501", "data_0600_001.mp4", 100),
            ("20260501", "data_20260501_0610_002.mp4", 100),
            ("20260501", "data_20260502_0620_003.mp4", 100),
        ),
    )
    assert report["counts"]["name_patterns"] == {
        "clock_only": 1,
        "date_and_clock": 2,
        "sequence_only": 1,
    }
    assert report["counts"]["date_label_mismatches"] == 1
    assert set(report["counts"]["raw_database_statuses"]) == {"database_not_checked"}


@pytest.mark.parametrize(
    "drive",
    [
        _drive(("bad-date", "video.mp4", 100)),
        _drive(("20260230", "video.mp4", 100)),
        _drive(("20260501", "../video.mp4", 100)),
        _drive(("20260501", "video.mp4", -1)),
    ],
)
def test_invalid_drive_metadata_is_rejected(drive: dict) -> None:
    with pytest.raises(ValueError):
        reconcile_assets(site="Sukiyabashi", drive_inventory=drive)


def test_wrong_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="site"):
        reconcile_assets(site="Shinjuku", drive_inventory=_drive())


def test_shinjuku_scope_is_supported() -> None:
    report = reconcile_assets(
        site="Shinjuku",
        drive_inventory={"schema_version": 1, "site": "Shinjuku", "files": []},
    )
    assert report["site"] == "Shinjuku"
    assert report["counts"]["drive_files"] == 0


def test_sqlite_source_is_read_only_and_cli_output_is_deterministic(tmp_path) -> None:
    database_path = tmp_path / "source.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE video_master (
                video_name TEXT, intersection_id TEXT, video_path TEXT,
                recording_datetime_utc TEXT, processed_datetime_utc TEXT
            )"""
        )
        connection.execute(
            """INSERT INTO video_master VALUES (
                'data_20260501_0600_001', 'Sukiyabashi',
                '/recordings/20260501/data_20260501_0600_001.mp4',
                '2026-04-30T21:00:00Z', '2026-07-01T01:00:00Z'
            )"""
        )
    original_bytes = database_path.read_bytes()
    assert load_video_master(database_path, "Sukiyabashi")["videos"] == [
        {
            "video_name": "data_20260501_0600_001",
            "video_path": "/recordings/20260501/data_20260501_0600_001.mp4",
            "recording_datetime_utc": "2026-04-30T21:00:00Z",
            "processed_datetime_utc": "2026-07-01T01:00:00Z",
        }
    ]

    drive_path = tmp_path / "drive.json"
    drive_path.write_text(
        json.dumps(_drive(("20260501", "data_20260501_0600_001.mp4", 100))),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    argv = [
        "--site",
        "Sukiyabashi",
        "--drive-inventory",
        str(drive_path),
        "--database",
        str(database_path),
        "--output",
        str(output_path),
    ]
    assert main(argv) == 0
    first = output_path.read_bytes()
    assert main(argv) == 0
    assert output_path.read_bytes() == first
    assert database_path.read_bytes() == original_bytes
    assert json.loads(first)["records"][0]["raw_database_status"] == "name_match_only"
    assert json.loads(first)["records"][0]["database_records"][0]["video_path"] == (
        "/recordings/20260501/data_20260501_0600_001.mp4"
    )

    with pytest.raises(ValueError, match="output must not replace an input"):
        main([*argv[:-1], str(database_path)])
    assert database_path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="output must not replace an input"):
        main([*argv[:-1], str(drive_path)])
    assert json.loads(drive_path.read_text(encoding="utf-8")) == _drive(
        ("20260501", "data_20260501_0600_001.mp4", 100)
    )

    output_path.write_text("different report\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
    assert output_path.read_text(encoding="utf-8") == "different report\n"
