from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from vehicle_emission import database
from vehicle_emission.inference.correlation import load_aligned_time_series
from vehicle_emission.load_history_csvs import load_air_pollution, load_weather
from vehicle_emission.time_utils import (
    compact_local_filename_timestamp_utc,
    normalize_utc_timestamp,
    parse_aware_timestamp,
    soramame_hour_interval_utc,
)


def test_soramame_hour_label_maps_to_half_open_utc_interval() -> None:
    assert soramame_hour_interval_utc("1/05/2026", 1, convention="end-labelled-hour") == (
        "2026-04-30T15:00:00Z",
        "2026-04-30T16:00:00Z",
    )
    assert soramame_hour_interval_utc("1/05/2026", 24, convention="end-labelled-hour") == (
        "2026-05-01T14:00:00Z",
        "2026-05-01T15:00:00Z",
    )
    assert soramame_hour_interval_utc("1/05/2026", 1, convention="centre-labelled-hour") == (
        "2026-04-30T15:30:00Z",
        "2026-04-30T16:30:00Z",
    )


def test_local_video_filename_requires_explicit_timezone() -> None:
    assert (
        compact_local_filename_timestamp_utc("20260524", "0005", timezone_name="Asia/Tokyo")
        == "2026-05-23T15:05:00Z"
    )
    assert (
        compact_local_filename_timestamp_utc("20260524", "0005", timezone_name="Pacific/Auckland")
        == "2026-05-23T12:05:00Z"
    )


def test_history_loaders_persist_true_utc_across_date_boundary(tmp_path) -> None:
    db_path = str(tmp_path / "history.db")
    weather_csv = tmp_path / "weather.csv"
    pollution_csv = tmp_path / "pollution.csv"
    weather_csv.write_text(
        "Date,Time,Tempareture,precipitation,wind speed,wind direction,cloud thinckness\n"
        "1/05/2026,01:00:00,11,0.5,2,North-northwest,\n",
        encoding="utf-8",
    )
    pollution_csv.write_text(
        "Date,Time,NO(ppm),NO2(ppm),NOx(ppm),CO(ppm),Ox(ppm)\n"
        "1/05/2026,1,0.001,0.004,0.005,0.1,0.04\n",
        encoding="utf-8",
    )
    database.init_db(db_path)

    load_weather(str(weather_csv), db_path)
    load_air_pollution(
        str(pollution_csv),
        "Sukiyabashi",
        db_path,
        hour_convention="end-labelled-hour",
    )

    connection = sqlite3.connect(db_path)
    weather_timestamp = connection.execute("SELECT timestamp_utc FROM weather_history").fetchone()[
        0
    ]
    monitor_timestamp = connection.execute(
        "SELECT timestamp_utc FROM air_pollution_history"
    ).fetchone()[0]
    connection.close()
    assert weather_timestamp == "2026-04-30T16:00:00Z"
    assert monitor_timestamp == "2026-04-30T15:00:00Z"
    assert parse_aware_timestamp(weather_timestamp).utcoffset().total_seconds() == 0
    assert parse_aware_timestamp(monitor_timestamp).utcoffset().total_seconds() == 0


def test_register_video_normalises_aware_time_and_rejects_naive(tmp_path) -> None:
    db_path = str(tmp_path / "video.db")
    video_path = tmp_path / "fixture.mp4"
    video_path.write_bytes(b"")

    video_id = database.register_video(
        db_path,
        str(video_path),
        "Sukiyabashi",
        "2026-05-01T00:00:00+09:00",
    )
    connection = sqlite3.connect(db_path)
    stored = connection.execute(
        "SELECT recording_datetime_utc FROM video_master WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    connection.close()
    assert stored == "2026-04-30T15:00:00Z"

    with pytest.raises(ValueError, match="explicit timezone offset"):
        normalize_utc_timestamp("2026-05-01T00:00:00")


def test_history_reload_is_atomic_when_a_source_row_is_invalid(tmp_path) -> None:
    db_path = str(tmp_path / "atomic.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO air_pollution_history VALUES (?, ?, NULL, NULL, ?, NULL, NULL)",
        ("2026-04-30T15:00:00Z", "Sukiyabashi", 0.005),
    )
    connection.commit()
    connection.close()

    invalid_csv = tmp_path / "invalid-pollution.csv"
    invalid_csv.write_text(
        "Date,Time,NO(ppm),NO2(ppm),NOx(ppm),CO(ppm),Ox(ppm)\n"
        "1/05/2026,1,0.001,0.004,0.006,0.1,0.04\n"
        "bad-date,2,0.001,0.004,0.007,0.1,0.04\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid air-pollution row 3"):
        load_air_pollution(
            str(invalid_csv),
            "Sukiyabashi",
            db_path,
            hour_convention="end-labelled-hour",
        )

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT timestamp_utc, nox_ppm FROM air_pollution_history WHERE monitoring_place = ?",
        ("Sukiyabashi",),
    ).fetchall()
    connection.close()
    assert rows == [("2026-04-30T15:00:00Z", 0.005)]


def test_inference_uses_persisted_utc_without_heuristic_shift(tmp_path) -> None:
    db_path = str(tmp_path / "alignment.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, group_id
        ) VALUES (1, 'aligned', 'aligned.mp4', 'Sukiyabashi',
                  '2026-04-30T15:00:00Z', '2026-04-30', '15:00:00',
                  '2026-04-30T15:01:00Z', 'utc-fixture')
        """
    )
    for place in ("Sukiyabashi", "Setagaya", "Shinjuku"):
        connection.execute(
            "INSERT INTO air_pollution_history VALUES (?, ?, NULL, NULL, 0.01, NULL, NULL)",
            ("2026-04-30T15:00:00Z", place),
        )
    connection.execute(
        "INSERT INTO weather_history VALUES (?, NULL, 0, 1, 'east', NULL, 1)",
        ("2026-04-30T15:00:00Z",),
    )
    connection.commit()
    connection.close()
    database.insert_entering_counts(
        db_path,
        1,
        [{"direction": "entering_from_left", "class_id": None, "vehicle_count": 1}],
    )

    aligned = load_aligned_time_series(db_path, "utc-fixture", resample_10min=False)

    assert len(aligned) == 1
    assert aligned.loc[0, "timestamp"] == pd.Timestamp("2026-04-30T15:00:00Z")


def test_inference_rejects_noncanonical_persisted_local_offsets(tmp_path) -> None:
    db_path = str(tmp_path / "local-offset.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, group_id
        ) VALUES (1, 'local-offset', 'local-offset.mp4', 'Sukiyabashi',
                  '2026-05-01T00:00:00+09:00', '2026-05-01', '00:00:00',
                  '2026-04-30T15:01:00Z', 'local-offset-fixture')
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="canonical UTC"):
        load_aligned_time_series(db_path, "local-offset-fixture", resample_10min=False)
