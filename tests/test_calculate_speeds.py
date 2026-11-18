from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import vehicle_emission.calculate_speeds as calculate_speeds_module
from vehicle_emission import database
from vehicle_emission.calculate_speeds import _calculate_speed_rows, calculate_speeds_for_all


def _seed_speed_database(db_path: str) -> None:
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, fps, duration_seconds
        ) VALUES (1, 'speeds', 'missing.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z', 1.0, 2.0)
        """
    )
    connection.executemany(
        "INSERT INTO vehicle_tracks VALUES (1, 1, ?, 1, 0.9, ?, 0, 1, 1)",
        [(0, 0.0), (1, 1.0)],
    )
    connection.executemany(
        "INSERT INTO vehicle_speeds VALUES (?, ?, ?, ?)",
        [(1, 1, 0, 999.0), (1, 1, 1, 999.0), (1, 99, 0, 999.0)],
    )
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 99.0, 1)
        """
    )
    connection.commit()
    connection.close()


def test_speed_refresh_replaces_complete_and_stale_rows(tmp_path) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)

    calculate_speeds_for_all(
        db_path,
        scale=1.0,
        site="Sukiyabashi",
        livestream_duration_sec=2.0,
    )

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT track_id, second, speed_kmh FROM vehicle_speeds ORDER BY track_id, second"
    ).fetchall()
    emissions = connection.execute("SELECT * FROM video_emissions").fetchall()
    connection.close()
    assert rows == [(1, 0, 0.0), (1, 1, 3.6)]
    assert emissions == []


def test_speed_failure_and_invalid_force_preserve_existing_rows(tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)

    with pytest.raises(ValueError, match="scale"):
        calculate_speeds_for_all(
            db_path,
            scale=0.0,
            site="Sukiyabashi",
            livestream_duration_sec=2.0,
            force_recalculate=True,
        )

    def fail_calculation(*args, **kwargs):
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(calculate_speeds_module, "_calculate_speed_rows", fail_calculation)
    with pytest.raises(RuntimeError, match="calculation failed"):
        calculate_speeds_for_all(
            db_path,
            scale=1.0,
            site="Sukiyabashi",
            livestream_duration_sec=2.0,
            force_recalculate=True,
        )

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT track_id, second, speed_kmh FROM vehicle_speeds ORDER BY track_id, second"
    ).fetchall()
    emissions = connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall()
    connection.close()
    assert rows == [(1, 0, 999.0), (1, 1, 999.0), (99, 0, 999.0)]
    assert emissions == [("MOVEStar", 99.0)]


def test_speed_write_failure_rolls_back_speed_and_emission_replacement(tmp_path) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TRIGGER reject_recalculated_speed
        BEFORE INSERT ON vehicle_speeds
        WHEN NEW.second = 1
        BEGIN
            SELECT RAISE(ABORT, 'speed write failure');
        END
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="speed write failure"):
        calculate_speeds_for_all(
            db_path,
            scale=1.0,
            site="Sukiyabashi",
            livestream_duration_sec=2.0,
        )

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT track_id, second, speed_kmh FROM vehicle_speeds ORDER BY track_id, second"
    ).fetchall() == [(1, 0, 999.0), (1, 1, 999.0), (99, 0, 999.0)]
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall() == [("MOVEStar", 99.0)]
    connection.close()


def test_speed_rows_reject_nonfinite_calculation() -> None:
    coordinates = pd.DataFrame(
        {
            "track_id": [1, 1],
            "frame_id": [0, 1],
            "x_centre": [0.0, float("inf")],
            "y_centre": [0.0, 0.0],
        }
    )

    with pytest.raises(RuntimeError, match="Invalid calculated speed"):
        _calculate_speed_rows(
            coordinates,
            total_frames=2,
            livestream_duration_sec=2.0,
            scale=1.0,
        )


def test_speed_refresh_requires_matching_supported_site(tmp_path) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)

    with pytest.raises(ValueError, match="site must be one of"):
        calculate_speeds_for_all(db_path, scale=1.0, site="all")
    with pytest.raises(ValueError, match="No videos found for requested site 'Shinjuku'"):
        calculate_speeds_for_all(db_path, scale=1.0, site="Shinjuku")


def test_speed_refresh_preserves_other_site_rows(tmp_path) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, fps, duration_seconds
        ) VALUES (2, 'shinjuku', 'missing.mp4', 'Shinjuku',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z', 1.0, 2.0)
        """
    )
    connection.executemany(
        "INSERT INTO vehicle_tracks VALUES (2, 2, ?, 1, 0.9, ?, 0, 1, 1)",
        [(0, 0.0), (1, 2.0)],
    )
    connection.execute("INSERT INTO vehicle_speeds VALUES (2, 2, 0, 88.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (2, 'MOVEStar', 88.0, 1)
        """
    )
    connection.commit()
    connection.close()

    calculate_speeds_for_all(
        db_path,
        scale=1.0,
        site="Sukiyabashi",
        livestream_duration_sec=2.0,
    )

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT track_id, second, speed_kmh FROM vehicle_speeds WHERE video_id = 2"
    ).fetchall() == [(2, 0, 88.0)]
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions WHERE video_id = 2"
    ).fetchall() == [("MOVEStar", 88.0)]
    connection.close()


def test_unproven_empty_video_preserves_existing_derived_rows(tmp_path) -> None:
    db_path = str(tmp_path / "speeds.db")
    _seed_speed_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("DELETE FROM vehicle_tracks WHERE video_id = 1")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="has no current track rows"):
        calculate_speeds_for_all(db_path, scale=1.0, site="Sukiyabashi")

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT track_id, second, speed_kmh FROM vehicle_speeds ORDER BY track_id, second"
    ).fetchall() == [(1, 0, 999.0), (1, 1, 999.0), (99, 0, 999.0)]
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall() == [("MOVEStar", 99.0)]
    connection.close()


def test_calculate_speeds_month_filter(tmp_path) -> None:
    db_path = str(tmp_path / "speeds_month.db")
    _seed_speed_database(db_path)
    connection = sqlite3.connect(db_path)
    # Update video 1 to May (2026-05-15)
    connection.execute(
        "UPDATE video_master SET recording_datetime_utc = '2026-05-15T03:00:00Z' WHERE video_id = 1"
    )
    # Insert video 2 in June (2026-06-05) with pre-existing speed
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, fps, duration_seconds
        ) VALUES (2, 'sukiyabashi_june', 'missing.mp4', 'Sukiyabashi',
                  '2026-06-05T03:00:00Z', '2026-06-05', '03:00:00',
                  '2026-06-05T03:01:00Z', 1.0, 2.0)
        """
    )
    connection.executemany(
        "INSERT INTO vehicle_tracks VALUES (2, 2, ?, 1, 0.9, ?, 0, 1, 1)",
        [(0, 0.0), (1, 2.0)],
    )
    connection.execute("INSERT INTO vehicle_speeds VALUES (2, 2, 0, 77.0)")
    connection.commit()
    connection.close()

    # Run for month '05' (May) only; video 2 (June) speed of 77.0 km/h must remain untouched
    calculate_speeds_for_all(
        db_path,
        scale=1.0,
        site="Sukiyabashi",
        livestream_duration_sec=2.0,
        month="05",
    )

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT speed_kmh FROM vehicle_speeds WHERE video_id = 2"
    ).fetchall() == [(77.0,)]
    # Video 1 (May) was recalculated
    video1_speeds = connection.execute(
        "SELECT speed_kmh FROM vehicle_speeds WHERE video_id = 1"
    ).fetchall()
    assert len(video1_speeds) > 0
    assert video1_speeds[0][0] != 999.0
    connection.close()


def test_calculate_speeds_date_filter(tmp_path) -> None:
    db_path = str(tmp_path / "speeds_date.db")
    _seed_speed_database(db_path)
    connection = sqlite3.connect(db_path)
    # Update video 1 to 2026-06-03
    connection.execute(
        "UPDATE video_master SET recording_datetime_utc = '2026-06-03T03:00:00Z', recording_date = '2026-06-03' WHERE video_id = 1"
    )
    # Insert video 2 on 2026-06-04 with pre-existing speed
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, fps, duration_seconds
        ) VALUES (2, 'sukiyabashi_day2', 'missing.mp4', 'Sukiyabashi',
                  '2026-06-04T03:00:00Z', '2026-06-04', '03:00:00',
                  '2026-06-04T03:01:00Z', 1.0, 2.0)
        """
    )
    connection.executemany(
        "INSERT INTO vehicle_tracks VALUES (2, 2, ?, 1, 0.9, ?, 0, 1, 1)",
        [(0, 0.0), (1, 2.0)],
    )
    connection.execute("INSERT INTO vehicle_speeds VALUES (2, 2, 0, 88.0)")
    connection.commit()
    connection.close()

    # Run for date '2026-06-03' only; video 2 (2026-06-04) speed of 88.0 km/h must remain untouched
    calculate_speeds_for_all(
        db_path,
        scale=1.0,
        site="Sukiyabashi",
        livestream_duration_sec=2.0,
        date="2026-06-03",
    )

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT speed_kmh FROM vehicle_speeds WHERE video_id = 2"
    ).fetchall() == [(88.0,)]
    # Video 1 (2026-06-03) was recalculated
    video1_speeds = connection.execute(
        "SELECT speed_kmh FROM vehicle_speeds WHERE video_id = 1"
    ).fetchall()
    assert len(video1_speeds) > 0
    assert video1_speeds[0][0] != 999.0
    connection.close()
