from __future__ import annotations

import sqlite3

import pytest

import vehicle_emission.calculate_emissions as calculate_emissions_module
from vehicle_emission import database
from vehicle_emission.calculate_emissions import calculate_emissions_for_all
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class


def _seed_single_track_database(db_path: str) -> None:
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'single', 'single.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 36.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 99.0, 1)
        """
    )
    connection.commit()
    connection.close()


def test_calculate_emissions_workflow(tmp_path) -> None:
    db_path = str(tmp_path / "test_emissions.db")

    # 1. Initialize DB
    database.init_db(db_path)

    # 2. Insert dummy video, track, and speed profiles
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert video
    cursor.execute(
        """
        INSERT INTO video_master (video_id, video_name, video_path, intersection_id, recording_datetime_utc, recording_date, recording_time, processed_datetime_utc, fps, duration_seconds)
        VALUES (1, 'test.mp4', 'test.mp4', 'Sukiyabashi', '2026-06-03T12:00:00Z', '2026-06-03', '12:00:00', '2026-06-03T12:00:00Z', 30.0, 10.0);
        """
    )

    # Insert tracks (class_id=1 for Car, class_id=3 for Light Truck)
    cursor.execute(
        """
        INSERT INTO vehicle_tracks (video_id, track_id, frame_id, class_id, confidence, x_centre, y_centre, width, height)
        VALUES (1, 1, 0, 1, 0.9, 100, 100, 50, 50);
        """
    )
    cursor.execute(
        """
        INSERT INTO vehicle_tracks (video_id, track_id, frame_id, class_id, confidence, x_centre, y_centre, width, height)
        VALUES (1, 2, 0, 3, 0.9, 200, 200, 60, 60);
        """
    )

    # Insert speeds for track 1 and track 2
    # track 1 has 5 seconds of constant 36 km/h speed (10 m/s)
    for sec in range(5):
        cursor.execute(
            """
            INSERT INTO vehicle_speeds (video_id, track_id, second, speed_kmh)
            VALUES (1, 1, ?, 36.0);
            """,
            (sec,),
        )
    # track 2 has 5 seconds of constant 72 km/h speed (20 m/s)
    for sec in range(5):
        cursor.execute(
            """
            INSERT INTO vehicle_speeds (video_id, track_id, second, speed_kmh)
            VALUES (1, 2, ?, 72.0);
            """,
            (sec,),
        )

    conn.commit()
    conn.close()

    # 3. Run emissions calculator
    calculate_emissions_for_all(db_path)

    # 4. Verify results were populated in video_emissions table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_emissions ORDER BY estimation_method;")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2

    # Columns: video_id, estimation_method, nox_grams, co_grams, hc_grams, pm25_grams, co2_grams, fuel_grams, energy_kj, vehicle_count, average_speed_kmh, congestion_level, nox_hybrid_adjusted_grams
    # First row is MOVEStar (alphabetical order)
    row_ms = rows[0]
    assert row_ms[0] == 1
    assert row_ms[1] == "MOVEStar"
    assert row_ms[2] > 0.0  # nox_grams should be positive
    assert row_ms[9] == 2  # vehicle_count should be 2
    assert row_ms[10] > 0.0  # average_speed_kmh should be positive
    assert row_ms[11] in ["Severe", "Moderate", "Smooth"]
    assert row_ms[12] > 0.0  # nox_hybrid_adjusted_grams should be positive

    # Second row is VT-CPFM
    row_cp = rows[1]
    assert row_cp[0] == 1
    assert row_cp[1] == "VT-CPFM"
    assert row_cp[2] > 0.0  # nox_grams should be positive
    assert row_cp[9] == 2  # vehicle_count should be 2
    assert row_cp[10] > 0.0  # average_speed_kmh should be positive
    assert row_cp[11] in ["Severe", "Moderate", "Smooth"]
    assert row_cp[12] > 0.0  # nox_hybrid_adjusted_grams should be positive


def test_calculate_emissions_uses_canonical_vtcpfm_mapping(tmp_path, monkeypatch) -> None:
    """Exercise all four mappings through the production aggregation loop."""

    db_path = str(tmp_path / "mapping.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'mapping', 'mapping.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    for track_id, class_id in enumerate(range(4), start=1):
        connection.execute(
            """
            INSERT INTO vehicle_tracks (
                video_id, track_id, frame_id, class_id, confidence,
                x_centre, y_centre, width, height
            ) VALUES (1, ?, 0, ?, 0.9, 1, 1, 1, 1)
            """,
            (track_id, class_id),
        )
        connection.execute(
            "INSERT INTO vehicle_speeds VALUES (1, ?, 0, 36.0)",
            (track_id,),
        )
    connection.commit()
    connection.close()

    class StubMOVEStar:
        def estimate_class_emissions(self, class_id, speed_mps):
            emission_vehicle_type_for_project_class(class_id)
            return {"Emission Rate": [[1.0] * 8]}

    class SpyVTCPFM:
        native_types: list[int] = []

        def estimate_class_emissions(self, class_id, speed_mps):
            self.native_types.append(emission_vehicle_type_for_project_class(class_id))
            return {"Emission Rate": [[1.0] * 8]}

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", StubMOVEStar)
    monkeypatch.setattr(calculate_emissions_module, "VTCPFMModel", SpyVTCPFM)

    calculate_emissions_for_all(db_path)

    assert SpyVTCPFM.native_types == [6, 1, 13, 2]

    connection = sqlite3.connect(db_path)
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 99, 0, 99, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 99, 0, 36.0)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="Invalid project class_id: 99"):
        calculate_emissions_for_all(db_path)


def test_emissions_replace_partial_rows_only_after_success(tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "atomic_emissions.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'atomic', 'atomic.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 36.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 99.0, 1)
        """
    )
    connection.commit()
    connection.close()

    class FailingMOVEStar:
        def estimate_class_emissions(self, class_id, speed_mps):
            raise RuntimeError("model failure")

    class StubEstimator:
        def estimate_class_emissions(self, class_id, speed_mps):
            return {"Emission Rate": [[1.0] * 8]}

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", FailingMOVEStar)
    monkeypatch.setattr(calculate_emissions_module, "VTCPFMModel", StubEstimator)
    with pytest.raises(RuntimeError, match="MOVEStar failed"):
        calculate_emissions_for_all(db_path, force_recalculate=True)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall() == [("MOVEStar", 99.0)]
    connection.close()

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", StubEstimator)
    calculate_emissions_for_all(db_path)
    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions ORDER BY estimation_method"
    ).fetchall() == [("MOVEStar", 1.0), ("VT-CPFM", 1.0)]
    connection.close()


def test_emissions_reject_orphan_speed_profile_and_preserve_existing_rows(tmp_path) -> None:
    db_path = str(tmp_path / "orphan_speed.db")
    _seed_single_track_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 999, 0, 100.0)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="speed profiles have no current track"):
        calculate_emissions_for_all(db_path)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall() == [("MOVEStar", 99.0)]
    connection.close()


def test_emissions_resolves_majority_class_for_jittered_tracks(tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "majority_track_class.db")
    _seed_single_track_database(db_path)
    connection = sqlite3.connect(db_path)
    # Track 1 has 1 frame of class 1, and 2 frames of class 3 (majority = class 3 / light truck)
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 1, 3, 0.8, 2, 2, 1, 1)")
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 2, 3, 0.9, 2, 2, 1, 1)")
    connection.commit()
    connection.close()

    recorded_classes = []

    class ClassSpyEstimator:
        def estimate_class_emissions(self, class_id, speed_mps):
            recorded_classes.append(class_id)
            return {"Emission Rate": [[1.0] * 8]}

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", ClassSpyEstimator)
    monkeypatch.setattr(calculate_emissions_module, "VTCPFMModel", ClassSpyEstimator)

    calculate_emissions_for_all(db_path)
    assert 3 in recorded_classes


@pytest.mark.parametrize("invalid_speed", [-1.0, float("inf")])
def test_emissions_reject_nonfinite_or_negative_speed(tmp_path, invalid_speed: float) -> None:
    db_path = str(tmp_path / "invalid_speed.db")
    _seed_single_track_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE vehicle_speeds SET speed_kmh = ? WHERE video_id = 1 AND track_id = 1",
        (invalid_speed,),
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="finite and non-negative"):
        calculate_emissions_for_all(db_path)


@pytest.mark.parametrize("invalid_emission", [-1.0, float("inf")])
def test_invalid_emission_output_preserves_existing_rows(
    tmp_path, monkeypatch, invalid_emission: float
) -> None:
    db_path = str(tmp_path / "invalid_emission.db")
    _seed_single_track_database(db_path)

    class InvalidEstimator:
        def estimate_class_emissions(self, class_id, speed_mps):
            return {"Emission Rate": [[invalid_emission] + [1.0] * 7]}

    class ValidEstimator:
        def estimate_class_emissions(self, class_id, speed_mps):
            return {"Emission Rate": [[1.0] * 8]}

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", InvalidEstimator)
    monkeypatch.setattr(calculate_emissions_module, "VTCPFMModel", ValidEstimator)

    with pytest.raises(RuntimeError, match="MOVEStar failed"):
        calculate_emissions_for_all(db_path)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT estimation_method, nox_grams FROM video_emissions"
    ).fetchall() == [("MOVEStar", 99.0)]
    connection.close()


def test_calculate_emissions_site_filter(tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "multi_site_emissions.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    # Seed Sukiyabashi (video 1) and Shinjuku (video 2)
    connection.execute(
        """
        INSERT INTO video_master (video_id, video_name, video_path, intersection_id, recording_datetime_utc, recording_date, recording_time, processed_datetime_utc)
        VALUES (1, 'suki.mp4', 'suki.mp4', 'Sukiyabashi', '2026-06-03T12:00:00Z', '2026-06-03', '12:00:00', '2026-06-03T12:00:00Z')
        """
    )
    connection.execute(
        """
        INSERT INTO video_master (video_id, video_name, video_path, intersection_id, recording_datetime_utc, recording_date, recording_time, processed_datetime_utc)
        VALUES (2, 'shinjuku.mp4', 'shinjuku.mp4', 'Shinjuku', '2026-06-05T12:00:00Z', '2026-06-05', '12:00:00', '2026-06-05T12:00:00Z')
        """
    )
    # Tracks and speeds for both
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 100, 100, 50, 50)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 30.0)")
    connection.execute("INSERT INTO vehicle_tracks VALUES (2, 1, 0, 1, 0.9, 100, 100, 50, 50)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (2, 1, 0, 40.0)")
    # Seed existing emission for video 1
    connection.execute(
        "INSERT INTO video_emissions (video_id, estimation_method, nox_grams) VALUES (1, 'MOVEStar', 50.0)"
    )
    connection.commit()
    connection.close()

    class StubEstimator:
        def estimate_class_emissions(self, class_id, speed_mps):
            return {"Emission Rate": [[10.0] * 8]}

    monkeypatch.setattr(calculate_emissions_module, "MOVEStarModel", StubEstimator)
    monkeypatch.setattr(calculate_emissions_module, "VTCPFMModel", StubEstimator)

    # Run only for Shinjuku
    calculate_emissions_for_all(db_path, site="Shinjuku")

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT video_id, estimation_method, nox_grams FROM video_emissions ORDER BY video_id, estimation_method"
    ).fetchall()
    connection.close()

    # Video 1 Sukiyabashi record preserved, Video 2 Shinjuku inserted
    assert (1, "MOVEStar", 50.0) in rows
    assert (2, "MOVEStar", 10.0) in rows
    assert (2, "VT-CPFM", 10.0) in rows
