from __future__ import annotations

import os
import sqlite3

import pytest

from vehicle_emission import database


class ReadableVideoCapture:
    """Small OpenCV metadata stub for registration-only tests."""

    def __init__(self, _path: str) -> None:
        self.properties = {
            database.cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            database.cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            database.cv2.CAP_PROP_FPS: 30.0,
            database.cv2.CAP_PROP_FRAME_COUNT: 300.0,
        }

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0.0)

    def release(self) -> None:
        return None


class UnreadableVideoCapture:
    def __init__(self, _path: str) -> None:
        pass

    def isOpened(self) -> bool:
        return False

    def get(self, _property_id: int) -> float:
        return 0.0

    def release(self) -> None:
        return None


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """Fixture to provide a clean temporary database file path."""
    db_file = tmp_path / "test_emissions.db"
    return str(db_file)


def test_init_db(temp_db_path) -> None:
    """Test database tables initialization and schemas."""
    database.init_db(temp_db_path)
    assert os.path.exists(temp_db_path)

    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Query all created tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}

    assert "video_master" in tables
    assert "vehicle_tracks" in tables
    assert "crossing_events" in tables
    assert "entering_counts" in tables

    conn.close()


def test_register_video_and_upsert(temp_db_path) -> None:
    """Test video registration, metadata extraction, date/time splitting, and UPSERT."""
    # Since we don't have a real video file guaranteed in test sandbox,
    # let's create a tiny dummy file to let cv2.VideoCapture try to read it.
    # It might fail properties extraction but will still insert row fallback values.
    dummy_video = os.path.join(os.path.dirname(temp_db_path), "dummy.mp4")
    with open(dummy_video, "wb") as f:
        f.write(b"")

    video_id = database.register_video(
        db_path=temp_db_path,
        video_path=dummy_video,
        intersection_id="Sukiyabashi",
        start_time_str="2026-06-27T10:30:15Z",
        duration_limit=300.0,
        notes="Sunny test run",
        group_id="June5th_6AM_8PM",
    )

    assert isinstance(video_id, int)

    # Verify registration data
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_master WHERE video_id = ?;", (video_id,))
    row = cursor.fetchone()
    assert row is not None

    # Get column names to assert values
    cursor.execute("PRAGMA table_info(video_master);")
    columns = [col[1] for col in cursor.fetchall()]
    row_dict = dict(zip(columns, row, strict=False))

    assert row_dict["video_name"] == "dummy"
    assert row_dict["intersection_id"] == "Sukiyabashi"
    assert row_dict["recording_datetime_utc"] == "2026-06-27T10:30:15Z"
    assert row_dict["recording_date"] == "2026-06-27"
    assert row_dict["recording_time"] == "10:30:15"
    assert row_dict["notes"] == "Sunny test run"
    assert row_dict["group_id"] == "June5th_6AM_8PM"

    # Test Upsert: register again with different notes and group ID
    new_video_id = database.register_video(
        db_path=temp_db_path,
        video_path=dummy_video,
        intersection_id="Sukiyabashi",
        start_time_str="2026-06-27T10:30:15Z",
        duration_limit=300.0,
        notes="Updated notes",
        group_id="June5th_6AM_8PM_v2",
    )

    assert video_id == new_video_id

    cursor.execute("SELECT notes, group_id FROM video_master WHERE video_id = ?;", (video_id,))
    notes_val, group_val = cursor.fetchone()
    assert notes_val == "Updated notes"
    assert group_val == "June5th_6AM_8PM_v2"

    conn.close()


def test_insert_tracks_and_events(temp_db_path) -> None:
    """Test bulk insertion of tracks and crossing events."""
    database.init_db(temp_db_path)

    # Insert a parent video first
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO video_master (
            video_name, video_path, intersection_id, recording_datetime_utc,
            recording_date, recording_time, processed_datetime_utc
        ) VALUES ('test_vid', 'path/test_vid.mp4', 'Shinjuku', '2026-06-27T10:00:00Z',
                  '2026-06-27', '10:00:00', '2026-06-27T15:00:00Z');
        """
    )
    conn.commit()
    cursor.execute("SELECT video_id FROM video_master WHERE video_name='test_vid';")
    video_id = cursor.fetchone()[0]

    # Test tracks insertion
    tracks = [
        {
            "vehicle_id": 1,
            "frame_id": 0,
            "class_id": 1,
            "confidence": 0.95,
            "x_centre": 100.5,
            "y_centre": 200.5,
            "width": 50.0,
            "height": 40.0,
        },
        {
            "vehicle_id": 1,
            "frame_id": 1,
            "class_id": 1,
            "confidence": 0.96,
            "x_centre": 102.0,
            "y_centre": 201.0,
            "width": 50.0,
            "height": 40.0,
        },
    ]

    database.insert_tracks(temp_db_path, video_id, tracks)

    cursor.execute("SELECT count(*) FROM vehicle_tracks WHERE video_id = ?;", (video_id,))
    assert cursor.fetchone()[0] == 2

    # Test duplicate prevention (ON CONFLICT REPLACE)
    updated_tracks = [
        {
            "vehicle_id": 1,
            "frame_id": 1,
            "class_id": 1,
            "confidence": 0.99,
            "x_centre": 102.5,
            "y_centre": 201.5,
            "width": 50.0,
            "height": 40.0,
        }
    ]
    database.insert_tracks(temp_db_path, video_id, updated_tracks)

    cursor.execute(
        "SELECT confidence, x_centre FROM vehicle_tracks WHERE video_id = ? AND frame_id = 1;",
        (video_id,),
    )
    row = cursor.fetchone()
    assert row[0] == 0.99
    assert row[1] == 102.5

    # Test crossing events insertion
    events = [
        {
            "vehicle_id": 1,
            "line_name": "exit1",
            "frame_id": 5,
            "direction": "negative_to_positive",
            "class_id": 1,
        }
    ]
    database.insert_crossing_events(temp_db_path, video_id, events)

    # Test entering counts insertion
    entering = [
        {"direction": "entering_from_left", "class_id": None, "vehicle_count": 10},
        {"direction": "entering_from_left", "class_id": 1, "vehicle_count": 8},
    ]
    database.insert_entering_counts(temp_db_path, video_id, entering)

    cursor.execute("SELECT count(*) FROM entering_counts WHERE video_id = ?;", (video_id,))
    assert cursor.fetchone()[0] == 2

    conn.close()


def test_foreign_key_constraint(temp_db_path) -> None:
    """Test that foreign key constraints prevent inserting orphans in details tables."""
    database.init_db(temp_db_path)

    # Try inserting a track with a non-existent video_id (e.g. 999)
    tracks = [
        {
            "vehicle_id": 1,
            "frame_id": 0,
            "class_id": 1,
            "confidence": 0.95,
            "x_centre": 100.5,
            "y_centre": 200.5,
            "width": 50.0,
            "height": 40.0,
        }
    ]

    with pytest.raises(sqlite3.IntegrityError):
        database.insert_tracks(temp_db_path, 999, tracks)


def test_check_video_id_exists(temp_db_path) -> None:
    """Test check_video_id_exists returns correct boolean."""
    # Initially doesn't exist
    assert not database.check_video_id_exists(temp_db_path, 1)

    # Register video to create ID 1
    dummy_video = os.path.join(os.path.dirname(temp_db_path), "dummy.mp4")
    with open(dummy_video, "wb") as f:
        f.write(b"")

    video_id = database.register_video(
        db_path=temp_db_path,
        video_path=dummy_video,
        intersection_id="Sukiyabashi",
        start_time_str="2026-06-27T10:00:00Z",
    )

    assert database.check_video_id_exists(temp_db_path, video_id)
    assert not database.check_video_id_exists(temp_db_path, video_id + 999)


def test_tracking_replacement_is_atomic_idempotent_and_invalidates_dependents(
    temp_db_path,
) -> None:
    database.init_db(temp_db_path)
    connection = sqlite3.connect(temp_db_path)
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
    connection.execute(
        """
        INSERT INTO crossing_events (
            video_id, vehicle_id, line_name, frame_id, direction, class_id
        ) VALUES (1, 1, 'old', 0, 'negative_to_positive', 1)
        """
    )
    connection.execute("INSERT INTO entering_counts VALUES (1, 'old', NULL, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 12.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 1.0, 1)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER reject_test_track
        BEFORE INSERT ON vehicle_tracks
        WHEN NEW.frame_id = 99
        BEGIN
            SELECT RAISE(ABORT, 'test failure');
        END
        """
    )
    connection.commit()
    connection.close()

    failing_track = {
        "vehicle_id": 2,
        "frame_id": 99,
        "class_id": 1,
        "confidence": 0.8,
        "x_centre": 2,
        "y_centre": 2,
        "width": 1,
        "height": 1,
    }
    with pytest.raises(sqlite3.IntegrityError, match="test failure"):
        database.replace_video_tracking_results(temp_db_path, 1, [failing_track], [], [])

    connection = sqlite3.connect(temp_db_path)
    assert connection.execute("SELECT track_id FROM vehicle_tracks").fetchall() == [(1,)]
    assert connection.execute("SELECT COUNT(*) FROM crossing_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM entering_counts").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM vehicle_speeds").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM video_emissions").fetchone()[0] == 1
    connection.execute("DROP TRIGGER reject_test_track")
    connection.commit()
    connection.close()

    replacement_track = {**failing_track, "frame_id": 1}
    replacement_event = {
        "vehicle_id": 2,
        "line_name": "exit1",
        "frame_id": 1,
        "direction": "positive_to_negative",
        "class_id": 1,
    }
    replacement_count = {
        "direction": "entering_from_left",
        "class_id": None,
        "vehicle_count": 1,
    }
    for _ in range(2):
        database.replace_video_tracking_results(
            temp_db_path,
            1,
            [replacement_track],
            [replacement_event],
            [replacement_count],
        )

    connection = sqlite3.connect(temp_db_path)
    assert connection.execute("SELECT track_id, frame_id FROM vehicle_tracks").fetchall() == [
        (2, 1)
    ]
    assert connection.execute("SELECT COUNT(*) FROM crossing_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM entering_counts").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM vehicle_speeds").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM video_emissions").fetchone()[0] == 0
    connection.close()


def test_registration_and_tracking_failure_preserves_existing_and_new_metadata(
    temp_db_path, monkeypatch
) -> None:
    monkeypatch.setattr(database.cv2, "VideoCapture", ReadableVideoCapture)
    database.init_db(temp_db_path)
    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'existing', 'old.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 12.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 1.0, 1)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER reject_atomic_track
        BEFORE INSERT ON vehicle_tracks
        WHEN NEW.frame_id = 99
        BEGIN
            SELECT RAISE(ABORT, 'atomic registration failure');
        END
        """
    )
    connection.commit()
    connection.close()

    failing_track = {
        "vehicle_id": 2,
        "frame_id": 99,
        "class_id": 1,
        "confidence": 0.8,
        "x_centre": 2,
        "y_centre": 2,
        "width": 1,
        "height": 1,
    }
    existing_path = os.path.join(os.path.dirname(temp_db_path), "existing.mp4")
    with pytest.raises(sqlite3.IntegrityError, match="atomic registration failure"):
        database.register_video_with_tracking_results(
            temp_db_path,
            existing_path,
            "Shinjuku",
            "2026-06-04T04:00:00Z",
            [failing_track],
            [],
            [],
        )

    connection = sqlite3.connect(temp_db_path)
    assert (
        connection.execute(
            """
        SELECT video_path, intersection_id, recording_datetime_utc,
               processed_datetime_utc
        FROM video_master WHERE video_id = 1
        """
        ).fetchone()
        == (
            "old.mp4",
            "Sukiyabashi",
            "2026-06-03T03:00:00Z",
            "2026-06-03T03:01:00Z",
        )
    )
    assert connection.execute("SELECT track_id FROM vehicle_tracks").fetchall() == [(1,)]
    assert connection.execute("SELECT COUNT(*) FROM vehicle_speeds").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM video_emissions").fetchone()[0] == 1
    connection.close()

    new_path = os.path.join(os.path.dirname(temp_db_path), "new.mp4")
    with pytest.raises(sqlite3.IntegrityError, match="atomic registration failure"):
        database.register_video_with_tracking_results(
            temp_db_path,
            new_path,
            "Shinjuku",
            "2026-06-04T04:00:00Z",
            [failing_track],
            [],
            [],
        )

    connection = sqlite3.connect(temp_db_path)
    assert (
        connection.execute("SELECT COUNT(*) FROM video_master WHERE video_name = 'new'").fetchone()[
            0
        ]
        == 0
    )
    connection.close()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("vehicle_id", -1),
        ("vehicle_id", float("inf")),
        ("vehicle_id", 1.5),
        ("frame_id", -1),
        ("frame_id", float("nan")),
        ("frame_id", 0.5),
        ("class_id", 1.5),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("x_centre", -1.0),
        ("y_centre", float("nan")),
        ("width", 0.0),
        ("height", float("-inf")),
    ],
)
def test_tracking_replacement_rejects_invalid_numeric_before_delete(
    temp_db_path, field: str, invalid_value: float
) -> None:
    database.init_db(temp_db_path)
    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'validated', 'validated.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 12.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 1.0, 1)
        """
    )
    connection.commit()
    connection.close()

    replacement = {
        "vehicle_id": 2,
        "frame_id": 1,
        "class_id": 1,
        "confidence": 0.8,
        "x_centre": 2.0,
        "y_centre": 2.0,
        "width": 1.0,
        "height": 1.0,
    }
    replacement[field] = invalid_value

    with pytest.raises(ValueError, match=f"track.{field}"):
        database.replace_video_tracking_results(temp_db_path, 1, [replacement], [], [])

    connection = sqlite3.connect(temp_db_path)
    assert connection.execute("SELECT track_id FROM vehicle_tracks").fetchall() == [(1,)]
    assert connection.execute("SELECT COUNT(*) FROM vehicle_speeds").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM video_emissions").fetchone()[0] == 1
    connection.close()


def test_tracking_registration_rejects_failed_metadata_read_without_writing(
    temp_db_path, monkeypatch
) -> None:
    database.init_db(temp_db_path)
    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id, width, height, resolution, fps,
            duration_seconds, recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'existing', 'old.mp4', 'Sukiyabashi', 1920, 1080, '1920x1080',
                  30.0, 10.0, '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute("INSERT INTO vehicle_speeds VALUES (1, 1, 0, 12.0)")
    connection.execute(
        """
        INSERT INTO video_emissions (
            video_id, estimation_method, nox_grams, vehicle_count
        ) VALUES (1, 'MOVEStar', 1.0, 1)
        """
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr(database.cv2, "VideoCapture", UnreadableVideoCapture)
    valid_track = {
        "vehicle_id": 2,
        "frame_id": 1,
        "class_id": 1,
        "confidence": 0.8,
        "x_centre": 2.0,
        "y_centre": 2.0,
        "width": 1.0,
        "height": 1.0,
    }
    parent = os.path.dirname(temp_db_path)
    for path in (os.path.join(parent, "existing.mp4"), os.path.join(parent, "new.mp4")):
        with pytest.raises(RuntimeError, match="Could not read complete video metadata"):
            database.register_video_with_tracking_results(
                temp_db_path,
                path,
                "Shinjuku",
                "2026-06-04T04:00:00Z",
                [valid_track],
                [],
                [],
                duration_limit=10.0,
            )

    connection = sqlite3.connect(temp_db_path)
    assert (
        connection.execute(
            """
        SELECT video_path, intersection_id, width, height, fps, duration_seconds
        FROM video_master WHERE video_id = 1
        """
        ).fetchone()
        == ("old.mp4", "Sukiyabashi", 1920, 1080, 30.0, 10.0)
    )
    assert connection.execute("SELECT track_id FROM vehicle_tracks").fetchall() == [(1,)]
    assert connection.execute("SELECT COUNT(*) FROM vehicle_speeds").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM video_emissions").fetchone()[0] == 1
    assert (
        connection.execute("SELECT COUNT(*) FROM video_master WHERE video_name = 'new'").fetchone()[
            0
        ]
        == 0
    )
    connection.close()


def test_entering_count_upsert_is_idempotent_with_null_class(temp_db_path) -> None:
    """A repeated direction-total write replaces, rather than duplicates, NULL."""

    database.init_db(temp_db_path)
    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'counts', 'counts.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.commit()
    connection.close()

    first = [{"direction": "entering_from_left", "class_id": None, "vehicle_count": 4}]
    updated = [{"direction": "entering_from_left", "class_id": None, "vehicle_count": 5}]
    database.insert_entering_counts(temp_db_path, 1, first)
    database.insert_entering_counts(temp_db_path, 1, updated)
    database.insert_entering_counts(temp_db_path, 1, updated)

    connection = sqlite3.connect(temp_db_path)
    rows = connection.execute(
        """
        SELECT class_id, vehicle_count
        FROM entering_counts
        WHERE video_id = 1 AND direction = 'entering_from_left'
        """
    ).fetchall()
    connection.close()
    assert rows == [(None, 5)]


def test_init_db_rejects_ambiguous_legacy_null_count_duplicates(temp_db_path) -> None:
    """Schema migration stops instead of choosing among duplicate NULL totals."""

    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        CREATE TABLE entering_counts (
            video_id INTEGER,
            direction TEXT NOT NULL,
            class_id INTEGER,
            vehicle_count INTEGER NOT NULL,
            PRIMARY KEY (video_id, direction, class_id)
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO entering_counts (video_id, direction, class_id, vehicle_count)
        VALUES (1, 'entering_from_left', NULL, ?)
        """,
        [(4,), (5,)],
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="Rebuild entering_counts"):
        database.init_db(temp_db_path)

    connection = sqlite3.connect(temp_db_path)
    rows = connection.execute(
        """
        SELECT class_id, vehicle_count
        FROM entering_counts
        ORDER BY rowid
        """
    ).fetchall()
    indexes = connection.execute("PRAGMA index_list(entering_counts)").fetchall()
    connection.close()

    assert rows == [(None, 4), (None, 5)]
    assert "entering_counts_logical_identity" not in {row[1] for row in indexes}


def test_entering_count_batch_validation_writes_nothing(temp_db_path) -> None:
    """An invalid repeated identity is rejected before opening a write transaction."""

    database.init_db(temp_db_path)
    duplicate_batch = [
        {"direction": "entering_from_left", "class_id": None, "vehicle_count": 4},
        {"direction": "entering_from_left", "class_id": None, "vehicle_count": 5},
    ]

    with pytest.raises(ValueError, match="Duplicate entering-count identity"):
        database.insert_entering_counts(temp_db_path, 1, duplicate_batch)

    connection = sqlite3.connect(temp_db_path)
    row_count = connection.execute("SELECT COUNT(*) FROM entering_counts").fetchone()[0]
    connection.close()
    assert row_count == 0


def test_entering_count_writer_rejects_noncanonical_legacy_sentinel(temp_db_path) -> None:
    database.init_db(temp_db_path)

    with pytest.raises(ValueError, match="Invalid project class_id: -1"):
        database.insert_entering_counts(
            temp_db_path,
            1,
            [{"direction": "entering_from_left", "class_id": -1, "vehicle_count": 1}],
        )


def test_init_db_rejects_noncanonical_legacy_class_rows(temp_db_path) -> None:
    connection = sqlite3.connect(temp_db_path)
    connection.execute(
        """
        CREATE TABLE entering_counts (
            video_id INTEGER,
            direction TEXT NOT NULL,
            class_id INTEGER,
            vehicle_count INTEGER NOT NULL,
            PRIMARY KEY (video_id, direction, class_id)
        )
        """
    )
    connection.execute("INSERT INTO entering_counts VALUES (1, 'entering_from_left', -1, 4)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="non-canonical class IDs"):
        database.init_db(temp_db_path)
