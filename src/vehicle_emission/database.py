from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from typing import Any

import cv2

from vehicle_emission.time_utils import normalize_utc_timestamp, utc_date_and_time

VALID_PROJECT_CLASS_IDS = frozenset({0, 1, 2, 3})
SQLITE_INTEGER_MAX = (1 << 63) - 1


def _validated_integer(value: object, field: str, *, minimum: int = 0) -> int:
    """Return an exact, SQLite-safe integer that satisfies the field domain."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be a finite integer greater than or equal to {minimum}"
        ) from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < minimum
        or numeric > SQLITE_INTEGER_MAX
    ):
        raise ValueError(f"{field} must be a finite integer greater than or equal to {minimum}")
    return int(numeric)


def _validated_real(
    value: object,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    """Return a finite real number within a field's physical domain."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite real number") from exc
    below_minimum = numeric < minimum if minimum_inclusive else numeric <= minimum
    if not math.isfinite(numeric) or below_minimum or (maximum is not None and numeric > maximum):
        qualifier = "at least" if minimum_inclusive else "greater than"
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be finite, {qualifier} {minimum}{upper}")
    return numeric


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def assert_entering_counts_integrity(conn: sqlite3.Connection) -> None:
    """Reject ambiguous or non-canonical count rows without modifying them."""

    invalid_rows = conn.execute(
        """
        SELECT video_id, direction, class_id
        FROM entering_counts
        WHERE class_id IS NOT NULL AND class_id NOT IN (0, 1, 2, 3)
        ORDER BY video_id, direction, class_id
        LIMIT 5
        """
    ).fetchall()
    if invalid_rows:
        examples = "; ".join(
            f"video_id={video_id}, direction={direction!r}, class_id={class_id!r}"
            for video_id, direction, class_id in invalid_rows
        )
        raise RuntimeError(
            "entering_counts contains non-canonical class IDs. Rebuild from verified "
            f"crossing events or apply a reviewed migration. Examples: {examples}"
        )

    duplicate_logical_keys = conn.execute(
        """
        SELECT video_id, direction, class_id, COUNT(*) AS duplicate_count
        FROM entering_counts
        GROUP BY video_id, direction, class_id
        HAVING COUNT(*) > 1
        ORDER BY video_id, direction, class_id
        LIMIT 5
        """
    ).fetchall()
    if duplicate_logical_keys:
        examples = "; ".join(
            f"video_id={video_id}, direction={direction!r}, "
            f"class_id={class_id!r}, rows={duplicate_count}"
            for video_id, direction, class_id, duplicate_count in duplicate_logical_keys
        )
        raise RuntimeError(
            "Legacy entering_counts contains duplicate logical keys. Rebuild entering_counts "
            "from verified crossing events or apply a reviewed migration; no duplicate row "
            f"was selected or deleted. Examples: {examples}"
        )


def init_db(db_path: str) -> None:
    """Initialize the SQLite database and create the tables if they do not exist."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = _connect(db_path)
    cursor = conn.cursor()

    # 1. vehicle_classification_master table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_classification_master (
            class_id INTEGER PRIMARY KEY,
            class_name TEXT NOT NULL UNIQUE
        );
        """
    )
    cursor.execute(
        """
        INSERT OR IGNORE INTO vehicle_classification_master (class_id, class_name) VALUES
        (0, 'bus'),
        (1, 'car'),
        (2, 'heavy truck'),
        (3, 'light truck');
        """
    )

    # 2. video_master table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_master (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL UNIQUE,
            video_path TEXT NOT NULL,
            intersection_id TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            resolution TEXT,
            fps REAL,
            duration_seconds REAL,
            recording_datetime_utc TEXT NOT NULL,
            recording_date TEXT NOT NULL,
            recording_time TEXT NOT NULL,
            processed_datetime_utc TEXT NOT NULL,
            notes TEXT,
            group_id TEXT
        );
        """
    )

    # Check if group_id column exists, if not, perform database migration on-the-fly
    cursor.execute("PRAGMA table_info(video_master);")
    columns = [col[1] for col in cursor.fetchall()]
    if "group_id" not in columns:
        cursor.execute("ALTER TABLE video_master ADD COLUMN group_id TEXT;")

    # 2. vehicle_tracks table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_tracks (
            video_id INTEGER,
            track_id INTEGER,
            frame_id INTEGER,
            class_id INTEGER,
            confidence REAL,
            x_centre REAL,
            y_centre REAL,
            width REAL,
            height REAL,
            PRIMARY KEY (video_id, track_id, frame_id),
            FOREIGN KEY (video_id) REFERENCES video_master(video_id) ON DELETE CASCADE
        );
        """
    )

    # 3. crossing_events table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS crossing_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            vehicle_id INTEGER,
            line_name TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            direction TEXT NOT NULL,
            class_id INTEGER,
            FOREIGN KEY (video_id) REFERENCES video_master(video_id) ON DELETE CASCADE
        );
        """
    )

    # 4. entering_counts table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS entering_counts (
            video_id INTEGER,
            direction TEXT NOT NULL,
            class_id INTEGER,
            vehicle_count INTEGER NOT NULL,
            PRIMARY KEY (video_id, direction, class_id),
            FOREIGN KEY (video_id) REFERENCES video_master(video_id) ON DELETE CASCADE
        );
        """
    )

    try:
        assert_entering_counts_integrity(conn)
    except RuntimeError:
        conn.rollback()
        conn.close()
        raise
    cursor.execute("DROP INDEX IF EXISTS entering_counts_logical_identity;")
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS entering_counts_direction_total_identity
        ON entering_counts (video_id, direction)
        WHERE class_id IS NULL;
        """
    )
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS entering_counts_class_identity
        ON entering_counts (video_id, direction, class_id)
        WHERE class_id IS NOT NULL;
        """
    )

    # 5. vehicle_speeds table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_speeds (
            video_id INTEGER,
            track_id INTEGER,
            second INTEGER,
            speed_kmh REAL NOT NULL,
            PRIMARY KEY (video_id, track_id, second),
            FOREIGN KEY (video_id) REFERENCES video_master(video_id) ON DELETE CASCADE
        );
        """
    )

    # 6. air_pollution_history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS air_pollution_history (
            timestamp_utc TEXT NOT NULL,
            monitoring_place TEXT NOT NULL,
            no_ppm REAL,
            no2_ppm REAL,
            nox_ppm REAL,
            co_ppm REAL,
            ox_ppm REAL,
            PRIMARY KEY (timestamp_utc, monitoring_place)
        );
        """
    )

    # 7. weather_history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_history (
            timestamp_utc TEXT PRIMARY KEY,
            temperature REAL,
            precipitation REAL,
            wind_speed REAL,
            wind_direction TEXT,
            cloud_thickness REAL,
            along_road_wind_speed REAL
        );
        """
    )

    # 8. video_emissions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS video_emissions (
            video_id INTEGER NOT NULL,
            estimation_method TEXT NOT NULL,
            nox_grams REAL,
            co_grams REAL,
            hc_grams REAL,
            pm25_grams REAL,
            co2_grams REAL,
            fuel_grams REAL,
            energy_kj REAL,
            vehicle_count INTEGER,
            average_speed_kmh REAL,
            congestion_level TEXT,
            nox_hybrid_adjusted_grams REAL,
            PRIMARY KEY (video_id, estimation_method),
            FOREIGN KEY (video_id) REFERENCES video_master(video_id) ON DELETE CASCADE
        );
        """
    )

    # Migration for new columns if they do not exist
    cursor.execute("PRAGMA table_info(video_emissions);")
    columns = [col[1] for col in cursor.fetchall()]
    if "average_speed_kmh" not in columns:
        cursor.execute("ALTER TABLE video_emissions ADD COLUMN average_speed_kmh REAL;")
    if "congestion_level" not in columns:
        cursor.execute("ALTER TABLE video_emissions ADD COLUMN congestion_level TEXT;")
    if "nox_hybrid_adjusted_grams" not in columns:
        cursor.execute("ALTER TABLE video_emissions ADD COLUMN nox_hybrid_adjusted_grams REAL;")

    cursor.execute("PRAGMA table_info(weather_history);")
    weather_columns = [col[1] for col in cursor.fetchall()]
    if "along_road_wind_speed" not in weather_columns:
        cursor.execute("ALTER TABLE weather_history ADD COLUMN along_road_wind_speed REAL;")

    conn.commit()
    conn.close()


def _video_registration_values(
    video_path: str,
    intersection_id: str,
    start_time_str: str,
    duration_limit: float | None = None,
    notes: str | None = None,
    group_id: str | None = None,
    *,
    require_complete_metadata: bool = False,
) -> dict[str, Any]:
    """Build validated video metadata without mutating the database."""

    norm_path = os.path.normpath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if duration_limit is not None:
        duration_limit = _validated_real(
            duration_limit,
            "duration_limit",
            minimum=0.0,
            minimum_inclusive=False,
        )

    cap = cv2.VideoCapture(video_path)
    try:
        opened = cap.isOpened()
        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else None
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else None
        raw_fps = cap.get(cv2.CAP_PROP_FPS) if opened else None
        raw_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if opened else None
    finally:
        cap.release()

    width = (
        int(raw_width)
        if raw_width is not None and math.isfinite(raw_width) and raw_width >= 1
        else None
    )
    height = (
        int(raw_height)
        if raw_height is not None and math.isfinite(raw_height) and raw_height >= 1
        else None
    )
    fps = float(raw_fps) if raw_fps is not None and math.isfinite(raw_fps) and raw_fps > 0 else None
    frame_count = (
        int(raw_frame_count)
        if raw_frame_count is not None and math.isfinite(raw_frame_count) and raw_frame_count >= 1
        else None
    )

    if require_complete_metadata and (
        not opened
        or width is None
        or height is None
        or fps is None
        or (frame_count is None and duration_limit is None)
    ):
        raise RuntimeError(
            f"Could not read complete video metadata for tracking registration: {norm_path}"
        )

    resolution = f"{width}x{height}" if (width and height) else None

    duration_seconds = None
    if frame_count and fps and fps > 0:
        duration_seconds = frame_count / fps
    if duration_limit is not None and (
        duration_seconds is None or duration_seconds > duration_limit
    ):
        duration_seconds = duration_limit

    recording_datetime_utc = normalize_utc_timestamp(start_time_str)
    recording_date_value, recording_time_value = utc_date_and_time(recording_datetime_utc)
    recording_date = recording_date_value.isoformat()
    recording_time = recording_time_value.isoformat(timespec="seconds")
    processed_datetime_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    return {
        "video_name": video_name,
        "video_path": norm_path,
        "intersection_id": intersection_id,
        "width": width,
        "height": height,
        "resolution": resolution,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "recording_datetime_utc": recording_datetime_utc,
        "recording_date": recording_date,
        "recording_time": recording_time,
        "processed_datetime_utc": processed_datetime_utc,
        "notes": notes,
        "group_id": group_id,
    }


def _upsert_video(
    conn: sqlite3.Connection,
    registration: dict[str, Any],
    *,
    video_id: int | None = None,
) -> int:
    """Upsert video metadata on an existing transaction and return its ID."""

    values = (
        registration["video_name"],
        registration["video_path"],
        registration["intersection_id"],
        registration["width"],
        registration["height"],
        registration["resolution"],
        registration["fps"],
        registration["duration_seconds"],
        registration["recording_datetime_utc"],
        registration["recording_date"],
        registration["recording_time"],
        registration["processed_datetime_utc"],
        registration["notes"],
        registration["group_id"],
    )

    if video_id is not None:
        conn.execute(
            """
            INSERT INTO video_master (
                video_id, video_name, video_path, intersection_id, width, height, resolution,
                fps, duration_seconds, recording_datetime_utc, recording_date,
                recording_time, processed_datetime_utc, notes, group_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_name) DO UPDATE SET
                video_id=excluded.video_id,
                video_path=excluded.video_path,
                intersection_id=excluded.intersection_id,
                width=excluded.width,
                height=excluded.height,
                resolution=excluded.resolution,
                fps=excluded.fps,
                duration_seconds=excluded.duration_seconds,
                recording_datetime_utc=excluded.recording_datetime_utc,
                recording_date=excluded.recording_date,
                recording_time=excluded.recording_time,
                processed_datetime_utc=excluded.processed_datetime_utc,
                notes=excluded.notes,
                group_id=excluded.group_id;
            """,
            (int(video_id), *values),
        )
    else:
        conn.execute(
            """
            INSERT INTO video_master (
                video_name, video_path, intersection_id, width, height, resolution,
                fps, duration_seconds, recording_datetime_utc, recording_date,
                recording_time, processed_datetime_utc, notes, group_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_name) DO UPDATE SET
                video_path=excluded.video_path,
                intersection_id=excluded.intersection_id,
                width=excluded.width,
                height=excluded.height,
                resolution=excluded.resolution,
                fps=excluded.fps,
                duration_seconds=excluded.duration_seconds,
                recording_datetime_utc=excluded.recording_datetime_utc,
                recording_date=excluded.recording_date,
                recording_time=excluded.recording_time,
                processed_datetime_utc=excluded.processed_datetime_utc,
                notes=excluded.notes,
                group_id=excluded.group_id;
            """,
            values,
        )

    row = conn.execute(
        "SELECT video_id FROM video_master WHERE video_name = ?;",
        (registration["video_name"],),
    ).fetchone()
    resolved_video_id = int(row[0]) if row else None
    if resolved_video_id is None:
        raise RuntimeError(
            f"Failed to register or retrieve video_id for {registration['video_name']}"
        )
    return resolved_video_id


def register_video(
    db_path: str,
    video_path: str,
    intersection_id: str,
    start_time_str: str,
    duration_limit: float | None = None,
    notes: str | None = None,
    video_id: int | None = None,
    group_id: str | None = None,
) -> int:
    """Register video metadata and return the auto-generated or requested ID."""

    init_db(db_path)
    registration = _video_registration_values(
        video_path,
        intersection_id,
        start_time_str,
        duration_limit,
        notes,
        group_id,
    )
    with closing(_connect(db_path)) as conn, conn:
        resolved_video_id = _upsert_video(conn, registration, video_id=video_id)

    return resolved_video_id


def insert_tracks(db_path: str, video_id: int, tracks: Iterable[dict[str, Any]]) -> None:
    """Insert vehicle tracking coordinate frames in bulk."""
    conn = _connect(db_path)
    cursor = conn.cursor()

    data_to_insert = [
        (
            video_id,
            int(t["vehicle_id"]),
            int(t["frame_id"]),
            int(t["class_id"]),
            float(t["confidence"]),
            float(t["x_centre"]),
            float(t["y_centre"]),
            float(t["width"]),
            float(t["height"]),
        )
        for t in tracks
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO vehicle_tracks (
            video_id, track_id, frame_id, class_id, confidence, x_centre, y_centre, width, height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        data_to_insert,
    )

    conn.commit()
    conn.close()


def insert_crossing_events(db_path: str, video_id: int, events: Iterable[dict[str, Any]]) -> None:
    """Insert crossing events in bulk."""
    conn = _connect(db_path)
    cursor = conn.cursor()

    data_to_insert = [
        (
            video_id,
            int(e["vehicle_id"]),
            str(e["line_name"]),
            int(e["frame_id"]),
            str(e["direction"]),
            int(e["class_id"]) if e.get("class_id") is not None else None,
        )
        for e in events
    ]

    cursor.executemany(
        """
        INSERT INTO crossing_events (
            video_id, vehicle_id, line_name, frame_id, direction, class_id
        ) VALUES (?, ?, ?, ?, ?, ?);
        """,
        data_to_insert,
    )

    conn.commit()
    conn.close()


def insert_entering_counts(db_path: str, video_id: int, counts: Iterable[dict[str, Any]]) -> None:
    """Insert aggregated entering counts in bulk."""
    data_to_insert = []
    identities: set[tuple[str, int | None]] = set()
    for count in counts:
        direction = str(count["direction"])
        class_id = int(count["class_id"]) if count.get("class_id") is not None else None
        if not direction.strip():
            raise ValueError("Entering-count direction must be non-empty")
        if class_id is not None and class_id not in VALID_PROJECT_CLASS_IDS:
            raise ValueError(f"Invalid project class_id: {class_id}")
        vehicle_count = int(count["vehicle_count"])
        if vehicle_count < 0:
            raise ValueError("Entering-count vehicle_count must be non-negative")
        identity = (direction, class_id)
        if identity in identities:
            raise ValueError(
                "Duplicate entering-count identity in one write: "
                f"video_id={video_id}, direction={direction!r}, class_id={class_id!r}"
            )
        identities.add(identity)
        data_to_insert.append((video_id, direction, class_id, vehicle_count))

    with closing(_connect(db_path)) as conn, conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO entering_counts (
                video_id, direction, class_id, vehicle_count
            ) VALUES (?, ?, ?, ?);
            """,
            data_to_insert,
        )


def replace_video_tracking_results(
    db_path: str,
    video_id: int,
    tracks: Iterable[dict[str, Any]],
    crossing_events: Iterable[dict[str, Any]],
    entering_counts: Iterable[dict[str, Any]],
    *,
    _connection: sqlite3.Connection | None = None,
) -> None:
    """Atomically replace one video's complete tracking-derived row set.

    Speed and emission rows are invalidated in the same transaction because they
    were derived from the replaced tracks.
    """

    video_id = _validated_integer(video_id, "video_id", minimum=1)
    track_rows = []
    track_identities: set[tuple[int, int]] = set()
    for track in tracks:
        track_id = _validated_integer(track["vehicle_id"], "track.vehicle_id")
        frame_id = _validated_integer(track["frame_id"], "track.frame_id")
        class_id = _validated_integer(track["class_id"], "track.class_id")
        if class_id not in VALID_PROJECT_CLASS_IDS:
            raise ValueError(f"Invalid project class_id: {class_id}")
        identity = (track_id, frame_id)
        if identity in track_identities:
            raise ValueError(
                "Duplicate track identity in one replacement: "
                f"video_id={video_id}, track_id={track_id}, frame_id={frame_id}"
            )
        track_identities.add(identity)
        track_rows.append(
            (
                video_id,
                track_id,
                frame_id,
                class_id,
                _validated_real(
                    track["confidence"],
                    "track.confidence",
                    minimum=0.0,
                    maximum=1.0,
                ),
                _validated_real(track["x_centre"], "track.x_centre"),
                _validated_real(track["y_centre"], "track.y_centre"),
                _validated_real(
                    track["width"],
                    "track.width",
                    minimum=0.0,
                    minimum_inclusive=False,
                ),
                _validated_real(
                    track["height"],
                    "track.height",
                    minimum=0.0,
                    minimum_inclusive=False,
                ),
            )
        )

    event_rows = []
    for event in crossing_events:
        line_name = str(event["line_name"])
        direction = str(event["direction"])
        if not line_name.strip() or not direction.strip():
            raise ValueError("Crossing-event line_name and direction must be non-empty")
        class_value = event.get("class_id")
        class_id = (
            _validated_integer(class_value, "crossing_event.class_id")
            if class_value is not None
            else None
        )
        if class_id is not None and class_id not in VALID_PROJECT_CLASS_IDS:
            raise ValueError(f"Invalid project class_id: {class_id}")
        event_rows.append(
            (
                video_id,
                _validated_integer(event["vehicle_id"], "crossing_event.vehicle_id"),
                line_name,
                _validated_integer(event["frame_id"], "crossing_event.frame_id"),
                direction,
                class_id,
            )
        )

    count_rows = []
    count_identities: set[tuple[str, int | None]] = set()
    for count in entering_counts:
        direction = str(count["direction"])
        class_value = count.get("class_id")
        class_id = (
            _validated_integer(class_value, "entering_count.class_id")
            if class_value is not None
            else None
        )
        vehicle_count = _validated_integer(count["vehicle_count"], "entering_count.vehicle_count")
        if not direction.strip():
            raise ValueError("Entering-count direction must be non-empty")
        if class_id is not None and class_id not in VALID_PROJECT_CLASS_IDS:
            raise ValueError(f"Invalid project class_id: {class_id}")
        if vehicle_count < 0:
            raise ValueError("Entering-count vehicle_count must be non-negative")
        identity = (direction, class_id)
        if identity in count_identities:
            raise ValueError(
                "Duplicate entering-count identity in one replacement: "
                f"video_id={video_id}, direction={direction!r}, class_id={class_id!r}"
            )
        count_identities.add(identity)
        count_rows.append((video_id, direction, class_id, vehicle_count))

    def replace_on_connection(conn: sqlite3.Connection) -> None:
        if (
            conn.execute("SELECT 1 FROM video_master WHERE video_id = ?", (video_id,)).fetchone()
            is None
        ):
            raise ValueError(f"Unknown video_id: {video_id}")
        conn.execute("DELETE FROM video_emissions WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM vehicle_speeds WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM crossing_events WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM entering_counts WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM vehicle_tracks WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO vehicle_tracks (
                video_id, track_id, frame_id, class_id, confidence,
                x_centre, y_centre, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            track_rows,
        )
        conn.executemany(
            """
            INSERT INTO crossing_events (
                video_id, vehicle_id, line_name, frame_id, direction, class_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        conn.executemany(
            """
            INSERT INTO entering_counts (video_id, direction, class_id, vehicle_count)
            VALUES (?, ?, ?, ?)
            """,
            count_rows,
        )

    if _connection is not None:
        replace_on_connection(_connection)
        return

    with closing(_connect(db_path)) as conn, conn:
        replace_on_connection(conn)


def register_video_with_tracking_results(
    db_path: str,
    video_path: str,
    intersection_id: str,
    start_time_str: str,
    tracks: Iterable[dict[str, Any]],
    crossing_events: Iterable[dict[str, Any]],
    entering_counts: Iterable[dict[str, Any]],
    *,
    duration_limit: float | None = None,
    notes: str | None = None,
    group_id: str | None = None,
    video_id: int | None = None,
) -> int:
    """Atomically upsert video metadata and replace all tracking-derived rows."""

    registration = _video_registration_values(
        video_path,
        intersection_id,
        start_time_str,
        duration_limit,
        notes,
        group_id,
        require_complete_metadata=True,
    )
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        resolved_video_id = _upsert_video(conn, registration, video_id=video_id)
        replace_video_tracking_results(
            db_path,
            resolved_video_id,
            tracks,
            crossing_events,
            entering_counts,
            _connection=conn,
        )
    return resolved_video_id


def check_video_id_exists(db_path: str, video_id: int) -> bool:
    """Check if the given video_id exists in the video_master table."""
    init_db(db_path)
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM video_master WHERE video_id = ?;", (video_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def check_video_name_processed(db_path: str, video_name: str) -> bool:
    """Check if the given video_name exists in video_master and has tracking or entering count records."""
    if not os.path.exists(db_path):
        return False
    init_db(db_path)
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1 FROM video_master v
        WHERE v.video_name = ?
          AND (
              EXISTS (SELECT 1 FROM vehicle_tracks t WHERE t.video_id = v.video_id)
              OR EXISTS (SELECT 1 FROM entering_counts ec WHERE ec.video_id = v.video_id)
          );
        """,
        (video_name,),
    )
    processed = cursor.fetchone() is not None
    conn.close()
    return processed


def insert_vehicle_speeds(db_path: str, video_id: int, speeds: Iterable[dict[str, Any]]) -> None:
    """Insert vehicle speed records in bulk."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    data_to_insert = [
        (
            int(video_id),
            int(s["track_id"]),
            int(s["second"]),
            float(s["speed_kmh"]),
        )
        for s in speeds
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO vehicle_speeds (
            video_id, track_id, second, speed_kmh
        ) VALUES (?, ?, ?, ?);
        """,
        data_to_insert,
    )
    conn.commit()
    conn.close()


def insert_video_emissions(db_path: str, emissions: Iterable[dict[str, Any]]) -> None:
    """Insert video-level emission records in bulk."""
    conn = _connect(db_path)
    cursor = conn.cursor()
    data_to_insert = [
        (
            int(e["video_id"]),
            e["estimation_method"],
            float(e["nox_grams"]),
            float(e["co_grams"]),
            float(e["hc_grams"]),
            float(e["pm25_grams"]),
            float(e["co2_grams"]),
            float(e["fuel_grams"]),
            float(e["energy_kj"]),
            int(e["vehicle_count"]),
            float(e["average_speed_kmh"]) if e.get("average_speed_kmh") is not None else None,
            e.get("congestion_level"),
            (
                float(e["nox_hybrid_adjusted_grams"])
                if e.get("nox_hybrid_adjusted_grams") is not None
                else None
            ),
        )
        for e in emissions
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO video_emissions (
            video_id, estimation_method,
            nox_grams, co_grams, hc_grams, pm25_grams, co2_grams, fuel_grams, energy_kj, vehicle_count,
            average_speed_kmh, congestion_level, nox_hybrid_adjusted_grams
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        data_to_insert,
    )
    conn.commit()
    conn.close()
