from __future__ import annotations

import sqlite3

import pytest

import vehicle_emission.tracking.rtdetr_bytetrack_entering as tracking_module
from vehicle_emission import database
from vehicle_emission.tracking.rtdetr_bytetrack_entering import (
    persist_tracking_outputs,
    should_collect_annotations,
    tracking_pass_is_complete,
)


def test_display_collects_annotations_without_debug_video() -> None:
    assert should_collect_annotations(display=True, save_debug_video=False)
    assert should_collect_annotations(display=False, save_debug_video=True)
    assert not should_collect_annotations(display=False, save_debug_video=False)


def test_tracking_completion_requires_known_boundary_and_decoded_frame() -> None:
    assert not tracking_pass_is_complete(
        expected_frames=None,
        frames_advanced=100,
        decoded_frame_count=100,
    )
    assert not tracking_pass_is_complete(
        expected_frames=100,
        frames_advanced=100,
        decoded_frame_count=0,
    )
    assert not tracking_pass_is_complete(
        expected_frames=100,
        frames_advanced=99,
        decoded_frame_count=99,
    )
    assert tracking_pass_is_complete(
        expected_frames=100,
        frames_advanced=100,
        decoded_frame_count=50,
    )
    assert tracking_pass_is_complete(
        expected_frames=600,
        frames_advanced=600,
        decoded_frame_count=150,
    )


def test_database_failure_does_not_publish_staged_tracking_csv(tmp_path) -> None:
    db_path = str(tmp_path / "tracking.db")
    output_csv = tmp_path / "tracking.csv"
    output_csv.write_text("old,published\n", encoding="utf-8")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc
        ) VALUES (1, 'atomic_csv', 'atomic_csv.mp4', 'Sukiyabashi',
                  '2026-06-03T03:00:00Z', '2026-06-03', '03:00:00',
                  '2026-06-03T03:01:00Z')
        """
    )
    connection.execute("INSERT INTO vehicle_tracks VALUES (1, 1, 0, 1, 0.9, 1, 1, 1, 1)")
    connection.execute(
        """
        CREATE TRIGGER reject_csv_publish_track
        BEFORE INSERT ON vehicle_tracks
        WHEN NEW.frame_id = 99
        BEGIN
            SELECT RAISE(ABORT, 'database publish failure');
        END
        """
    )
    connection.commit()
    connection.close()

    new_rows = [
        {
            "frame_id": 99,
            "vehicle_id": 2,
            "class_id": 1,
            "confidence": 0.8,
            "x_centre": 2.0,
            "y_centre": 2.0,
            "width": 1.0,
            "height": 1.0,
        }
    ]
    with pytest.raises(sqlite3.IntegrityError, match="database publish failure"):
        persist_tracking_outputs(
            db_path=db_path,
            output_csv=str(output_csv),
            csv_data=new_rows,
            video_id=1,
            pending_registration=None,
            crossing_events=[],
            entering_counts=[],
        )

    assert output_csv.read_text(encoding="utf-8") == "old,published\n"
    assert list(tmp_path.glob(".tracking.csv.*.tmp")) == []
    connection = sqlite3.connect(db_path)
    assert connection.execute("SELECT track_id FROM vehicle_tracks").fetchall() == [(1,)]
    connection.close()


def test_tracking_csv_staging_failure_preserves_published_file(tmp_path, monkeypatch) -> None:
    output_csv = tmp_path / "tracking.csv"
    output_csv.write_text("old,published\n", encoding="utf-8")

    def fail_during_staging(dataframe, path, *, index):
        del dataframe, index
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("csv staging failure")

    monkeypatch.setattr(tracking_module.pd.DataFrame, "to_csv", fail_during_staging)
    with pytest.raises(RuntimeError, match="csv staging failure"):
        persist_tracking_outputs(
            db_path=None,
            output_csv=str(output_csv),
            csv_data=[],
            video_id=None,
            pending_registration=None,
            crossing_events=[],
            entering_counts=[],
        )

    assert output_csv.read_text(encoding="utf-8") == "old,published\n"
    assert list(tmp_path.glob(".tracking.csv.*.tmp")) == []


def test_tracking_csv_is_sorted_before_atomic_publish(tmp_path) -> None:
    output_csv = tmp_path / "tracking.csv"
    output_csv.write_text("old,published\n", encoding="utf-8")
    rows = [
        {
            "frame_id": 2,
            "vehicle_id": 5,
            "class_id": 1,
            "confidence": 0.8,
            "x_centre": 2.0,
            "y_centre": 2.0,
            "width": 1.0,
            "height": 1.0,
        },
        {
            "frame_id": 1,
            "vehicle_id": 4,
            "class_id": 3,
            "confidence": 0.9,
            "x_centre": 3.0,
            "y_centre": 3.0,
            "width": 1.0,
            "height": 1.0,
        },
    ]

    persist_tracking_outputs(
        db_path=None,
        output_csv=str(output_csv),
        csv_data=rows,
        video_id=None,
        pending_registration=None,
        crossing_events=[],
        entering_counts=[],
    )

    lines = output_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(tracking_module.TRACKING_CSV_COLUMNS)
    assert lines[1].startswith("1,4,3,")
    assert lines[2].startswith("2,5,1,")
    assert list(tmp_path.glob(".tracking.csv.*.tmp")) == []
