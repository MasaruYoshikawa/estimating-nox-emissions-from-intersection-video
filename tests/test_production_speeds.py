from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vehicle_emission import database
from vehicle_emission.calculate_speeds import calculate_calibrated_speeds
from vehicle_emission.speed.calibration import project_image_to_world
from vehicle_emission.speed.production import (
    calculate_video_steps,
    replay_speed_bundle,
    second_speed_exports,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    # These bytes are test identities, not recorded empirical video or geometry.
    media = tmp_path / "synthetic.mp4"
    media.write_bytes(b"synthetic source identity only")
    evidence = tmp_path / "synthetic-geometry.txt"
    evidence.write_text("synthetic controls; not empirical calibration")
    timing = tmp_path / "synthetic-timing.csv"
    pd.DataFrame({"frame_id": [0, 1, 2, 3], "elapsed_seconds": [0, 0.5, 2.0, 3.0]}).to_csv(
        timing, index=False
    )
    db_path = tmp_path / "source.db"
    database.init_db(str(db_path))
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """INSERT INTO video_master (
                video_id,video_name,video_path,intersection_id,width,height,
                recording_datetime_utc,recording_date,recording_time,processed_datetime_utc,fps,duration_seconds
            ) VALUES (1,'synthetic',?,'Sukiyabashi',200,200,
                '2026-06-03T03:59:59.500Z','2026-06-03','03:59:59.500','2026-06-03T05:00:00Z',NULL,NULL)""",
            [str(media)],
        )
        connection.executemany(
            "INSERT INTO vehicle_tracks VALUES (1,1,?,1,0.9,?,30,10,20)",
            [(0, 20), (1, 30), (2, 60), (3, 60)],
        )
        connection.execute("INSERT INTO vehicle_speeds VALUES (1,1,0,999)")
        connection.execute(
            "INSERT INTO video_emissions(video_id,estimation_method,nox_grams,vehicle_count) VALUES (1,'MOVEStar',99,1)"
        )
    connection.close()
    controls = [[0, 0], [200, 0], [200, 200], [0, 200]]
    matrix = np.array([[0.1, 0.02, 2], [0.01, 0.2, -1], [0.001, 0.002, 1]])
    calibration = {
        "site": "Sukiyabashi",
        "camera_epoch": "synthetic-epoch",
        "calibration_id": "synthetic-h",
        "width": 200,
        "height": 200,
        "route": "homography",
        "image_points": controls,
        "world_points_metres": project_image_to_world(controls, matrix).tolist(),
        "image_polygon": controls,
        "geometry_evidence_path": str(evidence),
        "geometry_evidence_sha256": _digest(evidence),
    }
    source = {
        "site": "Sukiyabashi",
        "source_id": "synthetic-source",
        "video_id": 1,
        "source_path": str(media),
        "source_sha256": _digest(media),
        "camera_epoch": "synthetic-epoch",
        "calibration_id": "synthetic-h",
        "width": 200,
        "height": 200,
        "recording_start_utc": "2026-06-03T03:59:59.500Z",
        "timing_source_id": "synthetic-clock",
        "timing_path": str(timing),
        "timing_sha256": _digest(timing),
        "timing_basis": "source_elapsed_seconds",
    }
    config = {
        "schema_version": "production_speed_v4",
        "maximum_step_seconds": 2.0,
        "calibrations": [calibration],
        "sources": [source],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return db_path, config_path, config


def _calculate(db_path: Path, config: dict[str, Any], base: Path):
    with sqlite3.connect(db_path) as connection:
        tracks = pd.read_sql_query("SELECT * FROM vehicle_tracks", connection)
    return calculate_video_steps(
        tracks,
        pd.read_csv(config["sources"][0]["timing_path"]),
        source=config["sources"][0],
        calibration=config["calibrations"][0],
        maximum_step_seconds=config["maximum_step_seconds"],
        evidence_base=base,
    )


def test_production_database_non_affine_bottom_contact_real_timing_and_immutable_bundle(tmp_path):
    db_path, config_path, config = _fixture(tmp_path)
    old_database = db_path.read_bytes()
    output = calculate_calibrated_speeds(
        str(db_path), str(config_path), str(tmp_path / "kinematics")
    )
    steps = pd.read_csv(output / "speed_steps_v4.csv")
    assert steps.start_y_pixel.tolist() == [40, 40, 40]  # bottom centre, not y_centre=30
    matrix = np.array([[0.1, 0.02, 2], [0.01, 0.2, -1], [0.001, 0.002, 1]])
    world = project_image_to_world([[20, 40], [30, 40], [60, 40], [60, 40]], matrix)
    expected = np.linalg.norm(np.diff(world, axis=0), axis=1) / np.array([0.5, 1.5, 1])
    np.testing.assert_allclose(steps.speed_mps, expected, rtol=1e-12)
    assert steps.speed_mps.iloc[-1] == 0  # actual stationary interval
    assert len(steps) == 3  # no synthetic initial speed
    seconds = pd.read_csv(output / "speed_seconds_v4.csv")
    assert seconds.ts_utc.tolist() == [
        "2026-06-03T03:59:59Z",
        "2026-06-03T04:00:00Z",
        "2026-06-03T04:00:01Z",
        "2026-06-03T04:00:02Z",
    ]
    assert seconds.duration_seconds.tolist() == [0.5, 1, 1, 0.5]
    assert seconds.speed_mps.iloc[2] == pytest.approx(expected[1] / 2)
    profiles = pd.read_csv(output / "speed_profiles_v4.csv")
    assert profiles.second_index.tolist() == [0, 1]
    assert profiles.duration_seconds.eq(1).all()
    receipt = json.loads((output / "production_receipt_v4.json").read_text())
    assert receipt["database_access"] == "read_only"
    assert receipt["sources"][0]["source"]["sha256"] == config["sources"][0]["source_sha256"]
    assert receipt["code_revision"] and len(receipt["code_sha256"]) == 3
    assert receipt["exclusions"] == {"initial_observation": 1}
    for filename, expected_hash in receipt["outputs"].items():
        assert _digest(output / filename) == expected_hash
    assert db_path.read_bytes() == old_database
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT speed_kmh FROM vehicle_speeds").fetchall() == [(999,)]
        assert connection.execute("SELECT nox_grams FROM video_emissions").fetchall() == [(99,)]
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(output))
    assert original == {path.name: path.read_bytes() for path in output.iterdir()}
    repeat = calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "repeat"))
    assert (repeat / "speed_steps_v4.csv").read_bytes() == (
        output / "speed_steps_v4.csv"
    ).read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [("site", "Shinjuku"), ("camera_epoch", "other"), ("width", 100), ("calibration_id", "other")],
)
def test_source_geometry_identity_mismatch_fails_without_publication(tmp_path, field, value):
    db_path, config_path, config = _fixture(tmp_path)
    config["sources"][0][field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatch"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "times", [[0, 0, 2, 3], [0, float("nan"), 2, 3], [0, 2, 1, 3], [-1, 0, 2, 3]]
)
def test_invalid_source_timing_is_rejected(tmp_path, times):
    db_path, _, config = _fixture(tmp_path)
    timing_path = Path(config["sources"][0]["timing_path"])
    pd.DataFrame({"frame_id": [0, 1, 2, 3], "elapsed_seconds": times}).to_csv(
        timing_path, index=False
    )
    with pytest.raises(ValueError, match="timing"):
        _calculate(db_path, config, tmp_path)


def test_duplicate_and_missing_frame_timing_rejected(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    timing_path = Path(config["sources"][0]["timing_path"])
    pd.DataFrame({"frame_id": [0, 1, 1, 3], "elapsed_seconds": [0, 1, 2, 3]}).to_csv(
        timing_path, index=False
    )
    with pytest.raises(ValueError, match="timing"):
        _calculate(db_path, config, tmp_path)
    pd.DataFrame({"frame_id": [0, 1, 3], "elapsed_seconds": [0, 1, 3]}).to_csv(
        timing_path, index=False
    )
    with pytest.raises(ValueError, match="unavailable"):
        _calculate(db_path, config, tmp_path)


def test_missing_tracks_and_long_elapsed_intervals_are_not_zero_or_bridged(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    config["maximum_step_seconds"] = 1.0
    steps, rejected = _calculate(db_path, config, tmp_path)
    assert steps.end_frame.tolist() == [1, 3]
    assert rejected.reason.tolist() == ["initial_observation", "elapsed_time_gap"]
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM vehicle_tracks WHERE frame_id=1")
    steps, rejected = _calculate(db_path, config, tmp_path)
    assert steps.end_frame.tolist() == [3]
    assert rejected.reason.tolist() == ["initial_observation", "tracking_frame_gap"]


def _bounded(config):
    calibration = config["calibrations"][0]
    calibration["route"] = "bounded_zones"
    calibration["zones"] = [
        {
            "id": "left",
            "image_polygon": [[0, 0], [40, 0], [40, 100], [0, 100]],
            "scale_metres_per_pixel": 0.1,
        },
        {
            "id": "right",
            "image_polygon": [[50, 0], [100, 0], [100, 100], [50, 100]],
            "scale_metres_per_pixel": 0.2,
        },
    ]


def test_bounded_zone_crossing_outside_and_boundary_steps(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    _bounded(config)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE vehicle_tracks SET x_centre=40 WHERE frame_id=1")
    steps, rejected = _calculate(db_path, config, tmp_path)
    assert steps.end_frame.tolist() == [1, 3]
    assert steps.speed_mps.tolist() == [4.0, 0.0]
    assert rejected.reason.tolist() == ["initial_observation", "zone_crossing"]
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE vehicle_tracks SET x_centre=45 WHERE frame_id=1")
    steps, rejected = _calculate(db_path, config, tmp_path)
    assert steps.end_frame.tolist() == [3]
    assert rejected.reason.tolist() == ["initial_observation", "out_of_zone", "initial_observation"]


@pytest.mark.parametrize("right_edge", [40, 30])
def test_overlapping_or_touching_zones_rejected(tmp_path, right_edge):
    db_path, _, config = _fixture(tmp_path)
    _bounded(config)
    config["calibrations"][0]["zones"][1]["image_polygon"] = [
        [right_edge, 0],
        [100, 0],
        [100, 100],
        [right_edge, 100],
    ]
    with pytest.raises(ValueError, match="overlap"):
        _calculate(db_path, config, tmp_path)


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [100, 100], [0, 100], [100, 0]],
        [[0, 0], [100, 0], [50, 50], [100, 100], [0, 100]],
        [[0, 0], [0, 0], [10, 10]],
    ],
)
def test_invalid_or_unsupported_geometry_rejected(tmp_path, polygon):
    db_path, _, config = _fixture(tmp_path)
    config["calibrations"][0]["image_polygon"] = polygon
    with pytest.raises(ValueError, match="polygon"):
        _calculate(db_path, config, tmp_path)


def test_footprint_crossing_projective_horizon_is_rejected(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    calibration = config["calibrations"][0]
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, -1]])
    calibration["world_points_metres"] = project_image_to_world(
        calibration["image_points"], matrix
    ).tolist()
    with pytest.raises(ValueError, match="horizon"):
        _calculate(db_path, config, tmp_path)


def test_large_speed_is_not_capped_and_invalid_box_breaks_trajectory(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    _bounded(config)
    config["calibrations"][0]["zones"] = [config["calibrations"][0]["zones"][0]]
    config["calibrations"][0]["zones"][0]["scale_metres_per_pixel"] = 100
    steps, _ = _calculate(db_path, config, tmp_path)
    assert steps.speed_mps.tolist() == [2000]
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE vehicle_tracks SET height=0 WHERE frame_id=1")
    steps, rejected = _calculate(db_path, config, tmp_path)
    assert steps.empty
    assert "invalid_bounding_box" in rejected.reason.tolist()
    seconds, profiles = second_speed_exports(steps)
    assert seconds.empty and profiles.empty


def test_complete_profile_runs_split_at_missing_seconds_and_zones(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    steps, _ = _calculate(db_path, config, tmp_path)
    first = steps.iloc[0].to_dict()
    intervals = []
    for offset, zone in [(0, "road_plane"), (1, "road_plane"), (3, "road_plane"), (4, "other")]:
        row = copy.deepcopy(first)
        row.update(
            start_utc=f"2026-06-03T04:00:0{offset}Z",
            end_utc=f"2026-06-03T04:00:0{offset+1}Z",
            image_zone=zone,
        )
        intervals.append(row)
    _, profiles = second_speed_exports(pd.DataFrame(intervals))
    assert profiles.profile_id.tolist() == ["1/1/0", "1/1/0", "1/1/1", "1/1/2"]
    assert profiles.second_index.tolist() == [0, 1, 0, 0]


def test_ordinary_30fps_nonuniform_nanosecond_fragments_preserve_full_seconds(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    steps, _ = _calculate(db_path, config, tmp_path)
    template = steps.iloc[0].to_dict()
    start = pd.Timestamp("2026-06-03T04:00:00Z")
    observations = [start + pd.to_timedelta(index / 30, unit="s") for index in range(61)]
    intervals = []
    for first, second in zip(observations[:-1], observations[1:], strict=True):
        intervals.append(
            {
                **template,
                "start_utc": first.isoformat(),
                "end_utc": second.isoformat(),
                "speed_mps": 10.0,
            }
        )
    seconds, profiles = second_speed_exports(pd.DataFrame(intervals))
    assert seconds.duration_seconds.tolist() == [1.0, 1.0]
    assert profiles.second_index.tolist() == [0, 1]
    assert profiles.speed_mps.tolist() == [10.0, 10.0]
    from vehicle_emission.robustness.speed import SpeedPerturbation, calculate_speed_sensitivity

    sensitivity = calculate_speed_sensitivity(profiles, (SpeedPerturbation("identity", 1, 0),))
    assert len(sensitivity.runs) == 1
    assert sensitivity.runs.run_seconds.tolist() == [2]


def test_nanosecond_coverage_gap_is_not_a_complete_profile_second(tmp_path):
    db_path, _, config = _fixture(tmp_path)
    steps, _ = _calculate(db_path, config, tmp_path)
    row = steps.iloc[0].to_dict()
    row.update(start_utc="2026-06-03T04:00:00Z", end_utc="2026-06-03T04:00:00.999999999Z")
    seconds, profiles = second_speed_exports(pd.DataFrame([row]))
    assert seconds.duration_seconds.tolist() == [0.999999999]
    assert profiles.empty


def test_step_speed_samples_feed_duration_weighted_hourly_adapter(tmp_path):
    from vehicle_emission.inference.hourly_baselines import prepare_speed_samples

    db_path, config_path, _ = _fixture(tmp_path)
    bundle = calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    samples = pd.read_csv(bundle / "speed_samples_v4.csv")
    prepared = prepare_speed_samples(samples, site="Sukiyabashi")
    assert prepared is not None and len(prepared) == 3
    assert prepared._valid_speed.all() and prepared._valid_calibration.all()
    assert (prepared._end_ns - prepared._start_ns).tolist() == [
        500_000_000,
        1_500_000_000,
        1_000_000_000,
    ]


@pytest.mark.parametrize("prefix", ["source", "timing"])
def test_changed_evidence_hash_fails_and_preserves_database(tmp_path, prefix):
    db_path, config_path, config = _fixture(tmp_path)
    old_database = db_path.read_bytes()
    Path(config["sources"][0][f"{prefix}_path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    assert db_path.read_bytes() == old_database
    assert not (tmp_path / "output").exists()


def test_production_cli_uses_config_and_rejects_scalar_flags(tmp_path):
    db_path, config_path, _ = _fixture(tmp_path)
    command = [sys.executable, "-m", "vehicle_emission.calculate_speeds", "--db-path", str(db_path)]
    completed = subprocess.run(
        command + ["--config", str(config_path), "--output-dir", str(tmp_path / "cli")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli" / "production_receipt_v4.json").is_file()
    rejected = subprocess.run(
        command + ["--site", "Sukiyabashi", "--scale", "10"], capture_output=True, text=True
    )
    assert rejected.returncode != 0


@pytest.mark.parametrize("source_id", ["synthetic-source", "NA", "001"])
def test_receipt_replay_recomputes_without_database(tmp_path, source_id):
    db_path, config_path, config = _fixture(tmp_path)
    config["sources"][0]["source_id"] = source_id
    config_path.write_text(json.dumps(config))
    bundle = calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "first"))
    db_path.unlink()
    Path(config["sources"][0]["source_path"]).unlink()
    Path(config["sources"][0]["timing_path"]).unlink()
    Path(config["calibrations"][0]["geometry_evidence_path"]).unlink()
    config_path.unlink()
    output = replay_speed_bundle(str(bundle), str(tmp_path / "replayed"))
    for name in (
        "speed_steps_v4.csv",
        "speed_seconds_v4.csv",
        "speed_profiles_v4.csv",
        "speed_exclusions_v4.csv",
    ):
        assert (output / name).read_bytes() == (bundle / name).read_bytes()
    receipt = json.loads((output / "production_receipt_v4.json").read_text())
    assert receipt["replayed_from_receipt_sha256"] == _digest(bundle / "production_receipt_v4.json")
    assert (
        receipt["replay_external_evidence_verification"]
        == "raw_asset_identity_recorded_not_reverified"
    )


@pytest.mark.parametrize(
    "target",
    [
        "source_tracks_v4.csv",
        "source_timing_v4.csv",
        "production_config_v4.json",
        "speed_steps_v4.csv",
    ],
)
def test_replay_rejects_changed_copies(tmp_path, target):
    db_path, config_path, _ = _fixture(tmp_path)
    bundle = calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "first"))
    (bundle / target).write_bytes((bundle / target).read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_speed_bundle(str(bundle), str(tmp_path / "replayed"))
    assert not (tmp_path / "replayed").exists()


def test_replay_rejects_changed_geometry_and_recomputes_instead_of_copying(tmp_path):
    db_path, config_path, config = _fixture(tmp_path)
    bundle = calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "first"))
    geometry = Path(config["calibrations"][0]["geometry_evidence_path"])
    old_geometry = geometry.read_bytes()
    geometry.write_text("changed geometry")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_speed_bundle(str(bundle), str(tmp_path / "replayed"), reverify_source_evidence=True)
    geometry.write_bytes(old_geometry)
    output = bundle / "speed_steps_v4.csv"
    frame = pd.read_csv(output, float_precision="round_trip")
    frame.loc[0, "speed_mps"] = 999
    frame.to_csv(output, index=False, float_format="%.17g")
    receipt_path = bundle / "production_receipt_v4.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"][output.name] = _digest(output)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="does not reproduce"):
        replay_speed_bundle(str(bundle), str(tmp_path / "replayed"))


def test_mid_run_input_and_code_changes_fail_before_publication(tmp_path, monkeypatch):
    import vehicle_emission.speed.production as production

    db_path, config_path, _ = _fixture(tmp_path)
    original_exports = production.second_speed_exports

    def change_config(steps):
        config_path.write_text(config_path.read_text() + " ")
        return original_exports(steps)

    monkeypatch.setattr(production, "second_speed_exports", change_config)
    with pytest.raises(ValueError, match="configuration changed"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()
    monkeypatch.setattr(production, "second_speed_exports", original_exports)
    hashes = iter([production._IMPORTED_CODE_IDENTITY, {"source.py": "changed"}])
    monkeypatch.setattr(production, "_code_hashes", lambda: next(hashes))
    with pytest.raises(ValueError, match="code changed"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_committed_deleted_destination_is_refused(tmp_path, monkeypatch):
    import vehicle_emission.speed.production as production

    monkeypatch.setattr(production, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="archived/speed_steps_v4.csv\n"
        ),
    )
    with pytest.raises(FileExistsError, match="committed"):
        production._destination(str(tmp_path / "archived"))


def test_atomic_publication_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    db_path, config_path, _ = _fixture(tmp_path)
    original_rename = Path.rename

    def fail_rename(path, target):
        if path.name == "bundle":
            raise OSError("publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="publication failure"):
        calculate_calibrated_speeds(str(db_path), str(config_path), str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".production-speed-*"))
    assert not (tmp_path / ".output.publish.lock").exists()
