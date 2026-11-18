"""Read-only, evidence-bound production kinematics; see the calibration GLOSSARY entries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vehicle_emission.speed.calibration import (
    compute_homography_matrix,
    project_image_to_world,
)

SITES = {"Sukiyabashi", "Shinjuku"}
IDENTITY = [
    "site",
    "source_id",
    "video_id",
    "track_id",
    "class_id",
    "camera_epoch",
    "calibration_id",
]
STEP_COLUMNS = IDENTITY + [
    "image_zone",
    "start_frame",
    "end_frame",
    "start_elapsed_seconds",
    "end_elapsed_seconds",
    "start_utc",
    "end_utc",
    "start_x_pixel",
    "start_y_pixel",
    "end_x_pixel",
    "end_y_pixel",
    "start_x_metres",
    "start_y_metres",
    "end_x_metres",
    "end_y_metres",
    "duration_seconds",
    "distance_metres",
    "speed_mps",
]
SECOND_COLUMNS = IDENTITY + ["image_zone", "ts_utc", "duration_seconds", "speed_mps"]
PROFILE_COLUMNS = SECOND_COLUMNS + ["profile_id", "second_index"]
REJECTION_COLUMNS = IDENTITY + ["frame_id", "previous_frame_id", "reason"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("calibration.py"),
        Path(__file__).parent.parent / "calculate_speeds.py",
    ]
    return {path.relative_to(_repo_root()).as_posix(): _sha256(path) for path in paths}


def _capture_code_identity() -> dict[str, str]:
    current = _code_hashes()
    if current != _IMPORTED_CODE_IDENTITY:
        raise ValueError("imported production code differs from disk; restart before calculating")
    return current


def _destination(output_dir: str) -> Path:
    path = Path(output_dir).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if path.is_relative_to(_repo_root()):
        tracked = subprocess.run(
            ["git", "ls-files", "--", path.relative_to(_repo_root()).as_posix()],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if tracked.strip():
            raise FileExistsError(
                "output path contains committed artifacts, including deleted files"
            )
    return path


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _publish(destination: Path, contents: dict[str, bytes], verify: Callable[[], None]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".production-speed-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for filename, data in contents.items():
            (stage / filename).write_bytes(data)
        lock = destination.parent / f".{destination.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            _destination(str(destination))
            verify()
            stage.rename(destination)
        finally:
            os.close(descriptor)
            lock.unlink()
    return destination


def _iso_utc(stamp: pd.Timestamp) -> str:
    return stamp.isoformat().replace("+00:00", "Z")


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identifier(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _utc(value: Any, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an explicit UTC timestamp") from error
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    return stamp.tz_convert("UTC")


def _evidence(
    data: Mapping[str, Any], prefix: str, base: Path, *, verify: bool = True
) -> dict[str, str]:
    path = (base / _identifier(data, f"{prefix}_path")).resolve(strict=verify)
    expected = _identifier(data, f"{prefix}_sha256")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{prefix}_sha256 must be a lowercase SHA-256")
    if verify and _sha256(path) != expected:
        raise ValueError(f"{prefix} evidence hash mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def _polygon(value: Any, width: int, height: int) -> np.ndarray:
    """Accept strictly convex footprints so endpoint containment proves step containment."""
    polygon = np.asarray(value, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("image_polygon must be an Nx2 convex polygon")
    if (
        not np.isfinite(polygon).all()
        or len(np.unique(polygon, axis=0)) != len(polygon)
        or (polygon < 0).any()
        or (polygon[:, 0] > width).any()
        or (polygon[:, 1] > height).any()
    ):
        raise ValueError("image_polygon must be finite, unique and within the resolution")
    # All other vertices must lie strictly on one side of every directed edge.
    for index, point in enumerate(polygon):
        edge = polygon[(index + 1) % len(polygon)] - point
        offsets = np.delete(polygon - point, [index, (index + 1) % len(polygon)], axis=0)
        crosses = edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]
        if not (np.all(crosses > 1e-10) or np.all(crosses < -1e-10)):
            raise ValueError("image_polygon must be strictly convex and non-degenerate")
    return polygon


def _contains(polygon: np.ndarray, point: np.ndarray) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point - polygon
    crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(crosses >= -1e-10) or np.all(crosses <= 1e-10))


def _overlap(left: np.ndarray, right: np.ndarray) -> bool:
    # Separating axis test includes touching boundaries: these are ambiguous too.
    for polygon in (left, right):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array([-edge[1], edge[0]])
            a, b = left @ axis, right @ axis
            if a.max() < b.min() - 1e-10 or b.max() < a.min() - 1e-10:
                return False
    return True


def _geometry(data: dict[str, Any], base: Path, *, verify_evidence: bool = True) -> dict[str, Any]:
    for key in ("site", "camera_epoch", "calibration_id"):
        _identifier(data, key)
    if data["site"] not in SITES:
        raise ValueError("unsupported calibration site")
    width = _integer(data.get("width"), "calibration width", 1)
    height = _integer(data.get("height"), "calibration height", 1)
    evidence = _evidence(data, "geometry_evidence", base, verify=verify_evidence)
    route = data.get("route")
    if route == "homography":
        matrix = compute_homography_matrix(
            data.get("image_points", []), data.get("world_points_metres", [])
        )
        zones = [("road_plane", _polygon(data.get("image_polygon"), width, height), matrix)]
        # The whole footprint must remain on one finite side of the projective horizon.
        denominators = np.column_stack((zones[0][1], np.ones(len(zones[0][1])))) @ matrix[2]
        if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
            raise ValueError("calibrated footprint crosses the homography horizon")
    elif route == "bounded_zones":
        if not isinstance(data.get("zones"), list) or not data["zones"]:
            raise ValueError("bounded_zones must declare zones")
        zones = []
        for zone in data["zones"]:
            zone_id = _identifier(zone, "id")
            polygon = _polygon(zone.get("image_polygon"), width, height)
            scale = _positive(zone.get("scale_metres_per_pixel"), "zone scale")
            zones.append((zone_id, polygon, np.diag([scale, scale, 1.0])))
        if len({zone[0] for zone in zones}) != len(zones):
            raise ValueError("duplicate calibration zone id")
        for index, zone in enumerate(zones):
            if any(_overlap(zone[1], other[1]) for other in zones[index + 1 :]):
                raise ValueError("calibration zones overlap or share ambiguous boundaries")
    else:
        raise ValueError("calibration route must be homography or bounded_zones")
    return {"declaration": data, "zones": zones, "evidence": evidence}


def calculate_video_steps(
    tracks: pd.DataFrame,
    frame_timing: pd.DataFrame,
    *,
    source: dict[str, Any],
    calibration: dict[str, Any],
    maximum_step_seconds: float,
    evidence_base: Path,
    verify_geometry_evidence: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate source-timed ground-contact intervals and explicit exclusion rows.

    Frame IDs must be consecutive for a valid step. Missing observations never
    become stationary speeds or a displacement bridged over a tracking gap.
    """
    geometry = _geometry(calibration, evidence_base, verify_evidence=verify_geometry_evidence)
    for key in ("site", "camera_epoch", "calibration_id", "width", "height"):
        if source.get(key) != calibration.get(key):
            raise ValueError(f"source/calibration {key} mismatch")
    maximum = _positive(maximum_step_seconds, "maximum_step_seconds")
    start = _utc(source.get("recording_start_utc"), "recording_start_utc")
    required = {"track_id", "frame_id", "class_id", "x_centre", "y_centre", "width", "height"}
    if not required <= set(tracks) or tracks.empty:
        raise ValueError("tracks require nonempty database bounding-box observations")
    if not {"frame_id", "elapsed_seconds"} <= set(frame_timing) or frame_timing.empty:
        raise ValueError("explicit source frame timing is required")
    timing = frame_timing[["frame_id", "elapsed_seconds"]].copy()
    timing["frame_id"] = [_integer(value, "timing frame_id") for value in timing.frame_id]
    timing["elapsed_seconds"] = pd.to_numeric(timing.elapsed_seconds, errors="raise")
    timing = timing.sort_values("frame_id")
    if (
        timing.frame_id.duplicated().any()
        or not np.isfinite(timing.elapsed_seconds).all()
        or (timing.elapsed_seconds < 0).any()
        or (timing.elapsed_seconds.diff().dropna() <= 0).any()
    ):
        raise ValueError("frame timing must be unique, finite, nonnegative and strictly increasing")
    frame_times = dict(zip(timing.frame_id, timing.elapsed_seconds, strict=True))
    ordered = tracks.copy()
    for column in ("track_id", "frame_id", "class_id"):
        ordered[column] = [_integer(value, column) for value in ordered[column]]
    if not ordered.class_id.isin([0, 1, 2, 3]).all():
        raise ValueError("track class_id must use the project class mapping")
    if ordered.duplicated(["track_id", "frame_id"]).any():
        raise ValueError("duplicate track/frame observations")
    if not set(ordered.frame_id) <= set(frame_times):
        raise ValueError("source timing unavailable for observed track frames")
    ordered = ordered.sort_values(["track_id", "frame_id"])
    steps: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for track_id, group in ordered.groupby("track_id", sort=True):
        class_counts = group.class_id.value_counts()
        class_id = min(class_counts[class_counts.eq(class_counts.max())].index)
        identity = {key: source[key] for key in IDENTITY if key not in {"track_id", "class_id"}}
        identity.update(track_id=int(track_id), class_id=int(class_id))
        previous: tuple[int, float, np.ndarray, tuple[Any, ...]] | None = None
        for row in group.itertuples(index=False):
            frame = int(row.frame_id)
            elapsed = float(frame_times[frame])
            old_frame = previous[0] if previous else None
            box = np.asarray([row.x_centre, row.y_centre, row.width, row.height], dtype=float)
            reason = ""
            if not np.isfinite(box).all() or (box[2:] <= 0).any():
                reason = "invalid_bounding_box"
            elif (
                box[0] - box[2] / 2 < 0
                or box[0] + box[2] / 2 > source["width"]
                or box[1] - box[3] / 2 < 0
                or box[1] + box[3] / 2 > source["height"]
            ):
                reason = "bounding_box_out_of_resolution"
            point = np.array([box[0], box[1] + box[3] / 2])
            zones = (
                [zone for zone in geometry["zones"] if _contains(zone[1], point)]
                if not reason
                else []
            )
            if not reason and len(zones) != 1:
                reason = "out_of_zone" if not zones else "ambiguous_zone"
            if reason:
                rejected.append(
                    {
                        **identity,
                        "frame_id": frame,
                        "previous_frame_id": old_frame,
                        "reason": reason,
                    }
                )
                previous = None
                continue
            zone = zones[0]
            if previous is None:
                reason = "initial_observation"
            elif frame - previous[0] != 1:
                reason = "tracking_frame_gap"
            elif elapsed - previous[1] > maximum:
                reason = "elapsed_time_gap"
            elif zone[0] != previous[3][0]:
                reason = "zone_crossing"
            if reason:
                rejected.append(
                    {
                        **identity,
                        "frame_id": frame,
                        "previous_frame_id": old_frame,
                        "reason": reason,
                    }
                )
            else:
                assert previous is not None
                points = project_image_to_world(np.asarray([previous[2], point]), zone[2])
                duration = elapsed - previous[1]
                distance = float(np.linalg.norm(points[1] - points[0]))
                speed = distance / duration
                if not math.isfinite(speed):
                    raise ValueError("calculated speed must be finite; no clipping is permitted")
                steps.append(
                    {
                        **identity,
                        "image_zone": zone[0],
                        "start_frame": previous[0],
                        "end_frame": frame,
                        "start_elapsed_seconds": previous[1],
                        "end_elapsed_seconds": elapsed,
                        "start_utc": _iso_utc(start + pd.to_timedelta(previous[1], unit="s")),
                        "end_utc": _iso_utc(start + pd.to_timedelta(elapsed, unit="s")),
                        "start_x_pixel": previous[2][0],
                        "start_y_pixel": previous[2][1],
                        "end_x_pixel": point[0],
                        "end_y_pixel": point[1],
                        "start_x_metres": points[0, 0],
                        "start_y_metres": points[0, 1],
                        "end_x_metres": points[1, 0],
                        "end_y_metres": points[1, 1],
                        "duration_seconds": duration,
                        "distance_metres": distance,
                        "speed_mps": speed,
                    }
                )
            previous = (frame, elapsed, point, zone)
    return pd.DataFrame(steps, columns=STEP_COLUMNS), pd.DataFrame(
        rejected, columns=REJECTION_COLUMNS
    )


def second_speed_exports(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split intervals at UTC seconds; export complete contiguous 1 Hz profiles separately."""
    fragments: list[dict[str, Any]] = []
    for row in steps.to_dict("records"):
        start, end = _utc(row["start_utc"], "start_utc"), _utc(row["end_utc"], "end_utc")
        current = start
        while current < end:
            second = current.floor("s")
            next_time = min(second + pd.Timedelta(seconds=1), end)
            fragments.append(
                {
                    **{key: row[key] for key in IDENTITY},
                    "image_zone": row["image_zone"],
                    "ts_utc": _iso_utc(second),
                    "duration_ns": (next_time - current).value,
                    "speed_mps": row["speed_mps"],
                }
            )
            current = next_time
    keys = IDENTITY + ["image_zone", "ts_utc"]
    seconds: list[dict[str, Any]] = []
    if fragments:
        for _, group in pd.DataFrame(fragments).groupby(keys, sort=True):
            first = group.iloc[0]
            duration_ns = int(group.duration_ns.sum())
            if duration_ns > 1_000_000_000:
                raise ValueError("overlapping step intervals in a UTC-second bin")
            seconds.append(
                {
                    **{key: first[key] for key in keys},
                    "duration_seconds": duration_ns / 1_000_000_000,
                    "speed_mps": float((group.speed_mps * group.duration_ns).sum() / duration_ns),
                }
            )
    second_frame = pd.DataFrame(seconds, columns=SECOND_COLUMNS)
    profiles: list[dict[str, Any]] = []
    complete = second_frame[second_frame.duration_seconds.eq(1.0)]
    for (video_id, track_id), group in complete.groupby(["video_id", "track_id"], sort=True):
        previous: dict[str, Any] | None = None
        segment, index = -1, 0
        for row in group.sort_values("ts_utc").to_dict("records"):
            if (
                previous is None
                or row["image_zone"] != previous["image_zone"]
                or pd.Timestamp(row["ts_utc"]) - pd.Timestamp(previous["ts_utc"])
                != pd.Timedelta(seconds=1)
            ):
                segment, index = segment + 1, 0
            profiles.append(
                {
                    **row,
                    "duration_seconds": 1.0,
                    "profile_id": f"{video_id}/{track_id}/{segment}",
                    "second_index": index,
                }
            )
            previous, index = row, index + 1
    return second_frame, pd.DataFrame(profiles, columns=PROFILE_COLUMNS)


def hourly_speed_samples(steps: pd.DataFrame) -> pd.DataFrame:
    """Adapt valid actual intervals to hourly_baselines without attesting external acceptance."""
    columns = [
        "site",
        "source_id",
        "video_id",
        "sample_id",
        "track_id",
        "source_track_id",
        "sample_start_utc",
        "sample_end_utc",
        "speed_mps",
        "speed_valid",
        "calibration_id",
        "calibration_valid",
        "camera_epoch",
        "image_zone",
    ]
    rows = []
    for step in steps.to_dict("records"):
        rows.append(
            {
                "site": step["site"],
                "source_id": step["source_id"],
                "video_id": step["video_id"],
                "sample_id": json.dumps(
                    [step["source_id"], step["track_id"], step["start_frame"], step["end_frame"]],
                    separators=(",", ":"),
                ),
                "track_id": json.dumps(
                    [step["source_id"], step["track_id"]], separators=(",", ":")
                ),
                "source_track_id": step["track_id"],
                "sample_start_utc": step["start_utc"],
                "sample_end_utc": step["end_utc"],
                "speed_mps": step["speed_mps"],
                "speed_valid": True,
                "calibration_id": step["calibration_id"],
                "calibration_valid": True,
                "camera_epoch": step["camera_epoch"],
                "image_zone": step["image_zone"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calculate_calibrated_speeds(db_path: str, config_path: str, output_dir: str) -> Path:
    """Read actual SQLite tracks and atomically publish a fresh immutable v4 bundle.

    This route never writes to the source database or historical speed/emission
    rows. Evidence identities are checked, not treated as held-out acceptance.
    """
    code_identity = _capture_code_identity()
    database_path = Path(db_path).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    destination = _destination(output_dir)
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != "production_speed_v4":
        raise ValueError("configuration must use production_speed_v4")
    maximum = _positive(config.get("maximum_step_seconds"), "maximum_step_seconds")
    calibrations = config.get("calibrations")
    sources = config.get("sources")
    if (
        not isinstance(calibrations, list)
        or not calibrations
        or not isinstance(sources, list)
        or not sources
    ):
        raise ValueError("configuration requires calibration and source declarations")
    geometries = {}
    for item in calibrations:
        geometry = _geometry(item, config_file.parent)
        key = (
            _identifier(item, "site"),
            _identifier(item, "camera_epoch"),
            _identifier(item, "calibration_id"),
        )
        if key in geometries:
            raise ValueError("duplicate calibration identity")
        geometries[key] = geometry
    source_keys: set[int] = set()
    source_ids: set[str] = set()
    input_receipts: list[dict[str, Any]] = []
    all_steps, all_rejected, all_tracks, all_timings = [], [], [], []
    connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        for source in sources:
            video_id = _integer(source.get("video_id"), "video_id", 1)
            source_id = _identifier(source, "source_id")
            if video_id in source_keys or source_id in source_ids:
                raise ValueError("duplicate source/video identity")
            source_keys.add(video_id)
            source_ids.add(source_id)
            key = (
                _identifier(source, "site"),
                _identifier(source, "camera_epoch"),
                _identifier(source, "calibration_id"),
            )
            if key not in geometries:
                raise ValueError("source camera/site/calibration identity mismatch")
            video = connection.execute(
                "SELECT * FROM video_master WHERE video_id = ?", (video_id,)
            ).fetchone()
            if video is None or video["intersection_id"] != source["site"]:
                raise ValueError("source database site/video identity mismatch")
            for dimension in ("width", "height"):
                if _integer(video[dimension], f"database {dimension}", 1) != _integer(
                    source.get(dimension), dimension, 1
                ):
                    raise ValueError(f"source/database {dimension} resolution mismatch")
            if _utc(video["recording_datetime_utc"], "database recording time") != _utc(
                source.get("recording_start_utc"), "recording_start_utc"
            ):
                raise ValueError("source/database recording time mismatch")
            media = _evidence(source, "source", config_file.parent)
            if Path(video["video_path"]).resolve() != Path(media["path"]):
                raise ValueError("source path does not identify the selected database video")
            timing_evidence = _evidence(source, "timing", config_file.parent)
            _identifier(source, "timing_source_id")
            if source.get("timing_basis") != "source_elapsed_seconds":
                raise ValueError("timing_basis must explicitly be source_elapsed_seconds")
            timing = pd.read_csv(timing_evidence["path"])
            tracks = pd.read_sql_query(
                "SELECT track_id, frame_id, class_id, x_centre, y_centre, width, height FROM vehicle_tracks WHERE video_id = ? ORDER BY track_id, frame_id",
                connection,
                params=[video_id],
            )
            steps, rejected = calculate_video_steps(
                tracks,
                timing,
                source=source,
                calibration=geometries[key]["declaration"],
                maximum_step_seconds=maximum,
                evidence_base=config_file.parent,
            )
            all_steps.append(steps)
            all_rejected.append(rejected)
            all_tracks.append(tracks.assign(video_id=video_id, source_id=source_id))
            all_timings.append(timing.assign(video_id=video_id, source_id=source_id))
            input_receipts.append(
                {
                    "source_declaration": source,
                    "video_metadata": dict(video),
                    "source": media,
                    "timing": timing_evidence,
                    "geometry_evidence": geometries[key]["evidence"],
                }
            )
    finally:
        connection.close()
    steps = pd.concat(all_steps, ignore_index=True)
    rejected = pd.concat(all_rejected, ignore_index=True)
    seconds, profiles = second_speed_exports(steps)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Numeric source copies are the exact selected database observations used above,
    # including WAL-visible rows, rather than a potentially incoherent file hash.
    outputs = {
        "speed_steps_v4.csv": steps,
        "speed_seconds_v4.csv": seconds,
        "speed_profiles_v4.csv": profiles,
        "speed_samples_v4.csv": hourly_speed_samples(steps),
        "speed_exclusions_v4.csv": rejected,
        "source_tracks_v4.csv": pd.concat(all_tracks, ignore_index=True),
        "source_timing_v4.csv": pd.concat(all_timings, ignore_index=True),
    }
    contents = {filename: _csv_bytes(frame) for filename, frame in outputs.items()}
    contents["production_config_v4.json"] = config_bytes
    receipt = {
        "schema_version": "production_speed_receipt_v4",
        "database_access": "read_only",
        "source_database_path": str(database_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code_revision": revision,
        "code_sha256": code_identity,
        "config_directory": str(config_file.parent),
        "command": [
            "python",
            "-m",
            "vehicle_emission.calculate_speeds",
            "--db-path",
            str(database_path),
            "--config",
            str(config_file),
            "--output-dir",
            str(destination),
        ],
        "sources": input_receipts,
        "maximum_step_seconds": maximum,
        "ground_contact": "bounding_box_bottom_centre",
        "gap_policy": "consecutive_frame_ids_only",
        "class_assignment": "majority_observed_class_ties_lowest_id",
        "second_speed": "duration_weighted_interval_speed",
        "profile_policy": "fully_covered_UTC_seconds_only_contiguous_zone_runs_no_interpolation",
        "held_out_calibration_acceptance": "not_assessed_by_this_calculator",
        "calibration_valid_field": "geometry_and_zone_calculation_only_not_independent_reference_acceptance",
        "step_rows": len(steps),
        "second_rows": len(seconds),
        "profile_rows": len(profiles),
        "exclusions": rejected.reason.value_counts().sort_index().to_dict(),
        "outputs": {
            filename: hashlib.sha256(data).hexdigest() for filename, data in contents.items()
        },
    }

    def verify() -> None:
        for item in input_receipts:
            for role in ("source", "timing", "geometry_evidence"):
                if _sha256(Path(item[role]["path"])) != item[role]["sha256"]:
                    raise ValueError(f"{role} evidence changed during calculation")
        if config_file.read_bytes() != config_bytes:
            raise ValueError("configuration changed during calculation")
        if _code_hashes() != code_identity:
            raise ValueError("production code changed during calculation")

    contents["production_receipt_v4.json"] = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _publish(destination, contents, verify)


def replay_speed_bundle(
    bundle_dir: str, output_dir: str, *, reverify_source_evidence: bool = False
) -> Path:
    """Recompute all speeds from receipt-bound numeric copies, without SQLite.

    Numerical control points/scales, timing and tracks are self-contained.
    Raw media/geometry provenance is recorded, not reverified, unless requested.
    """
    code_identity = _capture_code_identity()
    destination = _destination(output_dir)
    bundle = Path(bundle_dir).resolve(strict=True)
    receipt_path = bundle / "production_receipt_v4.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema_version") != "production_speed_receipt_v4":
        raise ValueError("invalid production receipt schema")
    if receipt.get("code_sha256") != code_identity:
        raise ValueError("production code identity changed; replay requires the recorded code")
    expected_files = {
        "speed_steps_v4.csv",
        "speed_seconds_v4.csv",
        "speed_profiles_v4.csv",
        "speed_samples_v4.csv",
        "speed_exclusions_v4.csv",
        "source_tracks_v4.csv",
        "source_timing_v4.csv",
        "production_config_v4.json",
    }
    if set(receipt.get("outputs", {})) != expected_files:
        raise ValueError("receipt must bind exactly the production source/output files")
    contents = {}
    for filename in sorted(expected_files):
        data = (bundle / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["outputs"][filename]:
            raise ValueError(f"receipt-bound file hash mismatch: {filename}")
        contents[filename] = data
    config = json.loads(contents["production_config_v4.json"])
    if (
        hashlib.sha256(contents["production_config_v4.json"]).hexdigest()
        != receipt["config_sha256"]
    ):
        raise ValueError("configuration hash mismatch")
    if (
        config["sources"] != [item["source_declaration"] for item in receipt["sources"]]
        or config["maximum_step_seconds"] != receipt["maximum_step_seconds"]
    ):
        raise ValueError("receipt/config source or gap policy mismatch")
    base = Path(receipt["config_directory"])
    geometries = {
        (item["site"], item["camera_epoch"], item["calibration_id"]): item
        for item in config["calibrations"]
    }
    tracks = pd.read_csv(
        bundle / "source_tracks_v4.csv", float_precision="round_trip", converters={"source_id": str}
    )
    timing = pd.read_csv(
        bundle / "source_timing_v4.csv", float_precision="round_trip", converters={"source_id": str}
    )
    step_frames, rejection_frames = [], []
    for item in receipt["sources"]:
        source = item["source_declaration"]
        calibration = geometries[(source["site"], source["camera_epoch"], source["calibration_id"])]
        for role in ("source", "timing"):
            checked = _evidence(source, role, base, verify=reverify_source_evidence)
            if checked != item[role]:
                raise ValueError(f"receipt/config {role} evidence identity mismatch")
        if (
            _evidence(calibration, "geometry_evidence", base, verify=reverify_source_evidence)
            != item["geometry_evidence"]
        ):
            raise ValueError("receipt/config geometry evidence identity mismatch")
        source_tracks = tracks[
            tracks.video_id.eq(source["video_id"]) & tracks.source_id.eq(source["source_id"])
        ]
        source_timing = timing[
            timing.video_id.eq(source["video_id"]) & timing.source_id.eq(source["source_id"])
        ]
        steps, rejected = calculate_video_steps(
            source_tracks,
            source_timing,
            source=source,
            calibration=calibration,
            maximum_step_seconds=config["maximum_step_seconds"],
            evidence_base=base,
            verify_geometry_evidence=reverify_source_evidence,
        )
        step_frames.append(steps)
        rejection_frames.append(rejected)
    steps = pd.concat(step_frames, ignore_index=True)
    rejected = pd.concat(rejection_frames, ignore_index=True)
    seconds, profiles = second_speed_exports(steps)
    computed = {
        "speed_steps_v4.csv": steps,
        "speed_seconds_v4.csv": seconds,
        "speed_profiles_v4.csv": profiles,
        "speed_samples_v4.csv": hourly_speed_samples(steps),
        "speed_exclusions_v4.csv": rejected,
    }
    for filename, frame in computed.items():
        if _csv_bytes(frame) != contents[filename]:
            raise ValueError(f"numeric replay does not reproduce {filename}")

    def verify() -> None:
        if _code_hashes() != code_identity or receipt_path.read_bytes() != receipt_bytes:
            raise ValueError("code or receipt changed during replay")
        for filename in expected_files:
            if (bundle / filename).read_bytes() != contents[filename]:
                raise ValueError(f"receipt-bound file changed during replay: {filename}")
        if reverify_source_evidence:
            for item in receipt["sources"]:
                for role in ("source", "timing", "geometry_evidence"):
                    if _sha256(Path(item[role]["path"])) != item[role]["sha256"]:
                        raise ValueError(f"{role} evidence changed during replay")

    receipt["replayed_from_receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    receipt["replay_scope"] = "numeric_copy_recomputation"
    receipt["replay_external_evidence_verification"] = (
        "reverified_sha256"
        if reverify_source_evidence
        else "raw_asset_identity_recorded_not_reverified"
    )
    receipt["replay_command"] = [
        "python",
        "-m",
        "vehicle_emission.calculate_speeds",
        "--replay-bundle",
        str(bundle),
        "--output-dir",
        str(destination),
    ]
    if reverify_source_evidence:
        receipt["replay_command"].append("--reverify-source-evidence")
    contents["production_receipt_v4.json"] = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _publish(destination, contents, verify)


_IMPORTED_CODE_IDENTITY = _code_hashes()
