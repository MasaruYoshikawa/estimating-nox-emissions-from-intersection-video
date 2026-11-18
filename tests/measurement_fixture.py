"""Controlled observation fixtures, not empirical measurement evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from vehicle_emission.measurement.metrics import (
    CLASSES,
    EVENT_COLUMNS,
    FRAME_COLUMNS,
    OBJECT_COLUMNS,
    WINDOW_COLUMNS,
)
from vehicle_emission.measurement.speed import SPEED_PREDICTION_COLUMNS, SPEED_REFERENCE_COLUMNS


def write_measurement_fixture(
    root: Path,
    *,
    sites: tuple[str, ...] = ("Sukiyabashi", "Shinjuku"),
    acceptance: bool = True,
    checkpoint_sha256: str = "1" * 64,
    tracker_sha256: str = "2" * 64,
) -> Path:
    """Write perfect two-video, four-class observations with explicit identities."""
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    frames, objects, windows, events, references, predictions, segments = [], [], [], [], [], [], []
    for site in sites:
        calibration_id = f"{site}-calibration"
        segments.append(
            {
                "site": site,
                "camera_epoch": "camera-epoch-1",
                "calibration_id": calibration_id,
                "route": "homography",
                "valid_image_zones": ["near"],
                "geometry_source_id": "independent-survey",
                "geometry_sha256": "3" * 64,
            }
        )
        for video in range(2):
            video_id = f"{site}-video-{video}"
            digest = hashlib.sha256(video_id.encode()).hexdigest()
            key = {"site": site, "source_video_id": video_id}
            frames.append(
                {
                    **key,
                    "frame_index": 0,
                    "frame_time_seconds": 0,
                    "source_video_sha256": digest,
                    "split_id": "test",
                    "annotation_status": "complete",
                    "detector_checkpoint_sha256": checkpoint_sha256,
                    "tracker_config_sha256": tracker_sha256,
                }
            )
            windows.append(
                {
                    **key,
                    "window_id": "window",
                    "line_id": "entry",
                    "direction": "north",
                    "start_seconds": 0,
                    "end_seconds": 60,
                    "source_video_sha256": digest,
                    "split_id": "test",
                    "annotation_status": "complete",
                }
            )
            for class_index, class_name in enumerate(CLASSES):
                objects.append(
                    {
                        **key,
                        "frame_index": 0,
                        "object_id": f"object-{class_index}",
                        "class_name": class_name,
                        "x1": class_index * 0.2,
                        "x2": class_index * 0.2 + 0.1,
                        "y1": 0.1,
                        "y2": 0.2,
                    }
                )
                events.append(
                    {
                        **key,
                        "event_id": f"event-{class_index}",
                        "line_id": "entry",
                        "direction": "north",
                        "event_time_seconds": 1 + class_index,
                    }
                )
            speed_key = {
                **key,
                "reference_id": "speed-reference",
                "source_video_sha256": digest,
                "vehicle_id": "object-1",
                "direction": "north",
                "image_zone": "near",
                "camera_epoch": "camera-epoch-1",
                "calibration_id": calibration_id,
            }
            references.append(
                {
                    **speed_key,
                    "split_id": "test",
                    "reference_status": "actual",
                    "reference_method": "independent_speed",
                    "reference_source_id": "independent-timing",
                    "reference_units": "m/s",
                    "reference_speed": 10 + video,
                    "measured_distance_m": None,
                    "elapsed_seconds": None,
                }
            )
            predictions.append(
                {
                    **speed_key,
                    "predicted_speed": 10 + video,
                    "prediction_units": "m/s",
                    "prediction_status": "valid",
                }
            )
    tables = {
        "frames": pd.DataFrame(frames, columns=FRAME_COLUMNS),
        "reference_objects": pd.DataFrame(objects, columns=OBJECT_COLUMNS),
        "predicted_objects": pd.DataFrame(objects, columns=OBJECT_COLUMNS),
        "event_windows": pd.DataFrame(windows, columns=WINDOW_COLUMNS),
        "reference_events": pd.DataFrame(events, columns=EVENT_COLUMNS),
        "predicted_events": pd.DataFrame(events, columns=EVENT_COLUMNS),
        "speed_references": pd.DataFrame(references, columns=SPEED_REFERENCE_COLUMNS),
        "speed_predictions": pd.DataFrame(predictions, columns=SPEED_PREDICTION_COLUMNS),
    }
    files = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        files[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    calibration = directory / "calibration.json"
    calibration.write_text(
        json.dumps({"schema_version": "speed_reference_calibration_v4", "segments": segments}),
        encoding="utf-8",
    )
    files["calibration"] = {
        "path": calibration.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
    }
    configuration = {
        "schema_version": "measurement_validation_config_v4",
        "inputs": files,
        "parameters": {
            "iou_threshold": 0.5,
            "entry_tolerance_seconds": 0.5,
            "bootstrap_resamples": 100,
            "bootstrap_seed": 7,
            "confidence_level": 0.95,
        },
        "fit_source_video_sha256": [],
    }
    if acceptance:
        configuration["acceptance_criteria"] = {
            "minimum_source_videos": 2,
            "vehicle_class_min_f1": 0.9,
            "entry_min_f1": 0.9,
            "tracking_min_mota": 0.9,
            "speed_max_mae_mps": 1.0,
            "speed_max_abs_bias_mps": 0.5,
            "speed_max_mape_pct": 10.0,
            "speed_max_unpaired_fraction": 0.0,
        }
    config_path = directory / "measurement_config.json"
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return config_path
