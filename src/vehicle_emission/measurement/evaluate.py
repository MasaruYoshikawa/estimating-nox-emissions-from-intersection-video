"""Calculate reference-based measurement metrics into a fresh build directory."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_emission.inference.manifest_validation import stable_file_bytes
from vehicle_emission.measurement.metrics import (
    VIDEO_KEY,
    evaluate_entries,
    evaluate_objects,
    evaluate_tracking,
    validate_frames,
)
from vehicle_emission.measurement.speed import evaluate_speed_pairs, validate_calibration_segments

INPUT_NAMES = {
    "frames",
    "reference_objects",
    "predicted_objects",
    "event_windows",
    "reference_events",
    "predicted_events",
    "speed_references",
    "speed_predictions",
    "calibration",
}
PARAMETER_NAMES = {
    "iou_threshold",
    "entry_tolerance_seconds",
    "bootstrap_resamples",
    "bootstrap_seed",
    "confidence_level",
}
CRITERIA_NAMES = {
    "minimum_source_videos",
    "vehicle_class_min_f1",
    "entry_min_f1",
    "tracking_min_mota",
    "speed_max_mae_mps",
    "speed_max_abs_bias_mps",
    "speed_max_mape_pct",
    "speed_max_unpaired_fraction",
}


def sha256_file(path: Path) -> str:
    """Hash exact bytes without loading source assets into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("input paths must be canonical repository-relative paths")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("input paths must be canonical repository-relative paths")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"required measurement input is unavailable: {value}")
    return resolved


def _criteria_status(
    criteria: dict[str, Any] | None,
    objects: pd.DataFrame,
    entries: pd.DataFrame,
    tracking: pd.DataFrame,
    speed: pd.DataFrame,
) -> dict[str, Any]:
    if criteria is None:
        return {"status": "not_assessed", "criteria": {}, "site_results": []}
    if not isinstance(criteria, dict) or set(criteria) != CRITERIA_NAMES:
        raise ValueError(f"acceptance_criteria must declare exactly {sorted(CRITERIA_NAMES)}")
    if (
        isinstance(criteria["minimum_source_videos"], bool)
        or not isinstance(criteria["minimum_source_videos"], int)
        or criteria["minimum_source_videos"] < 2
    ):
        raise ValueError("minimum_source_videos must be an integer >= 2")
    for name, value in criteria.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"acceptance criterion {name} must be finite numeric")
        if (
            name in {"vehicle_class_min_f1", "entry_min_f1", "speed_max_unpaired_fraction"}
            and not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must lie in [0,1]")
        if name.startswith("speed_max_") and value < 0:
            raise ValueError(f"{name} must be nonnegative")
        if name == "tracking_min_mota" and value > 1:
            raise ValueError("tracking_min_mota cannot exceed 1")
    sites = sorted(objects.site.unique())
    reports: list[dict[str, Any]] = []
    for site in sites:
        classes = objects[objects.site == site]
        entry = entries[entries.site == site].iloc[0]
        track = tracking[tracking.site == site].iloc[0]
        velocities = speed[speed.site == site]
        complete_fraction = velocities.support / velocities.n_reference_rows.replace(
            0, float("nan")
        )
        checks = {
            "vehicle_classes": bool(
                (classes.f1 >= criteria["vehicle_class_min_f1"]).all()
                and (classes.class_source_video_support >= criteria["minimum_source_videos"]).all()
            ),
            "entry_counts": bool(
                entry.f1 >= criteria["entry_min_f1"]
                and entry.n_source_videos >= criteria["minimum_source_videos"]
            ),
            "tracking": bool(
                track.mota >= criteria["tracking_min_mota"]
                and track.n_source_videos >= criteria["minimum_source_videos"]
            ),
            "speed": bool(
                (velocities.mae_mps <= criteria["speed_max_mae_mps"]).all()
                and (velocities.bias_mps.abs() <= criteria["speed_max_abs_bias_mps"]).all()
                and (velocities.mape_pct <= criteria["speed_max_mape_pct"]).all()
                and (velocities.n_valid_source_videos >= criteria["minimum_source_videos"]).all()
                and (1 - complete_fraction <= criteria["speed_max_unpaired_fraction"]).all()
            ),
        }
        reports.append(
            {
                "site": site,
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
            }
        )
    return {
        "status": "passed" if all(row["status"] == "passed" for row in reports) else "failed",
        "criteria": criteria,
        "site_results": reports,
    }


def _calculate_measurements(
    config_path: Path,
    root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not config_path.resolve().is_relative_to(root):
        raise ValueError("measurement config must be inside repository root")
    config_bytes, _ = stable_file_bytes(config_path, display_path=str(config_path))
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    configuration = json.loads(config_bytes)
    required = {"schema_version", "inputs", "parameters", "fit_source_video_sha256"}
    if (
        not isinstance(configuration, dict)
        or not required <= set(configuration)
        or set(configuration) - required - {"acceptance_criteria"}
    ):
        raise ValueError("measurement configuration has missing/unknown keys")
    if configuration["schema_version"] != "measurement_validation_config_v4":
        raise ValueError("expected measurement_validation_config_v4")
    declarations = configuration["inputs"]
    parameters = configuration["parameters"]
    if not isinstance(declarations, dict) or set(declarations) != INPUT_NAMES:
        raise ValueError("configuration must declare all measurement inputs")
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_NAMES:
        raise ValueError("configuration must explicitly declare matching/bootstrap parameters")
    paths: dict[str, Path] = {}
    input_bytes: dict[str, bytes] = {}
    identities: dict[str, dict[str, str]] = {}
    for name, declaration in declarations.items():
        if not isinstance(declaration, dict) or set(declaration) != {"path", "sha256"}:
            raise ValueError(f"input {name} requires path and SHA-256")
        path = _relative_file(root, declaration["path"])
        content, _ = stable_file_bytes(path, display_path=declaration["path"])
        digest = hashlib.sha256(content).hexdigest()
        if digest != declaration["sha256"]:
            raise ValueError(f"measurement input hash mismatch: {name}")
        paths[name] = path
        input_bytes[name] = content
        identities[name] = {"path": path.relative_to(root).as_posix(), "sha256": digest}
    csv_types = {
        "source_video_id": str,
        "object_id": str,
        "event_id": str,
        "reference_id": str,
        "vehicle_id": str,
    }
    frames = validate_frames(
        pd.read_csv(
            io.BytesIO(input_bytes["frames"]), dtype=csv_types, float_precision="round_trip"
        )
    )
    tables = {
        name: pd.read_csv(io.BytesIO(content), dtype=csv_types, float_precision="round_trip")
        for name, content in input_bytes.items()
        if name not in {"frames", "calibration"}
    }
    calibration = json.loads(input_bytes["calibration"])
    segments = validate_calibration_segments(calibration)
    sites = set(frames.site)
    if {segment["site"] for segment in segments} != sites:
        raise ValueError("calibration site universe differs from frame universe")
    for name in ("event_windows", "speed_references"):
        if set(tables[name].site) != sites:
            raise ValueError(f"{name} site universe differs from frame universe")
    source_identity = frames[VIDEO_KEY + ["source_video_sha256"]].drop_duplicates()
    for name in ("event_windows", "speed_references", "speed_predictions"):
        joined = tables[name].merge(
            source_identity, on=VIDEO_KEY, how="left", suffixes=("", "_frames")
        )
        if not joined.source_video_sha256.eq(joined.source_video_sha256_frames).all():
            raise ValueError(f"{name} contains source identities outside frame universe")
    fit_hashes = configuration["fit_source_video_sha256"]
    if not isinstance(fit_hashes, list) or frames.source_video_sha256.isin(fit_hashes).any():
        raise ValueError("measurement test videos overlap fitting videos")
    bootstrap_parameters = {
        name: parameters[name]
        for name in ("bootstrap_resamples", "bootstrap_seed", "confidence_level")
    }
    objects, confusion = evaluate_objects(
        frames,
        tables["reference_objects"],
        tables["predicted_objects"],
        iou_threshold=parameters["iou_threshold"],
        **bootstrap_parameters,
    )
    tracking, track_details = evaluate_tracking(
        frames,
        tables["reference_objects"],
        tables["predicted_objects"],
        iou_threshold=parameters["iou_threshold"],
    )
    entries, entry_details = evaluate_entries(
        tables["event_windows"],
        tables["reference_events"],
        tables["predicted_events"],
        tolerance_seconds=parameters["entry_tolerance_seconds"],
    )
    speed, speed_details = evaluate_speed_pairs(
        tables["speed_references"],
        tables["speed_predictions"],
        calibration=calibration,
        fit_source_video_sha256=fit_hashes,
        **bootstrap_parameters,
    )
    acceptance = _criteria_status(
        configuration.get("acceptance_criteria"), objects, entries, tracking, speed
    )
    output_tables = {
        "vehicle_classes": objects,
        "class_confusion": confusion,
        "entry_counts": entries,
        "entry_details": entry_details,
        "tracking": tracking,
        "tracking_details": track_details,
        "speed": speed,
        "speed_details": speed_details,
    }
    site_identities: list[dict[str, Any]] = []
    for site, group in frames.groupby("site", sort=True):
        if (
            group.detector_checkpoint_sha256.nunique() != 1
            or group.tracker_config_sha256.nunique() != 1
        ):
            raise ValueError("each site requires one declared frozen detector/tracker")
        video_ids = (
            group[VIDEO_KEY + ["source_video_sha256"]]
            .drop_duplicates()
            .sort_values("source_video_id")
        )
        site_identities.append(
            {
                "site": site,
                "detector_checkpoint_sha256": group.detector_checkpoint_sha256.iloc[0],
                "tracker_config_sha256": group.tracker_config_sha256.iloc[0],
                "source_video_ids": video_ids.source_video_id.tolist(),
                "source_video_sha256": video_ids.source_video_sha256.tolist(),
                "calibration_segments": [
                    {key: value for key, value in segment.items() if key != "site"}
                    for segment in segments
                    if segment["site"] == site
                ],
            }
        )
    if (
        frames.detector_checkpoint_sha256.nunique() != 1
        or frames.tracker_config_sha256.nunique() != 1
    ):
        raise ValueError("sites must share the frozen detector/tracker identities")
    if sha256_file(config_path) != config_digest or any(
        sha256_file(paths[name]) != record["sha256"] for name, record in identities.items()
    ):
        raise ValueError("measurement input changed during calculation")
    source_root = Path(__file__).resolve().parents[3]
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("speed.py"),
        source_root / "src/vehicle_emission/detection/evaluate.py",
    ]
    code_hashes = {
        path.resolve().relative_to(source_root).as_posix(): sha256_file(path) for path in code_paths
    }
    receipt = {
        "schema_version": "measurement_validation_receipt_v4",
        "config": {
            "path": config_path.resolve().relative_to(root).as_posix(),
            "sha256": config_digest,
        },
        "inputs": identities,
        "code_sha256": code_hashes,
        "measurement_tasks": ["vehicle_classes", "entry_counts", "tracking", "speed"],
        "site_identities": site_identities,
        "acceptance": acceptance,
        "geometry_evidence_status": "identity_recorded_production_use_not_verified",
    }
    return output_tables, receipt


def evaluate_measurements(
    *,
    config_path: Path,
    output_directory: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Calculate all metrics and bind sources; a receipt is not independent review."""
    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    output = output_directory.resolve()
    if (
        (root / "build").is_symlink()
        or output_directory.is_symlink()
        or not output.is_relative_to(root / "build")
        or output == root / "build"
        or output.exists()
    ):
        raise ValueError("measurement output must be a fresh directory below build/")
    ancestor = output_directory.absolute()
    while ancestor != root and ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise ValueError("measurement output ancestors cannot be symlinks")
        ancestor = ancestor.parent
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--", output.relative_to(root).as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        if tracked.stdout.strip():
            raise ValueError("measurement output contains committed artifacts")
    output_tables, receipt = _calculate_measurements(config_path, root)
    serialized = {
        f"{name}_v4.csv": table.to_csv(index=False).encode("utf-8")
        for name, table in output_tables.items()
    }
    output_identities: dict[str, dict[str, str]] = {}
    for name in output_tables:
        path = output / f"{name}_v4.csv"
        output_identities[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(serialized[path.name]).hexdigest(),
        }
    receipt["outputs"] = output_identities
    serialized["measurement_receipt_v4.json"] = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".measurement-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, content in serialized.items():
            with (stage / name).open("xb") as handle:
                handle.write(content)
        output.mkdir(exist_ok=False)
        published: list[Path] = []
        try:
            for name in serialized:
                destination = output / name
                os.link(stage / name, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                destination.unlink()
            output.rmdir()
            raise
    return receipt


def validate_measurement_receipt(receipt: dict[str, Any], repository_root: Path) -> None:
    """Read-only replay verifies actual calculations, not just reported thresholds."""
    root = repository_root.resolve()
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "measurement_validation_receipt_v4"
    ):
        raise ValueError("expected measurement_validation_receipt_v4")
    config = receipt.get("config", {})
    if not isinstance(config, dict) or set(config) != {"path", "sha256"}:
        raise ValueError("measurement receipt requires a bound config")
    config_path = _relative_file(root, config["path"])
    if sha256_file(config_path) != config["sha256"]:
        raise ValueError("measurement config hash mismatch")
    tables, replay = _calculate_measurements(config_path, root)
    if set(receipt) != set(replay) | {"outputs"}:
        raise ValueError("measurement receipt has missing/unknown fields")
    for key, value in replay.items():
        if receipt.get(key) != value:
            raise ValueError(f"measurement receipt replay disagrees on {key}")
    outputs = receipt["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(tables):
        raise ValueError("measurement receipt must bind all calculated output tables")
    for name, table in tables.items():
        declaration = outputs[name]
        if not isinstance(declaration, dict) or set(declaration) != {"path", "sha256"}:
            raise ValueError(f"measurement output {name} requires path and hash")
        path = _relative_file(root, declaration["path"])
        if (
            sha256_file(path) != declaration["sha256"]
            or hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()
            != declaration["sha256"]
        ):
            raise ValueError(f"measurement output calculation/hash mismatch: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_measurements(config_path=arguments.config, output_directory=arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
