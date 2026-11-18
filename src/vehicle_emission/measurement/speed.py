"""Paired independent-reference speed errors and source-video bootstrap intervals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from vehicle_emission.measurement.metrics import VIDEO_KEY, require_columns, require_ids

SPEED_KEY = VIDEO_KEY + ["reference_id"]
SPEED_REFERENCE_COLUMNS = SPEED_KEY + [
    "source_video_sha256",
    "vehicle_id",
    "split_id",
    "reference_status",
    "reference_method",
    "reference_source_id",
    "reference_units",
    "reference_speed",
    "measured_distance_m",
    "elapsed_seconds",
    "direction",
    "image_zone",
    "camera_epoch",
    "calibration_id",
]
SPEED_PREDICTION_COLUMNS = SPEED_KEY + [
    "source_video_sha256",
    "vehicle_id",
    "predicted_speed",
    "prediction_units",
    "prediction_status",
    "direction",
    "image_zone",
    "camera_epoch",
    "calibration_id",
]


def validate_calibration_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Require explicit segment and geometry evidence identities, never a guessed scale."""
    if data.get("schema_version") != "speed_reference_calibration_v4":
        raise ValueError("calibration must use speed_reference_calibration_v4")
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("calibration must declare segments")
    identities: set[tuple[str, str, str]] = set()
    required = {
        "site",
        "camera_epoch",
        "calibration_id",
        "route",
        "valid_image_zones",
        "geometry_source_id",
        "geometry_sha256",
    }
    for segment in segments:
        if not isinstance(segment, dict) or not required <= set(segment):
            raise ValueError("calibration segment lacks geometry/identity fields")
        if segment["route"] not in {"homography", "bounded_zones"}:
            raise ValueError("calibration route must be perspective-aware")
        if any(
            not isinstance(segment[key], str) or not segment[key].strip()
            for key in ("site", "camera_epoch", "calibration_id", "geometry_source_id")
        ):
            raise ValueError("calibration identities must be nonempty strings")
        digest = segment["geometry_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("geometry_sha256 must be SHA-256")
        zones = segment["valid_image_zones"]
        if (
            not isinstance(zones, list)
            or not zones
            or any(not isinstance(zone, str) or not zone.strip() for zone in zones)
            or len(set(zones)) != len(zones)
        ):
            raise ValueError("valid_image_zones must be distinct nonempty zone IDs")
        identity = (segment["site"], segment["camera_epoch"], segment["calibration_id"])
        if identity in identities:
            raise ValueError("duplicate calibration segment identity")
        identities.add(identity)
    return segments


def _speed_mps(value: Any, units: str) -> float:
    if units not in {"m/s", "km/h"}:
        raise ValueError("speed units must be explicit m/s or km/h")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if units == "m/s" else numeric / 3.6


def _statistics(pairs: pd.DataFrame) -> dict[str, float]:
    if pairs.empty:
        return {"mae_mps": math.nan, "bias_mps": math.nan, "mape_pct": math.nan}
    errors = pairs.predicted_mps.to_numpy(dtype=float) - pairs.reference_mps.to_numpy(dtype=float)
    nonzero = pairs.reference_mps.to_numpy(dtype=float) > 0
    return {
        "mae_mps": float(np.mean(np.abs(errors))),
        "bias_mps": float(np.mean(errors)),
        "mape_pct": (
            float(
                np.mean(
                    np.abs(errors[nonzero]) / pairs.reference_mps.to_numpy(dtype=float)[nonzero]
                )
                * 100
            )
            if nonzero.any()
            else math.nan
        ),
    }


def evaluate_speed_pairs(
    references: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    calibration: dict[str, Any],
    fit_source_video_sha256: list[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve unmatched/invalid references and calculate paired errors, not summaries."""
    segments = validate_calibration_segments(calibration)
    require_columns(references, SPEED_REFERENCE_COLUMNS, "speed references")
    require_columns(predictions, SPEED_PREDICTION_COLUMNS, "speed predictions")
    if references.empty:
        raise ValueError("speed references must declare a nonempty evaluation universe")
    if not isinstance(fit_source_video_sha256, list) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in fit_source_video_sha256
    ):
        raise ValueError(
            "fit_source_video_sha256 must explicitly declare training/calibration-fit videos"
        )
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 100
    ):
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0,1)")
    calibration_lookup = {
        (row["site"], row["camera_epoch"], row["calibration_id"]): row for row in segments
    }
    identity_columns = VIDEO_KEY + [
        "reference_id",
        "source_video_sha256",
        "vehicle_id",
        "direction",
        "image_zone",
        "camera_epoch",
        "calibration_id",
    ]
    for name, frame in (("speed references", references), ("speed predictions", predictions)):
        require_ids(frame, identity_columns, name)
        if frame.duplicated(SPEED_KEY).any():
            raise ValueError(f"{name} has duplicate reference IDs within a video")
        if not frame.source_video_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"{name} requires source-video hashes")
        if frame.source_video_sha256.isin(fit_source_video_sha256).any():
            raise ValueError(
                "held-out speed references overlap training/calibration-fit source videos"
            )
        for row in frame.itertuples():
            segment = calibration_lookup.get((row.site, row.camera_epoch, row.calibration_id))
            if segment is None:
                raise ValueError("speed observation uses an undeclared calibration segment")
            if row.image_zone not in segment["valid_image_zones"]:
                raise ValueError("speed observation is outside declared valid image zones")
    combined = pd.concat(
        [references[identity_columns], predictions[identity_columns]], ignore_index=True
    )
    for _, group in combined.groupby(VIDEO_KEY, sort=True):
        if (
            group.source_video_sha256.nunique() != 1
            or group.camera_epoch.nunique() != 1
            or group.calibration_id.nunique() != 1
        ):
            raise ValueError("speed source video has conflicting identity/calibration")
    video_identity = combined[VIDEO_KEY + ["source_video_sha256"]].drop_duplicates()
    if video_identity.duplicated("source_video_sha256").any():
        raise ValueError("speed source video hash has reused IDs/sites")
    if not references.split_id.eq("test").all():
        raise ValueError("speed references must be held-out test observations")
    if not references.reference_status.isin({"actual", "missing", "invalid"}).all():
        raise ValueError("derived/synthetic speed references cannot validate measurements")
    if not references.reference_method.isin({"independent_speed", "distance_time"}).all():
        raise ValueError("speed references require independent_speed or distance_time")
    require_ids(references, ["reference_source_id"], "speed references")
    if not predictions.prediction_status.isin({"valid", "missing", "invalid", "out_of_zone"}).all():
        raise ValueError("unknown prediction_status")
    merged = references.merge(
        predictions,
        on=SPEED_KEY,
        how="outer",
        suffixes=("_reference", "_prediction"),
        indicator=True,
        validate="one_to_one",
    )
    for column in (
        "source_video_sha256",
        "vehicle_id",
        "direction",
        "image_zone",
        "camera_epoch",
        "calibration_id",
    ):
        both = merged._merge.eq("both")
        if (
            not merged.loc[both, f"{column}_reference"]
            .eq(merged.loc[both, f"{column}_prediction"])
            .all()
        ):
            raise ValueError(f"paired speed observations disagree on {column}")
        merged[column] = merged[f"{column}_reference"].combine_first(merged[f"{column}_prediction"])
    reference_values: list[float] = []
    prediction_values: list[float] = []
    statuses: list[str] = []
    for row in merged.itertuples():
        ref, pred = math.nan, math.nan
        if row.reference_status == "actual":
            if row.reference_method == "independent_speed":
                ref = _speed_mps(row.reference_speed, row.reference_units)
            else:
                if row.reference_units != "m,s":
                    raise ValueError("distance_time references require m,s units")
                try:
                    distance, elapsed = float(row.measured_distance_m), float(row.elapsed_seconds)
                    ref = (
                        distance / elapsed
                        if math.isfinite(distance)
                        and math.isfinite(elapsed)
                        and distance >= 0
                        and elapsed > 0
                        else math.nan
                    )
                except (TypeError, ValueError):
                    ref = math.nan
        if row.prediction_status == "valid":
            pred = _speed_mps(row.predicted_speed, row.prediction_units)
        if pd.isna(row.reference_status):
            status = "unmatched_prediction"
        elif row.reference_status == "missing":
            status = "missing_reference"
        elif row.reference_status == "invalid" or not math.isfinite(ref) or ref < 0:
            status = "invalid_reference"
        elif pd.isna(row.prediction_status):
            status = "unmatched_reference"
        elif row.prediction_status != "valid" or not math.isfinite(pred) or pred < 0:
            status = "invalid_prediction"
        else:
            status = "paired"
        reference_values.append(ref)
        prediction_values.append(pred)
        statuses.append(status)
    merged["reference_mps"], merged["predicted_mps"], merged["pair_status"] = (
        reference_values,
        prediction_values,
        statuses,
    )
    merged["signed_error_mps"] = np.where(
        merged.pair_status.eq("paired"), merged.predicted_mps - merged.reference_mps, np.nan
    )
    detail = (
        merged[
            SPEED_KEY
            + [
                "source_video_sha256",
                "vehicle_id",
                "direction",
                "image_zone",
                "camera_epoch",
                "calibration_id",
                "reference_mps",
                "predicted_mps",
                "pair_status",
                "signed_error_mps",
            ]
        ]
        .sort_values(SPEED_KEY)
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, Any]] = []
    groups = [
        (str(site), "__all__", "__all__", "__all__", "__all__", group)
        for site, group in detail.groupby("site", sort=True)
    ]
    for segment in sorted(
        segments, key=lambda value: (value["site"], value["camera_epoch"], value["calibration_id"])
    ):
        for zone in sorted(segment["valid_image_zones"]):
            selected = detail[
                detail.site.eq(segment["site"])
                & detail.camera_epoch.eq(segment["camera_epoch"])
                & detail.calibration_id.eq(segment["calibration_id"])
                & detail.image_zone.eq(zone)
            ]
            identity = (segment["site"], zone, segment["camera_epoch"], segment["calibration_id"])
            groups.append((identity[0], "__all__", *identity[1:], selected))
            groups.extend(
                (identity[0], str(direction), *identity[1:], group)
                for direction, group in selected.groupby("direction", sort=True)
            )
    for site, direction, image_zone, camera_epoch, calibration_id, group in groups:
        pairs = group[group.pair_status.eq("paired")]
        videos = sorted(group.source_video_id.unique())
        valid_videos = pairs.source_video_id.nunique()
        stats = _statistics(pairs)
        bootstrap: dict[str, list[float]] = {metric: [] for metric in stats}
        if valid_videos >= 2:
            clusters = {video: pairs[pairs.source_video_id == video] for video in videos}
            for _ in range(bootstrap_resamples):
                sampled = rng.choice(videos, size=len(videos), replace=True)
                draw_statistics = _statistics(
                    pd.concat([clusters[video] for video in sampled], ignore_index=True)
                )
                for metric, value in draw_statistics.items():
                    if math.isfinite(value):
                        bootstrap[metric].append(value)
        summary: dict[str, Any] = {
            "site": site,
            "direction": direction,
            "image_zone": image_zone,
            "camera_epoch": camera_epoch,
            "calibration_id": calibration_id,
            "n_reference_rows": int(group.pair_status.ne("unmatched_prediction").sum()),
            "support": len(pairs),
            "n_source_videos": len(videos),
            "n_valid_source_videos": valid_videos,
            "zero_reference_pairs": int(pairs.reference_mps.eq(0).sum()),
            "mape_support": int(pairs.reference_mps.gt(0).sum()),
            **stats,
            "bootstrap_unit": "source_video_id",
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
        }
        for status in (
            "missing_reference",
            "invalid_reference",
            "unmatched_reference",
            "invalid_prediction",
            "unmatched_prediction",
        ):
            summary[f"n_{status}"] = int(group.pair_status.eq(status).sum())
        for metric, values in bootstrap.items():
            estimable = len(values) >= math.ceil(0.9 * bootstrap_resamples)
            bounds = (
                np.quantile(values, [(1 - confidence_level) / 2, (1 + confidence_level) / 2])
                if estimable
                else [math.nan, math.nan]
            )
            summary[f"{metric}_ci_low"], summary[f"{metric}_ci_high"] = float(bounds[0]), float(
                bounds[1]
            )
            summary[f"{metric}_valid_resamples"] = len(values)
        summary["interval_status"] = (
            "estimated"
            if all(
                len(values) >= math.ceil(0.9 * bootstrap_resamples) for values in bootstrap.values()
            )
            else "not_estimable"
        )
        summaries.append(summary)
    return pd.DataFrame(summaries), detail
