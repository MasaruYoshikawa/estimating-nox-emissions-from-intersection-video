"""Timestamp-preserving emission runs from calibrated production speed seconds.

Definitions follow the production-speed and emission-profile GLOSSARY entries.
Only complete one-second observations enter the retained model implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.speed.production import IDENTITY, SECOND_COLUMNS
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

TRACK_KEY = ["site", "source_id", "video_id", "track_id"]
RUN_COLUMNS = IDENTITY + [
    "image_zone",
    "run_number",
    "start_utc",
    "end_utc",
    "run_seconds",
    "distance_metres",
    "speed_mps",
    "acceleration_mps2",
    "vsp_kw_per_tonne",
    "movestar_nox_g",
    "continuous_nox_g",
]
SUPPORT_COLUMNS = IDENTITY + [
    "observed_rows",
    "complete_valid_seconds",
    "partial_rows",
    "missing_speed_rows",
    "omitted_second_bins",
    "runs",
    "movestar_nox_g",
    "continuous_nox_g",
]
EXCLUSION_COLUMNS = SECOND_COLUMNS + ["reason"]


@dataclass(frozen=True)
class TimestampedEmissionTables:
    """Calculated contiguous runs, retained support and excluded observations."""

    runs: pd.DataFrame
    support: pd.DataFrame
    exclusions: pd.DataFrame


def _prepare_seconds(seconds: pd.DataFrame) -> pd.DataFrame:
    if seconds.columns.duplicated().any() or not set(SECOND_COLUMNS) <= set(seconds):
        raise ValueError("seconds require unique calibrated production-second columns")
    if seconds.empty:
        raise ValueError("seconds require explicit timestamped observations")
    columns = SECOND_COLUMNS + (["profile_id"] if "profile_id" in seconds else [])
    rows = seconds[columns].copy()
    for column in ("site", "source_id", "camera_epoch", "calibration_id", "image_zone"):
        if not rows[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must identify every observation")
    if not rows.site.isin(["Sukiyabashi", "Shinjuku"]).all():
        raise ValueError("unsupported site")
    if (
        "profile_id" in rows
        and not rows.profile_id.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError("profile_id must identify every supplied profile observation")
    for column in ("video_id", "track_id", "class_id"):
        values = pd.to_numeric(rows[column], errors="raise")
        if (
            rows[column].map(lambda value: isinstance(value, bool | np.bool_)).any()
            or not (np.isfinite(values) & values.ge(0) & values.eq(np.floor(values))).all()
        ):
            raise ValueError(f"{column} must contain nonnegative integers")
        rows[column] = values.astype("int64")
    for class_id in rows.class_id.unique():
        emission_vehicle_type_for_project_class(int(class_id))
    try:
        times = rows.ts_utc.map(pd.Timestamp)
    except (ValueError, TypeError) as error:
        raise ValueError("timestamps require explicit UTC-second boundaries") from error
    if not times.map(
        lambda stamp: pd.notna(stamp)
        and stamp.tzinfo is not None
        and stamp.utcoffset().total_seconds() == 0
        and stamp.value % 1_000_000_000 == 0
    ).all():
        raise ValueError("timestamps require explicit UTC-second boundaries")
    rows["ts_utc"] = pd.to_datetime(times, utc=True)
    if rows.duplicated([*TRACK_KEY, "ts_utc"]).any():
        raise ValueError("duplicate track seconds would double-count emissions")
    rows["duration_seconds"] = pd.to_numeric(rows.duration_seconds, errors="raise")
    if not (
        np.isfinite(rows.duration_seconds)
        & rows.duration_seconds.gt(0)
        & rows.duration_seconds.le(1)
    ).all():
        raise ValueError("duration_seconds must be finite and in (0, 1]")
    rows["speed_mps"] = pd.to_numeric(rows.speed_mps, errors="raise")
    if np.isinf(rows.speed_mps).any() or rows.speed_mps.dropna().lt(0).any():
        raise ValueError("speed_mps must be nonnegative, finite or explicitly missing")
    for _, group in rows.groupby(TRACK_KEY, sort=False):
        for column in ("class_id", "camera_epoch", "calibration_id"):
            if group[column].nunique() != 1:
                raise ValueError(f"a source track requires one {column}")
    return rows.sort_values([*TRACK_KEY, "ts_utc"], kind="stable").reset_index(drop=True)


def calculate_timestamped_emissions(seconds: pd.DataFrame) -> TimestampedEmissionTables:
    """Calculate both NOx representations on identical complete-second runs.

    Accepts ``speed_seconds_v4.csv`` or ``speed_profiles_v4.csv`` directly.
    Missing speeds, partial bins, omitted seconds and image-zone changes split
    runs. Input order does not affect results. No interpolation, zero filling,
    derivative across a gap or scaling of partial bins is performed. The
    retained engines use zero acceleration at each run endpoint, including
    singleton runs; every accepted bin contributes exactly one model second.
    """
    rows = _prepare_seconds(seconds)
    discrete, continuous = MOVEStarModel(), VTCPFMModel()
    run_rows, support_rows, excluded = [], [], []
    for _, track in rows.groupby(TRACK_KEY, sort=False):
        identity = {key: track.iloc[0][key] for key in IDENTITY}
        valid = track.duration_seconds.eq(1) & track.speed_mps.notna()
        for row in track.loc[~valid].to_dict("records"):
            row["ts_utc"] = row["ts_utc"].isoformat().replace("+00:00", "Z")
            row["reason"] = "partial_second" if row["duration_seconds"] != 1 else "missing_speed"
            excluded.append(row)
        selected = track.loc[valid].copy()
        boundary = selected.ts_utc.diff().ne(pd.Timedelta(seconds=1)) | selected.image_zone.ne(
            selected.image_zone.shift()
        )
        if "profile_id" in selected:
            boundary |= selected.profile_id.ne(selected.profile_id.shift())
        boundaries = boundary.cumsum()
        total_discrete, total_continuous = 0.0, 0.0
        run_count = 0
        native_type = emission_vehicle_type_for_project_class(int(identity["class_id"]))
        for run_number, (_, run) in enumerate(selected.groupby(boundaries, sort=False)):
            speed = run.speed_mps.to_numpy(dtype=float)
            acceleration = discrete.calculate_acceleration(speed)
            vsp = discrete.calculate_vsp(native_type, speed, acceleration)
            if not np.isfinite(acceleration).all() or not np.isfinite(vsp).all():
                raise ValueError("profile exceeds finite model numerical support")
            discrete_nox = float(
                discrete.estimate_emissions(native_type, speed)["Emission Rate"][0][2]
            )
            continuous_nox = float(
                continuous.estimate_emissions(native_type, speed)["Emission Rate"][0][2]
            )
            if not np.isfinite([discrete_nox, continuous_nox]).all():
                raise ValueError("profile exceeds finite model numerical support")
            run_rows.append(
                {
                    **identity,
                    "image_zone": run.image_zone.iloc[0],
                    "run_number": run_number,
                    "start_utc": run.ts_utc.iloc[0].isoformat().replace("+00:00", "Z"),
                    "end_utc": (run.ts_utc.iloc[-1] + pd.Timedelta(seconds=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "run_seconds": len(run),
                    "distance_metres": float(speed.sum()),
                    "speed_mps": json.dumps(speed.tolist(), separators=(",", ":")),
                    "acceleration_mps2": json.dumps(acceleration.tolist(), separators=(",", ":")),
                    "vsp_kw_per_tonne": json.dumps(vsp.tolist(), separators=(",", ":")),
                    "movestar_nox_g": discrete_nox,
                    "continuous_nox_g": continuous_nox,
                }
            )
            total_discrete += discrete_nox
            total_continuous += continuous_nox
            run_count += 1
        span_seconds = int((track.ts_utc.iloc[-1] - track.ts_utc.iloc[0]).total_seconds()) + 1
        support_rows.append(
            {
                **identity,
                "observed_rows": len(track),
                "complete_valid_seconds": int(valid.sum()),
                "partial_rows": int(track.duration_seconds.ne(1).sum()),
                "missing_speed_rows": int(track.speed_mps.isna().sum()),
                "omitted_second_bins": span_seconds - len(track),
                "runs": run_count,
                "movestar_nox_g": total_discrete if valid.any() else np.nan,
                "continuous_nox_g": total_continuous if valid.any() else np.nan,
            }
        )
    return TimestampedEmissionTables(
        pd.DataFrame(run_rows, columns=RUN_COLUMNS),
        pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS),
        pd.DataFrame(excluded, columns=EXCLUSION_COLUMNS),
    )
