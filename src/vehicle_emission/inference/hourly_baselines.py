"""Optional hourly baselines from explicit calibrated trajectory samples."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SPEED_COLUMNS = {
    "site",
    "sample_id",
    "track_id",
    "sample_start_utc",
    "sample_end_utc",
    "speed_mps",
    "speed_valid",
    "calibration_id",
    "calibration_valid",
}
SAMPLE_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")


def prepare_speed_samples(samples: pd.DataFrame | None, *, site: str) -> pd.DataFrame | None:
    """Validate timing and identity without treating calibration attestation as proof."""

    if samples is None:
        return None
    missing = SPEED_COLUMNS - set(samples.columns)
    if samples.columns.duplicated().any():
        raise ValueError(f"speed_samples require unique columns: {sorted(SPEED_COLUMNS)}")
    if missing:
        unavailable = pd.DataFrame()
        unavailable.attrs["speed_input_exclusion_reason"] = "missing_speed_columns:" + ",".join(
            sorted(missing)
        )
        return unavailable
    rows = samples.copy()
    if rows.empty:
        return rows.assign(_start_ns=pd.Series(dtype="int64"), _end_ns=pd.Series(dtype="int64"))
    if rows.site.isna().any() or not rows.site.eq(site).all():
        raise ValueError("speed_samples must identify only the extraction site")
    for column in ("sample_id", "track_id"):
        if rows[column].isna().any() or rows[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"speed_samples.{column} must identify every sample")
    if rows.duplicated(["site", "sample_id"]).any():
        raise ValueError("duplicate speed sample identities would double-count duration")
    for column, internal in (("sample_start_utc", "_start_ns"), ("sample_end_utc", "_end_ns")):
        if not rows[column].astype("string").str.fullmatch(SAMPLE_UTC.pattern).fillna(False).all():
            raise ValueError("speed sample boundaries must be explicit UTC timestamps ending Z")
        rows[internal] = pd.to_datetime(rows[column], utc=True, format="ISO8601").astype("int64")
    if rows._end_ns.le(rows._start_ns).any():
        raise ValueError("speed sample intervals must have positive actual duration")
    for _, track in rows.sort_values("_start_ns").groupby("track_id", sort=False):
        if (
            len(track) > 1
            and (
                track._start_ns.to_numpy()[1:]
                < np.maximum.accumulate(track._end_ns.to_numpy())[:-1]
            ).any()
        ):
            raise ValueError("overlapping samples for one track would double-count duration")
    for column in ("speed_valid", "calibration_valid"):
        if not rows[column].map(lambda value: isinstance(value, bool | np.bool_)).all():
            raise ValueError(f"speed_samples.{column} must be boolean")
    rows["speed_mps"] = pd.to_numeric(rows.speed_mps, errors="coerce")
    rows["_valid_speed"] = (
        rows.speed_valid
        & rows.speed_mps.notna()
        & np.isfinite(rows.speed_mps)
        & rows.speed_mps.ge(0)
    )
    rows["_valid_calibration"] = (
        rows.calibration_valid
        & rows.calibration_id.notna()
        & rows.calibration_id.astype("string").str.strip().ne("").fillna(False)
    )
    return rows


def aggregate_calibrated_speed(
    samples: pd.DataFrame | None,
    covered_seconds: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    """Weight each speed by its duration inside the same actual covered seconds."""

    result: dict[str, object] = {
        "mean_speed_mps": np.nan,
        "speed_valid_duration_seconds": 0.0,
        "speed_samples_contributing": 0,
        "speed_samples_invalid": 0,
        "speed_exclusion_reasons": "missing_speed_samples",
    }
    if samples is not None and "speed_input_exclusion_reason" in samples.attrs:
        result["speed_exclusion_reasons"] = samples.attrs["speed_input_exclusion_reason"]
    if samples is None or samples.empty or covered_seconds.empty:
        return result
    selected = samples.loc[samples._start_ns.lt(end.value) & samples._end_ns.gt(start.value)]
    covered_ns = np.sort(covered_seconds.astype("int64").to_numpy())
    weighted_speed = duration = 0.0
    contributors = invalid = 0
    reasons: set[str] = set()
    for _, sample in selected.iterrows():
        first = int(
            np.searchsorted(covered_ns, int(sample._start_ns) - 1_000_000_000, side="right")
        )
        last = int(np.searchsorted(covered_ns, int(sample._end_ns), side="left"))
        seconds = covered_ns[first:last]
        weight = (
            float(
                np.maximum(
                    0,
                    np.minimum(seconds + 1_000_000_000, int(sample._end_ns))
                    - np.maximum(seconds, int(sample._start_ns)),
                ).sum()
            )
            / 1_000_000_000
        )
        if weight == 0:
            continue
        if not sample._valid_calibration:
            reasons.add("missing_or_invalid_calibration")
        if not sample._valid_speed:
            reasons.add("missing_or_invalid_speed")
        if not sample._valid_calibration or not sample._valid_speed:
            invalid += 1
            continue
        duration += weight
        weighted_speed += float(sample.speed_mps) * weight
        contributors += 1
    result.update(
        {
            "speed_valid_duration_seconds": duration,
            "speed_samples_contributing": contributors,
            "speed_samples_invalid": invalid,
        }
    )
    if reasons:
        result["speed_exclusion_reasons"] = ";".join(sorted(reasons))
    elif duration > 0:
        result["mean_speed_mps"] = weighted_speed / duration
        result["speed_exclusion_reasons"] = ""
    return result
