"""Coverage from recorded clip intervals, without inferring a filename clock."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from vehicle_emission.time_utils import TOKYO_TIMEZONE

UTC_INTERVAL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")
CLIP_COLUMNS = {"site", "clip_id", "clip_start_utc", "clip_end_utc", "timing_evidence_id"}
HOUR_KEY = ["site", "monitor_start_utc", "monitor_end_utc"]


class ClipCoverageError(ValueError):
    """A clip lacks explicit timing evidence or has invalid intervals."""


def _utc(values: pd.Series, field: str) -> pd.Series:
    texts = values.astype("string")
    if not texts.str.fullmatch(UTC_INTERVAL_PATTERN.pattern).fillna(False).all():
        raise ClipCoverageError(f"{field} must contain explicit UTC timestamps ending in Z")
    try:
        return pd.to_datetime(texts, utc=True, format="ISO8601", errors="raise")
    except ValueError as exc:
        raise ClipCoverageError(f"{field} contains an invalid UTC timestamp") from exc


def _clips(clip_intervals: pd.DataFrame, site: str) -> pd.DataFrame:
    if site not in {"Sukiyabashi", "Shinjuku"}:
        raise ClipCoverageError(f"unsupported site: {site!r}")
    missing = sorted(CLIP_COLUMNS - set(clip_intervals.columns))
    if missing:
        raise ClipCoverageError(f"clip_intervals is missing required columns: {missing}")
    clips = clip_intervals.copy()
    if clips["site"].isna().any() or not clips["site"].eq(site).all():
        raise ClipCoverageError(f"clip_intervals must contain only {site!r}")
    for field in ("clip_id", "timing_evidence_id"):
        if not clips[field].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ClipCoverageError(f"clip_intervals.{field} must be recorded explicitly")
    if clips.duplicated(["site", "clip_id"]).any():
        raise ClipCoverageError("clip_intervals contains duplicate site-clip identities")
    clips["start"] = _utc(clips["clip_start_utc"], "clip_intervals.clip_start_utc")
    clips["end"] = _utc(clips["clip_end_utc"], "clip_intervals.clip_end_utc")
    if not clips["end"].gt(clips["start"]).all():
        raise ClipCoverageError("clip intervals must have positive duration")
    return clips


def _union(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def recorded_second_mask(
    clip_intervals: pd.DataFrame, second_start_utc: pd.Series, *, site: str
) -> pd.Series:
    """Identify complete one-second bins covered by the union of source clips."""

    clips = _clips(clip_intervals, site)
    seconds = _utc(second_start_utc, "video_seconds.second_start_utc")
    if not seconds.eq(seconds.dt.floor("s")).all():
        raise ClipCoverageError("video seconds must begin at whole-second UTC boundaries")
    intervals = _union(list(zip(clips["start"], clips["end"], strict=True)))
    if not intervals:
        return pd.Series(False, index=seconds.index, dtype=bool)
    starts = np.array([start.value for start, _ in intervals], dtype=np.int64)
    ends = np.array([end.value for _, end in intervals], dtype=np.int64)
    positions = np.searchsorted(starts, seconds.array.asi8, side="right") - 1
    stop_times = (seconds + pd.Timedelta(seconds=1)).array.asi8
    covered = (positions >= 0) & (stop_times <= ends[np.maximum(positions, 0)])
    return pd.Series(covered, index=seconds.index, dtype=bool)


def audit_clip_hour_coverage(
    *,
    clip_intervals: pd.DataFrame,
    candidate_hours: pd.DataFrame,
    site: str,
    predictor_offset_hours: int,
) -> pd.DataFrame:
    """Intersect clips with each half-open predictor hour and measure their union.

    ``timing_evidence_id`` identifies the source of the submitted UTC interval;
    this adapter checks its presence, not the contents of the external evidence.
    It does not select a coverage threshold or decide final eligibility.
    """

    clips = _clips(clip_intervals, site)
    if isinstance(predictor_offset_hours, bool) or not isinstance(predictor_offset_hours, int):
        raise ClipCoverageError("predictor_offset_hours must be an explicit whole hour")
    if predictor_offset_hours < 0:
        raise ClipCoverageError("predictor_offset_hours cannot be negative")
    missing = sorted(set(HOUR_KEY) - set(candidate_hours.columns))
    if missing:
        raise ClipCoverageError(f"candidate_hours is missing required columns: {missing}")
    if candidate_hours["site"].isna().any() or not candidate_hours["site"].eq(site).all():
        raise ClipCoverageError(f"candidate_hours must contain only {site!r}")
    if candidate_hours.duplicated(HOUR_KEY).any():
        raise ClipCoverageError("candidate_hours contains duplicate site-hour keys")
    starts = _utc(candidate_hours["monitor_start_utc"], "candidate_hours.monitor_start_utc")
    ends = _utc(candidate_hours["monitor_end_utc"], "candidate_hours.monitor_end_utc")
    if not (ends - starts).eq(pd.Timedelta(hours=1)).all():
        raise ClipCoverageError("candidate intervals must be half-open one-hour intervals")
    local_starts = starts.dt.tz_convert(TOKYO_TIMEZONE)
    if not local_starts.eq(local_starts.dt.floor("h")).all():
        raise ClipCoverageError("candidate intervals must use Tokyo calendar-hour boundaries")
    offset = pd.Timedelta(hours=predictor_offset_hours)
    rows: list[dict[str, object]] = []
    for position, (start, end) in enumerate(zip(starts - offset, ends - offset, strict=True)):
        included = clips.loc[clips["start"].lt(end) & clips["end"].gt(start)]
        intervals = _union(
            [
                (max(start, clip_start), min(end, clip_end))
                for clip_start, clip_end in zip(included["start"], included["end"], strict=True)
            ]
        )
        duration = sum((stop - begin).total_seconds() for begin, stop in intervals)
        rows.append(
            {
                **candidate_hours.iloc[position][HOUR_KEY].to_dict(),
                "recorded_covered_seconds": float(duration),
                "recorded_coverage_fraction": float(duration / 3600),
                "recorded_clip_count": len(included),
                "recorded_has_internal_gap": len(intervals) > 1,
                "recorded_clip_ids": json.dumps(sorted(included["clip_id"].tolist())),
                "recorded_timing_evidence_ids": json.dumps(
                    sorted(set(included["timing_evidence_id"]))
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=HOUR_KEY
        + [
            "recorded_covered_seconds",
            "recorded_coverage_fraction",
            "recorded_clip_count",
            "recorded_has_internal_gap",
            "recorded_clip_ids",
            "recorded_timing_evidence_ids",
        ],
    )


__all__ = ["ClipCoverageError", "audit_clip_hour_coverage", "recorded_second_mask"]
