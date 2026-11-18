from __future__ import annotations

import pandas as pd
import pytest

from vehicle_emission.inference.clip_coverage import (
    ClipCoverageError,
    audit_clip_hour_coverage,
    recorded_second_mask,
)


def _hours() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["Shinjuku", "Shinjuku"],
            "monitor_start_utc": ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
            "monitor_end_utc": ["2026-06-01T01:00:00Z", "2026-06-01T02:00:00Z"],
        }
    )


def _clips(intervals: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": "Shinjuku",
            "clip_id": [f"dated-path-{i}" for i in range(len(intervals))],
            "clip_start_utc": [start for start, _ in intervals],
            "clip_end_utc": [end for _, end in intervals],
            "timing_evidence_id": "recording-clock-fixture",
        }
    )


def _audit(clips: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    return audit_clip_hour_coverage(
        clip_intervals=clips,
        candidate_hours=_hours(),
        site="Shinjuku",
        predictor_offset_hours=offset,
    )


def test_boundary_spanning_clip_is_intersected_with_each_half_open_hour() -> None:
    audit = _audit(_clips([("2026-06-01T00:45:00Z", "2026-06-01T01:15:00Z")]))
    assert audit["recorded_covered_seconds"].tolist() == [900.0, 900.0]
    assert audit["recorded_coverage_fraction"].tolist() == [0.25, 0.25]


def test_overlapping_clips_are_unioned_and_never_summed() -> None:
    clips = _clips(
        [
            ("2026-06-01T00:00:00Z", "2026-06-01T00:45:00Z"),
            ("2026-06-01T00:15:00Z", "2026-06-01T01:00:00Z"),
            ("2026-06-01T01:00:00Z", "2026-06-01T01:30:00Z"),
        ]
    )
    audit = _audit(clips)
    assert audit["recorded_covered_seconds"].tolist() == [3600.0, 1800.0]
    assert audit["recorded_clip_count"].tolist() == [2, 1]
    assert not audit["recorded_has_internal_gap"].any()


def test_gaps_partial_seconds_and_shifted_windows_are_recorded_exactly() -> None:
    clips = _clips(
        [
            ("2026-05-31T23:00:00Z", "2026-05-31T23:15:00.500Z"),
            ("2026-05-31T23:45:00Z", "2026-06-01T00:00:00Z"),
        ]
    )
    audit = _audit(clips, offset=1)
    assert audit["recorded_covered_seconds"].tolist() == [1800.5, 0.0]
    assert audit["recorded_has_internal_gap"].tolist() == [True, False]
    mask = recorded_second_mask(
        clips,
        pd.Series(
            [
                "2026-05-31T23:15:00Z",
                "2026-05-31T23:45:00Z",
                "2026-06-01T00:00:00Z",
            ]
        ),
        site="Shinjuku",
    )
    assert mask.tolist() == [False, True, False]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clip_start_utc", "2026-06-01 00:00:00", "explicit UTC"),
        ("clip_start_utc", "2026-13-01T00:00:00Z", "invalid UTC"),
        ("timing_evidence_id", "", "recorded explicitly"),
        ("clip_id", None, "recorded explicitly"),
        ("clip_end_utc", "2026-05-31T23:59:59Z", "positive duration"),
    ],
)
def test_missing_clock_evidence_or_invalid_intervals_are_rejected(
    field: str, value: object, message: str
) -> None:
    clips = _clips([("2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z")])
    clips.loc[0, field] = value
    with pytest.raises(ClipCoverageError, match=message):
        _audit(clips)


def test_duplicate_clip_identity_is_rejected_without_guessing_which_record_to_use() -> None:
    clips = _clips([("2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z")])
    with pytest.raises(ClipCoverageError, match="duplicate site-clip"):
        _audit(pd.concat([clips, clips], ignore_index=True))


def test_fractional_second_feature_bin_is_rejected() -> None:
    clips = _clips([("2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z")])
    with pytest.raises(ClipCoverageError, match="whole-second"):
        recorded_second_mask(clips, pd.Series(["2026-06-01T00:00:00.500Z"]), site="Shinjuku")
