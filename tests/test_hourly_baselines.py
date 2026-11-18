from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vehicle_emission.inference.hourly_baselines import (
    aggregate_calibrated_speed,
    prepare_speed_samples,
)
from vehicle_emission.inference.hourly_measurements import (
    FrozenHourlyExtractionConfig,
    extract_hourly_measurements,
)


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["Sukiyabashi"] * 2,
            "sample_id": ["a", "b"],
            "track_id": ["one", "two"],
            "sample_start_utc": ["2025-06-01T00:00:00Z", "2025-06-01T00:00:00.500Z"],
            "sample_end_utc": ["2025-06-01T00:00:00.500Z", "2025-06-01T00:00:02Z"],
            "speed_mps": [10.0, 20.0],
            "speed_valid": [True, True],
            "calibration_id": ["held_out_test_calibration"] * 2,
            "calibration_valid": [True, True],
        }
    )


def _extract(samples: pd.DataFrame | None, increments: list[float] | None = None):
    config = FrozenHourlyExtractionConfig(
        method_config_id="fixture",
        method_config_sha256="a" * 64,
        source_snapshot_id="fixture",
        status="self_attested_scaffold_only",
        analysis_scope="locked_retrospective_two_site",
        site="Sukiyabashi",
        monitor_label_semantics="end-labelled-hour",
        predictor_offset_hours=0,
        coverage_rule_id="fixture",
        coverage_mode="threshold_normalize",
        minimum_coverage_fraction=1 / 1800,
        partial_hour_normalisation="same_covered_seconds_scale_for_all_predictors",
        gap_rule="require_contiguous_coverage",
        background_site=None,
    )
    candidates = pd.DataFrame(
        {
            "site": ["Sukiyabashi"],
            "monitor_start_utc": ["2025-06-01T00:00:00Z"],
            "monitor_end_utc": ["2025-06-01T01:00:00Z"],
        }
    )
    monitors = candidates.assign(
        ambient_nox_ppm=0.01,
        monitor_label_utc=candidates.monitor_end_utc,
        target_observation_status="actual_observation",
    )
    video = pd.DataFrame(
        {
            "site": ["Sukiyabashi"] * 2,
            "second_start_utc": ["2025-06-01T00:00:00Z", "2025-06-01T00:00:01Z"],
            "coverage_valid": [True, True],
            "count_only_increment": [1, 1],
            "relative_nox_emission_index_increment": [2.0, 3.0],
            "explicit_alt_increment": increments if increments is not None else [2.0, 4.0],
        }
    )
    return extract_hourly_measurements(
        config=config,
        candidate_hours=candidates,
        monitor_observations=monitors,
        video_seconds=video,
        background_observations=None,
        speed_samples=samples,
        secondary_increment_columns={"alternative_index": "explicit_alt_increment"},
    )


def test_speed_is_actual_duration_weighted_and_not_coverage_scaled() -> None:
    table = _extract(_samples())
    row = table.ambient_primary.iloc[0]
    assert row.mean_speed_mps == pytest.approx(17.5)
    assert row.speed_valid_duration_seconds == 2.0
    assert row.speed_samples_contributing == 2
    assert row.speed_exclusion_reasons == ""
    assert row.count_only_index == 3600
    assert row.alternative_index == 10800


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("calibration_valid", False, "missing_or_invalid_calibration"),
        ("calibration_id", "", "missing_or_invalid_calibration"),
        ("speed_valid", False, "missing_or_invalid_speed"),
        ("speed_mps", np.nan, "missing_or_invalid_speed"),
        ("speed_mps", -1.0, "missing_or_invalid_speed"),
        ("speed_mps", np.inf, "missing_or_invalid_speed"),
    ],
)
def test_invalid_optional_speed_never_changes_primary_eligibility(
    field: str, value: object, reason: str
) -> None:
    samples = _samples()
    samples.loc[0, field] = value
    row = _extract(samples).ambient_primary.iloc[0]
    assert row.primary_measurement_eligible
    assert pd.isna(row.mean_speed_mps)
    assert row.speed_exclusion_reasons == reason
    assert row.speed_samples_invalid == 1


def test_missing_optional_speed_or_increment_preserves_count_pair() -> None:
    row = _extract(None, [2.0, np.nan]).ambient_primary.iloc[0]
    assert row.primary_measurement_eligible and row.count_only_index == 3600
    assert pd.isna(row.alternative_index) and pd.isna(row.mean_speed_mps)
    assert row.speed_exclusion_reasons == "missing_speed_samples"


def test_missing_calibration_columns_only_narrow_the_speed_comparison() -> None:
    row = _extract(_samples().drop(columns="calibration_id")).ambient_primary.iloc[0]
    assert row.primary_measurement_eligible and row.count_only_index == 3600
    assert pd.isna(row.mean_speed_mps)
    assert row.speed_exclusion_reasons == "missing_speed_columns:calibration_id"


def test_sample_boundary_and_covered_intersection_are_half_open() -> None:
    samples = _samples()
    samples.loc[0, "sample_start_utc"] = "2025-05-31T23:59:59Z"
    samples.loc[0, "sample_end_utc"] = "2025-06-01T00:00:00Z"
    samples.loc[0, "speed_valid"] = False
    samples.loc[1, "sample_start_utc"] = "2025-06-01T00:00:01.500Z"
    samples.loc[1, "sample_end_utc"] = "2025-06-01T00:00:02.500Z"
    row = _extract(samples).ambient_primary.iloc[0]
    assert row.mean_speed_mps == 20.0
    assert row.speed_valid_duration_seconds == 0.5
    assert row.speed_samples_invalid == 0


def test_duplicate_or_overlapping_samples_cannot_double_count_time() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="duplicate"):
        prepare_speed_samples(pd.concat([samples, samples.iloc[[0]]]), site="Sukiyabashi")
    samples.loc[1, "track_id"] = "one"
    samples.loc[1, "sample_start_utc"] = "2025-06-01T00:00:00.250Z"
    with pytest.raises(ValueError, match="overlapping"):
        prepare_speed_samples(samples, site="Sukiyabashi")


def test_speed_weights_do_not_bridge_missing_video_seconds() -> None:
    samples = _samples()
    samples.loc[0, "sample_end_utc"] = "2025-06-01T00:00:03Z"
    samples.loc[1, "sample_start_utc"] = "2025-06-01T00:00:01Z"
    samples.loc[1, "sample_end_utc"] = "2025-06-01T00:00:02Z"
    samples.loc[1, "calibration_valid"] = False
    prepared = prepare_speed_samples(samples, site="Sukiyabashi")
    start = pd.Timestamp("2025-06-01T00:00:00Z")
    result = aggregate_calibrated_speed(
        prepared,
        pd.Series([start, start + pd.Timedelta(seconds=2)]),
        start=start,
        end=start + pd.Timedelta(hours=1),
    )
    assert result["mean_speed_mps"] == 10.0
    assert result["speed_valid_duration_seconds"] == 2.0
    assert result["speed_samples_invalid"] == 0
