from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.emissions.profiles import calculate_timestamped_emissions
from vehicle_emission.speed.production import STEP_COLUMNS, second_speed_exports


def _seconds(times=(0, 1, 2, 4, 5, 6), speeds=(1, 2, 3, 11, 12, 13)):
    return pd.DataFrame(
        {
            "site": "Sukiyabashi",
            "source_id": "fixture-source",
            "video_id": 1,
            "track_id": 1,
            "class_id": 1,
            "camera_epoch": "fixture-epoch",
            "calibration_id": "fixture-calibration",
            "image_zone": "road_plane",
            "ts_utc": [
                (pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(seconds=t)).isoformat()
                for t in times
            ],
            "duration_seconds": 1.0,
            "speed_mps": speeds,
        }
    )


def test_gap_preserves_time_and_separate_acceleration_and_both_models():
    seconds = _seconds()
    original = seconds.copy(deep=True)
    result = calculate_timestamped_emissions(seconds)
    pd.testing.assert_frame_equal(seconds, original)
    assert result.runs.run_seconds.tolist() == [3, 3]
    assert result.runs.start_utc.tolist() == ["2026-06-01T00:00:00Z", "2026-06-01T00:00:04Z"]
    assert result.runs.end_utc.tolist() == ["2026-06-01T00:00:03Z", "2026-06-01T00:00:07Z"]
    assert [json.loads(value) for value in result.runs.acceleration_mps2] == [[0, 1, 0]] * 2
    assert result.support.omitted_second_bins.tolist() == [1]
    assert result.support.complete_valid_seconds.tolist() == [6]
    for model, column in [(MOVEStarModel(), "movestar_nox_g"), (VTCPFMModel(), "continuous_nox_g")]:
        expected = sum(
            model.estimate_emissions(1, speed)["Emission Rate"][0][2]
            for speed in ([1, 2, 3], [11, 12, 13])
        )
        assert result.support[column].iloc[0] == pytest.approx(expected)
    pd.testing.assert_frame_equal(
        result.runs, calculate_timestamped_emissions(seconds.sample(frac=1, random_state=3)).runs
    )


def test_partial_missing_and_zone_boundaries_split_without_zero_fill():
    seconds = _seconds(times=range(7), speeds=[1, np.nan, 3, 4, 5, 6, 7])
    seconds.loc[2, "duration_seconds"] = 0.5
    seconds.loc[5:, "image_zone"] = "other-zone"
    result = calculate_timestamped_emissions(seconds)
    assert result.runs.run_seconds.tolist() == [1, 2, 2]
    assert result.exclusions.reason.tolist() == ["missing_speed", "partial_second"]
    assert result.support.complete_valid_seconds.iloc[0] == 5
    assert result.support.partial_rows.iloc[0] == 1
    assert result.support.missing_speed_rows.iloc[0] == 1
    assert json.loads(result.runs.acceleration_mps2.iloc[0]) == [0]


def test_all_excluded_is_unavailable_not_zero_emissions():
    seconds = _seconds(times=[0], speeds=[np.nan])
    result = calculate_timestamped_emissions(seconds)
    assert result.runs.empty
    assert result.support.runs.iloc[0] == 0
    assert pd.isna(result.support.movestar_nox_g.iloc[0])
    assert pd.isna(result.support.continuous_nox_g.iloc[0])


def test_explicit_profile_boundary_is_retained_even_when_times_are_adjacent():
    seconds = _seconds(times=[0, 1, 2, 3], speeds=[1, 2, 10, 11])
    seconds["profile_id"] = ["first", "first", "second", "second"]
    result = calculate_timestamped_emissions(seconds)
    assert result.runs.run_seconds.tolist() == [2, 2]
    assert [json.loads(value) for value in result.runs.acceleration_mps2] == [[0, 0]] * 2


def test_nonfinite_model_output_rejected():
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite model"),
    ):
        calculate_timestamped_emissions(_seconds(times=[0], speeds=[1e70]))


@pytest.mark.parametrize(
    "value",
    [None, "bad", "2026-06-01T00:00:00", "2026-06-01T09:00:00+09:00", "2026-06-01T00:00:00.5Z"],
)
def test_invalid_timestamps_rejected(value):
    seconds = _seconds()
    seconds.loc[0, "ts_utc"] = value
    with pytest.raises(ValueError, match="UTC-second"):
        calculate_timestamped_emissions(seconds)


def test_duplicate_seconds_rejected_even_across_class_or_zone():
    seconds = _seconds()
    duplicate = seconds.iloc[[0]].assign(class_id=2, image_zone="other-zone")
    with pytest.raises(ValueError, match="duplicate"):
        calculate_timestamped_emissions(pd.concat([seconds, duplicate]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("speed_mps", -1),
        ("speed_mps", np.inf),
        ("duration_seconds", 0),
        ("duration_seconds", 1.1),
        ("duration_seconds", np.nan),
        ("class_id", 7),
        ("track_id", True),
        ("source_id", ""),
        ("site", "other"),
    ],
)
def test_invalid_observation_rejected(column, value):
    seconds = _seconds()
    seconds[column] = seconds[column].astype(object)
    seconds.loc[0, column] = value
    with pytest.raises(ValueError):
        calculate_timestamped_emissions(seconds)


@pytest.mark.parametrize(
    "column,value", [("class_id", 2), ("calibration_id", "other"), ("camera_epoch", "other")]
)
def test_source_track_identity_is_fixed(column, value):
    seconds = _seconds()
    seconds.loc[1, column] = value
    with pytest.raises(ValueError, match="source track"):
        calculate_timestamped_emissions(seconds)


def test_real_production_second_exports_connect_directly():
    identity = _seconds().iloc[0].to_dict()
    steps = pd.DataFrame(
        [
            {
                **identity,
                "start_utc": "2026-06-01T00:00:00Z",
                "end_utc": "2026-06-01T00:00:03Z",
                "speed_mps": 5,
            },
            {
                **identity,
                "start_utc": "2026-06-01T00:00:04Z",
                "end_utc": "2026-06-01T00:00:06Z",
                "speed_mps": 10,
            },
        ]
    ).reindex(columns=STEP_COLUMNS)
    seconds, profiles = second_speed_exports(steps)
    left = calculate_timestamped_emissions(seconds)
    right = calculate_timestamped_emissions(profiles)
    pd.testing.assert_frame_equal(left.runs, right.runs)
    assert left.runs.run_seconds.tolist() == [3, 2]


@pytest.mark.parametrize("class_id,native_type", [(0, 6), (1, 1), (2, 13), (3, 2)])
def test_class_mapping_and_contiguous_run_match_retained_engines(class_id, native_type):
    seconds = _seconds(times=[0, 1, 2], speeds=[0, 5, 10]).assign(class_id=class_id)
    result = calculate_timestamped_emissions(seconds)
    for model, column in [(MOVEStarModel(), "movestar_nox_g"), (VTCPFMModel(), "continuous_nox_g")]:
        expected = model.estimate_emissions(native_type, [0, 5, 10])["Emission Rate"][0][2]
        assert result.support[column].iloc[0] == pytest.approx(expected)


def test_empty_or_missing_columns_rejected():
    with pytest.raises(ValueError, match="observations"):
        calculate_timestamped_emissions(_seconds().iloc[:0])
    with pytest.raises(ValueError, match="columns"):
        calculate_timestamped_emissions(_seconds().drop(columns="ts_utc"))
