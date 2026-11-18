"""Focused checks for recorded media clocks and fixed fleet-weight comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vehicle_emission.calculate_speeds import _total_frames_for_video
from vehicle_emission.inference.recorded_hourly_extension import reference_class_weights
from vehicle_emission.inference.recording_support import RecordingSupportConfig
from vehicle_emission.inference.timing_weight_sensitivity import (
    MODELS,
    SITES,
    _weight_values,
    fixed_class_weights,
    infer_weight_sensitivity,
    prepare_timing_clips,
    prepare_weight_panel,
    profile_entry_summary,
    run_timing_weight_sensitivity,
    timing_summaries,
)


def timing_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    clips = []
    videos = []
    for site in SITES:
        for index, duration in enumerate((300.0, 305.0, 306.0, 450.0, 600.0)):
            start = f"2026-05-0{index + 1}T01:05:00Z"
            clips.append(
                {
                    "site": site,
                    "video_id": index,
                    "monitor_start_utc": start.replace(":05:", ":00:"),
                    "clip_start_utc": start,
                }
            )
            videos.append(
                {
                    "source_site": site,
                    "video_id": index,
                    "duration_seconds": duration,
                    "fps": 30.0,
                    "recording_datetime_utc": start,
                }
            )
    return pd.DataFrame(clips), pd.DataFrame(videos)


def panel_fixture() -> pd.DataFrame:
    weights = fixed_class_weights()
    reference = (
        weights.loc[weights.reference_speed_kmh.eq(25)].set_index("class_id").normalised_weight
    )
    output = []
    for site in SITES:
        for index in range(12):
            counts = np.array([index % 3, 20 + index * 2, (index * 3) % 7, index % 5])
            output.append(
                {
                    "site": site,
                    "monitor_start_utc": f"2026-05-{index + 1:02}T01:00:00Z",
                    "local_calendar_date": f"2026-05-{index + 1:02}",
                    "ambient_nox_ppm": 0.005 + (index * 3 % 11) / 1000,
                    MODELS[0]: 2 + index**2,
                    MODELS[1]: 3 + (index * 7 % 13),
                    **{f"entering_class_{i}": counts[i] for i in range(4)},
                    "class_counts_known": index != 11,
                    "captured_clip_entering_count": counts.sum(),
                    "vehicle_count": int(counts.sum() * 4),
                    "fixed_rate_weighted_entries": counts @ reference.loc[range(4)].to_numpy(),
                }
            )
    return pd.DataFrame(output)


def test_clock_direction_and_short_rule_match_retained_callable() -> None:
    clips, videos = timing_fixture()
    result = prepare_timing_clips(clips, videos)
    for row in result.itertuples():
        expected = _total_frames_for_video(
            "",
            fps=row.fps,
            duration_seconds=row.duration_seconds,
            livestream_duration_sec=600,
            coords_df=pd.DataFrame(),
        )
        assert row.legacy_inferred_frames == expected
        assert (
            row.assigned_to_media_time_ratio * row.assigned_to_media_uncapped_step_speed_ratio
            == pytest.approx(1)
        )
        if row.duration_seconds <= 305:
            assert row.assigned_to_media_time_ratio == pytest.approx(1)
        else:
            assert row.assigned_to_media_time_ratio == pytest.approx(600 / row.duration_seconds)
            assert row.assigned_to_media_uncapped_step_speed_ratio == pytest.approx(
                row.duration_seconds / 600
            )
    summary, hourly = timing_summaries(result)
    assert summary.iloc[0].clips == 10
    assert summary.iloc[0].short_rule_clips == 4
    assert set(hourly.hour_jst) == {10}


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "bad_duration", "bad_fps", "wrong_start", "wrong_hour"]
)
def test_timing_rejects_invalid_membership_or_metadata(kind: str) -> None:
    clips, videos = timing_fixture()
    if kind == "duplicate":
        videos = pd.concat([videos, videos.iloc[[0]]])
    elif kind == "missing":
        videos = videos.iloc[1:]
    elif kind == "bad_duration":
        videos.loc[0, "duration_seconds"] = 0
    elif kind == "bad_fps":
        videos.loc[0, "fps"] = np.nan
    elif kind == "wrong_start":
        videos.loc[0, "recording_datetime_utc"] = "2026-05-01T01:06:00Z"
    else:
        clips.loc[0, "monitor_start_utc"] = "2026-05-01T02:00:00Z"
    with pytest.raises(ValueError):
        prepare_timing_clips(clips, videos)


def test_reference_weights_reproduce_existing_state() -> None:
    actual = fixed_class_weights()
    expected = reference_class_weights()
    assert set(actual.reference_speed_kmh) == {15, 25, 40}
    assert actual.groupby("reference_speed_kmh").size().tolist() == [4, 4, 4]
    np.testing.assert_allclose(
        actual.loc[actual.reference_speed_kmh.eq(25), "normalised_weight"],
        expected.normalised_weight,
    )
    assert actual.loc[actual.class_id.eq(1), "normalised_weight"].eq(1).all()


def test_weighted_correlations_use_same_mask_and_tied_rank_definition() -> None:
    panel = prepare_weight_panel(panel_fixture(), fixed_class_weights())
    values = _weight_values(panel)
    for site in SITES:
        selected = panel.loc[panel.site.eq(site) & panel.weight_sample_complete]
        assert len(selected) == 11
        for speed in (15, 25, 40):
            weighted = spearmanr(
                selected[f"weighted_entries_{speed}_kmh"], selected.ambient_nox_ppm
            ).statistic
            assert values[site, speed, "weighted_count_rho"] == pytest.approx(weighted)
            for model in MODELS:
                rho = spearmanr(selected[model], selected.ambient_nox_ppm).statistic
                assert values[site, speed, f"{model}_minus_weighted_count_rho"] == pytest.approx(
                    rho - weighted
                )
        assert values[site, 25, "weighted_count_minus_25_kmh_rho"] == 0
    units = profile_entry_summary(panel)
    assert units.profiles_per_entering_event.eq(4).all()


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "bad_date",
        "count_fraction",
        "count_missing",
        "count_sum",
        "reference_mismatch",
        "weight_missing",
        "weight_duplicate",
        "weight_car",
    ],
)
def test_weights_reject_inconsistent_records(kind: str) -> None:
    panel = panel_fixture()
    weights = fixed_class_weights()
    if kind == "duplicate":
        panel = pd.concat([panel, panel.iloc[[0]]])
    elif kind == "bad_date":
        panel.loc[0, "local_calendar_date"] = "2026-05-02"
    elif kind == "count_fraction":
        panel["entering_class_1"] = panel.entering_class_1.astype(float)
        panel.loc[0, "entering_class_1"] = 0.5
    elif kind == "count_missing":
        panel.loc[0, "entering_class_1"] = np.nan
    elif kind == "count_sum":
        panel.loc[0, "captured_clip_entering_count"] += 1
    elif kind == "reference_mismatch":
        panel.loc[0, "fixed_rate_weighted_entries"] += 1
    elif kind == "weight_missing":
        weights = weights.iloc[1:]
    elif kind == "weight_duplicate":
        weights = pd.concat([weights, weights.iloc[[0]]])
    else:
        weights.loc[weights.class_id.eq(1), "normalised_weight"] = 2
    with pytest.raises(ValueError):
        prepare_weight_panel(panel, weights)


def test_calendar_draws_are_shared_deterministic_and_retain_empty_dates() -> None:
    panel = prepare_weight_panel(panel_fixture(), fixed_class_weights())
    panel = panel.loc[~panel.local_calendar_date.eq("2026-05-06")]
    config = RecordingSupportConfig(resamples=19)
    first = infer_weight_sensitivity(panel, config)
    second = infer_weight_sensitivity(panel, config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 48
    assert first.calendar_days.eq(12).all()
    assert first.hours.eq(10).all()
    assert first.resamples_valid.eq(19).all()
    assert first.groupby("block_length_days").resample_sha256.nunique().eq(1).all()
    zero = first.loc[
        first.reference_speed_kmh.eq(25) & first.estimand.eq("weighted_count_minus_25_kmh_rho")
    ]
    assert zero[["estimate", "ci_low", "ci_high"]].eq(0).all().all()


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_timing_weight_sensitivity(
            source_directory=tmp_path,
            extension_bundle=tmp_path,
            recording_support_bundle=tmp_path,
            output_directory=output,
        )


def test_committed_diagnostic_receipt_hashes() -> None:
    from vehicle_emission.inference.recorded_hourly_reanalysis import _hash

    bundle = Path("results/timing_weight_sensitivity_20261117")
    if not bundle.exists():
        pytest.skip("diagnostic bundle not yet calculated")
    receipt = json.loads((bundle / "receipt.json").read_text())
    assert receipt["resamples"] == 2000
    for name, expected in receipt["outputs"].items():
        assert _hash(bundle / name) == expected
