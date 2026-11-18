from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from vehicle_emission.inference.hourly_measurements import (
    FrozenHourlyExtractionConfig,
    HourlyMeasurementError,
    extract_hourly_measurements,
    write_hourly_measurements_scaffold_v2,
)


def _config(
    *,
    site: str = "Sukiyabashi",
    background_site: str | None = "Setagaya",
) -> FrozenHourlyExtractionConfig:
    return FrozenHourlyExtractionConfig(
        method_config_id="fixture-method-v2",
        method_config_sha256="a" * 64,
        source_snapshot_id="fixture-source-v2",
        status="self_attested_scaffold_only",
        analysis_scope="locked_retrospective_two_site",
        site=site,
        monitor_label_semantics="end-labelled-hour",
        predictor_offset_hours=0,
        coverage_rule_id="complete-hour-fixture",
        coverage_mode="complete_only",
        minimum_coverage_fraction=1.0,
        partial_hour_normalisation="none",
        gap_rule="require_contiguous_coverage",
        background_site=background_site,
    )


def _candidate_hours(site: str, starts: list[str]) -> pd.DataFrame:
    timestamps = pd.to_datetime(starts, utc=True)
    return pd.DataFrame(
        {
            "site": site,
            "monitor_start_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "monitor_end_utc": (timestamps + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _monitor_rows(candidates: pd.DataFrame, values: list[float]) -> pd.DataFrame:
    result = candidates.copy()
    result["ambient_nox_ppm"] = values
    result["monitor_label_utc"] = result["monitor_end_utc"]
    result["target_observation_status"] = "actual_observation"
    return result


def _covered_hour(site: str, start: str, *, count: int = 4) -> pd.DataFrame:
    seconds = pd.date_range(start, periods=3600, freq="s", tz="UTC")
    count_increment = [0] * len(seconds)
    count_increment[0] = count
    return pd.DataFrame(
        {
            "site": site,
            "second_start_utc": seconds.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "coverage_valid": True,
            "count_only_increment": count_increment,
            "relative_nox_emission_index_increment": 0.01,
        }
    )


def test_missing_video_hour_is_flagged_and_never_zero_filled() -> None:
    candidates = _candidate_hours(
        "Sukiyabashi",
        ["2026-04-30T15:00:00Z", "2026-04-30T16:00:00Z"],
    )
    tables = extract_hourly_measurements(
        config=replace(
            _config(background_site=None),
            coverage_rule_id="partial-hour-fixture",
            coverage_mode="threshold_normalize",
            minimum_coverage_fraction=0.5,
            partial_hour_normalisation="same_covered_seconds_scale_for_all_predictors",
        ),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010, 0.012]),
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        background_observations=None,
    )

    missing = tables.audit.iloc[1]
    assert missing["covered_seconds"] == 0
    assert missing["coverage_fraction"] == 0.0
    assert pd.isna(missing["count_only_raw"])
    assert pd.isna(missing["relative_nox_emission_index_raw"])
    assert pd.isna(missing["count_only_index"])
    assert pd.isna(missing["relative_nox_emission_index"])
    assert not missing["primary_measurement_eligible"]
    assert missing["primary_exclusion_reasons"] == "missing_video_coverage"
    assert len(tables.ambient_primary) == 1
    assert tables.ambient_primary.iloc[0]["count_only_index"] == 4.0
    assert tables.ambient_primary.iloc[0]["site"] == "Sukiyabashi"
    assert tables.ambient_primary.iloc[0]["monitor_start_utc"] == "2026-04-30T15:00:00Z"
    assert tables.ambient_primary.iloc[0]["monitor_end_utc"] == "2026-04-30T16:00:00Z"
    assert not any("ground_truth" in column for column in tables.audit.columns)


def test_missing_monitor_and_background_remain_separate_from_primary() -> None:
    candidates = _candidate_hours(
        "Sukiyabashi",
        ["2026-04-30T15:00:00Z", "2026-04-30T16:00:00Z"],
    )
    monitor = _monitor_rows(candidates.iloc[[0]], [0.010])
    video = pd.concat(
        [
            _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
            _covered_hour("Sukiyabashi", "2026-04-30T16:00:00Z"),
        ],
        ignore_index=True,
    )

    tables = extract_hourly_measurements(
        config=_config(),
        candidate_hours=candidates,
        monitor_observations=monitor,
        video_seconds=video,
        background_observations=None,
    )

    assert pd.isna(tables.audit.iloc[1]["ambient_nox_ppm"])
    assert tables.audit.iloc[1]["primary_exclusion_reasons"] == "missing_ambient_reference"
    assert len(tables.ambient_primary) == 1
    assert pd.isna(tables.audit.iloc[0]["background_nox_ppm"])
    assert pd.isna(tables.audit.iloc[0]["background_differenced_nox_ppm"])
    assert tables.audit.iloc[0]["primary_measurement_eligible"]
    assert not tables.audit.iloc[0]["background_sensitivity_eligible"]
    assert tables.audit.iloc[0]["background_exclusion_reasons"] == ("missing_background_reference")
    assert tables.background_sensitivity.empty


def test_mixed_site_inputs_are_rejected_instead_of_pooled() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    video = _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z")
    other_site = video.iloc[[0]].copy()
    other_site["site"] = "Shinjuku"

    with pytest.raises(HourlyMeasurementError, match="must contain only 'Sukiyabashi'"):
        extract_hourly_measurements(
            config=_config(background_site=None),
            candidate_hours=candidates,
            monitor_observations=_monitor_rows(candidates, [0.010]),
            video_seconds=pd.concat([video, other_site], ignore_index=True),
            background_observations=None,
        )


def test_partial_coverage_islands_require_explicit_internal_gap_rule() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    complete_video = _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z")
    separated_islands = pd.concat(
        [complete_video.iloc[:1350], complete_video.iloc[2250:]], ignore_index=True
    )
    partial_config = replace(
        _config(),
        coverage_rule_id="partial-hour-fixture",
        coverage_mode="threshold_normalize",
        minimum_coverage_fraction=0.5,
        partial_hour_normalisation="same_covered_seconds_scale_for_all_predictors",
    )
    background = candidates.rename(columns={"site": "reference_site"}).copy()
    background["reference_site"] = "Setagaya"
    background["monitor_label_utc"] = background["monitor_end_utc"]
    background["background_nox_ppm"] = 0.004
    background["background_observation_status"] = "actual_observation"

    rejected = extract_hourly_measurements(
        config=partial_config,
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010]),
        video_seconds=separated_islands,
        background_observations=background,
    )
    assert rejected.audit.iloc[0]["coverage_status"] == "disallowed_internal_gap"
    assert rejected.audit.iloc[0]["has_internal_coverage_gap"]
    assert not rejected.audit.iloc[0]["gap_rule_passed"]
    assert rejected.audit.iloc[0]["primary_exclusion_reasons"] == ("disallowed_internal_video_gap")
    assert rejected.ambient_primary.empty
    assert rejected.background_sensitivity.empty

    allowed = extract_hourly_measurements(
        config=replace(partial_config, gap_rule="allow_internal_gaps"),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010]),
        video_seconds=separated_islands,
        background_observations=background,
    )
    assert len(allowed.ambient_primary) == 1
    assert len(allowed.background_sensitivity) == 1
    expected_semantics = {
        "count_only_raw": 4.0,
        "relative_nox_emission_index_raw": pytest.approx(27.0),
        "monitor_label_semantics": "end-labelled-hour",
        "predictor_offset_hours": 0,
        "predictor_window_sign_convention": ("positive_offset_means_predictor_precedes_receptor"),
        "coverage_mode": "threshold_normalize",
        "minimum_coverage_fraction": 0.5,
        "partial_hour_normalisation": ("same_covered_seconds_scale_for_all_predictors"),
        "gap_rule": "allow_internal_gaps",
        "observational_unit": "actual_hourly_ambient_monitor_observation",
        "primary_target": "ambient_nox_reference_observation",
        "background_sensitivity_target": ("ambient_minus_background_nox_reference_observation"),
        "target_interpolated": False,
        "target_is_derived": False,
        "target_observation_status": "actual_observation",
        "paired_predictor_set": "count_only_and_relative_nox_index_only",
        "analysis_scope": "locked_retrospective_two_site",
        "configuration_evidence_status": "self_attested_scaffold_only",
        "method_config_hash_verified": False,
    }
    for output in (allowed.ambient_primary, allowed.background_sensitivity):
        row = output.iloc[0]
        for column, expected in expected_semantics.items():
            assert row[column] == expected


def test_negative_ambient_or_background_nox_is_rejected() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    video = _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z")

    with pytest.raises(HourlyMeasurementError, match="ambient_nox_ppm cannot be negative"):
        extract_hourly_measurements(
            config=_config(background_site=None),
            candidate_hours=candidates,
            monitor_observations=_monitor_rows(candidates, [-0.001]),
            video_seconds=video,
            background_observations=None,
        )

    background = candidates.rename(columns={"site": "reference_site"}).copy()
    background["reference_site"] = "Setagaya"
    background["monitor_label_utc"] = background["monitor_end_utc"]
    background["background_nox_ppm"] = -0.001
    background["background_observation_status"] = "actual_observation"
    with pytest.raises(HourlyMeasurementError, match="background_nox_ppm cannot be negative"):
        extract_hourly_measurements(
            config=_config(),
            candidate_hours=candidates,
            monitor_observations=_monitor_rows(candidates, [0.010]),
            video_seconds=video,
            background_observations=background,
        )


def test_unresolved_lag_or_coverage_cannot_enter_the_config() -> None:
    values = dict(
        method_config_id="fixture-method-v2",
        method_config_sha256="a" * 64,
        source_snapshot_id="fixture-source-v2",
        status="self_attested_scaffold_only",
        analysis_scope="locked_retrospective_two_site",
        site="Sukiyabashi",
        monitor_label_semantics="end-labelled-hour",
        predictor_offset_hours=0,
        coverage_rule_id="fixture-rule",
        coverage_mode="complete_only",
        minimum_coverage_fraction=1.0,
        partial_hour_normalisation="none",
        gap_rule="require_contiguous_coverage",
        background_site=None,
    )
    values["predictor_offset_hours"] = None
    with pytest.raises(HourlyMeasurementError, match="predictor_offset_hours"):
        FrozenHourlyExtractionConfig(**values)

    values["predictor_offset_hours"] = 0
    values["minimum_coverage_fraction"] = None
    with pytest.raises(HourlyMeasurementError, match="minimum_coverage_fraction"):
        FrozenHourlyExtractionConfig(**values)

    values["minimum_coverage_fraction"] = 1.0
    values["gap_rule"] = "free-text-gap-policy"
    with pytest.raises(HourlyMeasurementError, match="unsupported gap rule"):
        FrozenHourlyExtractionConfig(**values)

    values["gap_rule"] = "require_contiguous_coverage"
    values["analysis_scope"] = "ambient_outcome_held_out_replication"
    with pytest.raises(HourlyMeasurementError, match="locked_retrospective_two_site"):
        FrozenHourlyExtractionConfig(**values)


def test_derived_monitor_values_are_retained_for_audit_but_never_eligible() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    monitor = _monitor_rows(candidates, [0.010])
    monitor["target_observation_status"] = "interpolated"

    tables = extract_hourly_measurements(
        config=_config(background_site=None),
        candidate_hours=candidates,
        monitor_observations=monitor,
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        background_observations=None,
    )

    row = tables.audit.iloc[0]
    assert row["target_interpolated"]
    assert row["target_is_derived"]
    assert row["observational_unit"] == "excluded_non_actual_or_missing_monitor_record"
    assert row["primary_exclusion_reasons"] == "non_actual_ambient_reference:interpolated"
    assert tables.ambient_primary.empty


def test_monitor_label_must_match_the_exact_calendar_hour_end() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    monitor = _monitor_rows(candidates, [0.010])
    video = _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z")

    accepted = extract_hourly_measurements(
        config=_config(background_site=None),
        candidate_hours=candidates,
        monitor_observations=monitor,
        video_seconds=video,
        background_observations=None,
    )
    assert len(accepted.ambient_primary) == 1

    monitor["monitor_label_utc"] = monitor["monitor_start_utc"]
    with pytest.raises(HourlyMeasurementError, match="does not match end-labelled-hour"):
        extract_hourly_measurements(
            config=_config(background_site=None),
            candidate_hours=candidates,
            monitor_observations=monitor,
            video_seconds=video,
            background_observations=None,
        )

    with pytest.raises(HourlyMeasurementError, match="end-labelled-hour.*exact calendar"):
        replace(_config(background_site=None), monitor_label_semantics="centre-labelled-hour")


def test_monitor_intervals_must_use_exact_tokyo_calendar_hour_boundaries() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:30:00Z"])

    with pytest.raises(HourlyMeasurementError, match="Asia/Tokyo.*calendar-hour boundaries"):
        extract_hourly_measurements(
            config=_config(background_site=None),
            candidate_hours=candidates,
            monitor_observations=_monitor_rows(candidates, [0.010]),
            video_seconds=pd.DataFrame(
                columns=[
                    "site",
                    "second_start_utc",
                    "coverage_valid",
                    "count_only_increment",
                    "relative_nox_emission_index_increment",
                ]
            ),
            background_observations=None,
        )


def test_derived_background_values_cannot_enter_sensitivity_output() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    background = candidates.rename(columns={"site": "reference_site"}).copy()
    background["reference_site"] = "Setagaya"
    background["monitor_label_utc"] = background["monitor_end_utc"]
    background["background_nox_ppm"] = 0.004
    background["background_observation_status"] = "forward_filled"

    tables = extract_hourly_measurements(
        config=_config(),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010]),
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        background_observations=background,
    )

    assert tables.audit.iloc[0]["primary_measurement_eligible"]
    assert not tables.audit.iloc[0]["background_sensitivity_eligible"]
    assert tables.audit.iloc[0]["background_exclusion_reasons"] == (
        "non_actual_background_reference:forward_filled"
    )
    assert tables.background_sensitivity.empty


def test_observed_zero_values_are_retained_but_absent_coverage_stays_missing() -> None:
    candidates = _candidate_hours(
        "Shinjuku",
        ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
    )
    observed_zero = _covered_hour("Shinjuku", "2026-06-01T00:00:00Z", count=0)
    observed_zero["relative_nox_emission_index_increment"] = 0.0
    tables = extract_hourly_measurements(
        config=_config(site="Shinjuku", background_site=None),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.0, 0.012]),
        video_seconds=observed_zero,
        background_observations=None,
    )

    observed = tables.audit.iloc[0]
    assert observed["coverage_status"] == "complete"
    assert observed["ambient_nox_ppm"] == 0.0
    assert observed["count_only_raw"] == 0.0
    assert observed["relative_nox_emission_index_raw"] == 0.0
    assert observed["count_only_index"] == 0.0
    assert observed["relative_nox_emission_index"] == 0.0
    assert observed["primary_measurement_eligible"]

    absent = tables.audit.iloc[1]
    assert absent["coverage_status"] == "missing_video_coverage"
    assert pd.isna(absent["count_only_raw"])
    assert pd.isna(absent["relative_nox_emission_index_raw"])
    assert len(tables.ambient_primary) == 1


def test_one_missing_predictor_excludes_the_hour_from_both_paired_predictors() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    video = _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z")
    video.loc[0, "relative_nox_emission_index_increment"] = pd.NA

    tables = extract_hourly_measurements(
        config=_config(background_site=None),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010]),
        video_seconds=video,
        background_observations=None,
    )

    row = tables.audit.iloc[0]
    assert not row["paired_predictors_complete"]
    assert row["primary_exclusion_reasons"] == "missing_paired_predictor"
    assert pd.isna(row["count_only_raw"])
    assert pd.isna(row["relative_nox_emission_index_raw"])
    assert tables.ambient_primary.empty


def _publishable_tables():
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    return extract_hourly_measurements(
        config=_config(background_site=None),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010]),
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        background_observations=None,
    )


def test_hourly_bundle_uses_versioned_names_and_hashes_exact_csv_bytes(tmp_path: Path) -> None:
    output_directory = tmp_path / "prepared"
    receipt = write_hourly_measurements_scaffold_v2(
        _publishable_tables(),
        output_directory=output_directory,
    )

    paths = sorted(output_directory.iterdir())
    assert {path.name for path in paths} == {
        "sukiyabashi_hourly_ambient_primary_scaffold_v2.csv",
        "sukiyabashi_hourly_audit_scaffold_v2.csv",
        "sukiyabashi_hourly_background_sensitivity_scaffold_v2.csv",
        "sukiyabashi_hourly_measurements_scaffold_v2.json",
    }
    assert all("_scaffold_v2." in path.name for path in paths)
    manifest = json.loads(
        (output_directory / "sukiyabashi_hourly_measurements_scaffold_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["analysis_scope"] == "locked_retrospective_two_site"
    assert manifest["manifest_type"] == "hourly_measurement_scaffold_bundle"
    assert manifest["configuration_evidence_status"] == "self_attested_scaffold_only"
    assert manifest["method_config_hash_verified"] is False
    for role, record in manifest["artifacts"].items():
        artifact = output_directory / record["path"]
        assert artifact.name.endswith("_scaffold_v2.csv")
        assert record["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert record["row_count"] == dict(receipt.row_counts)[role]
    assert dict(receipt.artifact_sha256) == {
        role: record["sha256"] for role, record in manifest["artifacts"].items()
    }


def test_hourly_bundle_never_overwrites_or_partially_publishes(tmp_path: Path) -> None:
    output_directory = tmp_path / "prepared"
    output_directory.mkdir()
    protected = output_directory / "sukiyabashi_hourly_audit_scaffold_v2.csv"
    protected.write_bytes(b"preserved exploratory bytes\n")

    with pytest.raises(FileExistsError, match="immutable"):
        write_hourly_measurements_scaffold_v2(
            _publishable_tables(),
            output_directory=output_directory,
        )

    assert protected.read_bytes() == b"preserved exploratory bytes\n"
    assert list(output_directory.iterdir()) == [protected]


def test_hourly_bundle_rejects_tracked_destination_deleted_locally(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    output = tmp_path / "build" / "tracked_preparation"
    output.mkdir(parents=True)
    tracked = output / "sukiyabashi_hourly_audit_scaffold_v2.csv"
    tracked.write_bytes(b"original tracked preparation\n")
    subprocess.run(["git", "add", "build"], cwd=tmp_path, check=True)
    tracked.unlink()
    output.rmdir()
    with pytest.raises(FileExistsError, match="tracked artifacts"):
        write_hourly_measurements_scaffold_v2(_publishable_tables(), output_directory=output)
    assert not output.exists()


def test_repository_hourly_outputs_require_a_fresh_build_directory(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    unsafe_output = tmp_path / "results" / "new_preparation"
    with pytest.raises(HourlyMeasurementError, match="new directory under build"):
        write_hourly_measurements_scaffold_v2(_publishable_tables(), output_directory=unsafe_output)
    assert not unsafe_output.exists()
    safe_output = tmp_path / "build" / "new_preparation"
    receipt = write_hourly_measurements_scaffold_v2(
        _publishable_tables(), output_directory=safe_output
    )
    assert receipt.output_directory == str(safe_output)


def test_repository_hourly_output_cannot_escape_through_a_build_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    build = repository / "build"
    build.mkdir()
    try:
        (build / "alias").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlinks not supported on system: {error}")
    output = build / "alias" / "new_preparation"
    with pytest.raises(HourlyMeasurementError, match="new directory under build"):
        write_hourly_measurements_scaffold_v2(_publishable_tables(), output_directory=output)
    assert not (outside / "new_preparation").exists()


def test_hourly_bundle_rejects_a_tampered_paired_support_table(tmp_path: Path) -> None:
    tables = _publishable_tables()
    tampered = replace(tables, ambient_primary=tables.ambient_primary.iloc[0:0].copy())

    with pytest.raises(HourlyMeasurementError, match="audited eligible support"):
        write_hourly_measurements_scaffold_v2(tampered, output_directory=tmp_path)
    assert not list(tmp_path.iterdir())


def _weather(candidates: pd.DataFrame, rain: list[float]) -> pd.DataFrame:
    weather = candidates.rename(
        columns={
            "monitor_start_utc": "weather_start_utc",
            "monitor_end_utc": "weather_end_utc",
        }
    ).copy()
    weather["weather_label_utc"] = weather["weather_end_utc"]
    weather["weather_source_id"] = "JMA-Kitanomaru-fixture"
    weather["weather_observation_status"] = "actual_observation"
    weather["precipitation_mm"] = rain
    weather["along_road_wind_speed"] = 1.5
    return weather


def test_recorded_wind_strata_retain_the_actual_weather_definition() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    weather = _weather(candidates, [0.0]).assign(
        wind_stratum="favourable", wind_definition_id="recorded_test_definition"
    )
    tables = extract_hourly_measurements(
        config=replace(
            _config(background_site=None),
            precipitation_rule="dry_only",
            weather_alignment="monitor_interval",
        ),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.01]),
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        weather_observations=weather,
        background_observations=None,
    )
    assert tables.ambient_primary.wind_stratum.tolist() == ["favourable"]
    assert tables.ambient_primary.wind_definition_id.tolist() == ["recorded_test_definition"]


def test_dry_hour_filter_retains_missingness_and_excludes_rain() -> None:
    candidates = _candidate_hours(
        "Sukiyabashi",
        ["2026-04-30T15:00:00Z", "2026-04-30T16:00:00Z", "2026-04-30T17:00:00Z"],
    )
    video = pd.concat(
        [_covered_hour("Sukiyabashi", start) for start in candidates["monitor_start_utc"]],
        ignore_index=True,
    )
    weather = _weather(candidates, [0.0, 0.1, float("nan")])
    tables = extract_hourly_measurements(
        config=replace(
            _config(background_site=None),
            precipitation_rule="dry_only",
            weather_alignment="predictor_window",
        ),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.010, 0.012, 0.014]),
        video_seconds=video,
        weather_observations=weather,
        background_observations=None,
    )
    assert len(tables.audit) == 3
    assert tables.audit["primary_exclusion_reasons"].tolist() == [
        "",
        "rainy_hour",
        "missing_precipitation_reference",
    ]
    assert len(tables.ambient_primary) == 1
    assert tables.ambient_primary["primary_measurement_eligible"].tolist() == [True]
    assert tables.ambient_primary["along_road_wind_speed"].tolist() == [1.5]


@pytest.mark.parametrize("alignment", ["predictor_window", "monitor_interval"])
def test_lag_shifts_predictors_and_only_the_explicit_weather_window(alignment: str) -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T16:00:00Z"])
    weather_hours = _candidate_hours(
        "Sukiyabashi", ["2026-04-30T15:00:00Z", "2026-04-30T16:00:00Z"]
    )
    tables = extract_hourly_measurements(
        config=replace(
            _config(background_site=None),
            predictor_offset_hours=1,
            precipitation_rule="dry_only",
            weather_alignment=alignment,
        ),
        candidate_hours=candidates,
        monitor_observations=_monitor_rows(candidates, [0.021]),
        video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z", count=9),
        weather_observations=_weather(weather_hours, [0.0, 0.1]),
        background_observations=None,
    )
    row = tables.audit.iloc[0]
    assert row["monitor_start_utc"] == "2026-04-30T16:00:00Z"
    assert row["ambient_nox_ppm"] == 0.021
    assert row["predictor_window_start_utc"] == "2026-04-30T15:00:00Z"
    assert row["count_only_index"] == 9
    if alignment == "predictor_window":
        assert row["weather_window_start_utc"] == "2026-04-30T15:00:00Z"
        assert row["primary_measurement_eligible"]
    else:
        assert row["weather_window_start_utc"] == "2026-04-30T16:00:00Z"
        assert row["primary_exclusion_reasons"] == "rainy_hour"
        assert tables.ambient_primary.empty


def test_derived_weather_and_absent_weather_do_not_establish_a_dry_hour() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    weather = _weather(candidates, [0.0])
    weather["weather_observation_status"] = "forward_filled"
    for supplied, expected in (
        (weather, "non_actual_weather_reference:forward_filled"),
        (None, "missing_precipitation_reference"),
    ):
        tables = extract_hourly_measurements(
            config=replace(
                _config(background_site=None),
                precipitation_rule="dry_only",
                weather_alignment="predictor_window",
            ),
            candidate_hours=candidates,
            monitor_observations=_monitor_rows(candidates, [0.01]),
            video_seconds=_covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
            weather_observations=supplied,
            background_observations=None,
        )
        assert tables.audit.iloc[0]["primary_exclusion_reasons"] == expected
        assert tables.ambient_primary.empty


def test_weather_policy_and_negative_precipitation_are_rejected() -> None:
    with pytest.raises(HourlyMeasurementError, match="explicit weather_alignment"):
        replace(_config(), precipitation_rule="dry_only")
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    weather = _weather(candidates, [-0.1])
    arguments = {
        "candidate_hours": candidates,
        "monitor_observations": _monitor_rows(candidates, [0.01]),
        "video_seconds": _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        "weather_observations": weather,
        "background_observations": None,
    }
    with pytest.raises(HourlyMeasurementError, match="explicit dry_only"):
        extract_hourly_measurements(config=_config(), **arguments)
    with pytest.raises(HourlyMeasurementError, match="precipitation_mm cannot be negative"):
        extract_hourly_measurements(
            config=replace(
                _config(), precipitation_rule="dry_only", weather_alignment="predictor_window"
            ),
            **arguments,
        )


def test_feature_coverage_must_be_contained_in_recorded_source_intervals() -> None:
    candidates = _candidate_hours("Sukiyabashi", ["2026-04-30T15:00:00Z"])
    clips = pd.DataFrame(
        {
            "site": ["Sukiyabashi", "Sukiyabashi"],
            "clip_id": ["dated-path-1", "dated-path-2"],
            "clip_start_utc": ["2026-04-30T15:00:00Z", "2026-04-30T15:15:00Z"],
            "clip_end_utc": ["2026-04-30T15:45:00Z", "2026-04-30T16:00:00Z"],
            "timing_evidence_id": "source-clock-fixture",
        }
    )
    arguments = {
        "config": _config(background_site=None),
        "candidate_hours": candidates,
        "monitor_observations": _monitor_rows(candidates, [0.01]),
        "video_seconds": _covered_hour("Sukiyabashi", "2026-04-30T15:00:00Z"),
        "clip_intervals": clips,
        "background_observations": None,
    }
    tables = extract_hourly_measurements(**arguments)
    assert tables.audit.iloc[0]["recorded_covered_seconds"] == 3600
    assert len(tables.ambient_primary) == 1
    clips.loc[1, "clip_end_utc"] = "2026-04-30T15:59:59Z"
    with pytest.raises(HourlyMeasurementError, match="outside recorded clip coverage"):
        extract_hourly_measurements(**arguments)
