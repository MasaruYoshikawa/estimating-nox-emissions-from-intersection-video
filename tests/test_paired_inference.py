from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import spearmanr

import vehicle_emission.inference.paired_inference as inference
from vehicle_emission.inference.paired_inference import (
    PairedInferenceConfig,
    classify_interval,
    compute_paired_inference,
    load_hourly_bundle,
    write_inference_artifacts,
)


def _config(**overrides: object) -> PairedInferenceConfig:
    values = {
        "resampling_method": "union_calendar_moving_block",
        "block_length_days": 2,
        "minimum_site_hours": 3,
        "minimum_site_days": 2,
        "minimum_stratum_hours": 3,
        "minimum_stratum_days": 2,
        "resamples": 49,
        "minimum_valid_resamples": 19,
        "seed": 81,
        "interval_method": "percentile",
        "interval_adjustment": "bonferroni",
        "confidence_level": 0.95,
        "primary_interval_family": "two_site_rq1_and_rq3",
        "primary_family_size": 4,
        "wind_interval_family": "two_site_conditional_rq2",
        "wind_family_size": 2,
        "wind_mode": "recorded_strata",
        "wind_definition_id": "synthetic_test_strata",
        "favourable_min_mps": None,
        "opposing_max_mps": None,
    }
    values.update(overrides)
    return PairedInferenceConfig(**values)  # type: ignore[arg-type]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(718)
    values = rng.integers(1, 30, size=(8, 4, 3))
    rows = []
    for site in inference.SITES:
        for day in range(8):
            for hour in range(4):
                start = pd.Timestamp("2025-06-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
                rows.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "ambient_nox_ppm": float(values[day, hour, 0]) / 1000,
                        "relative_nox_emission_index": float(values[day, hour, 1]),
                        "count_only_index": float(values[day, hour, 2]),
                        "primary_measurement_eligible": True,
                        "target_observation_status": "actual_observation",
                        "wind_stratum": "favourable" if hour < 2 else "opposing",
                    }
                )
    return pd.DataFrame(rows)


def _row(table: pd.DataFrame, site: str, estimand: str) -> pd.Series:
    return table.loc[(table["site"] == site) & (table["estimand"] == estimand)].iloc[0]


def _bundle_frame(site: str) -> pd.DataFrame:
    frame = _frame().loc[lambda data: data.site.eq(site)].copy()
    metadata = {
        "precipitation_rule": "dry_only",
        "precipitation_mm": 0.0,
        "weather_observation_status": "actual_observation",
        "weather_alignment": "monitor_interval",
        "weather_source_id": "synthetic_weather",
        "predictor_offset_hours": 0,
        "analysis_scope": "locked_retrospective_two_site",
        "configuration_evidence_status": "self_attested_scaffold_only",
        "method_config_id": "synthetic_shared_method",
        "method_config_sha256": "a" * 64,
        "method_config_hash_verified": False,
        "source_snapshot_id": f"synthetic_{site}",
        "coverage_mode": "complete_only",
        "minimum_coverage_fraction": 1.0,
        "partial_hour_normalisation": "none",
        "gap_rule": "allow_internal_gaps",
        "monitor_label_semantics": "end-labelled-hour",
    }
    for name, value in metadata.items():
        frame[name] = value
    frame["predictor_window_start_utc"] = frame["monitor_start_utc"]
    frame["predictor_window_end_utc"] = frame["monitor_end_utc"]
    return frame


def _write_bundle(directory: Path, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "primary.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "manifest_type": "hourly_measurement_scaffold_bundle",
        "schema_version": 2,
        "method_config_hash_verified": False,
        "site": frame.site.iloc[0],
        **{
            name: frame[name].iloc[0]
            for name in (
                "analysis_scope",
                "configuration_evidence_status",
                "method_config_id",
                "method_config_sha256",
                "source_snapshot_id",
            )
        },
        "artifacts": {
            "ambient_primary": {
                "path": csv_path.name,
                "row_count": len(frame),
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = directory / "bundle.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.mark.parametrize(
    "failure", ["missing_count", "duplicate_hour", "wrong_date", "derived_target", "excluded_hour"]
)
def test_rejects_unpaired_or_non_actual_hours(failure: str) -> None:
    frame = _frame()
    if failure == "missing_count":
        frame.loc[0, "count_only_index"] = np.nan
    elif failure == "duplicate_hour":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif failure == "wrong_date":
        frame.loc[0, "local_calendar_date"] = "2025-06-02"
    elif failure == "derived_target":
        frame.loc[0, "target_observation_status"] = "interpolated"
    else:
        frame.loc[0, "primary_measurement_eligible"] = False
    with pytest.raises(ValueError):
        compute_paired_inference(frame, _config())


def test_joint_draws_preserve_sites_predictors_and_deterministic_ties() -> None:
    frame = _frame()
    frame["count_only_index"] = frame["relative_nox_emission_index"]
    result = compute_paired_inference(frame, _config())
    pdt.assert_frame_equal(
        result, compute_paired_inference(frame.sample(frac=1, random_state=7), _config())
    )
    assert len(result) == 9
    assert result["resample_sha256"].nunique() == 1
    for site in inference.SITES:
        rows = result.loc[
            (result["site"] == site) & result["estimand"].isin(inference.PRIMARY_ESTIMANDS)
        ]
        assert rows["n_hours"].eq(32).all()
        assert rows["n_days"].eq(8).all()
        assert rows["observation_sha256"].nunique() == 1
        delta = _row(result, site, "rq3_paired_advantage")
        assert [delta.estimate, delta.ci_low, delta.ci_high] == [0, 0, 0]
        assert delta.classification == "inconclusive"
    heterogeneity = _row(result, "cross_site", inference.HETEROGENEITY_ESTIMAND)
    assert [heterogeneity.estimate, heterogeneity.ci_low, heterogeneity.ci_high] == [0, 0, 0]
    assert heterogeneity.classification == "descriptive"
    assert not result["classification"].str.contains("equival").any()


def test_uncertainty_is_computed_from_paired_day_blocks() -> None:
    frame = _frame()
    config = _config()
    result = compute_paired_inference(frame, config)
    site = frame.loc[frame.site.eq("Sukiyabashi")]
    rng = np.random.default_rng(config.seed)
    deltas = []
    for _ in range(config.resamples):
        starts = rng.integers(0, 7, size=4)
        dates = [f"2025-06-{day + 1:02d}" for start in starts for day in (start, start + 1)]
        sample = pd.concat([site.loc[site.local_calendar_date.eq(date)] for date in dates])
        index = spearmanr(sample.relative_nox_emission_index, sample.ambient_nox_ppm).statistic
        count = spearmanr(sample.count_only_index, sample.ambient_nox_ppm).statistic
        deltas.append(index - count)
    expected = np.quantile(
        deltas, [(1 - config.confidence_level) / 8, 1 - (1 - config.confidence_level) / 8]
    )
    actual = _row(result, "Sukiyabashi", "rq3_paired_advantage")
    assert [actual.ci_low, actual.ci_high] == pytest.approx(expected)
    assert actual.resamples_valid == config.resamples


def test_missing_calendar_days_are_empty_positions_and_whole_days_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame = frame.loc[frame.local_calendar_date.ne("2025-06-03")].copy()
    original = inference._point_values
    sampled_frames = []

    def capture(rows: pd.DataFrame, config: PairedInferenceConfig):
        if not rows.empty and rows["_cluster"].str.contains(":").any():
            sampled_frames.append(rows.copy())
        return original(rows, config)

    monkeypatch.setattr(inference, "_point_values", capture)
    compute_paired_inference(frame, _config())
    assert sampled_frames
    saw_empty_position = False
    for rows in sampled_frames:
        clusters = rows.groupby("_cluster", sort=False)
        observed = {}
        for label, cluster in clusters:
            number, date = label.split(":")
            observed[int(number)] = pd.Timestamp(date)
            assert cluster.site.value_counts().to_dict() == {"Sukiyabashi": 4, "Shinjuku": 4}
        for number in range(0, 8, 2):
            if number in observed and number + 1 in observed:
                assert observed[number + 1] - observed[number] == pd.Timedelta(days=1)
            elif number in observed or number + 1 in observed:
                saw_empty_position = True
    assert saw_empty_position


def test_wind_and_cross_site_contrast_signs() -> None:
    frame = _frame()
    target = np.tile(np.arange(1, 33), 2)
    frame["ambient_nox_ppm"] = target / 1000
    frame["relative_nox_emission_index"] = np.where(
        frame.wind_stratum.eq("favourable"), target, 33 - target
    )
    result = compute_paired_inference(frame, _config())
    assert _row(result, "Shinjuku", inference.WIND_ESTIMAND).estimate == pytest.approx(2)
    frame["relative_nox_emission_index"] = np.where(frame.site.eq("Shinjuku"), target, 33 - target)
    result = compute_paired_inference(frame, _config())
    assert _row(result, "cross_site", inference.HETEROGENEITY_ESTIMAND).estimate == pytest.approx(2)


def test_constant_and_insufficient_site_failures_are_isolated() -> None:
    frame = _frame()
    frame.loc[frame.site.eq("Sukiyabashi"), "count_only_index"] = 4
    result = compute_paired_inference(frame, _config())
    assert _row(result, "Sukiyabashi", "rq1_index").classification != "not_estimable"
    assert _row(result, "Sukiyabashi", "rq3_paired_advantage").classification == "not_estimable"
    assert _row(result, "Shinjuku", "rq3_paired_advantage").classification != "not_estimable"
    short = frame.loc[~frame.site.eq("Sukiyabashi") | frame.local_calendar_date.eq("2025-06-01")]
    result = compute_paired_inference(short, _config())
    assert _row(result, "Sukiyabashi", "rq1_index").failure_reason == "insufficient_site_support"
    assert _row(result, "Shinjuku", "rq1_index").classification != "not_estimable"
    assert (
        _row(result, "cross_site", inference.HETEROGENEITY_ESTIMAND).classification
        == "not_estimable"
    )


def test_missing_wind_does_not_block_primary_estimates() -> None:
    config = _config(wind_mode="unavailable", wind_definition_id=None)
    result = compute_paired_inference(_frame().drop(columns="wind_stratum"), config)
    assert (
        result.loc[result.estimand.eq(inference.WIND_ESTIMAND), "classification"]
        .eq("not_estimable")
        .all()
    )
    assert result.loc[result.estimand.eq("rq1_index"), "classification"].ne("not_estimable").all()


def test_explicit_wind_projection_thresholds_match_recorded_strata() -> None:
    frame = _frame()
    frame["along_road_wind_speed"] = np.where(frame.wind_stratum.eq("favourable"), 1, -1)
    projected = compute_paired_inference(
        frame, _config(wind_mode="projected_speed", favourable_min_mps=0.5, opposing_max_mps=-0.5)
    )
    recorded = compute_paired_inference(frame, _config())
    pdt.assert_frame_equal(projected, recorded)


@pytest.mark.parametrize(
    ("estimand", "lower", "upper", "expected"),
    [
        ("rq1_index", 0.4, 0.8, "positive"),
        ("rq1_index", 0.40001, 0.8, "practically_strong"),
        ("rq1_index", -0.2, 0, "non_positive"),
        ("rq3_paired_advantage", 0.1, 0.3, "inconclusive"),
        ("rq3_paired_advantage", 0.10001, 0.3, "meaningful_added_value"),
        ("rq3_paired_advantage", -0.3, -0.1, "inconclusive"),
        ("rq3_paired_advantage", -0.3, -0.10001, "meaningful_count_advantage"),
        ("rq3_paired_advantage", -0.05, 0.05, "inconclusive"),
        ("rq2_wind_contrast", 0.15, 0.4, "inconclusive"),
        ("rq2_wind_contrast", 0.15001, 0.4, "meaningful_moderation"),
        ("rq2_wind_contrast", 0, 0.14999, "proposed_magnitude_ruled_out"),
        ("rq2_wind_contrast", 0, 0.15, "inconclusive"),
    ],
)
def test_literal_practical_thresholds(
    estimand: str, lower: float, upper: float, expected: str
) -> None:
    assert classify_interval(estimand, lower, upper) == expected


def test_long_block_and_insufficient_valid_draws_cannot_create_intervals() -> None:
    result = compute_paired_inference(_frame(), _config(block_length_days=9))
    row = _row(result, "Shinjuku", "rq1_index")
    assert np.isfinite(row.estimate) and np.isnan(row.ci_low)
    assert row.failure_reason == "configured_block_exceeds_calendar"
    assert row.resamples_valid == 0
    short = _frame().loc[lambda data: data.local_calendar_date.isin(["2025-06-01", "2025-06-08"])]
    result = compute_paired_inference(
        short, _config(block_length_days=1, minimum_valid_resamples=49)
    )
    assert _row(result, "Shinjuku", "rq1_index").failure_reason == "insufficient_valid_resamples"


def test_hourly_bundle_hash_and_artifact_path_are_checked(tmp_path: Path) -> None:
    frame = _bundle_frame("Sukiyabashi")
    manifest_path = _write_bundle(tmp_path, frame)
    csv_path = tmp_path / "primary.csv"
    manifest = json.loads(manifest_path.read_text())
    loaded, identity = load_hourly_bundle(manifest_path)
    assert len(loaded) == len(frame)
    assert identity["site"] == "Sukiyabashi"
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt hash"):
        load_hourly_bundle(manifest_path)
    manifest["artifacts"]["ambient_primary"]["path"] = "../primary.csv"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local file"):
        load_hourly_bundle(manifest_path)


def test_hourly_bundle_parses_the_exact_hashed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _bundle_frame("Sukiyabashi")
    manifest_path = _write_bundle(tmp_path, frame)
    original_read = pd.read_csv

    def mutate_before_parse(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        (tmp_path / "primary.csv").write_text("corrupt replacement")
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(inference.pd, "read_csv", mutate_before_parse)
    loaded, _ = load_hourly_bundle(manifest_path)
    assert loaded.relative_nox_emission_index.tolist() == frame.relative_nox_emission_index.tolist()


def test_hourly_bundle_preserves_distinct_float_ranks(tmp_path: Path) -> None:
    frame = _bundle_frame("Sukiyabashi")
    frame.iloc[0, frame.columns.get_loc("relative_nox_emission_index")] = 1.0
    frame.iloc[1, frame.columns.get_loc("relative_nox_emission_index")] = np.nextafter(1.0, 2.0)
    loaded, _ = load_hourly_bundle(_write_bundle(tmp_path, frame))
    assert loaded.relative_nox_emission_index.iloc[0] < loaded.relative_nox_emission_index.iloc[1]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("coverage_mode", "threshold_normalize"),
        ("minimum_coverage_fraction", 0.8),
        ("partial_hour_normalisation", "same_covered_seconds_scale_for_all_predictors"),
        ("gap_rule", "require_contiguous_coverage"),
        ("monitor_label_semantics", "different-label"),
        ("method_config_id", "different_method"),
        ("method_config_sha256", "b" * 64),
    ],
)
def test_cli_rejects_mismatched_two_site_extraction_policy(
    tmp_path: Path, field: str, replacement: object
) -> None:
    first = _write_bundle(tmp_path / "suk", _bundle_frame("Sukiyabashi"))
    frame = _bundle_frame("Shinjuku")
    frame[field] = replacement
    second = _write_bundle(tmp_path / "shin", frame)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(_config())))
    with pytest.raises(ValueError, match=field):
        inference.main(
            [
                "--hourly-bundle",
                str(first),
                "--hourly-bundle",
                str(second),
                "--config",
                str(config_path),
                "--output-directory",
                str(tmp_path / "unused"),
            ]
        )


def test_artifact_writer_is_hash_bound_build_only_and_immutable(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    config = _config()
    result = compute_paired_inference(_frame(), config)
    output = tmp_path / "build" / "test_component"
    receipt_path = write_inference_artifacts(
        result,
        config,
        inputs=[{"site": "synthetic"}],
        output_directory=output,
        repository_root=tmp_path,
        code_revision="test",
    )
    receipt = json.loads(receipt_path.read_text())
    csv_path = output / receipt["results_csv"]
    assert receipt["results_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert (
        receipt["config_sha256"]
        == hashlib.sha256(
            json.dumps(
                asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
    original = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_inference_artifacts(
            result,
            config,
            inputs=[],
            output_directory=output,
            repository_root=tmp_path,
            code_revision="test",
        )
    assert csv_path.read_bytes() == original
    with pytest.raises(ValueError, match="build/"):
        write_inference_artifacts(
            result,
            config,
            inputs=[],
            output_directory=tmp_path / "results" / "v3",
            repository_root=tmp_path,
            code_revision="test",
        )
    missing_tracked = tmp_path / "build" / "tracked_missing"
    missing_tracked.mkdir()
    tracked_file = missing_tracked / "paired_inference.csv"
    tracked_file.write_text("preserved")
    subprocess.run(
        ["git", "add", "build/tracked_missing/paired_inference.csv"], cwd=tmp_path, check=True
    )
    tracked_file.unlink()
    missing_tracked.rmdir()
    with pytest.raises(FileExistsError, match="committed artifacts"):
        write_inference_artifacts(
            result,
            config,
            inputs=[],
            output_directory=missing_tracked,
            repository_root=tmp_path,
            code_revision="test",
        )
    assert not missing_tracked.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"primary_family_size": 1},
        {"block_length_days": 0},
        {"resamples": 18},
        {"minimum_valid_resamples": 50},
        {"wind_interval_family": "two_site_rq1_and_rq3"},
        {"seed": True},
    ],
)
def test_config_requires_explicit_valid_inference_choices(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_seed_changes_computed_uncertainty_and_draw_identity() -> None:
    first = compute_paired_inference(_frame(), _config())
    second = compute_paired_inference(_frame(), replace(_config(), seed=82))
    assert first.resample_sha256.iloc[0] != second.resample_sha256.iloc[0]
    assert not first.ci_low.equals(second.ci_low)
