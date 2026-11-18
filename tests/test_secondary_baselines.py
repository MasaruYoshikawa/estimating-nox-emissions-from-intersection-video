from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import spearmanr

from vehicle_emission.inference.paired_inference import (
    PairedInferenceConfig,
    compute_paired_inference,
)
from vehicle_emission.inference.secondary_baselines import (
    BaselineComparisonGroup,
    class_weighted_count_increments,
    compute_secondary_baseline_comparisons,
    main,
)


def _config() -> PairedInferenceConfig:
    return PairedInferenceConfig(
        resampling_method="union_calendar_moving_block",
        block_length_days=2,
        minimum_site_hours=3,
        minimum_site_days=2,
        minimum_stratum_hours=3,
        minimum_stratum_days=2,
        resamples=49,
        minimum_valid_resamples=19,
        seed=21,
        interval_method="percentile",
        interval_adjustment="bonferroni",
        confidence_level=0.95,
        primary_interval_family="primary",
        primary_family_size=4,
        wind_interval_family="wind",
        wind_family_size=2,
        wind_mode="unavailable",
        wind_definition_id=None,
        favourable_min_mps=None,
        opposing_max_mps=None,
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(761)
    output = []
    for site in ("Sukiyabashi", "Shinjuku"):
        for day in range(8):
            for hour in range(4):
                start = pd.Timestamp("2025-06-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
                output.append(
                    {
                        "site": site,
                        "monitor_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "monitor_end_utc": (start + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "local_calendar_date": start.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d"),
                        "ambient_nox_ppm": float(rng.integers(1, 25)) / 1000,
                        "relative_nox_emission_index": float(rng.integers(1, 25)),
                        "count_only_index": float(rng.integers(1, 25)),
                        "mean_speed_mps": float(rng.integers(0, 25)),
                        "alternative_index": float(rng.integers(1, 25)),
                        "primary_measurement_eligible": True,
                        "target_observation_status": "actual_observation",
                    }
                )
    return pd.DataFrame(output)


def _group(*columns: str, name: str = "secondary") -> BaselineComparisonGroup:
    return BaselineComparisonGroup(
        name, columns, tuple(f"test_{column}" for column in columns), name, 2 * len(columns)
    )


def test_computed_secondary_contrasts_share_support_and_primary_draws() -> None:
    frame = _frame()
    primary = compute_paired_inference(frame, _config())
    groups = (_group("count_only_index", "mean_speed_mps", "alternative_index"),)
    actual = compute_secondary_baseline_comparisons(frame, _config(), groups)
    pdt.assert_frame_equal(
        actual,
        compute_secondary_baseline_comparisons(
            frame.sample(frac=1, random_state=7), _config(), groups
        ),
    )
    assert len(actual) == 6
    assert set(actual.resample_sha256) == set(primary.resample_sha256)
    assert actual.paired_group_resample_sha256.nunique() == 1
    for site, selected in actual.groupby("site"):
        assert selected.observation_sha256.nunique() == 1
        assert selected.n_hours.eq(32).all()
        inputs = frame.loc[frame.site.eq(site)]
        for row in selected.itertuples():
            index = spearmanr(inputs.relative_nox_emission_index, inputs.ambient_nox_ppm).statistic
            baseline = spearmanr(inputs[row.predictor], inputs.ambient_nox_ppm).statistic
            assert [row.index_rho, row.baseline_rho, row.estimate] == pytest.approx(
                [index, baseline, index - baseline]
            )
            assert row.classification == "descriptive"
    assert not any("p_value" in column for column in actual.columns)


def test_optional_missingness_affects_declared_group_only_and_reports_reasons() -> None:
    frame = _frame()
    primary = compute_paired_inference(frame, _config())
    frame.loc[0, "mean_speed_mps"] = np.nan
    frame.loc[1, "alternative_index"] = np.inf
    groups = (
        _group("count_only_index", name="counts"),
        _group("count_only_index", "mean_speed_mps", "alternative_index"),
        _group("unavailable_index", name="unavailable"),
    )
    actual = compute_secondary_baseline_comparisons(frame, _config(), groups)
    pdt.assert_frame_equal(primary, compute_paired_inference(frame, _config()))
    narrowed = actual.loc[actual.comparison_group.eq("secondary") & actual.site.eq("Sukiyabashi")]
    assert narrowed.n_hours.eq(30).all() and narrowed.excluded_hours.eq(2).all()
    assert narrowed.observation_sha256.nunique() == 1
    assert json.loads(narrowed.exclusion_reasons.iloc[0]) == {
        "missing_or_invalid_predictor:mean_speed_mps": 1,
        "missing_or_invalid_predictor:alternative_index": 1,
    }
    assert actual.loc[actual.comparison_group.eq("counts"), "n_hours"].eq(32).all()
    missing = actual.loc[actual.comparison_group.eq("unavailable")]
    assert missing.n_hours.eq(0).all() and missing.classification.eq("not_estimable").all()
    assert missing.estimate.isna().all() and missing.ci_low.isna().all()
    assert actual.resample_sha256.nunique() == 1


def test_secondary_interval_is_recomputed_from_same_actual_blocks() -> None:
    frame = _frame()
    frame.loc[[0, 4], "mean_speed_mps"] = np.nan
    group = _group("count_only_index", "mean_speed_mps")
    actual = compute_secondary_baseline_comparisons(frame, _config(), (group,))
    source = frame.loc[frame.site.eq("Sukiyabashi") & frame.mean_speed_mps.notna()]
    rng = np.random.default_rng(_config().seed)
    draws = []
    for _ in range(_config().resamples):
        block_starts = rng.integers(0, 7, size=4)
        dates = [f"2025-06-{day+1:02d}" for start in block_starts for day in (start, start + 1)]
        sample = pd.concat([source.loc[source.local_calendar_date.eq(date)] for date in dates])
        draws.append(
            spearmanr(sample.relative_nox_emission_index, sample.ambient_nox_ppm).statistic
            - spearmanr(sample.mean_speed_mps, sample.ambient_nox_ppm).statistic
        )
    alpha = (1 - _config().confidence_level) / group.family_size
    expected = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    row = actual.loc[actual.site.eq("Sukiyabashi") & actual.predictor.eq("mean_speed_mps")].iloc[0]
    assert [row.ci_low, row.ci_high] == pytest.approx(expected)


def test_ties_constants_and_calendar_gaps_are_not_filled() -> None:
    frame = _frame().loc[lambda data: data.local_calendar_date.ne("2025-06-04")].copy()
    frame["tied_index"] = frame.relative_nox_emission_index
    frame["constant_speed"] = 0.0
    actual = compute_secondary_baseline_comparisons(
        frame, _config(), (_group("tied_index", "constant_speed"),)
    )
    tied = actual.loc[actual.predictor.eq("tied_index")]
    assert tied[["estimate", "ci_low", "ci_high"]].eq(0).all(axis=None)
    assert (
        actual.loc[actual.predictor.eq("constant_speed"), "classification"]
        .eq("not_estimable")
        .all()
    )
    assert actual.n_days.eq(7).all()
    assert set(actual.resample_sha256) == set(
        compute_paired_inference(frame, _config()).resample_sha256
    )
    failed = compute_secondary_baseline_comparisons(
        frame, replace(_config(), block_length_days=9), (_group("mean_speed_mps"),)
    )
    assert failed.failure_reason.eq("configured_block_exceeds_calendar").all()


def test_all_excluded_hours_report_not_estimable_without_resampling_observations() -> None:
    frame = _frame().iloc[:0].copy()
    dates = ("2025-06-01", "2025-06-08")
    primary = compute_paired_inference(frame, _config(), candidate_calendar_dates=dates)
    secondary = compute_secondary_baseline_comparisons(
        frame, _config(), (_group("mean_speed_mps"),), candidate_calendar_dates=dates
    )
    assert len(primary) == 9 and len(secondary) == 2
    for output in (primary, secondary):
        assert output.classification.eq("not_estimable").all()
        assert output[["estimate", "ci_low", "ci_high"]].isna().all(axis=None)
        assert output[["n_hours", "n_days", "resamples_valid"]].eq(0).all(axis=None)
    assert set(primary.resample_sha256) == set(secondary.resample_sha256)
    extended = compute_paired_inference(
        frame, _config(), candidate_calendar_dates=("2025-06-01", "2025-06-09")
    )
    assert extended.resample_sha256.iloc[0] != primary.resample_sha256.iloc[0]
    with pytest.raises(ValueError):
        compute_paired_inference(frame, _config())
    with pytest.raises(ValueError):
        compute_paired_inference(frame, _config(), candidate_calendar_dates=("not-a-date",))


@pytest.mark.parametrize(
    "field,value",
    [
        ("family_size", 1),
        ("predictor_columns", ("ambient_nox_ppm",)),
        ("definition_ids", ()),
        ("group_id", ""),
    ],
)
def test_group_definitions_fail_closed(field: str, value: object) -> None:
    values = dict(
        group_id="speed",
        predictor_columns=("mean_speed_mps",),
        definition_ids=("calibrated_speed",),
        interval_family="speed",
        family_size=2,
    )
    values[field] = value
    with pytest.raises(ValueError):
        BaselineComparisonGroup(**values)  # type: ignore[arg-type]


def test_class_weighted_count_uses_supplied_weights_and_preserves_unknowns() -> None:
    rows = pd.DataFrame(
        {
            "site": ["Sukiyabashi"] * 4,
            "event_id": ["a", "b", "c", "d"],
            "second_start_utc": ["2025-06-01T00:00:00Z"] * 3 + ["2025-06-01T00:00:01Z"],
            "vehicle_class": ["car", "truck", "car", "unknown"],
        }
    )
    result = class_weighted_count_increments(
        rows, {"car": 1.0, "truck": 3.0}, definition_id="explicit_test_weights"
    )
    assert result.class_weighted_count_increment.iloc[0] == 5.0
    assert pd.isna(result.class_weighted_count_increment.iloc[1])
    assert result.missing_weight_classes.iloc[1] == "unknown"
    assert result.weights_sha256.nunique() == 1
    with pytest.raises(ValueError, match="unique"):
        class_weighted_count_increments(
            pd.concat([rows, rows.iloc[[0]]]), {"car": 1.0}, definition_id="weights"
        )
    with pytest.raises(ValueError, match="non-negative"):
        class_weighted_count_increments(rows, {"car": -1.0}, definition_id="weights")


def _write_cli_bundle(directory: Path, frame: pd.DataFrame) -> Path:
    directory.mkdir()
    metadata = {
        "analysis_scope": "locked_retrospective_two_site",
        "configuration_evidence_status": "self_attested_scaffold_only",
        "method_config_hash_verified": False,
        "method_config_id": "test_method",
        "method_config_sha256": "a" * 64,
        "source_snapshot_id": str(frame.site.iloc[0]),
        "precipitation_rule": "dry_only",
        "precipitation_mm": 0.0,
        "weather_observation_status": "actual_observation",
        "weather_alignment": "monitor_interval",
        "weather_source_id": "test_weather",
        "predictor_offset_hours": 0,
        "coverage_mode": "complete_only",
        "minimum_coverage_fraction": 1.0,
        "partial_hour_normalisation": "none",
        "gap_rule": "allow_internal_gaps",
        "monitor_label_semantics": "end-labelled-hour",
    }
    for field, value in metadata.items():
        frame[field] = value
    frame["predictor_window_start_utc"] = frame.monitor_start_utc
    frame["predictor_window_end_utc"] = frame.monitor_end_utc
    csv_path = directory / "hourly.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        **metadata,
        "manifest_type": "hourly_measurement_scaffold_bundle",
        "schema_version": 2,
        "site": frame.site.iloc[0],
        "artifacts": {
            "ambient_primary": {
                "path": csv_path.name,
                "row_count": len(frame),
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_cli_binds_inputs_and_refuses_existing_or_nonbuild_outputs(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    frame = _frame()
    bundles = [
        _write_cli_bundle(tmp_path / site, frame.loc[frame.site.eq(site)].copy())
        for site in ("Sukiyabashi", "Shinjuku")
    ]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(_config())))
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps({"groups": [asdict(_group("count_only_index", "mean_speed_mps"))]})
    )
    output = tmp_path / "build" / "new_baselines"
    args = ["--config", str(config), "--groups", str(groups)]
    for bundle in bundles:
        args.extend(["--hourly-bundle", str(bundle)])
    assert main(args + ["--output-directory", str(output)]) == 0
    receipt = json.loads((output / "secondary_baselines_receipt.json").read_text())
    csv_path = output / "secondary_baselines.csv"
    original = csv_path.read_bytes()
    assert receipt["results_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["row_count"] == 4 and len(receipt["inputs"]) == 2
    with pytest.raises(FileExistsError):
        main(args + ["--output-directory", str(output)])
    assert csv_path.read_bytes() == original
    with pytest.raises(ValueError, match="build"):
        main(args + ["--output-directory", str(tmp_path / "results" / "unsafe")])
    assert not (tmp_path / "results").exists()
