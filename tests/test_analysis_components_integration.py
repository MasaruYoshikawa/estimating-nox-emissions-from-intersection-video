from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from vehicle_emission.inference import generate_v3_figures, paired_inference
from vehicle_emission.inference.hourly_measurements import (
    FrozenHourlyExtractionConfig,
    extract_hourly_measurements,
    write_hourly_measurements_scaffold_v2,
)
from vehicle_emission.inference.paired_inference import PairedInferenceConfig


def _inputs(site: str) -> dict[str, pd.DataFrame]:
    candidate_rows = []
    monitor_rows = []
    weather_rows = []
    clips = []
    seconds = []
    for day in range(6):
        for hour in range(4):
            rank = day * 4 + hour + 1
            source_start = pd.Timestamp("2025-06-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
            monitor_start = source_start + pd.Timedelta(hours=1)
            monitor_end = monitor_start + pd.Timedelta(hours=1)
            key = {
                "site": site,
                "monitor_start_utc": monitor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "monitor_end_utc": monitor_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            candidate_rows.append(key)
            monitor_rows.append(
                {
                    **key,
                    "ambient_nox_ppm": rank / 1000,
                    "monitor_label_utc": key["monitor_end_utc"],
                    "target_observation_status": "actual_observation",
                }
            )
            weather_rows.append(
                {
                    "site": site,
                    "weather_start_utc": source_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "weather_end_utc": monitor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "weather_label_utc": monitor_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "weather_source_id": f"fixture-weather-{site}",
                    "weather_observation_status": "actual_observation",
                    "precipitation_mm": 1.0 if rank == 24 else 0.0,
                    "along_road_wind_speed": 1.0 if hour < 2 else -1.0,
                }
            )
            clips.append(
                {
                    "site": site,
                    "clip_id": f"fixture/{site}/{day}/{hour}",
                    "clip_start_utc": source_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "clip_end_utc": (source_start + pd.Timedelta(seconds=2)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "timing_evidence_id": f"fixture-clock-{site}",
                }
            )
            index_rank = rank if site == "Sukiyabashi" else 25 - rank
            for second in range(2):
                seconds.append(
                    {
                        "site": site,
                        "second_start_utc": (source_start + pd.Timedelta(seconds=second)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "coverage_valid": True,
                        "count_only_increment": 25 - rank if second == 0 else 0,
                        "relative_nox_emission_index_increment": index_rank / 100_000,
                    }
                )
    return {
        "candidate_hours": pd.DataFrame(candidate_rows),
        "monitor_observations": pd.DataFrame(monitor_rows),
        "video_seconds": pd.DataFrame(seconds),
        "weather_observations": pd.DataFrame(weather_rows),
        "clip_intervals": pd.DataFrame(clips),
    }


def test_actual_scaffold_to_inference_and_figure_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    sentinel = root / "results" / "historical_v3.csv"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"immutable historical results\n")
    module_copy = root / "src" / "vehicle_emission" / "inference" / "paired_inference.py"
    module_copy.parent.mkdir(parents=True)
    module_copy.write_bytes(Path(paired_inference.__file__).read_bytes())
    subprocess.run(["git", "add", "results", "src"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Component Test",
            "-c",
            "user.email=component@example.invalid",
            "commit",
            "-qm",
            "Controlled integration fixture",
        ],
        cwd=root,
        check=True,
    )
    historical_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    monkeypatch.setattr(paired_inference, "__file__", str(module_copy))
    manifests = []
    for site in paired_inference.SITES:
        config = FrozenHourlyExtractionConfig(
            method_config_id="controlled-common-extraction",
            method_config_sha256="a" * 64,
            source_snapshot_id=f"controlled-source-{site}",
            status="self_attested_scaffold_only",
            analysis_scope="locked_retrospective_two_site",
            site=site,
            monitor_label_semantics="end-labelled-hour",
            predictor_offset_hours=1,
            coverage_rule_id="two-second-integration-fixture-only",
            coverage_mode="threshold_normalize",
            minimum_coverage_fraction=2 / 3600,
            partial_hour_normalisation="same_covered_seconds_scale_for_all_predictors",
            gap_rule="require_contiguous_coverage",
            background_site=None,
            precipitation_rule="dry_only",
            weather_alignment="predictor_window",
        )
        tables = extract_hourly_measurements(
            config=config, background_observations=None, **_inputs(site)
        )
        assert len(tables.audit) == 24
        assert len(tables.ambient_primary) == 23
        assert tables.audit.iloc[-1].primary_exclusion_reasons == "rainy_hour"
        assert tables.ambient_primary.covered_seconds.eq(2).all()
        assert tables.ambient_primary.count_only_index.iloc[0] == 24 * 1800
        output = root / "build" / "hourly_inputs" / site.lower()
        write_hourly_measurements_scaffold_v2(tables, output_directory=output)
        manifests.append(output / f"{site.lower()}_hourly_measurements_scaffold_v2.json")

    candidate_config = PairedInferenceConfig(
        resampling_method="union_calendar_moving_block",
        block_length_days=2,
        minimum_site_hours=3,
        minimum_site_days=2,
        minimum_stratum_hours=3,
        minimum_stratum_days=2,
        resamples=49,
        minimum_valid_resamples=19,
        seed=71,
        interval_method="percentile",
        interval_adjustment="bonferroni",
        confidence_level=0.95,
        primary_interval_family="controlled-two-site-rq1-rq3",
        primary_family_size=4,
        wind_interval_family="controlled-two-site-rq2",
        wind_family_size=2,
        wind_mode="projected_speed",
        wind_definition_id="controlled-signed-wind-fixture",
        favourable_min_mps=0.5,
        opposing_max_mps=-0.5,
    )
    config_path = root / "build" / "candidate_config.json"
    config_path.write_text(json.dumps(asdict(candidate_config)))
    analysis_output = root / "build" / "computed_analysis"
    assert (
        paired_inference.main(
            [
                "--hourly-bundle",
                str(manifests[0]),
                "--hourly-bundle",
                str(manifests[1]),
                "--config",
                str(config_path),
                "--output-directory",
                str(analysis_output),
            ]
        )
        == 0
    )
    csv_path = analysis_output / "paired_inference.csv"
    receipt_path = analysis_output / "paired_inference_receipt.json"
    results = pd.read_csv(csv_path)
    assert len(results) == 9
    site_rows = results.loc[results.site.ne("cross_site")]
    assert site_rows.n_days.eq(6).all()
    assert site_rows.n_hours.eq(23).all()
    for site, expected_index, expected_delta in [("Sukiyabashi", 1, 2), ("Shinjuku", -1, 0)]:
        index = results.loc[results.site.eq(site) & results.estimand.eq("rq1_index")].iloc[0]
        delta = results.loc[
            results.site.eq(site) & results.estimand.eq("rq3_paired_advantage")
        ].iloc[0]
        assert index.estimate == pytest.approx(expected_index)
        assert delta.estimate == pytest.approx(expected_delta)
        assert delta.ci_low == pytest.approx(expected_delta)
        assert delta.ci_high == pytest.approx(expected_delta)
    heterogeneity = results.loc[results.site.eq("cross_site")].iloc[0]
    assert heterogeneity.estimate == pytest.approx(-2)
    assert results.resample_sha256.nunique() == 1
    assert results.evidence_status.eq("provisional_components_only").all()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["row_count"] == 9
    assert receipt["results_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert {item["site"] for item in receipt["inputs"]} == set(paired_inference.SITES)

    figure_output = root / "build" / "computed_figures"
    assert (
        generate_v3_figures.main(
            [
                "--results-csv",
                str(csv_path),
                "--receipt",
                str(receipt_path),
                "--out-dir",
                str(figure_output),
            ]
        )
        == 0
    )
    figure_receipt = json.loads((figure_output / generate_v3_figures.RECEIPT_NAME).read_text())
    assert len(figure_receipt["artifacts"]) == 4
    for artifact in figure_receipt["artifacts"]:
        assert (
            artifact["sha256"]
            == hashlib.sha256((figure_output / artifact["path"]).read_bytes()).hexdigest()
        )
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == historical_hash
    assert not subprocess.run(
        ["git", "status", "--porcelain", "--", "results"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
