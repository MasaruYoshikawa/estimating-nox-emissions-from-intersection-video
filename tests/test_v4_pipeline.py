from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import vehicle_emission.inference.v4_pipeline as pipeline
from measurement_fixture import write_measurement_fixture
from vehicle_emission.inference.frozen_analysis import main as orchestrate
from vehicle_emission.inference.paired_inference import classify_interval
from vehicle_emission.inference.v4_contracts import (
    EVALUATION_DESIGN,
    THRESHOLDS,
    V4ContractError,
    canonical_sha256,
    scientific_config_sha256,
    validate_v4_config,
)
from vehicle_emission.inference.v4_smoke import main as smoke_cli
from vehicle_emission.measurement.evaluate import evaluate_measurements

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    res = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} in {root} failed code {res.returncode}:\n"
            f"stdout: {res.stdout!r}\nstderr: {res.stderr!r}"
        )
    return res.stdout.strip()


def _write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(root: Path, *, baselines: bool = False) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    _git(root, "init", "-q")
    shutil.copytree(
        ROOT / "src/vehicle_emission",
        root / "src/vehicle_emission",
        ignore=shutil.ignore_patterns("__pycache__", ".git"),
        ignore_dangling_symlinks=True,
    )
    (root / ".gitignore").write_text("build/\n", encoding="utf-8")
    config = {
        "schema_version": 4,
        "manifest_type": "numeric_export_analysis_v4",
        "analysis_id": "controlled-component-fixture",
        "mode": "preparation",
        "analysis_scope": "locked_retrospective_two_site",
        "evaluation_design": dict(EVALUATION_DESIGN),
        "thresholds": dict(THRESHOLDS),
        "hourly_policy": {
            "monitor_label_semantics": "end-labelled-hour",
            "predictor_offset_hours": 0,
            "coverage_rule_id": "synthetic-10-seconds-only",
            "coverage_mode": "threshold_normalize",
            "minimum_coverage_fraction": 10 / 3600,
            "partial_hour_normalisation": "same_covered_seconds_scale_for_all_predictors",
            "gap_rule": "require_contiguous_coverage",
            "precipitation_rule": "dry_only",
            "weather_alignment": "monitor_interval",
        },
        "inference": {
            "resampling_method": "union_calendar_moving_block",
            "block_length_days": 2,
            "minimum_site_hours": 3,
            "minimum_site_days": 2,
            "minimum_stratum_hours": 3,
            "minimum_stratum_days": 2,
            "resamples": 19,
            "minimum_valid_resamples": 19,
            "seed": 12,
            "interval_method": "percentile",
            "interval_adjustment": "bonferroni",
            "confidence_level": 0.95,
            "primary_interval_family": "two-site-rq1-rq3",
            "primary_family_size": 4,
            "wind_interval_family": "conditional-two-site-wind",
            "wind_family_size": 2,
            "wind_mode": "projected_speed",
            "wind_definition_id": "controlled-projection-only",
            "favourable_min_mps": 0.5,
            "opposing_max_mps": -0.5,
        },
        "sites": {},
        "baselines": {"secondary_increment_columns": {}, "groups": []},
        "evidence": {
            key: None
            for key in (
                "method_lock",
                "source_review",
                "measurement_receipt",
                "measurement_acceptance",
            )
        },
    }
    if baselines:
        config["baselines"] = {
            "secondary_increment_columns": {"class_weighted_count_index": "weighted_increment"},
            "groups": [
                {
                    "group_id": "targeted-count-speed",
                    "predictor_columns": [
                        "count_only_index",
                        "class_weighted_count_index",
                        "mean_speed_mps",
                    ],
                    "definition_ids": [
                        "unique-event-count",
                        "caller-fixed-weight-fixture",
                        "calibrated-speed-fixture",
                    ],
                    "interval_family": "secondary-two-site",
                    "family_size": 6,
                }
            ],
        }
    for site in pipeline.SITES:
        starts = [
            pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=day, hours=hour)
            for day in range(6)
            for hour in range(2)
        ]

        def utc(stamp: pd.Timestamp) -> str:
            return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        candidates = pd.DataFrame(
            {
                "site": site,
                "monitor_start_utc": [utc(value) for value in starts],
                "monitor_end_utc": [utc(value + pd.Timedelta(hours=1)) for value in starts],
            }
        )
        monitors = candidates.copy()
        monitors["station_id"] = f"{site}-fixture-monitor"
        monitors["ambient_nox_ppm"] = [
            0.002 + 0.001 * ((row * 7) % 13) for row in range(len(starts))
        ]
        monitors["monitor_label_utc"] = monitors["monitor_end_utc"]
        monitors["target_observation_status"] = "actual_observation"
        seconds = []
        for row, start in enumerate(starts):
            for number in range(10):
                seconds.append(
                    {
                        "site": site,
                        "second_start_utc": utc(start + pd.Timedelta(seconds=number)),
                        "coverage_valid": True,
                        "count_only_increment": row % 4 + 1 if number == 0 else 0,
                        "relative_nox_emission_index_increment": (row % 7 + 1) / 10,
                        "source_record_id": f"fixture-row-{row}-{number}",
                        "weighted_increment": (row % 4 + 1) * 2 if number == 0 else 0,
                    }
                )
        clips = pd.DataFrame(
            {
                "site": site,
                "clip_id": [f"{site}-clip-{number}" for number in range(len(starts))],
                "clip_start_utc": [utc(value) for value in starts],
                "clip_end_utc": [utc(value + pd.Timedelta(seconds=10)) for value in starts],
                "timing_evidence_id": "controlled-clock",
                "source_video_sha256": "1" * 64,
                "source_video_path": [
                    f"{site}/day-{number // 2}/video-{number}.mp4" for number in range(len(starts))
                ],
            }
        )
        weather = pd.DataFrame(
            {
                "site": site,
                "weather_start_utc": candidates["monitor_start_utc"],
                "weather_end_utc": candidates["monitor_end_utc"],
                "weather_label_utc": candidates["monitor_end_utc"],
                "weather_source_id": "controlled-weather",
                "weather_observation_status": "actual_observation",
                "precipitation_mm": 0.0,
                "along_road_wind_speed": [
                    0.8 if number % 2 else -0.8 for number in range(len(starts))
                ],
            }
        )
        frames = {
            "candidate_hours": candidates,
            "monitor_observations": monitors,
            "video_seconds": pd.DataFrame(seconds),
            "clip_intervals": clips,
            "weather_observations": weather,
        }
        if baselines:
            frames["speed_samples"] = pd.DataFrame(
                {
                    "site": site,
                    "sample_id": [f"sample-{number}" for number in range(len(starts))],
                    "track_id": [f"track-{number}" for number in range(len(starts))],
                    "sample_start_utc": [utc(value) for value in starts],
                    "sample_end_utc": [utc(value + pd.Timedelta(seconds=10)) for value in starts],
                    "speed_mps": [row % 5 + 2 for row in range(len(starts))],
                    "speed_valid": True,
                    "calibration_id": f"{site}-fixture-calibration",
                    "calibration_valid": True,
                }
            )
        sources = {"background_observations": None, "speed_samples": None}
        for role, frame in frames.items():
            path = root / "data" / site / f"{role}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            sources[role] = _reference(root, path)
        config["sites"][site] = {
            "source_snapshot_id": f"controlled-{site}-snapshot",
            "monitor_station_id": f"{site}-fixture-monitor",
            "weather_source_id": "controlled-weather",
            "background_site": None,
            "detector_checkpoint_sha256": "a" * 64,
            "tracker_config_sha256": "b" * 64,
            "estimator_definition_sha256": "c" * 64,
            "sources": sources,
        }
    config_path = root / "data/analysis-config-v4.json"
    _write_json(config_path, config)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "controlled inputs",
    )
    return config_path, config


def test_main_orchestration_calculates_complete_versioned_bundle(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root, baselines=True)
    assert (
        orchestrate(
            [
                "execute-v4",
                "--config",
                str(config_path),
                "--repository-root",
                str(root),
                "--output-directory",
                "build/calculated",
            ]
        )
        == 0
    )
    target = root / "build/calculated"
    manifest = pipeline.validate_v4_result_bundle(target)
    assert manifest["mode"] == "preparation"
    assert manifest["evidence_status"] == "provisional_components_only"
    assert {value["eligible_hours"] for value in manifest["support"].values()} == {12}
    assert len(list(target.glob("*.png"))) == 4
    assert all("_v4." in path.name for path in target.iterdir())
    rows = pd.read_csv(target / pipeline.RESULTS_NAME)
    assert len(rows) == 9
    assert rows["resample_sha256"].nunique() == 1
    assert rows["evaluation_status"].isin(["exploratory", "descriptive", "not_estimable"]).all()
    secondary = pd.read_csv(target / "secondary_baselines_v4.csv")
    assert len(secondary) == 6
    assert secondary["resample_sha256"].eq(rows.iloc[0]["resample_sha256"]).all()
    assert not (root / "results").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "threshold",
        "iid",
        "different_detector",
        "missing_evidence",
        "rain_bypass",
        "extra_engine",
    ],
)
def test_configuration_refuses_unsupported_or_unbound_choices(
    tmp_path: Path, mutation: str
) -> None:
    _, config = _fixture(tmp_path / "repository")
    value = copy.deepcopy(config)
    if mutation == "version":
        value["schema_version"] = 3
    elif mutation == "threshold":
        value["thresholds"]["rq3_advantage"] = 0.0
    elif mutation == "iid":
        value["inference"]["resampling_method"] = "iid_hours"
    elif mutation == "different_detector":
        value["sites"]["Shinjuku"]["detector_checkpoint_sha256"] = "d" * 64
    elif mutation == "missing_evidence":
        value["mode"] = "accepted_source"
    elif mutation == "rain_bypass":
        value["hourly_policy"]["precipitation_rule"] = "not_configured"
    else:
        value["engine"] = "arbitrary.module"
    with pytest.raises(ValueError):
        validate_v4_config(value)


def test_input_byte_identity_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, config = _fixture(root)
    reference = config["sites"]["Sukiyabashi"]["sources"]["candidate_hours"]
    path = root / reference["path"]
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(V4ContractError, match="input hash"):
        pipeline.run_v4_analysis(
            config_path, repository_root=root, output_directory=Path("build/changed")
        )
    assert not (root / "build/changed").exists()
    path.unlink()
    external = tmp_path / "external.csv"
    external.write_bytes(original)
    try:
        path.symlink_to(external)
    except OSError as err:
        pytest.skip(f"Symlinks not supported on system: {err}")
    with pytest.raises(V4ContractError, match="symlink"):
        pipeline.run_v4_analysis(
            config_path, repository_root=root, output_directory=Path("build/escaped")
        )


def test_tracked_deleted_and_existing_outputs_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)
    output = root / "build/retained"
    output.mkdir(parents=True)
    archived = output / "archive_v3.csv"
    archived.write_bytes(b"preserve this\n")
    _git(root, "add", "-f", archived.relative_to(root).as_posix())
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "archived result",
    )
    with pytest.raises(FileExistsError):
        pipeline.run_v4_analysis(config_path, repository_root=root, output_directory=output)
    assert archived.read_bytes() == b"preserve this\n"
    archived.unlink()
    output.rmdir()
    with pytest.raises(FileExistsError, match="committed"):
        pipeline.run_v4_analysis(config_path, repository_root=root, output_directory=output)


def test_failed_postcondition_never_publishes_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)

    def reject(*args: object, **kwargs: object) -> None:
        raise V4ContractError("controlled postcondition failure")

    monkeypatch.setattr(pipeline, "validate_v4_result_bundle", reject)
    with pytest.raises(V4ContractError, match="postcondition"):
        pipeline.run_v4_analysis(
            config_path, repository_root=root, output_directory=Path("build/not-published")
        )
    assert not (root / "build/not-published").exists()
    assert not list((root / "build").glob("v4-stage-*"))
    assert not list((root / "build").glob("*.publish-lock"))


def test_result_validator_rejects_modified_artifacts_and_unmanifested_files(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)
    manifest_path = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/validated")
    )
    target = manifest_path.parent
    result_path = target / pipeline.RESULTS_NAME
    original = result_path.read_bytes()
    result_path.write_bytes(original + b"\n")
    with pytest.raises(V4ContractError, match="artifact"):
        pipeline.validate_v4_result_bundle(target)
    result_path.write_bytes(original)
    (target / "unmanifested.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(V4ContractError, match="unmanifested"):
        pipeline.validate_v4_result_bundle(target)


def test_literal_thresholds_do_not_promote_boundary_or_null_intervals() -> None:
    assert classify_interval("rq3_paired_advantage", 0.10, 0.30) == "inconclusive"
    assert classify_interval("rq3_paired_advantage", 0.101, 0.30) == "meaningful_added_value"
    assert classify_interval("rq3_paired_advantage", -0.05, 0.05) == "inconclusive"
    assert classify_interval("rq2_wind_contrast", 0.15, 0.30) == "inconclusive"
    assert classify_interval("cross_site_index_heterogeneity", -0.05, 0.05) == "descriptive"


def _record_accepted_fixture(
    root: Path, *, baselines: bool = False, review_speed: bool = True
) -> Path:
    config_path, config = _fixture(root, baselines=baselines)
    preliminary = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/support-preparation")
    )
    support = json.loads(preliminary.read_text(encoding="utf-8"))["support"]
    measurement_config = write_measurement_fixture(
        root, checkpoint_sha256="a" * 64, tracker_sha256="b" * 64
    )
    receipt = evaluate_measurements(
        config_path=measurement_config,
        repository_root=root,
        output_directory=root / "build/measurement",
    )
    config["mode"] = "accepted_source"
    scientific_hash = scientific_config_sha256(config)
    evidence_directory = root / "data/evidence"
    evidence_directory.mkdir(parents=True)
    calibration_ids = {site: [f"{site}-calibration"] for site in pipeline.SITES}
    production = {
        "schema_version": 4,
        "record_type": "production_calibration_review",
        "status": "verified",
        "video_seconds_sha256": {
            site: config["sites"][site]["sources"]["video_seconds"]["sha256"]
            for site in pipeline.SITES
        },
        "calibration_ids": calibration_ids,
        "production_use_verified": True,
        "speed_samples_sha256": {
            site: (
                config["sites"][site]["sources"]["speed_samples"]["sha256"]
                if baselines and review_speed
                else None
            )
            for site in pipeline.SITES
        },
    }
    production_path = evidence_directory / "production-review.json"
    _write_json(production_path, production)
    design_path = evidence_directory / "design-review.json"
    _write_json(
        design_path,
        {
            "controlled_fixture": True,
            "interpretation": "estimation_only",
            "no_precision_claim": True,
        },
    )
    receipt_path = evidence_directory / "measurement-receipt.json"
    _write_json(receipt_path, receipt)
    records = {
        "method_lock": {
            "schema_version": 4,
            "record_type": "retrospective_method_lock",
            "status": "recorded",
            "scientific_config_sha256": scientific_hash,
            "evaluation_design": EVALUATION_DESIGN,
            "support": support,
            "interpretation": "estimation_only",
            "design_evidence": _reference(root, design_path),
        },
        "source_review": {
            "schema_version": 4,
            "record_type": "numeric_export_source_review",
            "status": "verified",
            "source_inventory_sha256": canonical_sha256(config["sites"]),
            "station_role": "spatially_separated_ambient_reference",
            "clip_timing_reviewed": True,
            "feature_records_traceable": True,
        },
        "measurement_receipt": receipt,
        "measurement_acceptance": {
            "schema_version": 4,
            "record_type": "measurement_acceptance",
            "status": "accepted",
            "measurement_receipt_sha256": _reference(root, receipt_path)["sha256"],
            "scientific_config_sha256": scientific_hash,
            "accepted_tasks": ["vehicle_classes", "entry_counts", "tracking", "speed"],
            "calibration_ids": calibration_ids,
            "criteria_sha256": canonical_sha256(receipt["acceptance"]["criteria"]),
            "production_calibration_review": _reference(root, production_path),
        },
    }
    for role, record in records.items():
        path = (
            receipt_path if role == "measurement_receipt" else evidence_directory / f"{role}.json"
        )
        _write_json(path, record)
        config["evidence"][role] = _reference(root, path)
    _write_json(config_path, config)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "controlled review records; no empirical evidence",
    )
    return config_path


def test_accepted_source_executes_with_replayed_measurements_and_recorded_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    path = _record_accepted_fixture(root)
    output = pipeline.run_v4_analysis(
        path, repository_root=root, output_directory=Path("results/controlled_v4")
    )
    manifest = pipeline.validate_v4_result_bundle(output.parent)
    assert manifest["evidence_status"] == "source_reproduced_unreviewed"
    assert manifest["evidence"]["interpretation"] == "estimation_only"
    assert {row["evaluation_status"] for row in manifest["result_rows"]} <= {
        "exploratory",
        "descriptive",
        "not_estimable",
    }


def test_unreviewed_optional_speed_does_not_block_accepted_primary(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    path = _record_accepted_fixture(root, baselines=True, review_speed=False)
    output = pipeline.run_v4_analysis(
        path, repository_root=root, output_directory=Path("results/optional-speed-unavailable")
    )
    manifest = pipeline.validate_v4_result_bundle(output.parent)
    assert {value["eligible_hours"] for value in manifest["support"].values()} == {12}
    primary = pd.read_csv(output.parent / pipeline.RESULTS_NAME)
    assert primary.loc[primary.estimand.eq("rq1_index"), "n_hours"].eq(12).all()
    secondary = pd.read_csv(output.parent / "secondary_baselines_v4.csv")
    speed = secondary.loc[secondary.predictor.eq("mean_speed_mps")]
    assert speed["classification"].eq("not_estimable").all()
    assert speed["n_hours"].eq(0).all()


@pytest.mark.parametrize(
    "mutation", ["support", "measurement", "production", "method", "precision"]
)
def test_accepted_source_does_not_trust_status_strings(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "repository"
    path = _record_accepted_fixture(root)
    config = json.loads(path.read_text(encoding="utf-8"))
    role = {
        "support": "method_lock",
        "measurement": "measurement_receipt",
        "production": "measurement_acceptance",
        "method": "method_lock",
        "precision": "method_lock",
    }[mutation]
    reference = config["evidence"][role]
    evidence_path = root / reference["path"]
    record = json.loads(evidence_path.read_text(encoding="utf-8"))
    if mutation == "support":
        record["support"]["Shinjuku"]["eligible_hours"] += 1
    elif mutation == "measurement":
        record["acceptance"]["criteria"]["speed_max_mae_mps"] = 500
    elif mutation == "production":
        record["calibration_ids"]["Shinjuku"] = ["wrong-calibration"]
    elif mutation == "precision":
        record["interpretation"] = "decision_quality"
    else:
        record["scientific_config_sha256"] = "0" * 64
    _write_json(evidence_path, record)
    config["evidence"][role] = _reference(root, evidence_path)
    _write_json(path, config)
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "invalid controlled review",
    )
    with pytest.raises(ValueError):
        pipeline.run_v4_analysis(
            path, repository_root=root, output_directory=Path("results/not-published")
        )
    assert not (root / "results/not-published").exists()


def test_controlled_smoke_generates_an_explicit_example_config(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _fixture(root)
    assert smoke_cli(["--output-directory", "build/smoke", "--repository-root", str(root)]) == 0
    output = root / "build/smoke/calculated/analysis_manifest_v4.json"
    manifest = pipeline.validate_v4_result_bundle(output.parent)
    config = json.loads(
        (root / "build/smoke/controlled_analysis_config_v4.json").read_text(encoding="utf-8")
    )
    assert config["analysis_id"] == "controlled_smoke_not_empirical"
    assert manifest["evidence_status"] == "provisional_components_only"
    wind = [row for row in manifest["result_rows"] if row["estimand"] == "rq2_wind_contrast"]
    assert all(row["classification"] == "not_estimable" for row in wind)


def test_execution_inventory_cannot_name_different_committed_code(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)
    changed = root / "src/vehicle_emission/inference/paired_inference.py"
    changed.write_text(
        changed.read_text(encoding="utf-8") + "\n# Different committed implementation.\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "different execution code",
    )
    with pytest.raises(V4ContractError, match="actually executing"):
        pipeline.run_v4_analysis(
            config_path, repository_root=root, output_directory=Path("build/mismatched-code")
        )


@pytest.mark.parametrize("field", ["scientific_config_sha256", "configuration", "mode", "evidence"])
def test_result_metadata_is_bound_to_the_staged_configuration(tmp_path: Path, field: str) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)
    output = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/metadata")
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    if field == "scientific_config_sha256":
        manifest[field] = "0" * 64
    elif field == "configuration":
        manifest[field]["sha256"] = "0" * 64
    elif field == "mode":
        manifest[field] = "accepted_source"
        manifest["evidence_status"] = "source_reproduced_unreviewed"
    else:
        manifest[field]["interpretation"] = "decision_quality"
    _write_json(output, manifest)
    with pytest.raises(ValueError):
        pipeline.validate_v4_result_bundle(output.parent)


def test_result_evaluation_status_cannot_be_promoted_by_rehashing(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root)
    output = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/row-status")
    )
    directory = output.parent
    csv_path = directory / pipeline.RESULTS_NAME
    rows = pd.read_csv(csv_path, float_precision="round_trip", keep_default_na=False)
    rows.loc[0, "evaluation_status"] = "confirmatory"
    rows.to_csv(csv_path, index=False, float_format="%.17g")
    receipt_path = directory / pipeline.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["results_sha256"] = _reference(root, csv_path)["sha256"]
    _write_json(receipt_path, receipt)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest["result_rows"] = json.loads(
        pd.read_csv(csv_path, float_precision="round_trip").to_json(
            orient="records", double_precision=15
        )
    )
    for artifact in manifest["artifacts"]:
        path = directory / artifact["path"]
        artifact["sha256"] = _reference(root, path)["sha256"]
        artifact["size_bytes"] = path.stat().st_size
    _write_json(output, manifest)
    with pytest.raises(V4ContractError, match="evaluation status"):
        pipeline.validate_v4_result_bundle(directory)


def test_secondary_numbers_cannot_be_fabricated_by_rehashing(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    config_path, _ = _fixture(root, baselines=True)
    output = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/secondary-replay")
    )
    csv_path = output.parent / "secondary_baselines_v4.csv"
    rows = pd.read_csv(csv_path, float_precision="round_trip", keep_default_na=False)
    rows.loc[0, "estimate"] = 0.5
    rows.to_csv(csv_path, index=False, float_format="%.17g")
    manifest = json.loads(output.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = output.parent / artifact["path"]
        artifact["sha256"] = _reference(root, path)["sha256"]
        artifact["size_bytes"] = path.stat().st_size
    _write_json(output, manifest)
    with pytest.raises(V4ContractError, match="secondary output does not reproduce"):
        pipeline.validate_v4_result_bundle(output.parent)


def test_repository_alias_support_does_not_allow_internal_symlink_inputs(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _fixture(root)
    alias = tmp_path / "repository-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError as err:
        pytest.skip(f"Symlinks not supported on system: {err}")
    output = pipeline.run_v4_analysis(
        alias / "data/analysis-config-v4.json",
        repository_root=alias,
        output_directory=alias / "build/alias-run",
    )
    assert output.is_file()


@pytest.mark.parametrize("missing_definition", [False, True])
def test_recorded_wind_uses_explicit_source_definition_or_narrows_only_wind(
    tmp_path: Path, missing_definition: bool
) -> None:
    root = tmp_path / "repository"
    config_path, config = _fixture(root)
    config["inference"]["wind_mode"] = "recorded_strata"
    config["inference"]["favourable_min_mps"] = None
    config["inference"]["opposing_max_mps"] = None
    for site in pipeline.SITES:
        reference = config["sites"][site]["sources"]["weather_observations"]
        path = root / reference["path"]
        weather = pd.read_csv(path)
        weather["wind_stratum"] = [
            "favourable" if number % 2 else "opposing" for number in range(len(weather))
        ]
        if not missing_definition:
            weather["wind_definition_id"] = config["inference"]["wind_definition_id"]
        weather.to_csv(path, index=False)
        config["sites"][site]["sources"]["weather_observations"] = _reference(root, path)
    _write_json(config_path, config)
    output = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/recorded-wind")
    )
    manifest = pipeline.validate_v4_result_bundle(output.parent)
    assert {value["eligible_hours"] for value in manifest["support"].values()} == {12}
    wind = [row for row in manifest["result_rows"] if row["estimand"] == "rq2_wind_contrast"]
    assert all((row["classification"] == "not_estimable") == missing_definition for row in wind)


def test_both_sites_zero_eligible_hours_publish_honest_audit_and_not_estimable_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    config_path, config = _fixture(root, baselines=True)
    for site in pipeline.SITES:
        reference = config["sites"][site]["sources"]["weather_observations"]
        path = root / reference["path"]
        weather = pd.read_csv(path)
        weather["precipitation_mm"] = 1.0
        weather.to_csv(path, index=False)
        config["sites"][site]["sources"]["weather_observations"] = _reference(root, path)
    _write_json(config_path, config)
    output = pipeline.run_v4_analysis(
        config_path, repository_root=root, output_directory=Path("build/zero-support")
    )
    manifest = pipeline.validate_v4_result_bundle(output.parent)
    assert len(manifest["result_rows"]) == 9
    assert {value["candidate_hours"] for value in manifest["support"].values()} == {12}
    assert {value["eligible_hours"] for value in manifest["support"].values()} == {0}
    primary = pd.read_csv(output.parent / pipeline.RESULTS_NAME)
    assert primary["classification"].eq("not_estimable").all()
    assert primary["resamples_valid"].eq(0).all()
    assert primary[["estimate", "ci_low", "ci_high"]].isna().all(axis=None)
    secondary = pd.read_csv(output.parent / "secondary_baselines_v4.csv")
    assert secondary["classification"].eq("not_estimable").all()
    assert secondary["resample_sha256"].eq(primary.iloc[0]["resample_sha256"]).all()
