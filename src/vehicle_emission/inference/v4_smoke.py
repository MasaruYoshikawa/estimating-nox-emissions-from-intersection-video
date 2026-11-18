"""Generate a small controlled-input v4 example, never empirical findings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_emission.inference.v4_contracts import EVALUATION_DESIGN, SCOPE, THRESHOLDS
from vehicle_emission.inference.v4_pipeline import _argument_path, _destination, run_v4_analysis


def run_controlled_smoke(output_directory: Path, *, repository_root: Path) -> Path:
    """Write an explicit example config/source package and calculate its bundle."""
    root = repository_root.resolve()
    target = _destination(
        root, _argument_path(root, output_directory, repository_root), "preparation"
    )
    target.mkdir(parents=True, exist_ok=False)
    config: dict[str, Any] = {
        "schema_version": 4,
        "manifest_type": "numeric_export_analysis_v4",
        "analysis_id": "controlled_smoke_not_empirical",
        "mode": "preparation",
        "analysis_scope": SCOPE,
        "evaluation_design": EVALUATION_DESIGN,
        "thresholds": THRESHOLDS,
        "hourly_policy": {
            "monitor_label_semantics": "end-labelled-hour",
            "predictor_offset_hours": 0,
            "coverage_rule_id": "controlled-10-seconds-not-an-empirical-rule",
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
            "seed": 0,
            "interval_method": "percentile",
            "interval_adjustment": "bonferroni",
            "confidence_level": 0.95,
            "primary_interval_family": "controlled-primary",
            "primary_family_size": 4,
            "wind_interval_family": "controlled-wind",
            "wind_family_size": 2,
            "wind_mode": "unavailable",
            "wind_definition_id": None,
            "favourable_min_mps": None,
            "opposing_max_mps": None,
        },
        "sites": {},
        "baselines": {"secondary_increment_columns": {}, "groups": []},
        "evidence": {
            role: None
            for role in (
                "method_lock",
                "source_review",
                "measurement_receipt",
                "measurement_acceptance",
            )
        },
    }
    for site in ("Sukiyabashi", "Shinjuku"):
        starts = pd.date_range("2026-05-01T00:00:00Z", periods=6, freq="D")
        candidates = pd.DataFrame(
            {
                "site": site,
                "monitor_start_utc": starts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "monitor_end_utc": (starts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        monitors = candidates.assign(
            station_id=f"controlled-{site}-station",
            ambient_nox_ppm=[0.002, 0.004, 0.003, 0.006, 0.005, 0.008],
            monitor_label_utc=candidates["monitor_end_utc"],
            target_observation_status="actual_observation",
        )
        clips = pd.DataFrame(
            {
                "site": site,
                "clip_id": [f"controlled-clip-{index}" for index in range(len(starts))],
                "clip_start_utc": starts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "clip_end_utc": (starts + pd.Timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timing_evidence_id": "controlled-generated-clock",
                "source_video_sha256": hashlib.sha256(
                    b"controlled fixture video identity; no empirical video"
                ).hexdigest(),
                "source_video_path": [
                    f"controlled/{site}/day-{index}.mp4" for index in range(len(starts))
                ],
            }
        )
        seconds = []
        for index, start in enumerate(starts):
            for number in range(10):
                seconds.append(
                    {
                        "site": site,
                        "second_start_utc": (start + pd.Timedelta(seconds=number)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "coverage_valid": True,
                        "count_only_increment": index % 3 + 1 if number == 0 else 0,
                        "relative_nox_emission_index_increment": (index + 1) / 10,
                        "source_record_id": f"controlled-row-{index}-{number}",
                    }
                )
        weather = pd.DataFrame(
            {
                "site": site,
                "weather_start_utc": candidates["monitor_start_utc"],
                "weather_end_utc": candidates["monitor_end_utc"],
                "weather_label_utc": candidates["monitor_end_utc"],
                "weather_source_id": "controlled-generated-weather",
                "weather_observation_status": "actual_observation",
                "precipitation_mm": 0.0,
            }
        )
        sources: dict[str, Any] = {"background_observations": None, "speed_samples": None}
        for role, frame in {
            "candidate_hours": candidates,
            "monitor_observations": monitors,
            "clip_intervals": clips,
            "video_seconds": pd.DataFrame(seconds),
            "weather_observations": weather,
        }.items():
            path = target / f"{site}_{role}.csv"
            frame.to_csv(path, index=False)
            sources[role] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        config["sites"][site] = {
            "source_snapshot_id": f"controlled-{site}-not-empirical",
            "monitor_station_id": f"controlled-{site}-station",
            "weather_source_id": "controlled-generated-weather",
            "background_site": None,
            "detector_checkpoint_sha256": hashlib.sha256(
                b"controlled detector identity; not weights"
            ).hexdigest(),
            "tracker_config_sha256": hashlib.sha256(
                b"controlled tracker identity; not empirical config"
            ).hexdigest(),
            "estimator_definition_sha256": hashlib.sha256(
                b"controlled estimator identity; not empirical model"
            ).hexdigest(),
            "sources": sources,
        }
    path = target / "controlled_analysis_config_v4.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_v4_analysis(path, repository_root=root, output_directory=target / "calculated")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    path = run_controlled_smoke(args.output_directory, repository_root=args.repository_root)
    print(f"Controlled smoke only; config and calculated bundle: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
