"""Recalculate source-dependent supplements after provider-quality correction."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from vehicle_emission.inference.monitor_quality import QUALITY_COLUMNS, apply_monitor_quality
from vehicle_emission.inference.panel_replay import _calculator_paths, load_panel
from vehicle_emission.inference.recorded_hourly_reanalysis import (
    _condition_points,
    _csv,
    _hash,
    _json,
    _load,
    _recorded_counts,
)
from vehicle_emission.inference.recording_support import (
    KEYS,
    SITES,
    SUPPORT_RULE,
    RecordingSupportConfig,
    infer_recording_support,
    prepare_recording_support,
)
from vehicle_emission.inference.timing_weight_sensitivity import (
    REFERENCE_SPEEDS,
    fixed_class_weights,
    infer_weight_sensitivity,
    prepare_timing_clips,
    prepare_weight_panel,
    profile_entry_summary,
    timing_summaries,
)

SOURCE_NAMES = (
    "video_master",
    "video_emissions",
    "entering_counts",
    "entering_count_audit",
    "air_pollution_history",
    "weather_history",
)


def correct_monitor_series(ambient: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Apply documented provider missingness to the exact monitoring intervals."""
    keys = ["monitoring_place", "monitor_start_utc"]
    if not set(QUALITY_COLUMNS).issubset(quality) or quality.empty:
        raise ValueError("quality evidence requires station, interval, value and status")
    if quality[keys].isna().any().any() or quality.duplicated(keys).any():
        raise ValueError("quality evidence requires distinct nonmissing station-hour keys")
    if not quality.quality_status.isin(["valid", "missing", "uncollected"]).all():
        raise ValueError("unrecognised provider quality status")
    if (
        not quality.monitor_start_utc.astype("string")
        .str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00:00Z")
        .fillna(False)
        .all()
    ):
        raise ValueError("quality evidence requires canonical UTC hour starts")
    pd.to_datetime(quality.monitor_start_utc, utc=True, errors="raise")
    values = pd.to_numeric(quality.nox_ppm, errors="raise")
    valid = quality.quality_status.eq("valid")
    if not (np.isfinite(values[valid]) & values[valid].ge(0)).all() or values[~valid].notna().any():
        raise ValueError("provider values disagree with their quality status")
    required = {"monitoring_place", "timestamp_utc", "nox_ppm"}
    if not required.issubset(ambient) or ambient.empty:
        raise ValueError("source monitoring series lacks its required fields")
    result = ambient.copy()
    result["monitor_start_utc"] = pd.to_datetime(
        result.timestamp_utc, utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if result.duplicated(keys).any():
        raise ValueError("duplicate source monitoring station-hour")
    evidence = quality[[*keys, "quality_status", "nox_ppm"]].rename(
        columns={"nox_ppm": "provider_nox_ppm"}
    )
    result = result.merge(evidence, on=keys, how="left", validate="one_to_one")
    observed = result.quality_status.eq("valid")
    if not np.allclose(
        result.loc[observed, "nox_ppm"],
        result.loc[observed, "provider_nox_ppm"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("provider numeric monitoring values differ from the source series")
    result.loc[result.quality_status.isin(["missing", "uncollected"]), "nox_ppm"] = np.nan
    return result.drop(columns="provider_nox_ppm")


def _verified_source(source: Path) -> dict[str, Path]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {name: source / f"{name}.csv" for name in SOURCE_NAMES}
    for path in paths.values():
        if _hash(path) != manifest.get("outputs", {}).get(path.name):
            raise ValueError(f"source hash differs from export manifest: {path.name}")
    return {**paths, "source_manifest": manifest_path}


def _quality_pool(rows: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = quality.loc[quality.quality_status.isin(["missing", "uncollected"])]
    keys = missing.rename(columns={"monitoring_place": "site"})[
        [*KEYS, "quality_status", "provider_station_id"]
    ]
    exclusions = rows[KEYS].merge(keys, on=KEYS, how="inner", validate="one_to_one")
    flagged = pd.MultiIndex.from_frame(keys[KEYS])
    keep = ~pd.MultiIndex.from_frame(rows[KEYS]).isin(flagged)
    return rows.loc[keep].reset_index(drop=True), exclusions


def run_supplements(
    *,
    panel: Path,
    quality: Path,
    source_directory: Path,
    all_weather_panel: Path,
    output_directory: Path,
    config: RecordingSupportConfig | None = None,
) -> dict[str, Any]:
    """Use the corrected core keys and original fixed supplement definitions."""
    config = config or RecordingSupportConfig()
    config.validate()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    if any(
        output_directory.resolve().is_relative_to(path.resolve())
        for path in (source_directory, panel.parent, quality.parent, all_weather_panel.parent)
    ):
        raise ValueError("output must be separate from source and existing result bundles")
    paths = {
        **_verified_source(source_directory),
        "corrected_panel": panel,
        "provider_quality": quality,
        "all_weather_predictor_panel": all_weather_panel,
    }
    input_hashes = {name: _hash(path) for name, path in paths.items()}
    root = Path(__file__).resolve().parents[3]
    code_paths = list(
        dict.fromkeys(
            [
                *_calculator_paths(),
                Path(__file__).resolve(),
                Path(__file__).with_name("monitor_quality.py"),
                Path(__file__).with_name("recording_support.py"),
                Path(__file__).with_name("timing_weight_sensitivity.py"),
            ]
        )
    )
    code_hashes = {str(path.relative_to(root)): _hash(path) for path in code_paths}
    corrected = load_panel(panel)
    evidence = pd.read_csv(
        quality, dtype={"provider_station_id": str}, float_precision="round_trip"
    )
    checked, _ = apply_monitor_quality(corrected, evidence)
    if len(checked) != len(corrected) or not np.allclose(
        checked.setagaya_nox_ppm, corrected.setagaya_nox_ppm, equal_nan=True, rtol=0, atol=0
    ):
        raise ValueError("input panel has not applied its provider-quality correction")
    ambient = pd.read_csv(paths["air_pollution_history"])
    corrected_ambient = correct_monitor_series(ambient, evidence)
    pool = _load(all_weather_panel, paths["air_pollution_history"], paths["weather_history"])
    pool, count_context = _recorded_counts(pool, source_directory)
    validation = pd.DataFrame(count_context["validation_material"])
    if not validation.empty:
        validation_keys = pd.MultiIndex.from_frame(validation[KEYS])
        pool = pool.loc[~pd.MultiIndex.from_frame(pool[KEYS]).isin(validation_keys)].reset_index(
            drop=True
        )
    pool, quality_exclusions = _quality_pool(pool, evidence)
    core_keys = pd.MultiIndex.from_frame(corrected[KEYS])
    pool_keys = pd.MultiIndex.from_frame(pool[KEYS])
    if not core_keys.isin(pool_keys).all():
        raise ValueError("corrected core contains hours outside the corrected all-weather pool")
    matched = corrected.merge(pool, on=KEYS, validate="one_to_one", suffixes=("", "_source"))
    for name in ("ambient_nox_ppm", "movestar_nox_grams", "vt_cpfm_nox_grams", "vehicle_count"):
        if not np.allclose(matched[name], matched[f"{name}_source"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"all-weather source does not reproduce the corrected core: {name}")
    videos = pd.read_csv(paths["video_master"], float_precision="round_trip")
    emissions = pd.read_csv(paths["video_emissions"], float_precision="round_trip")
    support_panel, support_clips = prepare_recording_support(corrected, videos, emissions)
    support_inference = infer_recording_support(support_panel, config)
    support_summary = []
    for site in SITES:
        for cohort in ("current", "restricted"):
            group = support_panel.loc[support_panel.site.eq(site)]
            if cohort == "restricted":
                group = group.loc[group.restricted_support]
            support_summary.append(
                {
                    "site": site,
                    "cohort": cohort,
                    "hours": len(group),
                    "site_days": group.local_calendar_date.nunique(),
                    "source_clips": int(group.source_clip_count.sum()),
                    "complete_entry_hours": int(group.entry_counts_complete.sum()),
                }
            )
    timing_clips = prepare_timing_clips(support_clips, videos)
    timing_summary, timing_hourly = timing_summaries(timing_clips)
    weights = fixed_class_weights()
    weight_panel = prepare_weight_panel(corrected, weights)
    outputs = {
        "corrected_all_weather_panel.csv": pool,
        "all_weather_quality_exclusions.csv": quality_exclusions,
        "recording_support_clips.csv": support_clips,
        "recording_support_hours.csv": support_panel[
            [
                *KEYS,
                "local_calendar_date",
                "captured_clip_count",
                "source_clip_count",
                "distinct_clip_starts",
                "distinct_ten_minute_bins",
                "restricted_support",
                "entry_counts_complete",
            ]
        ],
        "recording_support_summary.csv": pd.DataFrame(support_summary),
        "recording_support_inference.csv": support_inference,
        "timing_clips.csv": timing_clips,
        "timing_summary.csv": timing_summary,
        "timing_by_hour.csv": timing_hourly,
        "fixed_class_weights.csv": weights,
        "weight_sensitivity.csv": infer_weight_sensitivity(weight_panel, config),
        "profile_entry_units.csv": profile_entry_summary(weight_panel),
    }
    parameters = {
        "status": "provider_quality_corrected_source_supplements",
        **asdict(config),
        "input_sha256": input_hashes,
        "calculator_sha256": code_hashes,
        "interval_method": "percentile_95_pointwise_descriptive",
        "resampling": "shared_union_calendar_nonwrapping_moving_blocks",
        "calendar_start": corrected.local_calendar_date.min(),
        "calendar_end": corrected.local_calendar_date.max(),
        "support_rule": SUPPORT_RULE,
        "reference_speeds_kmh": list(REFERENCE_SPEEDS),
        "timing_definition": "media_metadata_and_retained_scalar_arithmetic_not_wall_clock_coverage",
        "numeric_zero_policy": "retain_unless_provider_marks_missing_or_uncollected",
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".quality-supplements-", dir=output_directory.parent
    ) as temporary:
        stage = Path(temporary) / "calculation"
        stage.mkdir()
        corrected_ambient_path = Path(temporary) / "corrected_ambient.csv"
        _csv(corrected_ambient_path, corrected_ambient)
        outputs["condition_point_estimates.csv"] = _condition_points(
            corrected, pool, corrected_ambient_path
        )
        for name, frame in outputs.items():
            _csv(stage / name, frame)
        _json(stage / "parameters.json", parameters)
        receipt = {
            **parameters,
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        _json(stage / "receipt.json", receipt)
        if any(_hash(path) != input_hashes[name] for name, path in paths.items()) or any(
            _hash(path) != code_hashes[str(path.relative_to(root))] for path in code_paths
        ):
            raise ValueError("an input or calculator changed during supplement calculation")
        lock = output_directory.parent / f".{output_directory.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if output_directory.exists() or output_directory.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
            stage.rename(output_directory)
        finally:
            os.close(descriptor)
            lock.unlink()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--all-weather-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    receipt = run_supplements(
        panel=args.panel,
        quality=args.quality,
        source_directory=args.source_directory,
        all_weather_panel=args.all_weather_panel,
        output_directory=args.output_directory,
        config=RecordingSupportConfig(args.resamples, args.seed),
    )
    print(
        json.dumps(
            {"outputs": len(receipt["outputs"]), "output_directory": str(args.output_directory)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
