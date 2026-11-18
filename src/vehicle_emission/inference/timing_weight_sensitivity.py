"""Media-clock summaries and fixed fleet-weight sensitivities on recorded hours."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.inference.paired_inference import joint_calendar_block_draws
from vehicle_emission.inference.recorded_hourly_extension import CLASS_COLUMNS, CLASS_NAMES
from vehicle_emission.inference.recorded_hourly_reanalysis import _config, _csv, _hash, _json
from vehicle_emission.inference.recording_support import (
    KEYS,
    MODELS,
    SITES,
    RecordingSupportConfig,
    _nonnegative,
    _require_columns,
    _timestamps,
    _verified_inputs,
)

REFERENCE_SPEEDS = (15, 25, 40)


def prepare_timing_clips(clips: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Join exact core membership and expose the retained scalar clock arithmetic."""
    _require_columns(clips, [*KEYS, "video_id", "clip_start_utc"], "core clips")
    _require_columns(
        videos,
        ["source_site", "video_id", "duration_seconds", "fps", "recording_datetime_utc"],
        "source videos",
    )
    videos = videos.rename(columns={"source_site": "site"})
    for frame in (clips, videos):
        if (
            frame[["site", "video_id"]].isna().any().any()
            or frame.duplicated(["site", "video_id"]).any()
        ):
            raise ValueError("clips require unique non-missing site/video identities")
    if clips.empty or set(clips.site) != set(SITES):
        raise ValueError("core clips require both sites")
    fields = ["site", "video_id", "duration_seconds", "fps", "recording_datetime_utc"]
    result = clips.merge(videos[fields], on=["site", "video_id"], how="left", validate="one_to_one")
    if result[fields].isna().any().any():
        raise ValueError("every core clip requires source timing metadata")
    for name in ("duration_seconds", "fps"):
        _nonnegative(result, [name], "source timing")
        if not result[name].gt(0).all():
            raise ValueError("source timing requires positive duration and fps")
    starts = _timestamps(result.clip_start_utc, "core clip starts")
    source_starts = _timestamps(result.recording_datetime_utc, "source clip starts")
    hours = _timestamps(result.monitor_start_utc, "core hour starts")
    if not starts.eq(source_starts).all() or not starts.dt.floor("h").eq(hours).all():
        raise ValueError("core/source clip timestamps or hour membership disagree")
    result["hour_jst"] = starts.dt.tz_convert("Asia/Tokyo").dt.hour
    result["nominal_to_media_duration_ratio"] = 600 / result.duration_seconds
    result["legacy_short_clip_rule"] = result.duration_seconds.le(305)
    frames = np.rint(result.duration_seconds * result.fps).astype(np.int64)
    short = result.legacy_short_clip_rule
    frames.loc[short] = np.rint(frames.loc[short] * 600 / result.loc[short, "duration_seconds"])
    if frames.le(0).any():
        raise ValueError("timing metadata must imply at least one frame")
    result["legacy_inferred_frames"] = frames
    result["legacy_assigned_seconds_per_frame"] = 600 / frames
    result["assigned_to_media_time_ratio"] = result.fps * 600 / frames
    result["assigned_to_media_uncapped_step_speed_ratio"] = frames / (600 * result.fps)
    return result.sort_values(["site", "video_id"], kind="stable").reset_index(drop=True)


def timing_summaries(clips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise playable duration, not independently measured recording coverage."""
    summaries = []
    hourly = []

    def summarise(group: pd.DataFrame, site: str) -> dict[str, Any]:
        return {
            "site": site,
            "clips": len(group),
            "duration_min_seconds": group.duration_seconds.min(),
            "duration_p01_seconds": group.duration_seconds.quantile(0.01),
            "duration_median_seconds": group.duration_seconds.median(),
            "duration_p99_seconds": group.duration_seconds.quantile(0.99),
            "duration_max_seconds": group.duration_seconds.max(),
            "nominal_to_media_ratio_p01": group.nominal_to_media_duration_ratio.quantile(0.01),
            "nominal_to_media_ratio_median": group.nominal_to_media_duration_ratio.median(),
            "nominal_to_media_ratio_p99": group.nominal_to_media_duration_ratio.quantile(0.99),
            "assigned_to_media_time_ratio_median": group.assigned_to_media_time_ratio.median(),
            "assigned_to_media_step_speed_ratio_median": (
                group.assigned_to_media_uncapped_step_speed_ratio.median()
            ),
            "short_rule_clips": int(group.legacy_short_clip_rule.sum()),
            "duration_below_600_clips": int(group.duration_seconds.lt(600).sum()),
        }

    summaries.append(summarise(clips, "All"))
    for site in SITES:
        group = clips.loc[clips.site.eq(site)]
        summaries.append(summarise(group, site))
        for hour, rows in group.groupby("hour_jst", sort=True):
            hourly.append({**summarise(rows, site), "hour_jst": int(hour)})
    return pd.DataFrame(summaries), pd.DataFrame(hourly)


def fixed_class_weights() -> pd.DataFrame:
    """Use all three ADR-034 steady states, independently of ambient outcomes."""
    model = MOVEStarModel()
    output = []
    for speed in REFERENCE_SPEEDS:
        rates = {
            i: float(model.estimate_class_emissions(i, [speed / 3.6] * 3)["Emission Rate"][0][2])
            / 3
            for i in range(4)
        }
        if not all(math.isfinite(rate) and rate > 0 for rate in rates.values()):
            raise ValueError("reference rates must be finite and positive")
        for class_id, rate in rates.items():
            output.append(
                {
                    "reference_speed_kmh": speed,
                    "reference_acceleration_mps2": 0,
                    "class_id": class_id,
                    "project_class": CLASS_NAMES[class_id],
                    "reference_nox_g_per_second": rate,
                    "normalised_weight": rate / rates[1],
                }
            )
    return pd.DataFrame(output)


def prepare_weight_panel(panel: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Retain the full core calendar and mark one common class-complete sample."""
    _require_columns(
        panel,
        [
            *KEYS,
            "local_calendar_date",
            "ambient_nox_ppm",
            *MODELS,
            *CLASS_COLUMNS,
            "class_counts_known",
            "captured_clip_entering_count",
            "vehicle_count",
            "fixed_rate_weighted_entries",
        ],
        "hourly panel",
    )
    if panel.empty or panel.duplicated(KEYS).any() or set(panel.site) != set(SITES):
        raise ValueError("hourly panel requires distinct two-site hour keys")
    if not panel.class_counts_known.isin([True, False]).all():
        raise ValueError("class completeness must be boolean")
    starts = _timestamps(panel.monitor_start_utc, "hour starts")
    if (
        not starts.eq(starts.dt.floor("h")).all()
        or not panel.local_calendar_date.eq(
            starts.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
        ).all()
    ):
        raise ValueError("hour keys and local calendar dates disagree")
    _nonnegative(panel, ["ambient_nox_ppm", *MODELS, "vehicle_count"], "hourly panel")
    selected = panel.class_counts_known & panel.captured_clip_entering_count.notna()
    counts = panel.loc[selected, [*CLASS_COLUMNS, "captured_clip_entering_count", "vehicle_count"]]
    _nonnegative(counts, list(counts.columns), "complete counts")
    if not counts.eq(np.floor(counts)).all().all():
        raise ValueError("complete counts must be integers")
    if not counts[list(CLASS_COLUMNS)].sum(axis=1).eq(counts.captured_clip_entering_count).all():
        raise ValueError("complete class totals disagree with entering-event counts")
    result = panel.copy()
    result["weight_sample_complete"] = selected
    if weights.duplicated(["reference_speed_kmh", "class_id"]).any():
        raise ValueError("reference weights require unique speed/class identities")
    for speed in REFERENCE_SPEEDS:
        group = weights.loc[weights.reference_speed_kmh.eq(speed)].set_index("class_id")
        if set(group.index) != set(range(4)):
            raise ValueError("each fixed speed requires all four class weights")
        values = group.loc[range(4), "normalised_weight"].to_numpy(float)
        if not np.isfinite(values).all() or not (values > 0).all() or values[1] != 1:
            raise ValueError("reference weights require positive car-normalised values")
        name = f"weighted_entries_{speed}_kmh"
        result[name] = np.nan
        result.loc[selected, name] = (
            result.loc[selected, list(CLASS_COLUMNS)].to_numpy(float) @ values
        )
    if not np.allclose(
        result.loc[selected, "weighted_entries_25_kmh"],
        result.loc[selected, "fixed_rate_weighted_entries"],
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError("25 km/h state does not reproduce the existing weighted benchmark")
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _weight_values(panel: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    output = {}
    names = [
        "ambient_nox_ppm",
        *MODELS,
        *(f"weighted_entries_{speed}_kmh" for speed in REFERENCE_SPEEDS),
    ]
    for site in SITES:
        group = panel.loc[panel.site.eq(site) & panel.weight_sample_complete, names]
        values = group.to_numpy(float)
        if len(group) < 3 or not np.isfinite(values).all():
            rhos = np.full(len(names) - 1, np.nan)
        else:
            ranks = rankdata(values, axis=0)
            centred = ranks - ranks.mean(axis=0)
            denom = np.sqrt(np.sum(centred**2, axis=0))
            rhos = np.divide(
                centred[:, 0] @ centred[:, 1:],
                denom[0] * denom[1:],
                out=np.full(len(names) - 1, np.nan),
                where=denom[0] * denom[1:] > 0,
            )
        reference = rhos[3]  # Ambient correlation with the fixed 25 km/h count.
        for index, speed in enumerate(REFERENCE_SPEEDS):
            weighted = float(rhos[2 + index])
            output[site, speed, "weighted_count_rho"] = weighted
            output[site, speed, "weighted_count_minus_25_kmh_rho"] = weighted - reference
            for model, rho in zip(MODELS, rhos[:2], strict=True):
                output[site, speed, f"{model}_minus_weighted_count_rho"] = float(rho) - weighted
    return output


def infer_weight_sensitivity(panel: pd.DataFrame, config: RecordingSupportConfig) -> pd.DataFrame:
    """Pair every model and fixed benchmark in shared union-calendar draws."""
    config.validate()
    panel = panel.sort_values(KEYS, kind="stable").reset_index(drop=True)
    calendar_days = (
        pd.Timestamp(panel.local_calendar_date.max())
        - pd.Timestamp(panel.local_calendar_date.min())
    ).days + 1
    if max(config.block_lengths) > calendar_days:
        raise ValueError("block length exceeds the recorded calendar")
    points = _weight_values(panel)
    support = {
        site: panel.loc[panel.site.eq(site) & panel.weight_sample_complete] for site in SITES
    }
    output = []
    for block in config.block_lengths:
        draws: dict[tuple[str, int, str], list[float]] = {key: [] for key in points}
        digest = hashlib.sha256(b"fixed-weight-joint-calendar-draws-v1\0")
        for selected, indices, _ in joint_calendar_block_draws(
            panel, _config(config.resamples, config.seed, block)
        ):
            digest.update(selected.astype("<i8").tobytes())
            digest.update(indices.astype("<i8").tobytes())
            for key, value in _weight_values(panel.iloc[indices]).items():
                draws[key].append(value)
        for (site, speed, estimand), point in points.items():
            array = np.asarray(draws[site, speed, estimand])
            valid = int(np.isfinite(array).sum())
            complete = math.isfinite(point) and valid == config.resamples
            low, high = np.quantile(array, [0.025, 0.975]) if complete else (math.nan, math.nan)
            output.append(
                {
                    "site": site,
                    "reference_speed_kmh": speed,
                    "estimand": estimand,
                    "estimate": point,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "hours": len(support[site]),
                    "site_days": support[site].local_calendar_date.nunique(),
                    "calendar_days": calendar_days,
                    "block_length_days": block,
                    "resamples_requested": config.resamples,
                    "resamples_valid": valid,
                    "resamples_invalid": config.resamples - valid,
                    "resample_sha256": digest.hexdigest(),
                    "interval_family": "pointwise_descriptive",
                    "interval_status": "complete" if complete else "not_estimable",
                }
            )
    return pd.DataFrame(output)


def profile_entry_summary(panel: pd.DataFrame) -> pd.DataFrame:
    output = []
    for site in SITES:
        group = panel.loc[panel.site.eq(site) & panel.weight_sample_complete]
        profiles = int(group.vehicle_count.sum())
        entries = int(group.captured_clip_entering_count.sum())
        output.append(
            {
                "site": site,
                "hours": len(group),
                "tracked_profiles": profiles,
                "entering_events": entries,
                "profiles_per_entering_event": profiles / entries if entries else math.nan,
            }
        )
    return pd.DataFrame(output)


def run_timing_weight_sensitivity(
    *,
    source_directory: Path,
    extension_bundle: Path,
    recording_support_bundle: Path,
    output_directory: Path,
    config: RecordingSupportConfig | None = None,
) -> dict[str, Any]:
    config = config or RecordingSupportConfig()
    config.validate()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    if any(
        output_directory.resolve().is_relative_to(path.resolve())
        for path in (source_directory, extension_bundle, recording_support_bundle)
    ):
        raise ValueError("output must be separate from existing source/result bundles")
    paths = _verified_inputs(source_directory, extension_bundle)
    paths["core_membership"] = recording_support_bundle / "recording_support_clips.csv"
    paths["membership_receipt"] = recording_support_bundle / "receipt.json"
    receipt = json.loads(paths["membership_receipt"].read_text())
    if receipt["outputs"].get(paths["core_membership"].name) != _hash(paths["core_membership"]):
        raise ValueError("core clip membership differs from its receipt")
    if receipt["input_sha256"].get("video_master") != _hash(paths["video_master"]):
        raise ValueError("core membership uses a different video source")
    hashes = {name: _hash(path) for name, path in paths.items()}
    calculators = [
        Path(__file__),
        Path(__file__).with_name("recording_support.py"),
        Path(__file__).with_name("paired_inference.py"),
        Path(__file__).with_name("recorded_hourly_reanalysis.py"),
        Path(__file__).with_name("recorded_hourly_extension.py"),
        Path(__file__).parents[1] / "calculate_speeds.py",
        Path(__file__).parents[1] / "vehicle_classes.py",
    ]
    calculators += [
        Path(__file__).parents[1] / "emissions" / name
        for name in (
            "movestar.py",
            "VehicleSrcCoeff.csv",
            "EmsRate_1.csv",
            "EmsRate_2.csv",
            "EmsRate_6.csv",
            "EmsRate_13.csv",
        )
    ]
    code_hashes = {path.name: _hash(path) for path in calculators}
    clips = prepare_timing_clips(
        pd.read_csv(paths["core_membership"]), pd.read_csv(paths["video_master"])
    )
    summary, hourly = timing_summaries(clips)
    weights = fixed_class_weights()
    panel = prepare_weight_panel(
        pd.read_csv(paths["current_panel"], float_precision="round_trip"), weights
    )
    membership = clips.groupby(KEYS).size().rename("actual_clips")
    matched = panel.merge(membership, on=KEYS, how="left", validate="one_to_one")
    if (
        len(matched) != len(membership)
        or not matched.captured_clip_count.eq(matched.actual_clips).all()
    ):
        raise ValueError("clip membership does not match the core hourly panel")
    inference = infer_weight_sensitivity(panel, config)
    outputs = {
        "timing_clips.csv": clips,
        "timing_summary.csv": summary,
        "timing_by_hour.csv": hourly,
        "fixed_class_weights.csv": weights,
        "weight_sensitivity.csv": inference,
        "profile_entry_units.csv": profile_entry_summary(panel),
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".timing-weight-", dir=output_directory.parent) as temp:
        stage = Path(temp) / "calculation"
        stage.mkdir()
        for name, frame in outputs.items():
            _csv(stage / name, frame)
        if hashes != {name: _hash(path) for name, path in paths.items()} or code_hashes != {
            path.name: _hash(path) for path in calculators
        }:
            raise ValueError("input or calculator changed during the calculation")
        receipt = {
            **asdict(config),
            "reference_speeds_kmh": list(REFERENCE_SPEEDS),
            "input_sha256": hashes,
            "calculator_sha256": code_hashes,
            "input_paths": {name: str(path) for name, path in paths.items()},
            "timing_definition": "media_metadata_and_retained_scalar_arithmetic_not_wall_clock_coverage",
            "weight_population": "same_class_complete_hours_for_every_fixed_speed_and_model",
            "interval_method": "percentile_95_pointwise_descriptive",
            "calendar_start": panel.local_calendar_date.min(),
            "calendar_end": panel.local_calendar_date.max(),
            "resampling": "shared_union_calendar_nonwrapping_moving_blocks",
            "invalid_draw_rule": "retain_invalid_count_and_withhold_interval_if_any_draw_invalid",
            "package_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        _json(stage / "receipt.json", receipt)
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(f"output appeared during calculation: {output_directory}")
        stage.rename(output_directory)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--extension-bundle", type=Path, required=True)
    parser.add_argument("--recording-support-bundle", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    receipt = run_timing_weight_sensitivity(
        source_directory=args.source_directory,
        extension_bundle=args.extension_bundle,
        recording_support_bundle=args.recording_support_bundle,
        output_directory=args.output_directory,
        config=RecordingSupportConfig(args.resamples, args.seed),
    )
    print(
        json.dumps(
            {"output_directory": str(args.output_directory), "outputs": len(receipt["outputs"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
