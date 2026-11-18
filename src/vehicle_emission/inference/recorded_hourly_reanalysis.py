"""Paired calendar uncertainty for an existing recorded-hour aggregate panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vehicle_emission.inference.paired_inference import (
    PairedInferenceConfig,
    _rho,
    joint_calendar_block_draws,
)
from vehicle_emission.robustness.temporal import (
    TemporalControlConfig,
    calculate_temporal_controls,
)

STATUS = "recorded_aggregate_statistical_reanalysis"
PREDICTORS = (
    "movestar_nox_grams",
    "vt_cpfm_nox_grams",
    "vehicle_count",
    "mean_speed_mps",
)
CONTRASTS = (
    ("movestar_minus_recorded_volume", "movestar_nox_grams", "vehicle_count"),
    ("continuous_minus_recorded_volume", "vt_cpfm_nox_grams", "vehicle_count"),
    ("continuous_minus_movestar", "vt_cpfm_nox_grams", "movestar_nox_grams"),
)
FAMILY = "two_models_two_sites_associations_and_paired_advantages"


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path, content: object) -> None:
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _load(panel: Path, ambient: Path, weather: Path) -> pd.DataFrame:
    frame = pd.read_csv(panel)
    required = {"site", "hour_timestamp_utc", "date_local_jst", *PREDICTORS}
    if frame.columns.duplicated().any() or not required.issubset(frame):
        raise ValueError("recorded panel requires distinct identity and predictor columns")
    if frame.empty or set(frame.site) != {"Sukiyabashi", "Shinjuku"}:
        raise ValueError("recorded panel must contain both sites")
    starts = pd.to_datetime(frame.hour_timestamp_utc, utc=True, errors="raise")
    if not starts.eq(starts.dt.floor("h")).all():
        raise ValueError("recorded hours require exact calendar-hour timestamps")
    dates = starts.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    if not frame.date_local_jst.eq(dates).all():
        raise ValueError("recorded dates disagree with UTC-to-JST conversion")
    rows = frame[["site", *PREDICTORS]].copy()
    rows["monitor_start_utc"] = starts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows["monitor_end_utc"] = (starts + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows["local_calendar_date"] = dates
    if rows.duplicated(["site", "monitor_start_utc"]).any():
        raise ValueError("duplicate recorded site-hour keys")
    for column in PREDICTORS:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not (np.isfinite(rows[column]) & rows[column].ge(0)).all():
            raise ValueError("recorded predictors must be finite and non-negative")
    observations = pd.read_csv(ambient)
    observations["monitor_start_utc"] = pd.to_datetime(
        observations.timestamp_utc, utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    observations = observations.rename(
        columns={"monitoring_place": "site", "nox_ppm": "ambient_nox_ppm"}
    )
    rows = rows.merge(
        observations[["site", "monitor_start_utc", "ambient_nox_ppm"]],
        on=["site", "monitor_start_utc"],
        how="left",
        validate="one_to_one",
    )
    if not (np.isfinite(rows.ambient_nox_ppm) & rows.ambient_nox_ppm.ge(0)).all():
        raise ValueError("every recorded hour requires an actual same-hour ambient value")
    rain = pd.read_csv(weather)
    rain["monitor_end_utc"] = pd.to_datetime(
        rain.timestamp_utc, utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rain = rain.rename(columns={"precipitation": "source_interval_precipitation_mm"})
    rows = rows.merge(
        rain[["monitor_end_utc", "source_interval_precipitation_mm"]],
        on="monitor_end_utc",
        how="left",
        validate="many_to_one",
    )
    rain_values = rows.source_interval_precipitation_mm
    if not (rain_values.isna() | (np.isfinite(rain_values) & rain_values.ge(0))).all():
        raise ValueError("precipitation must be missing or finite and non-negative")
    return rows.sort_values(["site", "monitor_start_utc"], kind="stable").reset_index(drop=True)


def _config(resamples: int, seed: int, block_days: int) -> PairedInferenceConfig:
    return PairedInferenceConfig(
        resampling_method="union_calendar_moving_block",
        block_length_days=block_days,
        minimum_site_hours=3,
        minimum_site_days=2,
        minimum_stratum_hours=3,
        minimum_stratum_days=2,
        resamples=resamples,
        minimum_valid_resamples=resamples,
        seed=seed,
        interval_method="percentile",
        interval_adjustment="bonferroni",
        confidence_level=0.95,
        primary_interval_family=FAMILY,
        primary_family_size=10,
        wind_interval_family="not_evaluated",
        wind_family_size=2,
        wind_mode="unavailable",
        wind_definition_id=None,
        favourable_min_mps=None,
        opposing_max_mps=None,
    )


def _values(rows: pd.DataFrame) -> dict[tuple[str, str], float]:
    values = {}
    for site in ("Sukiyabashi", "Shinjuku"):
        subset = rows.loc[rows.site.eq(site)]
        for column in PREDICTORS:
            values[site, column] = _rho(subset[column], subset.ambient_nox_ppm)
        for name, first, second in CONTRASTS:
            values[site, name] = values[site, first] - values[site, second]
    return values


def _inference(rows: pd.DataFrame, config: PairedInferenceConfig) -> pd.DataFrame:
    point = _values(rows)
    samples: dict[tuple[str, str], list[float]] = {key: [] for key in point}
    digest = hashlib.sha256(b"recorded-panel-joint-calendar-draws-v1\0")
    for selected, indices, _ in joint_calendar_block_draws(rows, config):
        digest.update(selected.astype("<i8").tobytes())
        digest.update(indices.astype("<i8").tobytes())
        for key, value in _values(rows.iloc[indices]).items():
            if math.isfinite(value):
                samples[key].append(value)
    output = []
    for (site, estimand), estimate in point.items():
        subset = rows.loc[rows.site.eq(site)]
        family_size = 1 if estimand in {"vehicle_count", "mean_speed_mps"} else 10
        draws = samples[site, estimand]
        if len(draws) != config.resamples or not math.isfinite(estimate):
            raise ValueError(f"insufficient finite paired resamples: {site}/{estimand}")
        nominal = np.quantile(draws, [0.025, 0.975])
        adjusted = np.quantile(draws, [0.025 / family_size, 1 - 0.025 / family_size])
        output.append(
            {
                "site": site,
                "estimand": estimand,
                "estimate": estimate,
                "ci_low": float(nominal[0]),
                "ci_high": float(nominal[1]),
                "adjusted_ci_low": float(adjusted[0]),
                "adjusted_ci_high": float(adjusted[1]),
                "interval_family": FAMILY if family_size == 10 else "descriptive",
                "family_size": family_size,
                "n_hours": len(subset),
                "n_site_days": subset.local_calendar_date.nunique(),
                "block_length_days": config.block_length_days,
                "resamples_requested": config.resamples,
                "resamples_valid": len(draws),
                "resample_sha256": digest.hexdigest(),
            }
        )
    return pd.DataFrame(output)


def _controls(rows: pd.DataFrame, offsets: tuple[int, ...]) -> pd.DataFrame:
    output = []
    config = TemporalControlConfig(
        str(rows.local_calendar_date.min()),
        str(rows.local_calendar_date.max()),
        offsets,
        (1, 24, 168),
    )
    for model in ("movestar_nox_grams", "vt_cpfm_nox_grams"):
        prepared = rows.rename(
            columns={model: "relative_nox_emission_index", "vehicle_count": "count_only_index"}
        ).copy()
        # These flags select the recorded panel for statistics, not source acceptance.
        prepared["primary_measurement_eligible"] = True
        prepared["target_observation_status"] = "actual_observation"
        tables = calculate_temporal_controls(prepared, config)
        output.append(tables.contrasts.assign(model=model))
    return pd.concat(output, ignore_index=True)


def _recorded_counts(rows: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    videos = pd.read_csv(source / "video_master.csv")
    emissions = pd.read_csv(source / "video_emissions.csv")
    audit = pd.read_csv(source / "entering_count_audit.csv")
    classes = pd.read_csv(source / "entering_counts.csv")
    videos["monitor_start_utc"] = (
        pd.to_datetime(videos.recording_datetime_utc, utc=True, format="ISO8601")
        .dt.floor("h")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    emissions = emissions.merge(
        videos[["source_site", "video_id", "monitor_start_utc"]],
        on=["source_site", "video_id"],
        validate="many_to_one",
    )
    for method, column in (("MOVEStar", PREDICTORS[0]), ("VT-CPFM", PREDICTORS[1])):
        selected = emissions.loc[emissions.estimation_method.eq(method)]
        if selected.duplicated(["source_site", "video_id"]).any():
            raise ValueError("source emission method has duplicate video records")
        summed = (
            selected.groupby(["source_site", "monitor_start_utc"])
            .agg(nox=("nox_hybrid_adjusted_grams", "sum"), profiles=("vehicle_count", "sum"))
            .reset_index()
            .rename(columns={"source_site": "site"})
        )
        replay = rows.merge(summed, on=["site", "monitor_start_utc"], validate="one_to_one")
        if len(replay) != len(rows) or not (
            np.isclose(replay[column], replay.nox, rtol=1e-12, atol=1e-12).all()
            and replay.vehicle_count.eq(replay.profiles).all()
        ):
            raise ValueError("raw clip aggregates do not reproduce the recorded predictor panel")
    audit["known"] = audit.recorded_entering_count.notna()
    audit["heavy_known"] = audit.count_status.isin(["total_and_classes_agree", "class_sum_only"])
    clip = (
        audit.groupby(["source_site", "video_id"])
        .agg(
            entering_known=("known", "all"),
            heavy_known=("heavy_known", "all"),
            entering_count=("recorded_entering_count", "sum"),
        )
        .reset_index()
    )
    classes = classes.loc[classes.class_id.notna()].copy()
    if (
        not classes.class_id.isin([0, 1, 2, 3]).all()
        or classes.duplicated(["source_site", "video_id", "direction", "class_id"]).any()
    ):
        raise ValueError("source class counts require canonical distinct class rows")
    heavy = (
        classes.loc[classes.class_id.isin([0, 2])]
        .groupby(["source_site", "video_id"])
        .vehicle_count.sum()
        .rename("bus_heavy_count")
        .reset_index()
    )
    clip = videos[["source_site", "video_id", "monitor_start_utc"]].merge(
        clip, on=["source_site", "video_id"], how="left", validate="one_to_one"
    )
    clip = clip.merge(heavy, on=["source_site", "video_id"], how="left", validate="one_to_one")
    clip["entering_known"] = clip.entering_known.fillna(False).astype(bool)
    clip["heavy_known"] = clip.heavy_known.fillna(False).astype(bool)
    clip["bus_heavy_count"] = clip.bus_heavy_count.fillna(0)
    hours = (
        clip.groupby(["source_site", "monitor_start_utc"])
        .agg(
            captured_clip_count=("video_id", "count"),
            entering_known=("entering_known", "all"),
            heavy_known=("heavy_known", "all"),
            captured_clip_entering_count=("entering_count", "sum"),
            captured_clip_bus_heavy_count=("bus_heavy_count", "sum"),
        )
        .reset_index()
        .rename(columns={"source_site": "site"})
    )
    hours.loc[~hours.entering_known, "captured_clip_entering_count"] = np.nan
    hours.loc[~hours.heavy_known, "captured_clip_bus_heavy_count"] = np.nan
    joined = rows.merge(hours, on=["site", "monitor_start_utc"], how="left", validate="one_to_one")
    validation = videos.loc[
        videos.video_path.str.replace("\\", "/", regex=False).str.contains(
            "/counting_test", regex=False
        ),
        ["source_site", "video_id", "video_path", "monitor_start_utc"],
    ].rename(columns={"source_site": "site"})
    return joined, {
        "raw_aggregate_replay": "both_hybrid_indices_and_track_profile_totals_match",
        "count_window": "whole_captured_clips_assigned_by_recorded_start_hour",
        "heavy_definition": "canonical_classes_0_and_2_bus_and_heavy_truck",
        "unknown_clip_count_handling": "whole_start_hour_count_unavailable",
        "validation_material": validation.to_dict(orient="records"),
    }


def _entry_inference(rows: pd.DataFrame, config: PairedInferenceConfig) -> pd.DataFrame:
    columns = ("captured_clip_entering_count", "captured_clip_bus_heavy_count")
    eligible = rows[list(columns)].notna().all(axis=1)
    keys = [
        (site, model, baseline)
        for site in ("Sukiyabashi", "Shinjuku")
        for model in PREDICTORS[:2]
        for baseline in columns
    ]

    def values(frame: pd.DataFrame) -> dict[tuple[str, str, str], float]:
        selected = frame.loc[frame[list(columns)].notna().all(axis=1)]
        return {
            (site, model, baseline): _rho(group[model], group.ambient_nox_ppm)
            - _rho(group[baseline], group.ambient_nox_ppm)
            for site, model, baseline in keys
            for group in [selected.loc[selected.site.eq(site)]]
        }

    points = values(rows)
    samples: dict[tuple[str, str, str], list[float]] = {key: [] for key in keys}
    digest = hashlib.sha256(b"recorded-panel-joint-calendar-draws-v1\0")
    for selected, indices, _ in joint_calendar_block_draws(rows, config):
        digest.update(selected.astype("<i8").tobytes())
        digest.update(indices.astype("<i8").tobytes())
        for key, value in values(rows.iloc[indices]).items():
            if math.isfinite(value):
                samples[key].append(value)
    output = []
    for key in keys:
        site, model, baseline = key
        subset = rows.loc[rows.site.eq(site) & eligible]
        draws = samples[key]
        supported = len(draws) == config.resamples and math.isfinite(points[key])
        nominal = np.quantile(draws, [0.025, 0.975]) if supported else [math.nan, math.nan]
        adjusted = (
            np.quantile(draws, [0.025 / 8, 1 - 0.025 / 8]) if supported else [math.nan, math.nan]
        )
        output.append(
            {
                "site": site,
                "model": model,
                "baseline": baseline,
                "index_rho": _rho(subset[model], subset.ambient_nox_ppm),
                "baseline_rho": _rho(subset[baseline], subset.ambient_nox_ppm),
                "estimate": points[key],
                "ci_low": float(nominal[0]),
                "ci_high": float(nominal[1]),
                "adjusted_ci_low": float(adjusted[0]),
                "adjusted_ci_high": float(adjusted[1]),
                "family_size": 8,
                "interval_family": "two_models_two_sites_captured_clip_count_comparisons",
                "n_hours": len(subset),
                "n_site_days": subset.local_calendar_date.nunique(),
                "block_length_days": config.block_length_days,
                "resamples_requested": config.resamples,
                "resamples_valid": len(draws),
                "resample_sha256": digest.hexdigest(),
                "status": "recorded_count_comparison" if supported else "not_estimable",
                "failure_reason": (
                    "" if supported else "insufficient_recorded_count_support_or_variation"
                ),
            }
        )
    return pd.DataFrame(output)


def _condition_points(rows: pd.DataFrame, pool: pd.DataFrame, ambient: Path) -> pd.DataFrame:
    observations = pd.read_csv(ambient)
    observations["time"] = pd.to_datetime(observations.timestamp_utc, utc=True)
    lookup = observations.set_index(["monitoring_place", "time"]).nox_ppm
    output = []
    for site, selected in rows.groupby("site", sort=True):
        conditions = {
            "same_hour_dry": selected,
            "all_weather": pool.loc[pool.site.eq(site)],
            "high_recorded_volume": selected.loc[
                selected.vehicle_count.gt(selected.vehicle_count.median())
            ],
            "high_recorded_mean_speed": selected.loc[
                selected.mean_speed_mps.gt(selected.mean_speed_mps.median())
            ],
            "weekdays": selected.loc[
                pd.to_datetime(selected.local_calendar_date).dt.dayofweek.lt(5)
            ],
            "next_hour_target": selected,
            "same_hour_minus_setagaya": selected,
        }
        if "captured_clip_bus_heavy_count" in selected:
            heavy = selected.captured_clip_bus_heavy_count
            conditions["high_captured_clip_bus_heavy_count"] = selected.loc[
                heavy.gt(heavy.median())
            ]
        for name, group in conditions.items():
            group = group.copy()
            times = pd.to_datetime(group.monitor_start_utc, utc=True)
            if name == "next_hour_target":
                group["ambient_nox_ppm"] = [
                    lookup.get((site, time + pd.Timedelta(hours=1)), np.nan) for time in times
                ]
            elif name == "same_hour_minus_setagaya":
                group["ambient_nox_ppm"] -= np.asarray(
                    [lookup.get(("Setagaya", time), np.nan) for time in times]
                )
            group = group.loc[group.ambient_nox_ppm.notna()]
            for predictor in (*PREDICTORS, "captured_clip_bus_heavy_count"):
                if predictor not in group:
                    continue
                complete = group.loc[group[predictor].notna()]
                output.append(
                    {
                        "site": site,
                        "condition": name,
                        "predictor": predictor,
                        "n_hours": len(complete),
                        "n_site_days": complete.local_calendar_date.nunique(),
                        "rho": _rho(complete[predictor], complete.ambient_nox_ppm),
                    }
                )
    return pd.DataFrame(output)


def _partial_rank_points(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for site, group in rows.groupby("site", sort=True):
        design = np.column_stack([np.ones(len(group)), rankdata(group.vehicle_count)])
        target_ranks = rankdata(group.ambient_nox_ppm)
        target = target_ranks - design @ np.linalg.lstsq(design, target_ranks, rcond=None)[0]
        for model in PREDICTORS[:2]:
            ranks = rankdata(group[model])
            residual = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
            rho = (
                float(np.corrcoef(residual, target)[0, 1])
                if np.std(residual) > 1e-12 and np.std(target) > 1e-12
                else math.nan
            )
            output.append(
                {
                    "site": site,
                    "model": model,
                    "controlled_predictor": "vehicle_count",
                    "n_hours": len(group),
                    "partial_rank_correlation": rho,
                }
            )
    return pd.DataFrame(output)


def _figures(table: pd.DataFrame, directory: Path) -> None:
    primary = table.loc[table.block_length_days.eq(1)]
    if primary.empty:
        return
    labels = {
        "movestar_nox_grams": "MOVEStar",
        "vt_cpfm_nox_grams": "Continuous comparator",
        "vehicle_count": "Recorded trajectory volume",
        "movestar_minus_recorded_volume": "MOVEStar minus volume",
        "continuous_minus_recorded_volume": "Continuous minus volume",
        "continuous_minus_movestar": "Continuous minus MOVEStar",
    }
    for name, selected, xlabel in (
        (
            "fig_recorded_correlations.png",
            list(PREDICTORS[:3]),
            "Spearman correlation with ambient NOx",
        ),
        (
            "fig_recorded_paired_advantages.png",
            [item[0] for item in CONTRASTS],
            "Paired difference in Spearman correlation",
        ),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
        for axis, site in zip(axes, ("Sukiyabashi", "Shinjuku"), strict=True):
            data = primary.loc[primary.site.eq(site)].set_index("estimand")
            for position, estimand in enumerate(selected):
                row = data.loc[estimand]
                color = ("#23618c", "#1d7a62", "#79664a")[position]
                axis.plot([row.ci_low, row.ci_high], [position, position], color=color, linewidth=2)
                axis.scatter([row.estimate], [position], color=color, s=40, zorder=3)
                axis.annotate(
                    f"{row.estimate:+.3f}",
                    (row.estimate, position),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=11,
                )
            axis.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
            axis.set_title(f"{site} (n = {int(data.n_hours.iloc[0]):,})", fontsize=12)
            axis.set_yticks(range(len(selected)), [labels[value] for value in selected])
            axis.tick_params(axis="both", labelsize=11)
            axis.set_ylim(len(selected) - 0.45, -0.55)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x", color="0.9", linewidth=0.6)
        fig.supxlabel(xlabel, fontsize=11, y=0.065)
        fig.text(
            0.5,
            0.015,
            f"95% calendar-day cluster bootstrap intervals; {int(primary.resamples_requested.iloc[0]):,} shared resamples",
            fontsize=11,
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=2.4)
        fig.savefig(directory / name, dpi=230, bbox_inches="tight")
        plt.close(fig)


def review_recorded_panel(
    *,
    panel: Path,
    ambient: Path,
    weather: Path,
    output_directory: Path,
    resamples: int,
    seed: int,
    block_lengths: tuple[int, ...] = (1, 7),
    day_offsets: tuple[int, ...] = (-7, -1, 1, 7),
    rainfall_mode: Literal["audit_only", "dry_only"] = "audit_only",
    raw_source_directory: Path | None = None,
) -> dict[str, object]:
    """Reanalyse recorded aggregates without accepting their upstream measurements."""
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("declare distinct calendar block lengths")
    configs = [_config(resamples, seed, value) for value in block_lengths]
    paths = {"recorded_panel": panel, "actual_ambient": ambient, "weather": weather}
    if raw_source_directory is not None:
        manifest = json.loads((raw_source_directory / "manifest.json").read_text(encoding="utf-8"))
        for name in ("video_master", "video_emissions", "entering_counts", "entering_count_audit"):
            path = raw_source_directory / f"{name}.csv"
            if _hash(path) != manifest["outputs"][path.name]:
                raise ValueError(f"raw source hash differs from its export manifest: {path.name}")
            paths[name] = path
        paths["raw_manifest"] = raw_source_directory / "manifest.json"
    hashes = {name: _hash(path) for name, path in paths.items()}
    rows = _load(panel, ambient, weather)
    input_hours = rows.groupby("site").size().to_dict()
    count_replay = None
    exclusions = pd.DataFrame(
        columns=["site", "monitor_start_utc", "video_id", "video_path", "reason"]
    )
    if raw_source_directory is not None:
        rows, count_replay = _recorded_counts(rows, raw_source_directory)
        material = pd.DataFrame(count_replay["validation_material"])
        if not material.empty:
            exclusions = (
                rows[["site", "monitor_start_utc"]]
                .merge(material, on=["site", "monitor_start_utc"], validate="one_to_many")
                .assign(reason="contains_explicit_counting_test_record")
            )
            exclude_keys = set(zip(exclusions.site, exclusions.monitor_start_utc, strict=True))
            keep = [
                (site, time) not in exclude_keys
                for site, time in zip(rows.site, rows.monitor_start_utc, strict=True)
            ]
            rows = rows.loc[keep].reset_index(drop=True)
    pool = rows.copy()
    if rainfall_mode == "dry_only":
        rows = rows.loc[rows.source_interval_precipitation_mm.eq(0)].reset_index(drop=True)
    elif rainfall_mode != "audit_only":
        raise ValueError("rainfall_mode must be explicitly audit_only or dry_only")
    if set(rows.site) != {"Sukiyabashi", "Shinjuku"}:
        raise ValueError("both sites require recorded hours after rainfall selection")
    calendar_days = (
        pd.Timestamp(rows.local_calendar_date.max()) - pd.Timestamp(rows.local_calendar_date.min())
    ).days + 1
    if any(length > calendar_days for length in block_lengths):
        raise ValueError("block length exceeds the complete calendar")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".recorded-panel-", dir=output_directory.parent
    ) as temp:
        stage = Path(temp) / "review"
        stage.mkdir()
        parameters = {
            "status": STATUS,
            "source_validation": "not_assessed_by_this_reanalysis",
            "input_paths": {name: str(path) for name, path in paths.items()},
            "input_sha256": hashes,
            "target_alignment": "actual_monitor_at_recorded_predictor_hour",
            "recorded_hour_mask": (
                "unchanged" if rainfall_mode == "audit_only" else "dry_at_interval_end"
            ),
            "rainfall_mode": rainfall_mode,
            "precipitation": "previous_hour_measurement_at_interval_end_label",
            "temporal_control_selection": "recorded_mask_for_statistics_not_source_validation",
            "inference": [asdict(config) for config in configs],
            "focal_estimands_per_site": [*PREDICTORS[:2], *(item[0] for item in CONTRASTS)],
            "descriptive_correlations": list(PREDICTORS[2:]),
            "temporal_day_offsets": list(day_offsets),
            "captured_clip_count_replay": count_replay,
            "source_hour_exclusion": "whole_recorded_hours_containing_explicit_counting_test_clips",
            "captured_clip_count_family": (
                {
                    "family_size": 8,
                    "models": list(PREDICTORS[:2]),
                    "baselines": ["captured_clip_entering_count", "captured_clip_bus_heavy_count"],
                    "sites": ["Sukiyabashi", "Shinjuku"],
                }
                if count_replay is not None
                else None
            ),
        }
        _json(stage / "parameters.json", parameters)
        _csv(stage / "recorded_hourly_panel.csv", rows)
        _csv(stage / "excluded_recorded_source_hours.csv", exclusions)
        inference = pd.concat([_inference(rows, config) for config in configs], ignore_index=True)
        controls = _controls(rows, day_offsets)
        _csv(stage / "association_and_paired_contrasts.csv", inference)
        _csv(stage / "temporal_controls.csv", controls)
        _csv(stage / "condition_point_estimates.csv", _condition_points(rows, pool, ambient))
        _csv(stage / "partial_rank_point_estimates.csv", _partial_rank_points(rows))
        if count_replay is not None:
            _csv(
                stage / "captured_clip_count_comparisons.csv",
                pd.concat(
                    [_entry_inference(rows, config) for config in configs], ignore_index=True
                ),
            )
        _figures(inference, stage)
        support = {
            site: {
                "input_recorded_hours": int(input_hours[site]),
                "excluded_validation_start_hours": int(
                    exclusions.loc[exclusions.site.eq(site), "monitor_start_utc"].nunique()
                ),
                "recorded_hours": len(group),
                "site_days": int(group.local_calendar_date.nunique()),
                "rainy_source_intervals": int(group.source_interval_precipitation_mm.gt(0).sum()),
                "missing_source_precipitation": int(
                    group.source_interval_precipitation_mm.isna().sum()
                ),
                **(
                    {
                        "recorded_clips": int(group.captured_clip_count.sum()),
                        "recorded_trajectory_volume": int(group.vehicle_count.sum()),
                        "hours_with_complete_captured_clip_counts": int(group.entering_known.sum()),
                        "hours_with_complete_bus_heavy_counts": int(group.heavy_known.sum()),
                        "model_index_aggregates": {
                            model: {
                                "sum_recorded_index_grams": float(group[model].sum()),
                                "mean_recorded_index_grams": float(group[model].mean()),
                            }
                            for model in PREDICTORS[:2]
                        },
                    }
                    if count_replay is not None
                    else {}
                ),
            }
            for site, group in rows.groupby("site", sort=True)
        }
        summary = {
            "status": STATUS,
            "sites": support,
            "distinct_observed_calendar_dates": int(rows.local_calendar_date.nunique()),
            "full_union_calendar_days": calendar_days,
            "calendar_start_date": str(rows.local_calendar_date.min()),
            "calendar_end_date": str(rows.local_calendar_date.max()),
            "controls": [
                {
                    "site": site,
                    "model": model,
                    "controls": len(group),
                    "control_rho_mean": float(group.control_index_rho.mean()),
                    "control_rho_min": float(group.control_index_rho.min()),
                    "control_rho_max": float(group.control_index_rho.max()),
                    "retained_hours_min": int(group.retained_hours.min()),
                    "retained_hours_max": int(group.retained_hours.max()),
                }
                for (site, model), group in controls.groupby(["site", "model"], sort=True)
            ],
        }
        _json(stage / "summary.json", summary)
        receipt = {
            **parameters,
            "calculator_sha256": {
                path.name: _hash(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("paired_inference.py"),
                    Path(__file__).parents[1] / "robustness" / "temporal.py",
                )
            },
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        _json(stage / "receipt.json", receipt)
        if any(_hash(path) != hashes[name] for name, path in paths.items()):
            raise ValueError("an input changed during the calculation")
        lock = output_directory.parent / f".{output_directory.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if output_directory.exists() or output_directory.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
            stage.rename(output_directory)
        finally:
            os.close(descriptor)
            lock.unlink()
    return summary


def main() -> None:
    """Calculate one explicitly configured, fresh recorded-panel reanalysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ambient", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rainfall-mode", choices=("audit_only", "dry_only"), required=True)
    parser.add_argument("--raw-source-directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(review_recorded_panel(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
