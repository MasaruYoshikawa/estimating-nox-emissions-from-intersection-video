"""Fixed information comparisons on the recorded-hour thesis cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.inference.paired_inference import joint_calendar_block_draws
from vehicle_emission.inference.recorded_hourly_reanalysis import (
    CONTRASTS,
    FAMILY,
    PREDICTORS,
    _config,
    _csv,
    _hash,
    _json,
    _values,
)
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

SITES = ("Sukiyabashi", "Shinjuku")
MODELS = PREDICTORS[:2]
CLASS_NAMES = {0: "bus", 1: "car", 2: "heavy_truck", 3: "light_truck"}
CLASS_COLUMNS = tuple(f"entering_class_{i}" for i in range(4))
SHARES = ("bus_share", "heavy_truck_share", "light_truck_share")
REGIONAL = ("temperature", "wind_speed", "setagaya_nox_ppm")
STAGES = ("marginal", "calendar", "calendar_traffic", "calendar_traffic_regional")
COUNT_BASELINES = (
    "captured_clip_entering_count",
    "captured_clip_bus_heavy_count",
    "fixed_rate_weighted_entries",
)
DIRECTIONS = {f"entering_from_{name}" for name in ("top", "bottom", "left", "right")}
STATUS = "recorded_aggregate_information_extension"


@dataclass(frozen=True)
class ExtensionConfig:
    """Fixed settings for the bounded retrospective extension."""

    core_resamples: int = 20000
    extension_resamples: int = 2000
    seed: int = 42
    block_lengths: tuple[int, ...] = (1, 7)

    def validate(self) -> None:
        if (
            self.core_resamples < 19
            or self.extension_resamples < 2
            or self.core_resamples < self.extension_resamples
        ):
            raise ValueError(
                "require core resamples >= 19 and 2 <= extension resamples <= core resamples"
            )
        if not self.block_lengths or len(set(self.block_lengths)) != len(self.block_lengths):
            raise ValueError("declare distinct calendar block lengths")


def _require_counts(frame: pd.DataFrame, columns: list[str]) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
        raise ValueError("source counts must be finite non-negative integers")


def reference_class_weights() -> pd.DataFrame:
    """Use one fixed steady-speed MOVEStar state, independently of the outcome."""
    model = MOVEStarModel()
    rates = {
        i: float(model.estimate_class_emissions(i, [25 / 3.6] * 3)["Emission Rate"][0][2]) / 3
        for i in range(4)
    }
    if not all(math.isfinite(value) and value > 0 for value in rates.values()):
        raise ValueError("reference class rates require finite positive values")
    return pd.DataFrame(
        [
            {
                "class_id": i,
                "project_class": CLASS_NAMES[i],
                "native_vehicle_type": emission_vehicle_type_for_project_class(i),
                "reference_speed_kmh": 25.0,
                "reference_acceleration_mps2": 0.0,
                "reference_nox_g_per_second": rates[i],
                "normalised_weight": rates[i] / rates[1],
            }
            for i in range(4)
        ]
    )


def prepare_extension_panel(
    core: pd.DataFrame, source: Path, weights: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct class, speed and raw-index summaries on the unchanged core keys."""
    keys = ["site", "monitor_start_utc"]
    if core.duplicated(keys).any() or set(core.site) != set(SITES):
        raise ValueError("core panel requires distinct two-site hour keys")
    videos = pd.read_csv(source / "video_master.csv").rename(columns={"source_site": "site"})
    if videos.duplicated(["site", "video_id"]).any():
        raise ValueError("duplicate source video identity")
    videos["monitor_start_utc"] = (
        pd.to_datetime(videos.recording_datetime_utc, utc=True, format="ISO8601")
        .dt.floor("h")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    videos = videos.merge(core[keys], on=keys, validate="many_to_one")
    emissions = pd.read_csv(source / "video_emissions.csv").rename(columns={"source_site": "site"})
    if emissions.duplicated(["site", "video_id", "estimation_method"]).any():
        raise ValueError("duplicate source emission method/video")
    records = emissions.merge(videos[[*keys, "video_id"]], on=["site", "video_id"])
    panel = core.copy()
    for method, model in (("MOVEStar", MODELS[0]), ("VT-CPFM", MODELS[1])):
        selected = records.loc[records.estimation_method.eq(method)]
        if len(selected) != len(videos):
            raise ValueError("every included clip requires both model records")
        values = selected[["nox_grams", "nox_hybrid_adjusted_grams", "average_speed_kmh"]]
        if not (np.isfinite(values) & values.ge(0)).all().all():
            raise ValueError("clip indices and speed means must be finite and non-negative")
        hourly = selected.groupby(keys).agg(
            raw=("nox_grams", "sum"),
            adjusted=("nox_hybrid_adjusted_grams", "sum"),
            clip_count=("video_id", "count"),
            clip_speed_mean=("average_speed_kmh", "mean"),
            clip_speed_sd=("average_speed_kmh", "std"),
        )
        matched = core.merge(hourly.reset_index(), on=keys, validate="one_to_one")
        if len(matched) != len(core) or not np.allclose(
            matched[model], matched.adjusted, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("source clip indices do not reproduce the core model")
        if not matched.clip_count.eq(matched.captured_clip_count).all():
            raise ValueError("clip support differs from core support")
        panel = panel.merge(
            hourly[["raw"]].rename(columns={"raw": f"{model}_raw"}),
            on=keys,
            validate="one_to_one",
        )
        if method == "MOVEStar":
            if not np.allclose(matched.mean_speed_mps, matched.clip_speed_mean / 3.6):
                raise ValueError(
                    "source recorded mean speeds do not reproduce the core speed proxy"
                )
            speed = hourly[["clip_speed_mean", "clip_speed_sd"]].copy()
            speed["between_clip_speed_cv"] = speed.clip_speed_sd / speed.clip_speed_mean.where(
                speed.clip_speed_mean.gt(0)
            )
            panel = panel.merge(speed, on=keys, validate="one_to_one")
    audit = pd.read_csv(source / "entering_count_audit.csv").rename(columns={"source_site": "site"})
    audit = audit.merge(videos[["site", "video_id"]], on=["site", "video_id"])
    if audit.duplicated(["site", "video_id", "direction"]).any():
        raise ValueError("duplicate entering audit direction")
    universe = audit.groupby(["site", "video_id"]).size()
    if len(universe) != len(videos) or not audit.direction.dropna().isin(DIRECTIONS).all():
        raise ValueError("each included clip requires a canonical recorded-direction audit")
    audit["class_counts_known"] = audit.count_status.isin(
        ["total_and_classes_agree", "class_sum_only"]
    )
    classes = pd.read_csv(source / "entering_counts.csv").rename(columns={"source_site": "site"})
    classes = classes.loc[classes.class_id.notna()].copy()
    classes = classes.merge(videos[["site", "video_id"]], on=["site", "video_id"])
    if (
        not classes.class_id.isin(range(4)).all()
        or not classes.direction.isin(DIRECTIONS).all()
        or classes.duplicated(["site", "video_id", "direction", "class_id"]).any()
    ):
        raise ValueError("class counts require distinct canonical class identities")
    audit_keys = set(audit[["site", "video_id", "direction"]].itertuples(index=False, name=None))
    class_keys = set(classes[["site", "video_id", "direction"]].itertuples(index=False, name=None))
    if not class_keys.issubset(audit_keys):
        raise ValueError("class direction rows require their own source audit")
    _require_counts(classes, ["vehicle_count"])
    arithmetic = classes.groupby(["site", "video_id", "direction"]).vehicle_count.sum()
    checked = audit.merge(
        arithmetic.rename("replayed_class_sum"), on=["site", "video_id", "direction"], how="left"
    )
    known = checked.class_counts_known
    if not checked.loc[known, "class_row_sum"].eq(checked.loc[known, "replayed_class_sum"]).all():
        raise ValueError("class-row sums disagree with the source audit")
    class_totals = (
        classes.groupby(["site", "video_id", "class_id"]).vehicle_count.sum().unstack()
    ).reindex(columns=range(4), fill_value=0)
    # Sparse absent classes denote zero only after the complete class audit succeeds.
    class_totals = class_totals.fillna(0).rename(columns={i: CLASS_COLUMNS[i] for i in range(4)})
    clips = videos[[*keys, "video_id"]].merge(class_totals, on=["site", "video_id"], how="left")
    class_known = audit.groupby(["site", "video_id"]).class_counts_known.all()
    clips = clips.merge(class_known, on=["site", "video_id"], validate="one_to_one")
    hourly_class = clips.groupby(keys)[list(CLASS_COLUMNS)].sum(min_count=1)
    hourly_known = clips.groupby(keys).class_counts_known.all()
    hourly_class.loc[~hourly_known, :] = np.nan
    hourly_class["class_counts_known"] = hourly_known
    panel = panel.merge(hourly_class, on=keys, validate="one_to_one")
    total = panel[list(CLASS_COLUMNS)].sum(axis=1, min_count=4)
    complete = panel.class_counts_known
    if not total.loc[complete].eq(panel.loc[complete, "captured_clip_entering_count"]).all():
        raise ValueError("class counts disagree with captured entering totals")
    lookup = weights.set_index("class_id").normalised_weight
    panel["fixed_rate_weighted_entries"] = sum(
        panel[CLASS_COLUMNS[i]] * lookup.loc[i] for i in range(4)
    )
    for class_id, name in ((0, "bus_share"), (2, "heavy_truck_share"), (3, "light_truck_share")):
        panel[name] = panel[CLASS_COLUMNS[class_id]] / total.where(total.gt(0))
    weather = pd.read_csv(source / "weather_history.csv")
    weather["monitor_start_utc"] = pd.to_datetime(
        weather.timestamp_utc, utc=True, format="ISO8601"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    panel = panel.merge(
        weather[["monitor_start_utc", "temperature", "wind_speed"]],
        on="monitor_start_utc",
        how="left",
        validate="many_to_one",
    )
    ambient = pd.read_csv(source / "air_pollution_history.csv")
    background = ambient.loc[ambient.monitoring_place.eq("Setagaya")].copy()
    background["monitor_start_utc"] = pd.to_datetime(
        background.timestamp_utc, utc=True, format="ISO8601"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    panel = panel.merge(
        background[["monitor_start_utc", "nox_ppm"]].rename(
            columns={"nox_ppm": "setagaya_nox_ppm"}
        ),
        on="monitor_start_utc",
        how="left",
        validate="many_to_one",
    )
    starts = pd.to_datetime(panel.monitor_start_utc, utc=True).dt.tz_convert("Asia/Tokyo")
    panel["hour_jst"] = starts.dt.hour
    panel["weekday_jst"] = starts.dt.dayofweek
    adjustment = ["vehicle_count", *SHARES, *REGIONAL]
    panel["residual_complete"] = np.isfinite(panel[adjustment]).all(axis=1)
    return panel.sort_values(keys, kind="stable").reset_index(drop=True)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def _rho(first: Any, second: Any) -> float:
    return _correlation(rankdata(np.asarray(first)), rankdata(np.asarray(second)))


def rank_residual_values(group: pd.DataFrame) -> dict[str, float]:
    """Refit ranks and all linear projections on one complete resampled population."""
    selected = group.loc[group.residual_complete]
    if len(selected) < 3:
        return {
            f"{stage}:{name}": math.nan
            for stage in STAGES
            for name in (*MODELS, "continuous_minus_movestar")
        }
    values = selected[["ambient_nox_ppm", *MODELS, "vehicle_count", *SHARES, *REGIONAL]].to_numpy(
        float
    )
    ranked = rankdata(values, axis=0)
    calendar = np.column_stack(
        [
            np.ones(len(selected)),
            *[selected.hour_jst.eq(i).to_numpy(float) for i in range(10, 21)],
            *[selected.weekday_jst.eq(i).to_numpy(float) for i in range(1, 7)],
        ]
    )
    designs = [
        np.ones((len(selected), 1)),
        calendar,
        np.column_stack([calendar, ranked[:, 3:7]]),
        np.column_stack([calendar, ranked[:, 3:]]),
    ]
    output = {}
    for stage, design in zip(STAGES, designs, strict=True):
        coefficients, _, design_rank, _ = np.linalg.lstsq(design, ranked[:, :3], rcond=None)
        residuals = ranked[:, :3] - design @ coefficients
        model_values = [_correlation(residuals[:, i], residuals[:, 0]) for i in (1, 2)]
        if len(selected) - design_rank < 3:
            model_values = [math.nan, math.nan]
        for model, value in zip(MODELS, model_values, strict=True):
            output[f"{stage}:{model}"] = value
        output[f"{stage}:continuous_minus_movestar"] = model_values[1] - model_values[0]
    return output


def extension_values(rows: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """Calculate all fixed extension quantities on a paired calendar sample."""
    output = {}
    for site in SITES:
        group = rows.loc[rows.site.eq(site)]
        counts = group.loc[group[list(COUNT_BASELINES)].notna().all(axis=1)]
        for model in MODELS:
            index_rho = _rho(counts[model], counts.ambient_nox_ppm)
            for baseline in COUNT_BASELINES:
                output["count", site, f"{model}:{baseline}"] = index_rho - _rho(
                    counts[baseline], counts.ambient_nox_ppm
                )
            adjusted = _rho(group[model], group.ambient_nox_ppm)
            raw = _rho(group[f"{model}_raw"], group.ambient_nox_ppm)
            output["hybrid", site, f"{model}:adjusted_minus_raw"] = adjusted - raw
            output["hybrid", site, f"{model}:raw"] = raw
            output["hybrid", site, f"{model}:adjusted"] = adjusted
        for estimand, value in rank_residual_values(group).items():
            output["residual", site, estimand] = value
        volatility = group.loc[group.between_clip_speed_cv.notna()]
        cutpoint = volatility.between_clip_speed_cv.median()
        lower = volatility.between_clip_speed_cv.le(cutpoint)
        contrasts = {}
        for label, subset in (("lower", volatility.loc[lower]), ("higher", volatility.loc[~lower])):
            correlations = {
                model: _rho(subset[model], subset.ambient_nox_ppm)
                for model in (*MODELS, "vehicle_count")
            }
            for model, value in correlations.items():
                output["volatility", site, f"{label}:{model}"] = value
            for name, first, second in CONTRASTS:
                value = correlations[first] - correlations[second]
                output["volatility", site, f"{label}:{name}"] = value
                contrasts[label, name] = value
        for name, _, _ in CONTRASTS:
            output["volatility", site, f"lower_minus_higher:{name}"] = (
                contrasts["lower", name] - contrasts["higher", name]
            )
    return output


def _cross_site(values: dict[tuple[str, str], float]) -> dict[str, float]:
    return {
        name: values["Shinjuku", name] - values["Sukiyabashi", name]
        for name in (*MODELS, *(item[0] for item in CONTRASTS))
    }


def _interval(values: list[float], family: int = 1) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        return math.nan, math.nan
    low, high = np.quantile(array, [0.025 / family, 1 - 0.025 / family])
    return float(low), float(high)


def infer_extensions(rows: pd.DataFrame, config: ExtensionConfig) -> dict[str, pd.DataFrame]:
    """Preserve the full union calendar and use common draws throughout."""
    config.validate()
    point = _values(rows)
    extension_point = extension_values(rows)
    cross_point = _cross_site(point)
    core_output, cross_output, extension_output, convergence = [], [], [], []
    checkpoints = sorted(
        {n for n in (1000, 5000, 10000, 20000, config.core_resamples) if n <= config.core_resamples}
    )
    for block in config.block_lengths:
        core_samples: dict[tuple[str, str], list[float]] = {key: [] for key in point}
        extension_samples: dict[tuple[str, str, str], list[float]] = {
            key: [] for key in extension_point
        }
        cross_samples: dict[str, list[float]] = {key: [] for key in cross_point}
        digest = hashlib.sha256(b"recorded-panel-joint-calendar-draws-v1\0")
        extension_digest = None
        for draw, (selected, indices, _) in enumerate(
            joint_calendar_block_draws(rows, _config(config.core_resamples, config.seed, block)), 1
        ):
            digest.update(selected.astype("<i8").tobytes())
            digest.update(indices.astype("<i8").tobytes())
            sample = rows.iloc[indices]
            values = _values(sample)
            for core_key, value in values.items():
                core_samples[core_key].append(value)
            for cross_key, value in _cross_site(values).items():
                cross_samples[cross_key].append(value)
            if draw <= config.extension_resamples:
                for extension_key, value in extension_values(sample).items():
                    extension_samples[extension_key].append(value)
                if draw == config.extension_resamples:
                    extension_digest = digest.hexdigest()
            if draw in checkpoints:
                for (site, estimand), samples in core_samples.items():
                    family = 1 if estimand in PREDICTORS[2:] else 10
                    lo, hi = _interval(samples, family)
                    nominal = _interval(samples)
                    convergence.append(
                        {
                            "site": site,
                            "estimand": estimand,
                            "block_length_days": block,
                            "resamples": draw,
                            "family_size": family,
                            "adjusted_ci_low": lo,
                            "adjusted_ci_high": hi,
                            "ci_low": nominal[0],
                            "ci_high": nominal[1],
                            "resample_sha256": digest.hexdigest(),
                            "expected_lower_tail_draws": draw * 0.025 / family,
                        }
                    )
        for (site, estimand), samples in core_samples.items():
            if not np.isfinite(samples).all():
                raise ValueError("nonfinite core bootstrap draw")
            family = 1 if estimand in PREDICTORS[2:] else 10
            nominal = _interval(samples)
            adjusted = _interval(samples, family)
            group = rows.loc[rows.site.eq(site)]
            core_output.append(
                {
                    "site": site,
                    "estimand": estimand,
                    "estimate": point[site, estimand],
                    "ci_low": nominal[0],
                    "ci_high": nominal[1],
                    "adjusted_ci_low": adjusted[0],
                    "adjusted_ci_high": adjusted[1],
                    "family_size": family,
                    "interval_family": FAMILY if family == 10 else "descriptive",
                    "n_hours": len(group),
                    "n_site_days": group.local_calendar_date.nunique(),
                    "block_length_days": block,
                    "resamples_requested": config.core_resamples,
                    "resamples_valid": len(samples),
                    "resample_sha256": digest.hexdigest(),
                }
            )
        for estimand, samples in cross_samples.items():
            nominal = _interval(samples)
            cross_output.append(
                {
                    "estimand": estimand,
                    "contrast_direction": "Shinjuku_minus_Sukiyabashi",
                    "estimate": cross_point[estimand],
                    "ci_low": nominal[0],
                    "ci_high": nominal[1],
                    "block_length_days": block,
                    "resamples": config.core_resamples,
                    "resample_sha256": digest.hexdigest(),
                    "interval_family": "pointwise_descriptive",
                }
            )
        for (package, site, estimand), samples in extension_samples.items():
            finite = np.isfinite(samples)
            supported = bool(
                finite.all() and math.isfinite(extension_point[package, site, estimand])
            )
            nominal = _interval(samples) if supported else (math.nan, math.nan)
            group = rows.loc[rows.site.eq(site)]
            support = (
                group.loc[group.residual_complete]
                if package == "residual"
                else (
                    group.loc[group[list(COUNT_BASELINES)].notna().all(axis=1)]
                    if package == "count"
                    else (
                        group.loc[group.between_clip_speed_cv.notna()]
                        if package == "volatility"
                        else group
                    )
                )
            )
            point_support = support
            if package == "volatility":
                median = support.between_clip_speed_cv.median()
                if estimand.startswith("lower:"):
                    point_support = support.loc[support.between_clip_speed_cv.le(median)]
                elif estimand.startswith("higher:"):
                    point_support = support.loc[support.between_clip_speed_cv.gt(median)]
            extension_output.append(
                {
                    "package": package,
                    "site": site,
                    "estimand": estimand,
                    "estimate": extension_point[package, site, estimand],
                    "ci_low": nominal[0],
                    "ci_high": nominal[1],
                    "n_hours": len(support),
                    "n_site_days": support.local_calendar_date.nunique(),
                    "resampling_population_hours": len(group),
                    "point_stratum_hours": len(point_support),
                    "point_stratum_days": point_support.local_calendar_date.nunique(),
                    "block_length_days": block,
                    "resamples_requested": config.extension_resamples,
                    "resamples_valid": int(finite.sum()),
                    "resample_sha256": extension_digest,
                    "interval_family": "pointwise_descriptive",
                    "status": "estimated" if supported else "not_estimable",
                }
            )
        print(
            f"Completed calendar block length {block}: {config.core_resamples} core / {config.extension_resamples} extension draws",
            flush=True,
        )
    extension = pd.DataFrame(extension_output)
    tables = {
        f"{name}_comparisons.csv": extension.loc[extension.package.eq(name)].drop(columns="package")
        for name in ("count", "residual", "volatility", "hybrid")
    }
    return {
        "core_inference.csv": pd.DataFrame(core_output),
        "cross_site_contrasts.csv": pd.DataFrame(cross_output),
        "core_interval_convergence.csv": pd.DataFrame(convergence),
        **tables,
    }


def descriptive_tables(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarise recorded support and distributions without additional hypothesis searches."""
    distributions, diurnal, volatility = [], [], []
    columns = ["ambient_nox_ppm", *PREDICTORS, *COUNT_BASELINES, *SHARES, "between_clip_speed_cv"]
    for site, group in rows.groupby("site", sort=True):
        for column in dict.fromkeys(columns):
            values = group[column].dropna()
            distributions.append(
                {
                    "site": site,
                    "variable": column,
                    "n_hours": len(values),
                    "n_unique": values.nunique(),
                    "minimum": values.min(),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "sd": values.std(),
                }
            )
        for hour, subset in group.groupby("hour_jst", sort=True):
            for column in (
                "ambient_nox_ppm",
                *MODELS,
                "vehicle_count",
                "captured_clip_bus_heavy_count",
            ):
                values = subset[column].dropna()
                diurnal.append(
                    {
                        "site": site,
                        "hour_jst": hour,
                        "variable": column,
                        "n_hours": len(values),
                        "mean": values.mean(),
                        "median": values.median(),
                        "q25": values.quantile(0.25),
                        "q75": values.quantile(0.75),
                    }
                )
        selected = group.loc[group.between_clip_speed_cv.notna()]
        median = selected.between_clip_speed_cv.median()
        for label, subset in (
            ("lower", selected.loc[selected.between_clip_speed_cv.le(median)]),
            ("higher", selected.loc[selected.between_clip_speed_cv.gt(median)]),
        ):
            volatility.append(
                {
                    "site": site,
                    "stratum": label,
                    "cutpoint": median,
                    "n_hours": len(subset),
                    "n_site_days": subset.local_calendar_date.nunique(),
                    "mean_recorded_volume": subset.vehicle_count.mean(),
                    "mean_speed_mps": subset.mean_speed_mps.mean(),
                    "mean_bus_heavy_share": (subset.bus_share + subset.heavy_truck_share).mean(),
                }
            )
    return {
        "dataset_descriptives.csv": pd.DataFrame(distributions),
        "diurnal_profiles.csv": pd.DataFrame(diurnal),
        "volatility_support.csv": pd.DataFrame(volatility),
    }


def _figures(rows: pd.DataFrame, tables: dict[str, pd.DataFrame], directory: Path) -> None:
    plt.rcParams.update({"font.size": 11})
    colors = ("#23618c", "#1d7a62")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True)
    dates = pd.date_range(rows.local_calendar_date.min(), rows.local_calendar_date.max()).strftime(
        "%Y-%m-%d"
    )
    for axis, site in zip(axes, SITES, strict=True):
        group = rows.loc[rows.site.eq(site)]
        presence = pd.crosstab(group.hour_jst, group.local_calendar_date).reindex(
            index=range(9, 21), columns=dates, fill_value=0
        )
        axis.imshow(
            presence,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axis.set_title(f"{site} (n = {len(group)})")
        axis.set_yticks([0, 3, 6, 9, 11], [9, 12, 15, 18, 20])
        axis.set_ylabel("Hour start (JST)")
        positions = sorted({i for i in (0, 14, 30, 45, len(dates) - 1) if i < len(dates)})
        axis.set_xticks(positions, [dates[i][5:] for i in positions])
    fig.supxlabel("Calendar date (2026)")
    fig.tight_layout()
    fig.savefig(directory / "fig_recorded_calendar_support.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    primary = tables["core_inference.csv"].loc[lambda value: value.block_length_days.eq(1)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    labels = ("MOVEStar", "VT-CPFM", "Tracked profiles")
    for axis, site in zip(axes, SITES, strict=True):
        selected = (
            primary.loc[primary.site.eq(site)].set_index("estimand").loc[list(PREDICTORS[:3])]
        )
        for j, (_, row) in enumerate(selected.iterrows()):
            color = ("#23618c", "#1d7a62", "#79664a")[j]
            axis.hlines(j, row.ci_low, row.ci_high, color=color, linewidth=2)
            axis.scatter(row.estimate, j, color=color, s=40)
            axis.annotate(
                f"{row.estimate:+.3f}",
                (row.estimate, j),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
            )
        axis.set_yticks(range(3), labels)
        axis.set_ylim(2.45, -0.55)
        axis.axvline(0, color=".6", linestyle="--", linewidth=0.8)
        axis.set_title(f"{site} (n = {int(selected.n_hours.iloc[0])})")
        axis.grid(axis="x", color=".9")
    fig.supxlabel("Spearman correlation with ambient NOx (pointwise 95% interval)")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(directory / "fig_core_associations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    residual = tables["residual_comparisons.csv"].loc[lambda x: x.block_length_days.eq(1)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.1), sharex=True)
    residual_labels = ["Marginal", "Calendar", "+ Tracked profiles / fleet", "+ Regional context"]
    for axis, site in zip(axes, SITES, strict=True):
        for offset, (model, color) in enumerate(zip(MODELS, colors, strict=True)):
            records = residual.loc[residual.site.eq(site)].set_index("estimand")
            selected = records.loc[[f"{stage}:{model}" for stage in STAGES]]
            y = np.arange(4) + (offset - 0.5) * 0.15
            axis.hlines(y, selected.ci_low, selected.ci_high, color=color)
            axis.scatter(selected.estimate, y, color=color, label=("MOVEStar", "VT-CPFM")[offset])
        axis.set_yticks(range(4), residual_labels)
        axis.invert_yaxis()
        axis.axvline(0, color=".6", linestyle="--", linewidth=0.8)
        axis.set_title(site)
        axis.grid(axis="x", color=".9")
    axes[0].legend(loc="best")
    fig.supxlabel("Rank-residual association with ambient NOx (pointwise 95% interval)")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(directory / "fig_rank_residual_associations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    core = tables["core_inference.csv"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    names = [item[0] for item in CONTRASTS]
    for axis, site in zip(axes, SITES, strict=True):
        for j, block in enumerate(sorted(core.block_length_days.unique())):
            selected = (
                core.loc[core.site.eq(site) & core.block_length_days.eq(block)]
                .set_index("estimand")
                .loc[names]
            )
            y = np.arange(3) + (j - 0.5) * 0.17
            axis.hlines(y, selected.adjusted_ci_low, selected.adjusted_ci_high, color=colors[j])
            axis.scatter(selected.estimate, y, color=colors[j], label=f"{block}-day blocks")
        axis.set_yticks(
            range(3),
            [
                "MOVEStar minus tracked profiles",
                "VT-CPFM minus tracked profiles",
                "VT-CPFM minus MOVEStar",
            ],
        )
        axis.invert_yaxis()
        axis.axvline(0, color=".6", linestyle="--", linewidth=0.8)
        axis.set_title(site)
        axis.grid(axis="x", color=".9")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.65, 1), ncol=2, frameon=False
    )
    fig.supxlabel("Paired correlation difference (ten-estimand adjusted 95% interval)")
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(directory / "fig_block_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_extension(
    *, base_bundle: Path, source_directory: Path, output_directory: Path, config: ExtensionConfig
) -> dict[str, Any]:
    """Write one fresh, source-hashed calculation bundle."""
    config.validate()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    base_receipt = json.loads((base_bundle / "receipt.json").read_text())
    for name, expected in base_receipt["outputs"].items():
        if _hash(base_bundle / name) != expected:
            raise ValueError(f"base calculation output hash differs: {name}")
    source_manifest = json.loads((source_directory / "manifest.json").read_text())
    names = (
        "video_master",
        "video_emissions",
        "entering_counts",
        "entering_count_audit",
        "weather_history",
        "air_pollution_history",
    )
    paths = {name: source_directory / f"{name}.csv" for name in names}
    for path in paths.values():
        if _hash(path) != source_manifest["outputs"][path.name]:
            raise ValueError(f"source hash differs from its export manifest: {path.name}")
    paths.update(
        {
            "source_manifest": source_directory / "manifest.json",
            "base_receipt": base_bundle / "receipt.json",
            "base_panel": base_bundle / "recorded_hourly_panel.csv",
            "base_controls": base_bundle / "temporal_controls.csv",
        }
    )
    model_dir = Path(__file__).parents[1] / "emissions"
    for name in (
        "VehicleSrcCoeff.csv",
        "EmsRate_1.csv",
        "EmsRate_2.csv",
        "EmsRate_6.csv",
        "EmsRate_13.csv",
    ):
        paths[name] = model_dir / name
    hashes = {name: _hash(path) for name, path in paths.items()}
    bindings = {
        "actual_ambient": "air_pollution_history",
        "weather": "weather_history",
        "raw_manifest": "source_manifest",
        **{
            name: name
            for name in (
                "video_master",
                "video_emissions",
                "entering_counts",
                "entering_count_audit",
            )
        },
    }
    for base_name, source_name in bindings.items():
        if base_receipt.get("input_sha256", {}).get(base_name) != hashes[source_name]:
            raise ValueError(f"base/source snapshot hash differs: {base_name}")
    calculators = (
        Path(__file__),
        Path(__file__).with_name("recorded_hourly_reanalysis.py"),
        Path(__file__).with_name("paired_inference.py"),
        model_dir / "movestar.py",
        Path(__file__).parents[1] / "vehicle_classes.py",
    )
    calculator_hashes = {str(path): _hash(path) for path in calculators}
    core = pd.read_csv(base_bundle / "recorded_hourly_panel.csv", float_precision="round_trip")
    weights = reference_class_weights()
    panel = prepare_extension_panel(core, source_directory, weights)
    calendar_days = (
        pd.Timestamp(panel.local_calendar_date.max())
        - pd.Timestamp(panel.local_calendar_date.min())
    ).days + 1
    if any(value > calendar_days for value in config.block_lengths):
        raise ValueError("block length exceeds the complete calendar")
    parameters = {
        "status": STATUS,
        "source_validation": "not_assessed_by_this_statistical_extension",
        **asdict(config),
        "core_interval_family": FAMILY,
        "core_family_size": 10,
        "extension_interval_family": "pointwise_descriptive",
        "base_panel_float_parser": "round_trip_preserves_stored_binary_values",
        "weighted_count_reference": "MOVEStar_steady_25kmh_zero_acceleration_normalised_to_car",
        "residual_stages": list(STAGES),
        "residual_calendar": "fixed_hour_JST_and_weekday_indicator_columns",
        "residual_traffic": ["ranked_profile_volume", *[f"ranked_{name}" for name in SHARES]],
        "residual_regional": [f"ranked_{name}" for name in REGIONAL],
        "residual_support": "common_complete_case_all_stages",
        "residual_estimator": "Pearson_correlation_of_OLS_residuals_of_within_sample_average_ranks",
        "minimum_residual_degrees_of_freedom": 3,
        "weather_time": "temperature_and_wind_observations_at_monitor_interval_start",
        "background_time": "Setagaya_actual_same_hour",
        "volatility": "sample_SD_over_mean_across_MOVEStar_clip_average_speeds",
        "stored_clip_speed_definition": "mean_of_speed_rows_at_least_1_kmh_else_all_rows_mean",
        "volatility_zero_mean": "undefined_excluded_only_from_volatility_comparison",
        "volatility_split": "site_median_recomputed_each_draw_ties_lower",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": hashes,
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".information-extension-", dir=output_directory.parent
    ) as temp:
        stage = Path(temp) / "calculation"
        stage.mkdir()
        _json(stage / "parameters.json", parameters)
        _csv(stage / "extension_hourly_panel.csv", panel)
        _csv(stage / "class_reference_weights.csv", weights)
        tables = {**infer_extensions(panel, config), **descriptive_tables(panel)}
        controls = pd.read_csv(base_bundle / "temporal_controls.csv")
        controls["weekday_preserved_for_nonwrapped_pairs"] = controls.day_offset.mod(7).eq(0)
        nonwrapped = controls.retained_hours - controls.retained_wrapped_hours
        controls["retained_weekday_preserving_hours"] = nonwrapped * controls.day_offset.mod(7).eq(
            0
        ) + controls.retained_wrapped_hours * (controls.day_offset.abs() - calendar_days).mod(7).eq(
            0
        )
        tables["offset_control_contrasts.csv"] = controls
        for name, frame in tables.items():
            _csv(stage / name, frame)
        _figures(panel, tables, stage)
        summary = {
            "status": STATUS,
            "sites": {
                site: {
                    "core_hours": len(group),
                    "core_days": group.local_calendar_date.nunique(),
                    "weighted_count_hours": int(group.fixed_rate_weighted_entries.notna().sum()),
                    "residual_common_hours": int(group.residual_complete.sum()),
                    "volatility_hours": int(group.between_clip_speed_cv.notna().sum()),
                    "volatility_median": float(group.between_clip_speed_cv.median()),
                }
                for site, group in panel.groupby("site", sort=True)
            },
        }
        _json(stage / "summary.json", summary)
        receipt = {
            **parameters,
            "calculator_sha256": calculator_hashes,
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        _json(stage / "receipt.json", receipt)
        if any(_hash(path) != hashes[name] for name, path in paths.items()):
            raise ValueError("an input changed during the calculation")
        if any(_hash(path) != calculator_hashes[str(path)] for path in calculators):
            raise ValueError("a calculator changed during the calculation")
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
    """Run the fixed extension on a validated recorded-hour bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--core-resamples", type=int, default=20000)
    parser.add_argument("--extension-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = ExtensionConfig(args.core_resamples, args.extension_resamples, args.seed)
    print(
        json.dumps(
            run_extension(
                base_bundle=args.base_bundle,
                source_directory=args.source_directory,
                output_directory=args.output_directory,
                config=config,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
