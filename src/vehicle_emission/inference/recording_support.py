"""Recording-schedule sensitivity on an unchanged recorded-hour panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

from vehicle_emission.inference.paired_inference import joint_calendar_block_draws
from vehicle_emission.inference.recorded_hourly_reanalysis import _config, _csv, _hash, _json

SITES = ("Sukiyabashi", "Shinjuku")
KEYS = ["site", "monitor_start_utc"]
MODELS = ("movestar_nox_grams", "vt_cpfm_nox_grams")
COUNT_BASELINES = (
    "captured_clip_entering_count",
    "captured_clip_bus_heavy_count",
    "fixed_rate_weighted_entries",
)
ESTIMANDS = (
    ("movestar_nox_grams", "recorded_core"),
    ("vt_cpfm_nox_grams", "recorded_core"),
    ("vehicle_count", "recorded_core"),
    ("continuous_minus_movestar", "recorded_core"),
    ("continuous_minus_recorded_volume", "recorded_core"),
    ("vt_cpfm_nox_grams", "complete_entry_counts"),
    ("fixed_rate_weighted_entries", "complete_entry_counts"),
    ("continuous_minus_fixed_rate_weighted_entries", "complete_entry_counts"),
)
SUPPORT_RULE = "six_records_six_distinct_starts_one_in_each_ten_minute_clock_bin"


@dataclass(frozen=True)
class RecordingSupportConfig:
    resamples: int = 2000
    seed: int = 42
    block_lengths: tuple[int, ...] = (1, 7)

    def validate(self) -> None:
        if (
            isinstance(self.resamples, bool)
            or not isinstance(self.resamples, int)
            or self.resamples < 19
        ):
            raise ValueError("resamples must be an integer >= 19")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.block_lengths or len(set(self.block_lengths)) != len(self.block_lengths):
            raise ValueError("block lengths must be non-empty and distinct")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.block_lengths):
            raise ValueError("block lengths must be positive integers")


def _timestamps(values: pd.Series, label: str) -> pd.Series:
    if (
        values.isna().any()
        or not values.map(
            lambda value: isinstance(value, str)
            and re.search(r"(?:Z|[+-]\d{2}:\d{2})$", value) is not None
        ).all()
    ):
        raise ValueError(f"{label} requires explicit time zones")
    try:
        result = pd.to_datetime(values, utc=True, format="ISO8601", errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid {label}") from error
    if result.isna().any():
        raise ValueError(f"invalid {label}")
    return result


def _require_columns(frame: pd.DataFrame, names: list[str], label: str) -> None:
    if missing := set(names).difference(frame.columns):
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _nonnegative(
    frame: pd.DataFrame, names: list[str], label: str, *, nullable: bool = False
) -> None:
    try:
        values = frame[names].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} requires numeric values") from error
    valid = np.isfinite(values) & (values >= 0)
    if nullable:
        valid |= np.isnan(values)
    if not valid.all():
        raise ValueError(f"{label} requires finite non-negative values")


def prepare_recording_support(
    panel: pd.DataFrame, videos: pd.DataFrame, emissions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile source membership before deriving the outcome-independent subset."""
    required = [
        *KEYS,
        "monitor_end_utc",
        "local_calendar_date",
        "captured_clip_count",
        "ambient_nox_ppm",
        *MODELS,
        "vehicle_count",
        *COUNT_BASELINES,
    ]
    _require_columns(panel, required, "current panel")
    panel = panel.copy()
    if panel.empty or set(panel.site) != set(SITES) or panel.duplicated(KEYS).any():
        raise ValueError("current panel requires distinct two-site hour keys")
    starts = _timestamps(panel.monitor_start_utc, "panel hour starts")
    ends = _timestamps(panel.monitor_end_utc, "panel hour ends")
    canonical = starts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if (
        not starts.eq(starts.dt.floor("h")).all()
        or not ends.eq(starts + pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("panel intervals must be exact clock hours")
    panel["monitor_start_utc"] = canonical
    if panel.duplicated(KEYS).any():
        raise ValueError("duplicate canonical panel hour")
    dates = starts.dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    if not panel.local_calendar_date.eq(dates).all():
        raise ValueError("panel dates do not match the recorded JST hours")
    _nonnegative(
        panel, ["ambient_nox_ppm", *MODELS, "vehicle_count", "captured_clip_count"], "panel"
    )
    _nonnegative(panel, list(COUNT_BASELINES), "entry baselines", nullable=True)
    for name in ("captured_clip_count", "vehicle_count", *COUNT_BASELINES[:2]):
        values = panel[name].dropna().to_numpy(float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("recorded counts must be integers")
    if not panel.captured_clip_count.gt(0).all():
        raise ValueError("every current hour requires source clips")

    videos = videos.rename(columns={"source_site": "site"}).copy()
    _require_columns(videos, ["site", "video_id", "recording_datetime_utc"], "source videos")
    if (
        videos[["site", "video_id"]].isna().any().any()
        or videos.duplicated(["site", "video_id"]).any()
    ):
        raise ValueError("source clips require distinct non-missing site/video identities")
    video_starts = _timestamps(videos.recording_datetime_utc, "source clip starts")
    videos["monitor_start_utc"] = video_starts.dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    videos["clip_start_utc"] = video_starts.map(lambda value: value.isoformat())
    videos["ten_minute_clock_bin"] = video_starts.dt.minute // 10
    clips = videos.merge(panel[KEYS], on=KEYS, validate="many_to_one")
    hours = clips.groupby(KEYS, sort=True).agg(
        source_clip_count=("video_id", "size"),
        distinct_clip_starts=("clip_start_utc", "nunique"),
        distinct_ten_minute_bins=("ten_minute_clock_bin", "nunique"),
    )
    panel = panel.merge(hours, on=KEYS, how="left", validate="one_to_one")
    if not panel.source_clip_count.eq(panel.captured_clip_count).all():
        raise ValueError("source clip membership/count does not match the current panel")
    panel["restricted_support"] = (
        panel.source_clip_count.eq(6)
        & panel.distinct_clip_starts.eq(6)
        & panel.distinct_ten_minute_bins.eq(6)
    )
    panel["entry_counts_complete"] = panel[list(COUNT_BASELINES)].notna().all(axis=1)

    emissions = emissions.rename(columns={"source_site": "site"}).copy()
    emission_fields = [
        "site",
        "video_id",
        "estimation_method",
        "nox_hybrid_adjusted_grams",
        "vehicle_count",
    ]
    _require_columns(emissions, emission_fields, "source emissions")
    if (
        emissions[emission_fields[:3]].isna().any().any()
        or emissions.duplicated(emission_fields[:3]).any()
    ):
        raise ValueError("duplicate or missing source model/video identity")
    records = emissions.merge(
        clips[[*KEYS, "video_id"]], on=["site", "video_id"], validate="many_to_one"
    )
    for method, model in zip(("MOVEStar", "VT-CPFM"), MODELS, strict=True):
        selected = records.loc[records.estimation_method.eq(method)]
        if len(selected) != len(clips):
            raise ValueError("each included source clip requires both model records")
        _nonnegative(selected, ["nox_hybrid_adjusted_grams", "vehicle_count"], "source models")
        if not selected.vehicle_count.eq(np.floor(selected.vehicle_count)).all():
            raise ValueError("source profile counts must be integers")
        hourly = selected.groupby(KEYS).agg(
            source_index=("nox_hybrid_adjusted_grams", "sum"),
            source_profiles=("vehicle_count", "sum"),
        )
        matched = panel.merge(hourly, on=KEYS, how="left", validate="one_to_one")
        if not np.allclose(matched[model], matched.source_index, rtol=1e-12, atol=1e-12):
            raise ValueError("source indices do not reproduce the current panel")
        if not matched.vehicle_count.eq(matched.source_profiles).all():
            raise ValueError("source profile counts do not reproduce the current panel")
    clips = clips[[*KEYS, "video_id", "clip_start_utc", "ten_minute_clock_bin"]]
    return (
        panel.sort_values(KEYS, kind="stable").reset_index(drop=True),
        clips.sort_values([*KEYS, "clip_start_utc", "video_id"], kind="stable").reset_index(
            drop=True
        ),
    )


def _rho(first: pd.Series, second: pd.Series) -> float:
    values = np.column_stack([first.to_numpy(float), second.to_numpy(float)])
    if len(values) < 3 or not np.isfinite(values).all():
        return math.nan
    ranked = rankdata(values, axis=0)
    if np.any(np.std(ranked, axis=0) < 1e-12):
        return math.nan
    return float(np.corrcoef(ranked, rowvar=False)[0, 1])


def recording_support_values(rows: pd.DataFrame) -> dict[tuple[str, str, str, str], float]:
    """Evaluate paired predictors and nested support populations on one draw."""
    output = {}
    for site in SITES:
        site_rows = rows.loc[rows.site.eq(site)]
        for cohort in ("current", "restricted"):
            group = (
                site_rows if cohort == "current" else site_rows.loc[site_rows.restricted_support]
            )
            core = {
                name: _rho(group[name], group.ambient_nox_ppm)
                for name in (*MODELS, "vehicle_count")
            }
            core["continuous_minus_movestar"] = core[MODELS[1]] - core[MODELS[0]]
            core["continuous_minus_recorded_volume"] = core[MODELS[1]] - core["vehicle_count"]
            counts = group.loc[group.entry_counts_complete]
            entry = {
                name: _rho(counts[name], counts.ambient_nox_ppm)
                for name in (MODELS[1], COUNT_BASELINES[2])
            }
            entry["continuous_minus_fixed_rate_weighted_entries"] = (
                entry[MODELS[1]] - entry[COUNT_BASELINES[2]]
            )
            for population, values in (("recorded_core", core), ("complete_entry_counts", entry)):
                for estimand, value in values.items():
                    output[site, cohort, population, estimand] = value
        for estimand, population in ESTIMANDS:
            output[site, "restricted_minus_current", population, estimand] = (
                output[site, "restricted", population, estimand]
                - output[site, "current", population, estimand]
            )
    return output


def infer_recording_support(panel: pd.DataFrame, config: RecordingSupportConfig) -> pd.DataFrame:
    """Use the original union calendar for every paired descriptive contrast."""
    config.validate()
    panel = panel.sort_values(KEYS, kind="stable").reset_index(drop=True)
    calendar = pd.date_range(
        panel.local_calendar_date.min(), panel.local_calendar_date.max(), freq="D"
    )
    if any(block > len(calendar) for block in config.block_lengths):
        raise ValueError("block length exceeds the current panel calendar")
    points = recording_support_values(panel)
    support = {}
    for site in SITES:
        for population in ("recorded_core", "complete_entry_counts"):
            current = panel.loc[panel.site.eq(site)]
            if population == "complete_entry_counts":
                current = current.loc[current.entry_counts_complete]
            restricted = current.loc[current.restricted_support]
            support[site, population] = {
                "current_hours": len(current),
                "restricted_hours": len(restricted),
                "current_site_days": current.local_calendar_date.nunique(),
                "restricted_site_days": restricted.local_calendar_date.nunique(),
            }
    output = []
    for block in config.block_lengths:
        samples: dict[tuple[str, str, str, str], list[float]] = {key: [] for key in points}
        digest = hashlib.sha256(b"recording-support-joint-calendar-draws-v1\0")
        for selected, indices, _ in joint_calendar_block_draws(
            panel, _config(config.resamples, config.seed, block)
        ):
            digest.update(selected.astype("<i8").tobytes())
            digest.update(indices.astype("<i8").tobytes())
            for key, value in recording_support_values(panel.iloc[indices]).items():
                samples[key].append(value)
        for (site, cohort, population, estimand), point in points.items():
            array = np.asarray(samples[site, cohort, population, estimand], dtype=float)
            valid = int(np.isfinite(array).sum())
            complete = (
                math.isfinite(point)
                and len(array) == config.resamples
                and valid == config.resamples
            )
            low, high = np.quantile(array, [0.025, 0.975]) if complete else (math.nan, math.nan)
            output.append(
                {
                    "site": site,
                    "cohort": cohort,
                    "population": population,
                    "estimand": estimand,
                    "estimate": point,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    **support[site, population],
                    "calendar_days": len(calendar),
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


def _verified_inputs(source: Path, extension: Path) -> dict[str, Path]:
    paths = {
        "extension_receipt": extension / "receipt.json",
        "current_panel": extension / "extension_hourly_panel.csv",
        "source_manifest": source / "manifest.json",
        "video_master": source / "video_master.csv",
        "video_emissions": source / "video_emissions.csv",
    }
    receipt = json.loads(paths["extension_receipt"].read_text())
    manifest = json.loads(paths["source_manifest"].read_text())
    if receipt["outputs"].get(paths["current_panel"].name) != _hash(paths["current_panel"]):
        raise ValueError("current panel differs from its extension receipt")
    for name in ("source_manifest", "video_master", "video_emissions"):
        if receipt["input_sha256"].get(name) != _hash(paths[name]):
            raise ValueError(f"source snapshot differs from the current extension: {name}")
        if name != "source_manifest" and manifest["outputs"].get(paths[name].name) != _hash(
            paths[name]
        ):
            raise ValueError(f"source differs from its export manifest: {name}")
    return paths


def run_recording_support(
    *,
    source_directory: Path,
    extension_bundle: Path,
    output_directory: Path,
    config: RecordingSupportConfig | None = None,
) -> dict[str, Any]:
    """Write a fresh bundle without modifying the source or existing results."""
    config = config or RecordingSupportConfig()
    config.validate()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    target = output_directory.resolve()
    if any(target.is_relative_to(path.resolve()) for path in (source_directory, extension_bundle)):
        raise ValueError("output must be separate from source and existing result bundles")
    paths = _verified_inputs(source_directory, extension_bundle)
    hashes = {name: _hash(path) for name, path in paths.items()}
    calculators = (
        Path(__file__),
        Path(__file__).with_name("paired_inference.py"),
        Path(__file__).with_name("recorded_hourly_reanalysis.py"),
    )
    code_hashes = {str(path): _hash(path) for path in calculators}
    panel, clips = prepare_recording_support(
        pd.read_csv(paths["current_panel"], float_precision="round_trip"),
        pd.read_csv(paths["video_master"], float_precision="round_trip"),
        pd.read_csv(paths["video_emissions"], float_precision="round_trip"),
    )
    inference = infer_recording_support(panel, config)
    support_columns = [
        *KEYS,
        "local_calendar_date",
        "captured_clip_count",
        "source_clip_count",
        "distinct_clip_starts",
        "distinct_ten_minute_bins",
        "restricted_support",
        "entry_counts_complete",
    ]
    summary = []
    for site in SITES:
        for cohort in ("current", "restricted"):
            group = panel.loc[panel.site.eq(site)]
            if cohort == "restricted":
                group = group.loc[group.restricted_support]
            summary.append(
                {
                    "site": site,
                    "cohort": cohort,
                    "hours": len(group),
                    "site_days": group.local_calendar_date.nunique(),
                    "source_clips": int(group.source_clip_count.sum()),
                    "complete_entry_hours": int(group.entry_counts_complete.sum()),
                }
            )
    parameters = {
        **asdict(config),
        "support_rule": SUPPORT_RULE,
        "calendar_start": panel.local_calendar_date.min(),
        "calendar_end": panel.local_calendar_date.max(),
        "calendar_days": (
            pd.Timestamp(panel.local_calendar_date.max())
            - pd.Timestamp(panel.local_calendar_date.min())
        ).days
        + 1,
        "resampling": "shared_union_calendar_nonwrapping_moving_blocks",
        "interval_method": "percentile_95_pointwise_descriptive",
        "invalid_draw_rule": "retain_invalid_count_and_withhold_interval_if_any_draw_invalid",
        "entry_population": "common_complete_case_all_three_existing_count_baselines",
        "comparison_direction": "restricted_minus_current",
        "recording_support": "recorded_start_schedule_not_continuous_media_coverage",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": hashes,
        "calculator_sha256": code_hashes,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".recording-support-", dir=output_directory.parent
    ) as temp:
        stage = Path(temp) / "calculation"
        stage.mkdir()
        _csv(stage / "recording_support_clips.csv", clips)
        _csv(stage / "recording_support_hours.csv", panel[support_columns])
        _csv(stage / "recording_support_summary.csv", pd.DataFrame(summary))
        _csv(stage / "recording_support_inference.csv", inference)
        if hashes != {name: _hash(path) for name, path in paths.items()} or code_hashes != {
            str(path): _hash(path) for path in calculators
        }:
            raise ValueError("an input or calculator changed during recording-support calculation")
        receipt = {
            **parameters,
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
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-days", type=int, nargs="+", default=[1, 7])
    args = parser.parse_args()
    receipt = run_recording_support(
        source_directory=args.source_directory,
        extension_bundle=args.extension_bundle,
        output_directory=args.output_directory,
        config=RecordingSupportConfig(args.resamples, args.seed, tuple(args.block_days)),
    )
    print(
        json.dumps(
            {"output_directory": str(args.output_directory), "outputs": len(receipt["outputs"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
