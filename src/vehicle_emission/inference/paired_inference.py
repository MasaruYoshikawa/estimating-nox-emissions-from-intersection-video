"""Explicit-configuration inference components for T-128.

Estimands and decisions follow ``coordination/GLOSSARY.md`` and ADR-024.
These components do not establish a final empirical method freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vehicle_emission.inference.contracts import LOCKED_RETROSPECTIVE_SCOPE
from vehicle_emission.time_utils import TOKYO_TIMEZONE

SITES = ("Sukiyabashi", "Shinjuku")
EVIDENCE_STATUS = "provisional_components_only"
ROW_KEY = ["site", "monitor_start_utc", "monitor_end_utc"]
PRIMARY_ESTIMANDS = ("rq1_index", "rq1_count", "rq3_paired_advantage")
WIND_ESTIMAND = "rq2_wind_contrast"
HETEROGENEITY_ESTIMAND = "cross_site_index_heterogeneity"
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHARED_EXTRACTION_FIELDS = (
    "predictor_offset_hours",
    "precipitation_rule",
    "weather_alignment",
    "coverage_mode",
    "minimum_coverage_fraction",
    "partial_hour_normalisation",
    "gap_rule",
    "monitor_label_semantics",
    "method_config_id",
    "method_config_sha256",
)


@dataclass(frozen=True)
class PairedInferenceConfig:
    """Caller-specified candidate settings, without implied method acceptance."""

    resampling_method: Literal["union_calendar_moving_block"]
    block_length_days: int
    minimum_site_hours: int
    minimum_site_days: int
    minimum_stratum_hours: int
    minimum_stratum_days: int
    resamples: int
    minimum_valid_resamples: int
    seed: int
    interval_method: Literal["percentile"]
    interval_adjustment: Literal["bonferroni"]
    confidence_level: float
    primary_interval_family: str
    primary_family_size: int
    wind_interval_family: str
    wind_family_size: int
    wind_mode: Literal["unavailable", "recorded_strata", "projected_speed"]
    wind_definition_id: str | None
    favourable_min_mps: float | None
    opposing_max_mps: float | None

    def __post_init__(self) -> None:
        if self.resampling_method != "union_calendar_moving_block":
            raise ValueError("resampling_method must preserve joint calendar-day blocks")
        minima = {
            "block_length_days": 1,
            "minimum_site_hours": 3,
            "minimum_site_days": 2,
            "minimum_stratum_hours": 3,
            "minimum_stratum_days": 2,
            "resamples": 19,
            "minimum_valid_resamples": 19,
            "primary_family_size": 4,
            "wind_family_size": 2,
            "seed": 0,
        }
        for name, minimum in minima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an explicit integer >= {minimum}")
        if self.minimum_valid_resamples > self.resamples:
            raise ValueError("minimum_valid_resamples cannot exceed requested resamples")
        if self.interval_method != "percentile" or self.interval_adjustment != "bonferroni":
            raise ValueError("only explicit Bonferroni percentile intervals are implemented")
        if isinstance(self.confidence_level, bool) or not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        for name in ("primary_interval_family", "wind_interval_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be explicitly named")
        if self.primary_interval_family == self.wind_interval_family:
            raise ValueError("primary and conditional wind interval families must differ")
        if self.wind_mode == "unavailable":
            if any(
                value is not None
                for value in (
                    self.wind_definition_id,
                    self.favourable_min_mps,
                    self.opposing_max_mps,
                )
            ):
                raise ValueError("unavailable wind must not claim a stratum definition")
        elif self.wind_mode in {"recorded_strata", "projected_speed"}:
            if not isinstance(self.wind_definition_id, str) or not self.wind_definition_id.strip():
                raise ValueError("wind_definition_id is required for wind strata")
            if self.wind_mode == "projected_speed":
                if (
                    self.favourable_min_mps is None
                    or self.opposing_max_mps is None
                    or not math.isfinite(self.favourable_min_mps)
                    or not math.isfinite(self.opposing_max_mps)
                    or self.favourable_min_mps <= 0
                    or self.opposing_max_mps >= 0
                ):
                    raise ValueError("projected wind thresholds must be finite, positive/negative")
            elif self.favourable_min_mps is not None or self.opposing_max_mps is not None:
                raise ValueError("recorded strata do not accept projected-speed thresholds")
        else:
            raise ValueError("unsupported wind_mode")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validated_rows(
    frame: pd.DataFrame, config: PairedInferenceConfig, *, allow_empty: bool = False
) -> pd.DataFrame:
    required = set(ROW_KEY) | {
        "local_calendar_date",
        "ambient_nox_ppm",
        "relative_nox_emission_index",
        "count_only_index",
        "primary_measurement_eligible",
        "target_observation_status",
    }
    if frame.columns.duplicated().any() or not required.issubset(frame.columns):
        raise ValueError(f"eligible hourly rows require unique columns: {sorted(required)}")
    rows = frame.copy().sort_values(ROW_KEY, kind="stable").reset_index(drop=True)
    if (
        (rows.empty and not allow_empty)
        or rows["site"].isna().any()
        or not set(rows["site"]).issubset(SITES)
    ):
        raise ValueError("hourly rows must contain supported, explicit sites")
    if rows.duplicated(ROW_KEY).any():
        raise ValueError("duplicate site-hour keys break the paired comparison")
    for column in ("monitor_start_utc", "monitor_end_utc"):
        if not rows[column].astype("string").str.fullmatch(UTC_PATTERN.pattern).fillna(False).all():
            raise ValueError(f"{column} must use canonical UTC timestamps")
    starts = pd.to_datetime(rows["monitor_start_utc"], utc=True)
    ends = pd.to_datetime(rows["monitor_end_utc"], utc=True)
    if not (ends - starts).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("each target must identify exactly one actual monitor hour")
    if not starts.dt.minute.eq(0).all() or not starts.dt.second.eq(0).all():
        raise ValueError("monitor intervals must use calendar-hour boundaries")
    dates = starts.dt.tz_convert(TOKYO_TIMEZONE).dt.strftime("%Y-%m-%d")
    if not rows["local_calendar_date"].eq(dates).all():
        raise ValueError("local_calendar_date disagrees with the explicit monitor interval")
    if (
        not rows["primary_measurement_eligible"]
        .map(lambda value: isinstance(value, bool | np.bool_) and bool(value))
        .all()
    ):
        raise ValueError("consume the audited eligible projection, not candidate or excluded rows")
    if not rows["target_observation_status"].eq("actual_observation").all():
        raise ValueError("only actual hourly monitor observations can enter inference")
    if "target_is_derived" in rows and rows["target_is_derived"].ne(False).any():
        raise ValueError("interpolated or derived targets cannot enter inference")
    for column in ("ambient_nox_ppm", "relative_nox_emission_index", "count_only_index"):
        values = pd.to_numeric(rows[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} requires finite non-negative paired observations")
        rows[column] = values
    if config.wind_mode == "recorded_strata":
        if "wind_stratum" not in rows:
            raise ValueError("recorded_strata requires the audited wind_stratum column")
        if not set(rows["wind_stratum"].dropna()).issubset({"favourable", "opposing", "other"}):
            raise ValueError("wind_stratum contains an unsupported label")
        rows["_wind"] = rows["wind_stratum"].fillna("missing")
    elif config.wind_mode == "projected_speed":
        if "along_road_wind_speed" not in rows:
            raise ValueError("projected_speed requires along_road_wind_speed")
        speeds = pd.to_numeric(rows["along_road_wind_speed"], errors="raise")
        if np.isinf(speeds.to_numpy(dtype=float)).any():
            raise ValueError("projected wind cannot contain infinite values")
        rows["_wind"] = np.select(
            [speeds.ge(config.favourable_min_mps), speeds.le(config.opposing_max_mps)],
            ["favourable", "opposing"],
            default="other",
        )
        rows.loc[speeds.isna(), "_wind"] = "missing"
    else:
        rows["_wind"] = "missing"
    rows["_cluster"] = rows["local_calendar_date"]
    return rows


def _rho(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(np.clip(np.corrcoef(rankdata(x), rankdata(y))[0, 1], -1.0, 1.0))


def _point_values(
    rows: pd.DataFrame, config: PairedInferenceConfig
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    values: dict[tuple[str, str], float] = {}
    failures: dict[tuple[str, str], str] = {}
    for site in SITES:
        selected = rows.loc[rows["site"] == site]
        if (
            len(selected) < config.minimum_site_hours
            or selected["_cluster"].nunique() < config.minimum_site_days
        ):
            for estimand in PRIMARY_ESTIMANDS:
                values[site, estimand] = math.nan
                failures[site, estimand] = "insufficient_site_support"
        else:
            index = _rho(selected["relative_nox_emission_index"], selected["ambient_nox_ppm"])
            count = _rho(selected["count_only_index"], selected["ambient_nox_ppm"])
            values[site, "rq1_index"] = index
            values[site, "rq1_count"] = count
            values[site, "rq3_paired_advantage"] = index - count
            for estimand in PRIMARY_ESTIMANDS:
                if not math.isfinite(values[site, estimand]):
                    failures[site, estimand] = "constant_or_insufficient_rank_variation"
        strata = [selected.loc[selected["_wind"] == label] for label in ("favourable", "opposing")]
        if any(
            len(stratum) < config.minimum_stratum_hours
            or stratum["_cluster"].nunique() < config.minimum_stratum_days
            for stratum in strata
        ):
            values[site, WIND_ESTIMAND] = math.nan
            failures[site, WIND_ESTIMAND] = "insufficient_wind_stratum_support"
        else:
            correlations = [
                _rho(stratum["relative_nox_emission_index"], stratum["ambient_nox_ppm"])
                for stratum in strata
            ]
            values[site, WIND_ESTIMAND] = correlations[0] - correlations[1]
            if not math.isfinite(values[site, WIND_ESTIMAND]):
                failures[site, WIND_ESTIMAND] = "constant_wind_stratum_or_target"
    key = ("cross_site", HETEROGENEITY_ESTIMAND)
    values[key] = values["Shinjuku", "rq1_index"] - values["Sukiyabashi", "rq1_index"]
    if not math.isfinite(values[key]):
        failures[key] = "site_index_not_estimable"
    return values, failures


def classify_interval(estimand: str, lower: float, upper: float) -> str:
    """Apply literal ADR-024 boundaries without an equivalence decision."""

    if not math.isfinite(lower) or not math.isfinite(upper):
        return "not_estimable"
    if lower > upper:
        raise ValueError("confidence interval bounds are reversed")
    if estimand == "rq1_index":
        if lower > 0.40:
            return "practically_strong"
        if lower > 0:
            return "positive"
        return "non_positive" if upper <= 0 else "inconclusive"
    if estimand == WIND_ESTIMAND:
        if lower > 0.15:
            return "meaningful_moderation"
        return "proposed_magnitude_ruled_out" if upper < 0.15 else "inconclusive"
    if estimand == "rq3_paired_advantage":
        if lower > 0.10:
            return "meaningful_added_value"
        return "meaningful_count_advantage" if upper < -0.10 else "inconclusive"
    return "descriptive"


def _observation_sha256(rows: pd.DataFrame) -> str:
    columns = ROW_KEY + [
        "local_calendar_date",
        "ambient_nox_ppm",
        "relative_nox_emission_index",
        "count_only_index",
        "_wind",
    ]
    return hashlib.sha256(
        rows[columns].to_csv(index=False, float_format="%.17g").encode()
    ).hexdigest()


def joint_calendar_block_draws(
    rows: pd.DataFrame, config: PairedInferenceConfig
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield calendar positions, paired row positions and repeated-day identities.

    The input must be the sorted, validated primary projection. Secondary
    comparisons filter each resulting draw, not the calendar or draw itself.
    """

    calendar = tuple(
        pd.date_range(
            rows["local_calendar_date"].min(), rows["local_calendar_date"].max(), freq="D"
        ).strftime("%Y-%m-%d")
    )
    if config.block_length_days > len(calendar):
        return
    members = {
        date: rows.index[rows["local_calendar_date"].eq(date)].to_numpy(dtype=np.int64)
        for date in calendar
    }
    rng = np.random.default_rng(config.seed)
    for _ in range(config.resamples):
        selected: list[int] = []
        while len(selected) < len(calendar):
            start = int(rng.integers(0, len(calendar) - config.block_length_days + 1))
            selected.extend(range(start, start + config.block_length_days))
        selected = selected[: len(calendar)]
        indices = [members[calendar[position]] for position in selected]
        flattened = np.concatenate(indices)
        clusters = np.concatenate(
            [
                np.repeat(f"{number}:{calendar[position]}", len(indices[number]))
                for number, position in enumerate(selected)
            ]
        )
        yield np.asarray(selected, dtype=np.int64), flattened, clusters


def inference_calendar_dates(
    rows: pd.DataFrame, candidate_calendar_dates: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Retain an explicit candidate calendar when every actual hour is excluded."""

    if rows.empty:
        if not candidate_calendar_dates or any(
            not isinstance(date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) is None
            for date in candidate_calendar_dates
        ):
            raise ValueError(
                "empty eligible support requires explicit canonical candidate calendar dates"
            )
        dates = pd.to_datetime(list(candidate_calendar_dates), format="%Y-%m-%d", errors="raise")
        start, end = dates.min(), dates.max()
    else:
        start, end = rows["local_calendar_date"].min(), rows["local_calendar_date"].max()
    return tuple(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))


def compute_paired_inference(
    frame: pd.DataFrame,
    config: PairedInferenceConfig,
    *,
    candidate_calendar_dates: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compute estimates on one audited wide hourly projection.

    Both sites receive the same consecutive-calendar moving-block draw. Days
    without observations remain explicit empty positions; the calendar is never
    compressed across gaps. All observations within a selected day stay paired.
    """

    rows = _validated_rows(frame, config, allow_empty=candidate_calendar_dates is not None)
    values, failures = _point_values(rows, config)
    samples: dict[tuple[str, str], list[float]] = {key: [] for key in values}
    calendar = inference_calendar_dates(rows, candidate_calendar_dates)
    digest = hashlib.sha256(b"paired-union-calendar-moving-block-v1\0")
    digest.update(json.dumps(calendar).encode())
    digest.update(str(config.block_length_days).encode())
    calendar_failure = config.block_length_days > len(calendar)
    if rows.empty:
        digest.update(b"no_eligible_observations")
    elif not calendar_failure:
        for selected, flattened, clusters in joint_calendar_block_draws(rows, config):
            digest.update(selected.astype("<i8").tobytes())
            digest.update(flattened.astype("<i8").tobytes())
            sampled = rows.iloc[flattened].copy()
            sampled["_cluster"] = clusters
            drawn, _ = _point_values(sampled, config)
            for key, value in drawn.items():
                if math.isfinite(value):
                    samples[key].append(value)
    else:
        digest.update(b"configured_block_exceeds_calendar")
    resample_hash = digest.hexdigest()
    result_rows: list[dict[str, object]] = []
    for (site, estimand), estimate in values.items():
        site_rows = rows if site == "cross_site" else rows.loc[rows["site"] == site]
        if estimand == WIND_ESTIMAND:
            site_rows = site_rows.loc[site_rows["_wind"].isin(["favourable", "opposing"])]
        family = "descriptive_counts" if estimand == "rq1_count" else "descriptive_heterogeneity"
        family_size = 1
        if estimand in {"rq1_index", "rq3_paired_advantage"}:
            family, family_size = config.primary_interval_family, config.primary_family_size
        elif estimand == WIND_ESTIMAND:
            family, family_size = config.wind_interval_family, config.wind_family_size
        reason = failures.get((site, estimand), "")
        if not reason and calendar_failure:
            reason = "configured_block_exceeds_calendar"
        if not reason and len(samples[site, estimand]) < config.minimum_valid_resamples:
            reason = "insufficient_valid_resamples"
        lower = upper = math.nan
        if not reason:
            alpha = (1.0 - config.confidence_level) / family_size
            lower, upper = (
                float(value)
                for value in np.quantile(samples[site, estimand], [alpha / 2, 1.0 - alpha / 2])
            )
        result_rows.append(
            {
                "site": site,
                "estimand": estimand,
                "estimate": estimate,
                "ci_low": lower,
                "ci_high": upper,
                "n_hours": len(site_rows),
                "n_days": site_rows["local_calendar_date"].nunique(),
                "resamples_requested": config.resamples,
                "resamples_valid": len(samples[site, estimand]),
                "interval_method": "union_calendar_moving_block_percentile",
                "interval_adjustment": config.interval_adjustment,
                "interval_family": family,
                "family_size": family_size,
                "confidence_level": config.confidence_level,
                "classification": classify_interval(estimand, lower, upper),
                "analysis_scope": LOCKED_RETROSPECTIVE_SCOPE,
                "evidence_status": EVIDENCE_STATUS,
                "observation_sha256": _observation_sha256(site_rows),
                "resample_sha256": resample_hash,
                "failure_reason": reason,
                "favourable_hours": int(site_rows["_wind"].eq("favourable").sum()),
                "opposing_hours": int(site_rows["_wind"].eq("opposing").sum()),
                "favourable_days": site_rows.loc[
                    site_rows["_wind"].eq("favourable"), "local_calendar_date"
                ].nunique(),
                "opposing_days": site_rows.loc[
                    site_rows["_wind"].eq("opposing"), "local_calendar_date"
                ].nunique(),
            }
        )
    return pd.DataFrame(result_rows)


def load_hourly_bundle(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only the hash-bound primary artifact in an hourly scaffold bundle."""

    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    if (
        manifest.get("manifest_type") != "hourly_measurement_scaffold_bundle"
        or manifest.get("schema_version") != 2
        or manifest.get("analysis_scope") != LOCKED_RETROSPECTIVE_SCOPE
        or manifest.get("configuration_evidence_status") != "self_attested_scaffold_only"
        or manifest.get("method_config_hash_verified") is not False
        or manifest.get("site") not in SITES
    ):
        raise ValueError("input must be an explicit hourly-measurement scaffold bundle")
    artifact = manifest["artifacts"]["ambient_primary"]
    artifact_name = artifact["path"]
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        raise ValueError("hourly bundle artifact path must name a local file")
    csv_path = manifest_path.parent / artifact_name
    csv_bytes = csv_path.read_bytes()
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if csv_hash != artifact["sha256"]:
        raise ValueError("hourly primary artifact does not match its receipt hash")
    header = next(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))
    if len(header) != len(set(header)):
        raise ValueError("hourly primary artifact contains duplicate columns")
    frame = pd.read_csv(io.BytesIO(csv_bytes), float_precision="round_trip")
    if len(frame) != artifact["row_count"] or (
        not frame.empty and set(frame["site"]) != {manifest["site"]}
    ):
        raise ValueError("hourly primary artifact support disagrees with its bundle")
    metadata_fields = (
        "analysis_scope",
        "configuration_evidence_status",
        "method_config_id",
        "method_config_sha256",
        "source_snapshot_id",
    )
    if not (
        set(SHARED_EXTRACTION_FIELDS) | set(metadata_fields) | {"method_config_hash_verified"}
    ).issubset(frame.columns):
        raise ValueError("hourly primary artifact is missing extraction policy metadata")
    for field in metadata_fields:
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"hourly bundle requires explicit {field}")
        if not frame[field].eq(manifest[field]).all():
            raise ValueError(f"hourly artifact {field} disagrees with its bundle")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["method_config_sha256"]) is None:
        raise ValueError("hourly method_config_sha256 must be a lowercase SHA-256")
    if not frame["method_config_hash_verified"].eq(False).all():
        raise ValueError("hourly scaffold cannot assert a verified method hash")
    for field in SHARED_EXTRACTION_FIELDS:
        if not frame.empty and (frame[field].isna().any() or frame[field].nunique() != 1):
            raise ValueError(f"hourly artifact must record one explicit {field}")
    weather_columns = {
        "precipitation_rule",
        "precipitation_mm",
        "weather_observation_status",
        "weather_alignment",
        "weather_source_id",
    }
    if not weather_columns.issubset(frame.columns):
        raise ValueError("CLI hourly bundles require an explicit precipitation eligibility rule")
    if not frame.empty and (
        not frame["precipitation_rule"].eq("dry_only").all()
        or not frame["precipitation_mm"].eq(0).all()
        or not frame["weather_observation_status"].eq("actual_observation").all()
        or not frame["weather_alignment"].isin(["monitor_interval", "predictor_window"]).all()
        or frame["weather_alignment"].nunique() != 1
        or frame["weather_source_id"].isna().any()
        or frame["weather_source_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("CLI hourly bundles must contain actual dry-weather eligible observations")
    window_columns = {
        "predictor_window_start_utc",
        "predictor_window_end_utc",
        "predictor_offset_hours",
    }
    if not window_columns.issubset(frame.columns):
        raise ValueError("hourly bundles require explicit predictor windows and offset semantics")
    if not frame.empty:
        for name in window_columns - {"predictor_offset_hours"}:
            if (
                not frame[name]
                .astype("string")
                .str.fullmatch(UTC_PATTERN.pattern)
                .fillna(False)
                .all()
            ):
                raise ValueError("predictor windows must contain canonical UTC intervals")
        offsets = pd.to_numeric(frame["predictor_offset_hours"], errors="raise")
        if (
            not np.isfinite(offsets).all()
            or offsets.lt(0).any()
            or offsets.ne(np.floor(offsets)).any()
            or offsets.nunique() != 1
        ):
            raise ValueError(
                "hourly predictor offset must be a single non-negative whole-hour rule"
            )
        delta = pd.to_timedelta(offsets, unit="h")
        for boundary in ("start", "end"):
            monitor = pd.to_datetime(frame[f"monitor_{boundary}_utc"], utc=True)
            predictor = pd.to_datetime(frame[f"predictor_window_{boundary}_utc"], utc=True)
            if not predictor.eq(monitor - delta).all():
                raise ValueError("predictor windows do not follow their recorded timestamp offset")
    return frame, {
        "site": manifest["site"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "ambient_primary_sha256": csv_hash,
        "source_snapshot_id": manifest["source_snapshot_id"],
        "method_config_id": manifest["method_config_id"],
        "method_config_sha256": manifest["method_config_sha256"],
    }


def write_inference_artifacts(
    results: pd.DataFrame,
    config: PairedInferenceConfig,
    *,
    inputs: list[dict[str, object]],
    output_directory: Path,
    repository_root: Path,
    code_revision: str,
) -> Path:
    """Publish an immutable component bundle only into a new build directory."""

    root = repository_root.resolve()
    build_root = root / "build"
    target = output_directory.resolve()
    if build_root.is_symlink() or not target.is_relative_to(build_root) or target == build_root:
        raise ValueError("component artifacts must use a fresh subdirectory under build/")
    if output_directory.is_symlink() or target.exists():
        raise FileExistsError("component output directory already exists; choose a new run path")
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(target.relative_to(root))],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("cannot verify committed output paths without a valid repository")
    if tracked.stdout.strip():
        raise FileExistsError("component output path contains committed artifacts")
    csv_bytes = results.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    receipt = {
        "schema_version": 1,
        "artifact_kind": "paired_inference_component",
        "results_csv": "paired_inference.csv",
        "results_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "row_count": len(results),
        "analysis_scope": LOCKED_RETROSPECTIVE_SCOPE,
        "evidence_status": EVIDENCE_STATUS,
        "input_sha256": _json_sha256(inputs),
        "inputs": inputs,
        "config_sha256": _json_sha256(asdict(config)),
        "config": asdict(config),
        "code_revision": code_revision,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calendar_gap_policy": "consecutive_calendar_with_explicit_empty_days",
        "observational_unit": "actual_hourly_ambient_monitor_observation",
        "resampling_unit": "joint_calendar_day_blocks_preserving_intra_day_rows",
    }
    target.mkdir(parents=True, exist_ok=False)
    with (target / "paired_inference.csv").open("xb") as stream:
        stream.write(csv_bytes)
    receipt_path = target / "paired_inference_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    """Run the candidate component from two explicit hash-bound hourly bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-bundle", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PairedInferenceConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    loaded = [load_hourly_bundle(path) for path in args.hourly_bundle]
    if len(loaded) != 2 or {item[1]["site"] for item in loaded} != set(SITES):
        raise ValueError("provide exactly one hourly bundle per site")
    for field in ("method_config_id", "method_config_sha256"):
        if len({item[1][field] for item in loaded}) != 1:
            raise ValueError(f"two-site inputs must share the same {field}")
    combined = pd.concat([item[0] for item in loaded], ignore_index=True)
    for field in SHARED_EXTRACTION_FIELDS:
        if combined[field].nunique(dropna=False) != 1:
            raise ValueError(f"two-site inputs must share the same {field}")
    results = compute_paired_inference(combined, config)
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    receipt = write_inference_artifacts(
        results,
        config,
        inputs=sorted([item[1] for item in loaded], key=lambda item: str(item["site"])),
        output_directory=args.output_directory,
        repository_root=root,
        code_revision=revision,
    )
    print(f"Provisional component bundle: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
