"""Site-specific, fail-closed construction of hourly measurement rows.

This module aligns measurements and records coverage only. It does not select
an analysis lag or coverage threshold and is not connected to empirical
execution; callers must supply those choices explicitly for this scaffold.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from vehicle_emission.inference.clip_coverage import (
    audit_clip_hour_coverage,
    recorded_second_mask,
)
from vehicle_emission.inference.contracts import (
    LOCKED_RETROSPECTIVE_SCOPE,
    MONITOR_LABEL_SEMANTICS,
    VIDEO_COVERAGE_GAP_RULES,
)
from vehicle_emission.inference.hourly_baselines import (
    aggregate_calibrated_speed,
    prepare_speed_samples,
)
from vehicle_emission.time_utils import TOKYO_TIMEZONE

SITES = frozenset({"Sukiyabashi", "Shinjuku"})
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ROW_KEY = ["site", "monitor_start_utc", "monitor_end_utc"]
EXPECTED_SECONDS = 3600
ACTUAL_OBSERVATION = "actual_observation"
NON_ACTUAL_OBSERVATIONS = frozenset({"interpolated", "forward_filled", "repeated"})
OBSERVATION_STATUSES = frozenset({ACTUAL_OBSERVATION, *NON_ACTUAL_OBSERVATIONS})

AnalysisScope = Literal["locked_retrospective_two_site"]
CoverageMode = Literal["complete_only", "threshold_normalize"]
Normalisation = Literal["none", "same_covered_seconds_scale_for_all_predictors"]
GapRule = Literal["require_contiguous_coverage", "allow_internal_gaps"]
PrecipitationRule = Literal["not_configured", "dry_only"]
WeatherAlignment = Literal["predictor_window", "monitor_interval"]


class HourlyMeasurementError(ValueError):
    """Raised when an hourly measurement contract fails closed."""


@dataclass(frozen=True, slots=True)
class FrozenHourlyExtractionConfig:
    """Caller-attested choices for the standalone extraction scaffold.

    The SHA-256 is syntax-checked but is not verified against a file here. This
    value neither proves a freeze nor authorises an empirical run.
    """

    method_config_id: str
    method_config_sha256: str
    source_snapshot_id: str
    status: str
    analysis_scope: AnalysisScope
    site: str
    monitor_label_semantics: str
    predictor_offset_hours: int
    coverage_rule_id: str
    coverage_mode: CoverageMode
    minimum_coverage_fraction: float
    partial_hour_normalisation: Normalisation
    gap_rule: GapRule
    background_site: str | None
    precipitation_rule: PrecipitationRule = "not_configured"
    weather_alignment: WeatherAlignment | None = None

    def __post_init__(self) -> None:
        for name in ("method_config_id", "source_snapshot_id", "coverage_rule_id", "gap_rule"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise HourlyMeasurementError(f"{name} must be recorded explicitly")
        if not re.fullmatch(r"[0-9a-f]{64}", self.method_config_sha256):
            raise HourlyMeasurementError("method_config_sha256 must be a lowercase SHA-256")
        if self.status != "self_attested_scaffold_only":
            raise HourlyMeasurementError(
                "hourly extraction requires a self-attested scaffold-only config"
            )
        if self.analysis_scope != LOCKED_RETROSPECTIVE_SCOPE:
            raise HourlyMeasurementError("analysis_scope must be locked_retrospective_two_site")
        if self.site not in SITES:
            raise HourlyMeasurementError(f"unsupported analysis site: {self.site!r}")
        if self.monitor_label_semantics not in MONITOR_LABEL_SEMANTICS:
            raise HourlyMeasurementError(
                "monitor label semantics must be end-labelled-hour for exact calendar hours"
            )
        if (
            isinstance(self.predictor_offset_hours, bool)
            or not isinstance(self.predictor_offset_hours, int)
            or self.predictor_offset_hours < 0
        ):
            raise HourlyMeasurementError(
                "predictor_offset_hours must be an explicit non-negative whole hour"
            )
        if (
            isinstance(self.minimum_coverage_fraction, bool)
            or not isinstance(self.minimum_coverage_fraction, Real)
            or not 0.0 < float(self.minimum_coverage_fraction) <= 1.0
        ):
            raise HourlyMeasurementError("minimum_coverage_fraction must be explicit and in (0, 1]")
        if self.coverage_mode == "complete_only":
            if (
                float(self.minimum_coverage_fraction) != 1.0
                or self.partial_hour_normalisation != "none"
            ):
                raise HourlyMeasurementError(
                    "complete_only requires full coverage and no normalisation"
                )
        elif self.coverage_mode == "threshold_normalize":
            if self.partial_hour_normalisation != "same_covered_seconds_scale_for_all_predictors":
                raise HourlyMeasurementError(
                    "partial coverage must use one scale for every paired predictor"
                )
        else:
            raise HourlyMeasurementError(f"unsupported coverage mode: {self.coverage_mode!r}")
        if self.gap_rule not in VIDEO_COVERAGE_GAP_RULES:
            raise HourlyMeasurementError(f"unsupported gap rule: {self.gap_rule!r}")
        if self.background_site is not None and (
            not self.background_site.strip() or self.background_site == self.site
        ):
            raise HourlyMeasurementError("background_site must name a distinct station")
        if self.precipitation_rule == "dry_only":
            if self.weather_alignment not in {"predictor_window", "monitor_interval"}:
                raise HourlyMeasurementError("dry_only requires explicit weather_alignment")
        elif self.precipitation_rule != "not_configured" or self.weather_alignment is not None:
            raise HourlyMeasurementError("unsupported precipitation rule or weather alignment")


@dataclass(frozen=True)
class HourlyMeasurementTables:
    """All candidate rows plus explicitly separated eligible tables."""

    audit: pd.DataFrame
    ambient_primary: pd.DataFrame
    background_sensitivity: pd.DataFrame


@dataclass(frozen=True, slots=True)
class HourlyMeasurementArtifactReceipt:
    """Identity of one immutable hourly-measurement bundle."""

    site: str
    analysis_scope: str
    output_directory: str
    artifact_sha256: tuple[tuple[str, str], ...]
    row_counts: tuple[tuple[str, int], ...]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise HourlyMeasurementError(f"{name} is missing required columns: {missing}")


def _require_site(frame: pd.DataFrame, column: str, expected: str, name: str) -> None:
    values = set(frame[column].dropna().astype(str))
    if not frame.empty and (frame[column].isna().any() or values != {expected}):
        raise HourlyMeasurementError(
            f"{name}.{column} must contain only {expected!r}; found {sorted(values)!r}"
        )


def _utc(values: pd.Series, field: str) -> pd.Series:
    text = values.astype("string")
    valid = text.str.fullmatch(UTC_PATTERN.pattern).fillna(False)
    if not valid.all():
        raise HourlyMeasurementError(f"{field} must use canonical UTC text ending in Z")
    return pd.to_datetime(text, utc=True, errors="raise")


def _hour_rows(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    starts = _utc(result["monitor_start_utc"], f"{name}.monitor_start_utc")
    ends = _utc(result["monitor_end_utc"], f"{name}.monitor_end_utc")
    if not ((ends - starts).dt.total_seconds() == EXPECTED_SECONDS).all():
        raise HourlyMeasurementError(f"{name} intervals must be explicit half-open UTC hours")
    local_starts = starts.dt.tz_convert(TOKYO_TIMEZONE)
    local_ends = ends.dt.tz_convert(TOKYO_TIMEZONE)
    if not (
        local_starts.eq(local_starts.dt.floor("h")).all()
        and local_ends.eq(local_ends.dt.floor("h")).all()
    ):
        raise HourlyMeasurementError(
            f"{name} intervals must use exact Asia/Tokyo local calendar-hour boundaries"
        )
    if result.duplicated(ROW_KEY).any():
        raise HourlyMeasurementError(f"{name} contains duplicate site-hour keys")
    return result


def _numeric(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if (frame[column].notna() & values.isna()).any() or not values.dropna().map(np.isfinite).all():
        raise HourlyMeasurementError(f"{name}.{column} must contain finite numeric values")
    return values.astype(float)


def _observation_status(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = frame[column].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    unsupported = ~values.isin(OBSERVATION_STATUSES) & ~missing
    if missing.any() or unsupported.any():
        observed = sorted(set(values[~missing].astype(str)) - OBSERVATION_STATUSES)
        detail = f": {observed}" if observed else ""
        raise HourlyMeasurementError(
            f"{name}.{column} must explicitly identify actual or derived observations{detail}"
        )
    return values


def _validate_monitor_labels(
    frame: pd.DataFrame,
    *,
    label_column: str,
    semantics: str,
    name: str,
) -> None:
    labels = _utc(frame[label_column], f"{name}.{label_column}")
    if semantics not in MONITOR_LABEL_SEMANTICS:
        raise HourlyMeasurementError(f"unsupported monitor label semantics: {semantics!r}")
    expected_labels = _utc(frame["monitor_end_utc"], f"{name}.monitor_end_utc")
    if not labels.equals(expected_labels):
        raise HourlyMeasurementError(
            f"{name}.{label_column} does not match {semantics} interval boundaries"
        )


def _reasons(values: list[str]) -> str:
    return ";".join(value for value in values if value)


def _align_weather(
    audit: pd.DataFrame,
    *,
    config: FrozenHourlyExtractionConfig,
    weather_observations: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "weather_source_id",
        "weather_label_utc",
        "weather_observation_status",
        "precipitation_mm",
        "along_road_wind_speed",
        "wind_stratum",
        "wind_definition_id",
    ]
    if config.precipitation_rule == "not_configured":
        if weather_observations is not None:
            raise HourlyMeasurementError("weather input requires an explicit dry_only rule")
        for column in ["weather_window_start_utc", "weather_window_end_utc", *columns]:
            audit[column] = pd.NA
        audit["weather_exclusion_reasons"] = ""
        return audit

    window_prefix = (
        "predictor_window" if config.weather_alignment == "predictor_window" else "monitor"
    )
    audit["weather_window_start_utc"] = audit[f"{window_prefix}_start_utc"]
    audit["weather_window_end_utc"] = audit[f"{window_prefix}_end_utc"]
    merge_key = ["site", "weather_window_start_utc", "weather_window_end_utc"]
    if weather_observations is None:
        weather = pd.DataFrame(columns=[*merge_key, *columns])
    else:
        required = {
            "site",
            "weather_start_utc",
            "weather_end_utc",
            "weather_source_id",
            "weather_label_utc",
            "weather_observation_status",
            "precipitation_mm",
        }
        _require_columns(weather_observations, required, "weather_observations")
        _require_site(weather_observations, "site", config.site, "weather_observations")
        weather = weather_observations[
            sorted(required)
            + [
                column
                for column in ("along_road_wind_speed", "wind_stratum", "wind_definition_id")
                if column in weather_observations
            ]
        ].rename(
            columns={"weather_start_utc": "monitor_start_utc", "weather_end_utc": "monitor_end_utc"}
        )
        weather = _hour_rows(weather, "weather_observations")
        _validate_monitor_labels(
            weather,
            label_column="weather_label_utc",
            semantics=config.monitor_label_semantics,
            name="weather_observations",
        )
        if (
            not weather["weather_source_id"]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise HourlyMeasurementError("weather_source_id must be recorded explicitly")
        weather["weather_observation_status"] = _observation_status(
            weather, "weather_observation_status", "weather_observations"
        )
        weather["precipitation_mm"] = _numeric(weather, "precipitation_mm", "weather_observations")
        if weather["precipitation_mm"].dropna().lt(0).any():
            raise HourlyMeasurementError("weather_observations.precipitation_mm cannot be negative")
        if "along_road_wind_speed" in weather:
            weather["along_road_wind_speed"] = _numeric(
                weather, "along_road_wind_speed", "weather_observations"
            )
        else:
            weather["along_road_wind_speed"] = np.nan
        for column in ("wind_stratum", "wind_definition_id"):
            if column not in weather:
                weather[column] = pd.NA
        weather = weather.rename(
            columns={
                "monitor_start_utc": "weather_window_start_utc",
                "monitor_end_utc": "weather_window_end_utc",
            }
        )
    result = audit.merge(weather, on=merge_key, how="left", validate="one_to_one")
    result["weather_exclusion_reasons"] = [
        (
            "missing_precipitation_reference"
            if pd.isna(rain)
            else (
                f"non_actual_weather_reference:{status}"
                if status != ACTUAL_OBSERVATION
                else ("rainy_hour" if rain > 0 else "")
            )
        )
        for rain, status in zip(
            result["precipitation_mm"], result["weather_observation_status"], strict=True
        )
    ]
    return result


def _single_metadata_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        raise HourlyMeasurementError(f"audit is missing required metadata column: {column}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise HourlyMeasurementError(f"audit.{column} must contain one constant value")
    return values.iloc[0]


def _validate_publishable_tables(tables: HourlyMeasurementTables) -> tuple[str, dict[str, object]]:
    audit = tables.audit
    if audit.empty:
        raise HourlyMeasurementError("hourly measurement audit cannot be empty")
    site = str(_single_metadata_value(audit, "site"))
    if site not in SITES:
        raise HourlyMeasurementError(f"unsupported analysis site: {site!r}")
    metadata = {
        "analysis_scope": _single_metadata_value(audit, "analysis_scope"),
        "configuration_evidence_status": _single_metadata_value(
            audit, "configuration_evidence_status"
        ),
        "method_config_hash_verified": _single_metadata_value(audit, "method_config_hash_verified"),
        "method_config_id": _single_metadata_value(audit, "method_config_id"),
        "method_config_sha256": _single_metadata_value(audit, "method_config_sha256"),
        "source_snapshot_id": _single_metadata_value(audit, "source_snapshot_id"),
    }
    if metadata["analysis_scope"] != LOCKED_RETROSPECTIVE_SCOPE:
        raise HourlyMeasurementError("hourly bundle must record locked retrospective scope")
    if metadata["configuration_evidence_status"] != "self_attested_scaffold_only":
        raise HourlyMeasurementError("hourly bundle has an unsupported evidence status")
    hash_verified = metadata["method_config_hash_verified"]
    if not isinstance(hash_verified, bool | np.bool_) or bool(hash_verified):
        raise HourlyMeasurementError("scaffold output cannot claim a verified method hash")

    for name, frame, eligibility_column in (
        ("ambient_primary", tables.ambient_primary, "primary_measurement_eligible"),
        (
            "background_sensitivity",
            tables.background_sensitivity,
            "background_sensitivity_eligible",
        ),
    ):
        if frame.duplicated(ROW_KEY).any():
            raise HourlyMeasurementError(f"{name} contains duplicate site-hour keys")
        if not frame.empty and set(frame["site"].astype(str)) != {site}:
            raise HourlyMeasurementError(f"{name} must contain only {site!r}")
        expected_keys = set(
            map(
                tuple,
                audit.loc[audit[eligibility_column], ROW_KEY].itertuples(index=False, name=None),
            )
        )
        actual_keys = set(map(tuple, frame[ROW_KEY].itertuples(index=False, name=None)))
        if actual_keys != expected_keys:
            raise HourlyMeasurementError(f"{name} does not match its audited eligible support")
        missing_projection_columns = set(frame.columns) - set(audit.columns)
        if missing_projection_columns:
            raise HourlyMeasurementError(
                f"{name} contains columns absent from the audit projection: "
                f"{sorted(missing_projection_columns)}"
            )
        expected_projection = audit.loc[audit[eligibility_column], list(frame.columns)].reset_index(
            drop=True
        )
        actual_projection = frame.reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                actual_projection,
                expected_projection,
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise HourlyMeasurementError(
                f"{name} values do not exactly match the audited eligible projection"
            ) from exc

    primary = tables.ambient_primary
    if not primary.empty:
        if not primary["target_observation_status"].eq(ACTUAL_OBSERVATION).all():
            raise HourlyMeasurementError("ambient primary output may contain only actual targets")
        if primary["target_is_derived"].astype(bool).any():
            raise HourlyMeasurementError("derived targets cannot enter ambient primary output")
        if primary[["count_only_index", "relative_nox_emission_index"]].isna().any(axis=None):
            raise HourlyMeasurementError("paired predictors must share identical complete support")

    missing_video = audit["coverage_status"].eq("missing_video_coverage")
    missing_columns = [
        "count_only_raw",
        "relative_nox_emission_index_raw",
        "count_only_index",
        "relative_nox_emission_index",
    ]
    if audit.loc[missing_video, missing_columns].notna().any(axis=None):
        raise HourlyMeasurementError("missing video coverage cannot be represented as zero")
    return site, metadata


def _serialise_frame_csv(frame: pd.DataFrame) -> bytes:
    if frame.columns.duplicated().any():
        raise HourlyMeasurementError("hourly output cannot contain duplicate columns")
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    ).encode("utf-8")


def _write_immutable_outputs(outputs: dict[Path, bytes]) -> None:
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Hourly output already exists and is immutable: {path}. "
                "Use a new version rather than replacing it."
            )
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _check_output_destination(output_directory: Path) -> Path:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError("Hourly output directory is immutable; choose a new directory")
    ancestor = output_directory.parent
    while not ancestor.exists() or ancestor.is_symlink():
        ancestor = ancestor.parent
    for candidate in (ancestor, *ancestor.parents):
        if (candidate / ".git").exists():
            ancestor = candidate
            break
    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ancestor,
        capture_output=True,
        text=True,
        check=False,
    )
    if (ancestor / ".git").exists() and repository.returncode != 0:
        raise HourlyMeasurementError("cannot verify committed hourly output paths")
    target = output_directory.resolve()
    if repository.returncode == 0:
        root = Path(repository.stdout.strip()).resolve()
        build_root = root / "build"
        if build_root.is_symlink() or not target.is_relative_to(build_root) or target == build_root:
            raise HourlyMeasurementError(
                "repository hourly outputs must use a new directory under build/"
            )
        tracked = subprocess.run(
            ["git", "ls-files", "--", target.relative_to(root).as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise HourlyMeasurementError("cannot verify committed hourly output paths")
        if tracked.stdout:
            raise FileExistsError("Hourly output destination contains tracked artifacts")
    return target


def write_hourly_measurements_scaffold_v2(
    tables: HourlyMeasurementTables,
    *,
    output_directory: Path,
) -> HourlyMeasurementArtifactReceipt:
    """Write immutable fixture/preparation bytes, never an empirical result."""

    site, metadata = _validate_publishable_tables(tables)
    output_directory = _check_output_destination(output_directory)
    slug = site.lower()
    frames = {
        "audit": tables.audit,
        "ambient_primary": tables.ambient_primary,
        "background_sensitivity": tables.background_sensitivity,
    }
    csv_bytes = {role: _serialise_frame_csv(frame) for role, frame in frames.items()}
    csv_paths = {
        role: output_directory / f"{slug}_hourly_{role}_scaffold_v2.csv" for role in frames
    }
    artifact_records = {
        role: {
            "path": path.name,
            "row_count": len(frames[role]),
            "sha256": hashlib.sha256(csv_bytes[role]).hexdigest(),
        }
        for role, path in csv_paths.items()
    }
    manifest = {
        "analysis_scope": metadata["analysis_scope"],
        "artifacts": artifact_records,
        "configuration_evidence_status": metadata["configuration_evidence_status"],
        "manifest_type": "hourly_measurement_scaffold_bundle",
        "method_config_hash_verified": bool(metadata["method_config_hash_verified"]),
        "method_config_id": metadata["method_config_id"],
        "method_config_sha256": metadata["method_config_sha256"],
        "schema_version": 2,
        "site": site,
        "source_snapshot_id": metadata["source_snapshot_id"],
    }
    manifest_path = output_directory / f"{slug}_hourly_measurements_scaffold_v2.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    outputs = {csv_paths[role]: csv_bytes[role] for role in frames}
    outputs[manifest_path] = manifest_bytes
    _write_immutable_outputs(outputs)
    return HourlyMeasurementArtifactReceipt(
        site=site,
        analysis_scope=str(metadata["analysis_scope"]),
        output_directory=str(output_directory),
        artifact_sha256=tuple(
            (role, str(record["sha256"])) for role, record in artifact_records.items()
        ),
        row_counts=tuple((role, len(frame)) for role, frame in frames.items()),
    )


def extract_hourly_measurements(
    *,
    config: FrozenHourlyExtractionConfig,
    candidate_hours: pd.DataFrame,
    monitor_observations: pd.DataFrame,
    video_seconds: pd.DataFrame,
    background_observations: pd.DataFrame | None,
    weather_observations: pd.DataFrame | None = None,
    clip_intervals: pd.DataFrame | None = None,
    speed_samples: pd.DataFrame | None = None,
    secondary_increment_columns: dict[str, str] | None = None,
) -> HourlyMeasurementTables:
    """Align one site's candidate hours without inventing missing measurements."""

    _require_columns(candidate_hours, set(ROW_KEY), "candidate_hours")
    _require_site(candidate_hours, "site", config.site, "candidate_hours")
    if candidate_hours.empty:
        raise HourlyMeasurementError("candidate_hours cannot be empty")
    candidates = _hour_rows(candidate_hours[ROW_KEY], "candidate_hours")
    candidates = candidates.sort_values("monitor_start_utc").reset_index(drop=True)
    candidate_starts = _utc(candidates["monitor_start_utc"], "candidate_hours.monitor_start_utc")
    candidate_ends = _utc(candidates["monitor_end_utc"], "candidate_hours.monitor_end_utc")
    if (
        len(candidates) > 1
        and (candidate_starts.iloc[1:].array < candidate_ends.iloc[:-1].array).any()
    ):
        raise HourlyMeasurementError("candidate_hours contains overlapping intervals")

    monitor_columns = set(ROW_KEY) | {
        "ambient_nox_ppm",
        "monitor_label_utc",
        "target_observation_status",
    }
    _require_columns(monitor_observations, monitor_columns, "monitor_observations")
    _require_site(monitor_observations, "site", config.site, "monitor_observations")
    monitors = _hour_rows(
        monitor_observations[ROW_KEY + sorted(monitor_columns - set(ROW_KEY))],
        "monitor_observations",
    )
    _validate_monitor_labels(
        monitors,
        label_column="monitor_label_utc",
        semantics=config.monitor_label_semantics,
        name="monitor_observations",
    )
    monitors["target_observation_status"] = _observation_status(
        monitors,
        "target_observation_status",
        "monitor_observations",
    )
    monitors["ambient_nox_ppm"] = _numeric(monitors, "ambient_nox_ppm", "monitor_observations")
    if (monitors["ambient_nox_ppm"].dropna() < 0).any():
        raise HourlyMeasurementError("monitor_observations.ambient_nox_ppm cannot be negative")

    video_columns = {
        "site",
        "second_start_utc",
        "coverage_valid",
        "count_only_increment",
        "relative_nox_emission_index_increment",
    }
    _require_columns(video_seconds, video_columns, "video_seconds")
    _require_site(video_seconds, "site", config.site, "video_seconds")
    secondary_columns = secondary_increment_columns or {}
    reserved = (
        set(video_columns)
        | set(candidate_hours.columns)
        | {
            "relative_nox_emission_index",
            "count_only_index",
            "mean_speed_mps",
            "ambient_nox_ppm",
            "primary_measurement_eligible",
        }
    )
    for output, source in secondary_columns.items():
        if (
            not isinstance(output, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", output)
            or output in reserved
            or not isinstance(source, str)
            or not source
        ):
            raise HourlyMeasurementError(
                "secondary increment columns require distinct explicit names"
            )
    present_columns = video_columns | (set(secondary_columns.values()) & set(video_seconds.columns))
    seconds = video_seconds[sorted(present_columns)].copy()
    for column in set(secondary_columns.values()) - set(seconds.columns):
        seconds[column] = np.nan
    seconds["second_start"] = _utc(seconds["second_start_utc"], "video_seconds.second_start_utc")
    if seconds.duplicated(["site", "second_start_utc"]).any():
        raise HourlyMeasurementError("video_seconds contains duplicate site-second keys")
    if not seconds["coverage_valid"].map(lambda value: isinstance(value, bool | np.bool_)).all():
        raise HourlyMeasurementError("video_seconds.coverage_valid must be boolean")
    seconds["coverage_valid"] = seconds["coverage_valid"].astype(bool)
    if clip_intervals is not None:
        source_covered = recorded_second_mask(
            clip_intervals, seconds["second_start_utc"], site=config.site
        )
        if (seconds["coverage_valid"] & ~source_covered).any():
            raise HourlyMeasurementError("valid video seconds fall outside recorded clip coverage")
    for column in ("count_only_increment", "relative_nox_emission_index_increment"):
        seconds[column] = _numeric(seconds, column, "video_seconds")
        if (seconds.loc[seconds["coverage_valid"], column].dropna() < 0).any():
            raise HourlyMeasurementError(f"video_seconds.{column} cannot be negative")
    counts = seconds.loc[seconds["coverage_valid"], "count_only_increment"].dropna()
    if not np.isclose(counts, np.round(counts)).all():
        raise HourlyMeasurementError("count_only_increment must contain whole vehicle counts")
    for column in set(secondary_columns.values()):
        seconds[column] = pd.to_numeric(seconds[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    speeds = prepare_speed_samples(speed_samples, site=config.site)

    audit = candidates.merge(monitors, on=ROW_KEY, how="left", validate="one_to_one")
    offset = pd.Timedelta(hours=config.predictor_offset_hours)
    window_starts = candidate_starts - offset
    window_ends = candidate_ends - offset
    audit["predictor_window_start_utc"] = window_starts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    audit["predictor_window_end_utc"] = window_ends.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    audit["local_calendar_date"] = candidate_starts.dt.tz_convert(TOKYO_TIMEZONE).dt.strftime(
        "%Y-%m-%d"
    )

    aggregates: list[dict[str, object]] = []
    for start, end in zip(window_starts, window_ends, strict=True):
        window = seconds[(seconds["second_start"] >= start) & (seconds["second_start"] < end)]
        valid = window[window["coverage_valid"]]
        covered = len(valid)
        fraction = covered / EXPECTED_SECONDS
        ordered_valid_seconds = valid["second_start"].sort_values()
        has_internal_gap = bool(
            len(ordered_valid_seconds) > 1
            and not ordered_valid_seconds.diff().dropna().eq(pd.Timedelta(seconds=1)).all()
        )
        gap_rule_passed = config.gap_rule == "allow_internal_gaps" or not has_internal_gap
        complete = covered > 0 and valid[
            ["count_only_increment", "relative_nox_emission_index_increment"]
        ].notna().all(axis=None)
        coverage_pass = fraction >= float(config.minimum_coverage_fraction)
        raw_count = float(valid["count_only_increment"].sum()) if complete else np.nan
        raw_index = (
            float(valid["relative_nox_emission_index_increment"].sum()) if complete else np.nan
        )
        if covered == 0:
            status = "missing_video_coverage"
        elif not coverage_pass:
            status = "below_self_attested_threshold"
        elif not gap_rule_passed:
            status = "disallowed_internal_gap"
        elif covered == EXPECTED_SECONDS:
            status = "complete"
        else:
            status = "partial_above_self_attested_threshold"
        scale = (
            EXPECTED_SECONDS / covered
            if config.coverage_mode == "threshold_normalize" and covered > 0
            else 1.0
        )
        usable = coverage_pass and gap_rule_passed and complete
        aggregates.append(
            {
                "covered_seconds": covered,
                "expected_seconds": EXPECTED_SECONDS,
                "coverage_fraction": fraction,
                "coverage_status": status,
                "has_internal_coverage_gap": has_internal_gap,
                "gap_rule_passed": gap_rule_passed,
                "paired_predictors_complete": complete,
                "count_only_raw": raw_count,
                "relative_nox_emission_index_raw": raw_index,
                "count_only_index": raw_count * scale if usable else np.nan,
                "relative_nox_emission_index": raw_index * scale if usable else np.nan,
            }
        )
        aggregates[-1].update(
            aggregate_calibrated_speed(speeds, valid["second_start"], start=start, end=end)
        )
        for output, source in secondary_columns.items():
            if output in aggregates[-1]:
                raise HourlyMeasurementError(
                    "secondary predictors cannot replace hourly audit fields"
                )
            complete_optional = covered > 0 and valid[source].notna().all()
            aggregates[-1][output] = (
                float(valid[source].sum()) * scale if usable and complete_optional else np.nan
            )
            aggregates[-1][f"{output}_exclusion_reasons"] = (
                ""
                if usable and complete_optional
                else "missing_optional_increment_or_primary_coverage"
            )
    audit = pd.concat([audit, pd.DataFrame(aggregates)], axis=1)
    if clip_intervals is not None:
        clip_audit = audit_clip_hour_coverage(
            clip_intervals=clip_intervals,
            candidate_hours=candidates,
            site=config.site,
            predictor_offset_hours=config.predictor_offset_hours,
        )
        audit = audit.merge(clip_audit, on=ROW_KEY, how="left", validate="one_to_one")
    audit["clip_timing_evidence_status"] = (
        "submitted_utc_intervals_with_evidence_ids"
        if clip_intervals is not None
        else "not_provided"
    )
    audit = _align_weather(audit, config=config, weather_observations=weather_observations)
    audit["target_interpolated"] = (
        audit["target_observation_status"].eq("interpolated").fillna(False)
    )
    audit["target_is_derived"] = audit["target_observation_status"].isin(NON_ACTUAL_OBSERVATIONS)
    audit["observational_unit"] = np.where(
        audit["target_observation_status"].eq(ACTUAL_OBSERVATION).fillna(False),
        "actual_hourly_ambient_monitor_observation",
        "excluded_non_actual_or_missing_monitor_record",
    )

    primary_reasons: list[str] = []
    for row in audit.itertuples(index=False):
        reasons = []
        if pd.isna(row.ambient_nox_ppm):
            reasons.append("missing_ambient_reference")
        elif row.target_observation_status != ACTUAL_OBSERVATION:
            reasons.append(f"non_actual_ambient_reference:{row.target_observation_status}")
        if row.coverage_status == "missing_video_coverage":
            reasons.append("missing_video_coverage")
        elif row.coverage_status == "below_self_attested_threshold":
            reasons.append("insufficient_video_coverage")
        elif row.coverage_status == "disallowed_internal_gap":
            reasons.append("disallowed_internal_video_gap")
        if row.covered_seconds > 0 and not row.paired_predictors_complete:
            reasons.append("missing_paired_predictor")
        if row.weather_exclusion_reasons:
            reasons.append(row.weather_exclusion_reasons)
        primary_reasons.append(_reasons(reasons))
    audit["primary_exclusion_reasons"] = primary_reasons
    audit["primary_measurement_eligible"] = audit["primary_exclusion_reasons"] == ""

    audit["background_site"] = config.background_site
    if config.background_site is None:
        audit["background_nox_ppm"] = np.nan
        audit["background_monitor_label_utc"] = pd.NA
        audit["background_observation_status"] = pd.NA
        audit["background_exclusion_reasons"] = "not_configured"
    else:
        background_columns = {
            "reference_site",
            "monitor_start_utc",
            "monitor_end_utc",
            "monitor_label_utc",
            "background_nox_ppm",
            "background_observation_status",
        }
        if background_observations is None:
            backgrounds = pd.DataFrame(
                columns=ROW_KEY
                + [
                    "background_monitor_label_utc",
                    "background_nox_ppm",
                    "background_observation_status",
                ]
            )
        else:
            _require_columns(background_observations, background_columns, "background_observations")
            _require_site(
                background_observations,
                "reference_site",
                config.background_site,
                "background_observations",
            )
            backgrounds = background_observations[sorted(background_columns)].rename(
                columns={"reference_site": "site"}
            )
            backgrounds["site"] = config.site
            backgrounds = _hour_rows(backgrounds, "background_observations")
            _validate_monitor_labels(
                backgrounds,
                label_column="monitor_label_utc",
                semantics=config.monitor_label_semantics,
                name="background_observations",
            )
            backgrounds["background_observation_status"] = _observation_status(
                backgrounds,
                "background_observation_status",
                "background_observations",
            )
            backgrounds = backgrounds.rename(
                columns={"monitor_label_utc": "background_monitor_label_utc"}
            )
            backgrounds["background_nox_ppm"] = _numeric(
                backgrounds, "background_nox_ppm", "background_observations"
            )
            if (backgrounds["background_nox_ppm"].dropna() < 0).any():
                raise HourlyMeasurementError(
                    "background_observations.background_nox_ppm cannot be negative"
                )
        audit = audit.merge(backgrounds, on=ROW_KEY, how="left", validate="one_to_one")
        audit["background_exclusion_reasons"] = [
            _reasons(
                [
                    primary,
                    "" if pd.notna(background) else "missing_background_reference",
                    (
                        ""
                        if pd.isna(status) or status == ACTUAL_OBSERVATION
                        else f"non_actual_background_reference:{status}"
                    ),
                ]
            )
            for primary, background, status in zip(
                audit["primary_exclusion_reasons"],
                audit["background_nox_ppm"],
                audit["background_observation_status"],
                strict=True,
            )
        ]
    audit["background_differenced_nox_ppm"] = audit["ambient_nox_ppm"] - audit["background_nox_ppm"]
    audit["background_sensitivity_eligible"] = audit["background_exclusion_reasons"] == ""

    metadata = {
        "analysis_scope": config.analysis_scope,
        "primary_target": "ambient_nox_reference_observation",
        "background_sensitivity_target": "ambient_minus_background_nox_reference_observation",
        "paired_predictor_set": "count_only_and_relative_nox_index_only",
        "configuration_evidence_status": "self_attested_scaffold_only",
        "method_config_hash_verified": False,
        "method_config_id": config.method_config_id,
        "method_config_sha256": config.method_config_sha256,
        "source_snapshot_id": config.source_snapshot_id,
        "monitor_label_semantics": config.monitor_label_semantics,
        "predictor_offset_hours": config.predictor_offset_hours,
        "predictor_window_sign_convention": "positive_offset_means_predictor_precedes_receptor",
        "coverage_rule_id": config.coverage_rule_id,
        "coverage_mode": config.coverage_mode,
        "minimum_coverage_fraction": float(config.minimum_coverage_fraction),
        "partial_hour_normalisation": config.partial_hour_normalisation,
        "gap_rule": config.gap_rule,
        "precipitation_rule": config.precipitation_rule,
        "weather_alignment": config.weather_alignment,
    }
    for column, value in metadata.items():
        if column in secondary_columns:
            raise HourlyMeasurementError("secondary predictors cannot replace hourly metadata")
        audit[column] = value

    primary_columns = [
        *ROW_KEY,
        "local_calendar_date",
        "predictor_window_start_utc",
        "predictor_window_end_utc",
        "ambient_nox_ppm",
        "monitor_label_utc",
        "target_observation_status",
        "target_is_derived",
        "count_only_raw",
        "relative_nox_emission_index_raw",
        "count_only_index",
        "relative_nox_emission_index",
        "covered_seconds",
        "expected_seconds",
        "coverage_fraction",
        "coverage_status",
        "has_internal_coverage_gap",
        "gap_rule_passed",
        "paired_predictors_complete",
        "primary_measurement_eligible",
        "clip_timing_evidence_status",
        "weather_window_start_utc",
        "weather_window_end_utc",
        "weather_source_id",
        "weather_label_utc",
        "weather_observation_status",
        "precipitation_mm",
        "along_road_wind_speed",
        "wind_stratum",
        "wind_definition_id",
        "weather_exclusion_reasons",
        "precipitation_rule",
        "weather_alignment",
        "paired_predictor_set",
        "observational_unit",
        "primary_target",
        "background_sensitivity_target",
        "target_interpolated",
        "analysis_scope",
        "monitor_label_semantics",
        "predictor_offset_hours",
        "predictor_window_sign_convention",
        "coverage_mode",
        "minimum_coverage_fraction",
        "partial_hour_normalisation",
        "gap_rule",
        "method_config_id",
        "method_config_sha256",
        "configuration_evidence_status",
        "method_config_hash_verified",
        "source_snapshot_id",
        "coverage_rule_id",
    ]
    primary_columns += [
        "mean_speed_mps",
        "speed_valid_duration_seconds",
        "speed_samples_contributing",
        "speed_samples_invalid",
        "speed_exclusion_reasons",
        *secondary_columns,
        *(f"{output}_exclusion_reasons" for output in secondary_columns),
    ]
    if clip_intervals is not None:
        primary_columns += [
            "recorded_covered_seconds",
            "recorded_coverage_fraction",
            "recorded_clip_count",
            "recorded_has_internal_gap",
            "recorded_clip_ids",
            "recorded_timing_evidence_ids",
        ]
    sensitivity_columns = primary_columns + [
        "background_site",
        "background_monitor_label_utc",
        "background_observation_status",
        "background_nox_ppm",
        "background_differenced_nox_ppm",
    ]
    primary = audit.loc[audit["primary_measurement_eligible"], primary_columns].copy()
    sensitivity = audit.loc[audit["background_sensitivity_eligible"], sensitivity_columns].copy()
    return HourlyMeasurementTables(
        audit=audit.reset_index(drop=True),
        ambient_primary=primary.reset_index(drop=True),
        background_sensitivity=sensitivity.reset_index(drop=True),
    )


__all__ = [
    "AnalysisScope",
    "FrozenHourlyExtractionConfig",
    "GapRule",
    "HourlyMeasurementArtifactReceipt",
    "HourlyMeasurementError",
    "HourlyMeasurementTables",
    "PrecipitationRule",
    "WeatherAlignment",
    "extract_hourly_measurements",
    "write_hourly_measurements_scaffold_v2",
]
