"""Versioned contracts for calculated, retrospective numeric-export analyses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any, Literal

from vehicle_emission.inference.hourly_measurements import FrozenHourlyExtractionConfig
from vehicle_emission.inference.paired_inference import PairedInferenceConfig

SITES = ("Sukiyabashi", "Shinjuku")
SCOPE: Literal["locked_retrospective_two_site"] = "locked_retrospective_two_site"
THRESHOLDS = {"rq1_strong": 0.4, "rq2_moderation": 0.15, "rq3_advantage": 0.10}
SOURCE_ROLES = (
    "candidate_hours",
    "monitor_observations",
    "video_seconds",
    "clip_intervals",
    "weather_observations",
)
OPTIONAL_SOURCE_ROLES = ("background_observations", "speed_samples")
EVALUATION_DESIGN = {
    "shinjuku_outcome_accessed_before_retrospective_lock": True,
    "method_revised_after_outcome_access": True,
    "no_further_outcome_guided_tuning_after_lock": True,
    "untouched_replication_claim_permitted": False,
}


class V4ContractError(ValueError):
    """A versioned configuration or calculated bundle is inconsistent."""


def canonical_sha256(value: object) -> str:
    """Hash an unambiguous finite JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def exact_keys(value: Any, expected: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise V4ContractError(f"{context} requires exactly {sorted(expected)}")


def require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V4ContractError(f"{context} must be explicit non-empty text")
    return value


def require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise V4ContractError(f"{context} must be a lowercase SHA-256")
    return value


def require_relative_path(value: Any, context: str) -> str:
    text = require_text(value, context)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or any(part in {".", ".."} for part in text.split("/"))
        or path.as_posix() != text
    ):
        raise V4ContractError(f"{context} must be a canonical repository-relative path")
    return text


def validate_reference(value: Any, context: str) -> None:
    exact_keys(value, {"path", "sha256"}, context)
    require_relative_path(value["path"], f"{context}.path")
    require_sha256(value["sha256"], f"{context}.sha256")


def scientific_config_sha256(config: dict[str, Any]) -> str:
    """Bind scientific choices and source bytes without cyclic evidence hashes."""
    return canonical_sha256(
        {key: value for key, value in config.items() if key not in {"evidence", "mode"}}
    )


def validate_v4_config(config: Any) -> dict[str, Any]:
    """Reject unsupported choices; no implicit defaults select a method."""
    exact_keys(
        config,
        {
            "schema_version",
            "manifest_type",
            "analysis_id",
            "mode",
            "analysis_scope",
            "evaluation_design",
            "thresholds",
            "hourly_policy",
            "inference",
            "sites",
            "baselines",
            "evidence",
        },
        "v4 config",
    )
    if (
        type(config["schema_version"]) is not int
        or config["schema_version"] != 4
        or config["manifest_type"] != "numeric_export_analysis_v4"
    ):
        raise V4ContractError("configuration must use the v4 numeric-export contract")
    if (
        config["mode"] not in {"preparation", "accepted_source"}
        or config["analysis_scope"] != SCOPE
    ):
        raise V4ContractError(
            "only preparation or accepted_source retrospective execution is supported"
        )
    require_text(config["analysis_id"], "analysis_id")
    if config["evaluation_design"] != EVALUATION_DESIGN or any(
        type(value) is not bool for value in config["evaluation_design"].values()
    ):
        raise V4ContractError("the prior outcome access and retrospective scope must be disclosed")
    if config["thresholds"] != THRESHOLDS or any(
        type(value) not in {int, float} for value in config["thresholds"].values()
    ):
        raise V4ContractError("literal ADR-024 thresholds cannot be changed")
    shared_fields = {field.name for field in fields(FrozenHourlyExtractionConfig)} - {
        "method_config_id",
        "method_config_sha256",
        "source_snapshot_id",
        "status",
        "analysis_scope",
        "site",
        "background_site",
    }
    exact_keys(config["hourly_policy"], shared_fields, "hourly_policy")
    if config["hourly_policy"]["precipitation_rule"] != "dry_only":
        raise V4ContractError("v4 primary execution requires an explicit actual dry-weather rule")
    exact_keys(
        config["inference"], {field.name for field in fields(PairedInferenceConfig)}, "inference"
    )
    PairedInferenceConfig(**config["inference"])
    exact_keys(config["sites"], set(SITES), "sites")
    for site, value in config["sites"].items():
        exact_keys(
            value,
            {
                "source_snapshot_id",
                "monitor_station_id",
                "weather_source_id",
                "background_site",
                "detector_checkpoint_sha256",
                "tracker_config_sha256",
                "estimator_definition_sha256",
                "sources",
            },
            site,
        )
        for key in ("source_snapshot_id", "monitor_station_id", "weather_source_id"):
            require_text(value[key], f"{site}.{key}")
        for key in (
            "detector_checkpoint_sha256",
            "tracker_config_sha256",
            "estimator_definition_sha256",
        ):
            require_sha256(value[key], f"{site}.{key}")
        exact_keys(value["sources"], set(SOURCE_ROLES + OPTIONAL_SOURCE_ROLES), f"{site}.sources")
        for role in SOURCE_ROLES + OPTIONAL_SOURCE_ROLES:
            reference = value["sources"][role]
            if reference is None and role in OPTIONAL_SOURCE_ROLES:
                continue
            validate_reference(reference, f"{site}.{role}")
        FrozenHourlyExtractionConfig(
            method_config_id=config["analysis_id"],
            method_config_sha256=scientific_config_sha256(config),
            source_snapshot_id=value["source_snapshot_id"],
            status="self_attested_scaffold_only",
            analysis_scope=SCOPE,
            site=site,
            background_site=value["background_site"],
            **config["hourly_policy"],
        )
    for key in (
        "detector_checkpoint_sha256",
        "tracker_config_sha256",
        "estimator_definition_sha256",
    ):
        if len({config["sites"][site][key] for site in SITES}) != 1:
            raise V4ContractError(f"both sites must share the same {key}")
    exact_keys(config["baselines"], {"secondary_increment_columns", "groups"}, "baselines")
    increments = config["baselines"]["secondary_increment_columns"]
    if not isinstance(increments, dict):
        raise V4ContractError("secondary_increment_columns must be an explicit mapping")
    for output, input_column in increments.items():
        require_text(output, "secondary output column")
        require_text(input_column, "secondary input column")
        if output in {
            "ambient_nox_ppm",
            "count_only_index",
            "relative_nox_emission_index",
            "mean_speed_mps",
        }:
            raise V4ContractError("secondary predictors cannot replace primary columns")
    groups = config["baselines"]["groups"]
    if not isinstance(groups, list):
        raise V4ContractError("baseline groups must be an explicit list")
    identities: set[str] = set()
    for group in groups:
        exact_keys(
            group,
            {"group_id", "predictor_columns", "definition_ids", "interval_family", "family_size"},
            "baseline group",
        )
        identity = require_text(group["group_id"], "group_id")
        if identity in identities:
            raise V4ContractError("baseline group IDs must be unique")
        identities.add(identity)
        require_text(group["interval_family"], "baseline interval_family")
        predictors, definitions = group["predictor_columns"], group["definition_ids"]
        if (
            not isinstance(predictors, list)
            or not predictors
            or len(set(predictors)) != len(predictors)
            or not isinstance(definitions, list)
            or len(definitions) != len(predictors)
        ):
            raise V4ContractError("baseline columns and definition IDs must be paired unique lists")
        for predictor, definition in zip(predictors, definitions, strict=True):
            require_text(definition, "baseline definition")
            if predictor not in {"count_only_index", "mean_speed_mps", *increments}:
                raise V4ContractError(
                    f"baseline column is not generated by this pipeline: {predictor}"
                )
        if type(group["family_size"]) is not int or group["family_size"] < 2 * len(predictors):
            raise V4ContractError(
                "baseline family size must include both sites and every comparator"
            )
        if group["interval_family"] in {
            config["inference"]["primary_interval_family"],
            config["inference"]["wind_interval_family"],
        }:
            raise V4ContractError("secondary families cannot silently reuse primary/wind families")
    evidence = config["evidence"]
    exact_keys(
        evidence,
        {"method_lock", "source_review", "measurement_receipt", "measurement_acceptance"},
        "evidence",
    )
    for role, reference in evidence.items():
        if reference is None and config["mode"] == "preparation":
            continue
        validate_reference(reference, role)
    return config
