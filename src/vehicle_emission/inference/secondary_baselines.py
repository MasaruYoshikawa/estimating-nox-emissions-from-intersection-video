"""Paired secondary comparisons using explicit supplied baseline definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_emission.inference.contracts import LOCKED_RETROSPECTIVE_SCOPE
from vehicle_emission.inference.hourly_measurements import _check_output_destination
from vehicle_emission.inference.paired_inference import (
    EVIDENCE_STATUS,
    ROW_KEY,
    SHARED_EXTRACTION_FIELDS,
    SITES,
    PairedInferenceConfig,
    _json_sha256,
    _observation_sha256,
    _rho,
    _validated_rows,
    inference_calendar_dates,
    joint_calendar_block_draws,
    load_hourly_bundle,
)


@dataclass(frozen=True)
class BaselineComparisonGroup:
    """One declared complete-case group; no implicit predictor definitions."""

    group_id: str
    predictor_columns: tuple[str, ...]
    definition_ids: tuple[str, ...]
    interval_family: str
    family_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictor_columns", tuple(self.predictor_columns))
        object.__setattr__(self, "definition_ids", tuple(self.definition_ids))
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("secondary comparison group_id must be explicit")
        if (
            not self.predictor_columns
            or len(self.predictor_columns) != len(self.definition_ids)
            or len(set(self.predictor_columns)) != len(self.predictor_columns)
        ):
            raise ValueError("group predictors require distinct columns and matching definitions")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (*self.predictor_columns, *self.definition_ids, self.interval_family)
        ):
            raise ValueError("predictor definitions and interval family must be explicit")
        if set(self.predictor_columns) & {"ambient_nox_ppm", "relative_nox_emission_index"}:
            raise ValueError("secondary baselines cannot redefine the target or full index")
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", column)
            or column
            in {
                *ROW_KEY,
                "local_calendar_date",
                "target_observation_status",
                "primary_measurement_eligible",
                "wind_stratum",
                "along_road_wind_speed",
            }
            for column in self.predictor_columns
        ):
            raise ValueError(
                "secondary predictor columns cannot replace timing, identity or eligibility"
            )
        if (
            isinstance(self.family_size, bool)
            or not isinstance(self.family_size, int)
            or self.family_size < len(SITES) * len(self.predictor_columns)
        ):
            raise ValueError("family_size must include both sites and every declared contrast")


def class_weighted_count_increments(
    events: pd.DataFrame, weights: dict[str, float], *, definition_id: str
) -> pd.DataFrame:
    """Calculate event counts using only supplied weights; unknown classes stay missing."""

    required = {"site", "event_id", "second_start_utc", "vehicle_class"}
    if not required.issubset(events.columns) or events.columns.duplicated().any():
        raise ValueError(f"class event rows require unique columns: {sorted(required)}")
    if not isinstance(definition_id, str) or not definition_id.strip() or not weights:
        raise ValueError("weights and their definition_id must be explicit")
    if any(
        not isinstance(label, str)
        or not label.strip()
        or isinstance(weight, bool)
        or not isinstance(weight, int | float)
        or not math.isfinite(weight)
        or weight < 0
        for label, weight in weights.items()
    ):
        raise ValueError("class weights must be finite, non-negative explicit numbers")
    if (
        events[list(required)].isna().any(axis=None)
        or events.duplicated(["site", "event_id"]).any()
        or events["event_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("entering event identities and classes must be complete and unique")
    if not set(events.site).issubset(SITES):
        raise ValueError("class event rows must identify a supported site")
    timestamps = pd.to_datetime(events.second_start_utc, utc=True, errors="raise")
    if not events.second_start_utc.eq(timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).all():
        raise ValueError("class event seconds must use canonical UTC text")
    rows = events.assign(_weight=events.vehicle_class.map(weights))
    output = []
    for (site, second), group in rows.groupby(["site", "second_start_utc"], sort=True):
        unknown = sorted(set(group.loc[group._weight.isna(), "vehicle_class"].astype(str)))
        output.append(
            {
                "site": site,
                "second_start_utc": second,
                "class_weighted_count_increment": (
                    math.nan if unknown else float(group._weight.sum())
                ),
                "weight_definition_id": definition_id,
                "weights_sha256": _json_sha256(weights),
                "missing_weight_classes": ";".join(unknown),
            }
        )
    return pd.DataFrame(
        output,
        columns=[
            "site",
            "second_start_utc",
            "class_weighted_count_increment",
            "weight_definition_id",
            "weights_sha256",
            "missing_weight_classes",
        ],
    )


def _paired_values(
    rows: pd.DataFrame, column: str, config: PairedInferenceConfig
) -> tuple[float, float, str]:
    if len(rows) < config.minimum_site_hours or rows._cluster.nunique() < config.minimum_site_days:
        return math.nan, math.nan, "insufficient_group_support"
    index = _rho(rows.relative_nox_emission_index, rows.ambient_nox_ppm)
    baseline = _rho(rows[column], rows.ambient_nox_ppm)
    reason = (
        ""
        if math.isfinite(index) and math.isfinite(baseline)
        else "constant_or_insufficient_rank_variation"
    )
    return index, baseline, reason


def compute_secondary_baseline_comparisons(
    frame: pd.DataFrame,
    inference_config: PairedInferenceConfig,
    groups: tuple[BaselineComparisonGroup, ...],
    *,
    candidate_calendar_dates: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compute explicit grouped contrasts without modifying primary support or results."""

    rows = _validated_rows(
        frame, inference_config, allow_empty=candidate_calendar_dates is not None
    )
    if not groups or len({group.group_id for group in groups}) != len(groups):
        raise ValueError("provide distinct declared secondary comparison groups")
    for group in groups:
        if group.interval_family in {
            inference_config.primary_interval_family,
            inference_config.wind_interval_family,
        }:
            raise ValueError(
                "secondary interval families must not masquerade as primary or wind families"
            )
        members = [other for other in groups if other.interval_family == group.interval_family]
        if len({member.family_size for member in members}) != 1 or group.family_size < 2 * sum(
            len(member.predictor_columns) for member in members
        ):
            raise ValueError(
                "a shared secondary family must cover all its site/predictor contrasts"
            )
    masks: dict[str, pd.Series] = {}
    exclusions: dict[tuple[str, str], dict[str, int]] = {}
    supplied_columns = set(rows.columns)
    for group in groups:
        complete = pd.Series(True, index=rows.index)
        reason_masks = {}
        for column in group.predictor_columns:
            missing_column = column not in supplied_columns
            values = (
                pd.Series(np.nan, index=rows.index)
                if missing_column
                else pd.to_numeric(rows[column], errors="coerce")
            )
            valid = values.notna() & np.isfinite(values)
            reason_masks[
                f"{'missing_column' if missing_column else 'missing_or_invalid_predictor'}:{column}"
            ] = ~valid
            rows[column] = values
            complete &= valid
        masks[group.group_id] = complete
        for site in SITES:
            exclusions[group.group_id, site] = {
                reason: int((mask & rows.site.eq(site)).sum())
                for reason, mask in reason_masks.items()
                if (mask & rows.site.eq(site)).any()
            }
    samples: dict[tuple[str, str, str], list[float]] = {
        (group.group_id, site, column): []
        for group in groups
        for site in SITES
        for column in group.predictor_columns
    }
    calendar = inference_calendar_dates(rows, candidate_calendar_dates)
    digest = hashlib.sha256(b"paired-union-calendar-moving-block-v1\0")
    digest.update(json.dumps(calendar).encode())
    digest.update(str(inference_config.block_length_days).encode())
    support_digests = {
        group.group_id: hashlib.sha256(b"secondary-paired-group-draws-v1\0") for group in groups
    }
    calendar_failure = inference_config.block_length_days > len(calendar)
    if rows.empty:
        digest.update(b"no_eligible_observations")
        for support_digest in support_digests.values():
            support_digest.update(json.dumps(calendar).encode())
            support_digest.update(b"no_eligible_observations")
    elif calendar_failure:
        digest.update(b"configured_block_exceeds_calendar")
    else:
        for selected, indices, clusters in joint_calendar_block_draws(rows, inference_config):
            digest.update(selected.astype("<i8").tobytes())
            digest.update(indices.astype("<i8").tobytes())
            for group in groups:
                keep = masks[group.group_id].iloc[indices].to_numpy()
                paired_indices = indices[keep]
                support_digests[group.group_id].update(paired_indices.astype("<i8").tobytes())
                support_digests[group.group_id].update(json.dumps(clusters[keep].tolist()).encode())
                sampled = rows.iloc[paired_indices].copy()
                sampled["_cluster"] = clusters[keep]
                for site in SITES:
                    site_rows = sampled.loc[sampled.site.eq(site)]
                    for column in group.predictor_columns:
                        index, baseline, _ = _paired_values(site_rows, column, inference_config)
                        if math.isfinite(index) and math.isfinite(baseline):
                            samples[group.group_id, site, column].append(index - baseline)
    result = []
    config_hash = _json_sha256(
        {"inference": asdict(inference_config), "groups": [asdict(group) for group in groups]}
    )
    for group in groups:
        for site in SITES:
            primary_rows = rows.loc[rows.site.eq(site)]
            paired = rows.loc[rows.site.eq(site) & masks[group.group_id]]
            observation_columns = (
                ROW_KEY
                + [
                    "local_calendar_date",
                    "ambient_nox_ppm",
                    "relative_nox_emission_index",
                    "count_only_index",
                ]
                + list(group.predictor_columns)
            )
            observation_columns = list(dict.fromkeys(observation_columns))
            observation_hash = hashlib.sha256(
                paired[observation_columns].to_csv(index=False, float_format="%.17g").encode()
            ).hexdigest()
            for column, definition in zip(
                group.predictor_columns, group.definition_ids, strict=True
            ):
                index, baseline, reason = _paired_values(paired, column, inference_config)
                drawn = samples[group.group_id, site, column]
                if exclusions[group.group_id, site] and paired.empty:
                    reason = "missing_secondary_predictor_support"
                if not reason and calendar_failure:
                    reason = "configured_block_exceeds_calendar"
                if not reason and len(drawn) < inference_config.minimum_valid_resamples:
                    reason = "insufficient_valid_resamples"
                low = high = math.nan
                if not reason:
                    alpha = (1 - inference_config.confidence_level) / group.family_size
                    low, high = (
                        float(value) for value in np.quantile(drawn, [alpha / 2, 1 - alpha / 2])
                    )
                result.append(
                    {
                        "site": site,
                        "comparison_group": group.group_id,
                        "predictor": column,
                        "definition_id": definition,
                        "estimand": "secondary_paired_association_advantage",
                        "index_rho": index,
                        "baseline_rho": baseline,
                        "estimate": index - baseline,
                        "ci_low": low,
                        "ci_high": high,
                        "n_hours": len(paired),
                        "n_days": paired.local_calendar_date.nunique(),
                        "primary_n_hours": len(primary_rows),
                        "excluded_hours": len(primary_rows) - len(paired),
                        "exclusion_reasons": json.dumps(
                            exclusions[group.group_id, site], sort_keys=True
                        ),
                        "resamples_requested": inference_config.resamples,
                        "resamples_valid": len(drawn),
                        "interval_method": "union_calendar_moving_block_percentile",
                        "interval_adjustment": inference_config.interval_adjustment,
                        "interval_family": group.interval_family,
                        "family_size": group.family_size,
                        "confidence_level": inference_config.confidence_level,
                        "classification": "not_estimable" if reason else "descriptive",
                        "analysis_scope": LOCKED_RETROSPECTIVE_SCOPE,
                        "evidence_status": EVIDENCE_STATUS,
                        "failure_reason": reason,
                        "observation_sha256": observation_hash,
                        "primary_observation_sha256": _observation_sha256(primary_rows),
                        "resample_sha256": digest.hexdigest(),
                        "paired_group_resample_sha256": support_digests[group.group_id].hexdigest(),
                        "config_sha256": config_hash,
                        "group_sha256": _json_sha256(asdict(group)),
                    }
                )
    return pd.DataFrame(result)


def main(argv: list[str] | None = None) -> int:
    """Write a new build-only secondary bundle from explicit hash-bound hourly bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-bundle", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PairedInferenceConfig(**json.loads(args.config.read_text()))
    groups = tuple(
        BaselineComparisonGroup(**item) for item in json.loads(args.groups.read_text())["groups"]
    )
    loaded = [load_hourly_bundle(path) for path in args.hourly_bundle]
    if len(loaded) != 2 or {item[1]["site"] for item in loaded} != set(SITES):
        raise ValueError("provide exactly one hourly bundle per site")
    for field in ("method_config_id", "method_config_sha256"):
        if len({item[1][field] for item in loaded}) != 1:
            raise ValueError("two-site secondary inputs must share their method identity")
    combined = pd.concat([item[0] for item in loaded], ignore_index=True)
    for field in SHARED_EXTRACTION_FIELDS:
        if combined[field].nunique(dropna=False) != 1:
            raise ValueError(f"two-site secondary inputs must share the same {field}")
    result = compute_secondary_baseline_comparisons(combined, config, groups)
    target = _check_output_destination(args.output_directory)
    csv_bytes = result.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    receipt = {
        "schema_version": 1,
        "artifact_kind": "secondary_baseline_component",
        "analysis_scope": LOCKED_RETROSPECTIVE_SCOPE,
        "evidence_status": EVIDENCE_STATUS,
        "results_csv": "secondary_baselines.csv",
        "results_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "row_count": len(result),
        "inputs": [item[1] for item in loaded],
        "config": asdict(config),
        "groups": [asdict(group) for group in groups],
        "config_sha256": _json_sha256(
            {"inference": asdict(config), "groups": [asdict(group) for group in groups]}
        ),
        "inputs_sha256": _json_sha256([item[1] for item in loaded]),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calendar_gap_policy": "consecutive_calendar_with_explicit_empty_days",
    }
    target.mkdir(parents=True, exist_ok=False)
    with (target / "secondary_baselines.csv").open("xb") as stream:
        stream.write(csv_bytes)
    with (target / "secondary_baselines_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Secondary component bundle: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
