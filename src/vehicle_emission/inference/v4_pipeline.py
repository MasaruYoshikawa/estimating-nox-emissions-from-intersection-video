"""Calculate immutable v4 bundles from explicit, hash-bound numeric exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from vehicle_emission.inference.generate_v3_figures import build_figures, load_figure_inputs
from vehicle_emission.inference.hourly_measurements import (
    FrozenHourlyExtractionConfig,
    extract_hourly_measurements,
)
from vehicle_emission.inference.manifest_validation import (
    stable_file_bytes,
    verified_clean_repository,
)
from vehicle_emission.inference.paired_inference import (
    EVIDENCE_STATUS,
    PairedInferenceConfig,
    classify_interval,
    compute_paired_inference,
)
from vehicle_emission.inference.v4_contracts import (
    EVALUATION_DESIGN,
    SCOPE,
    SITES,
    V4ContractError,
    canonical_sha256,
    exact_keys,
    require_relative_path,
    require_sha256,
    scientific_config_sha256,
    validate_v4_config,
)

MANIFEST_NAME = "analysis_manifest_v4.json"
RESULTS_NAME = "paired_inference_v4.csv"
RECEIPT_NAME = "paired_inference_receipt_v4.json"
PRIMARY_SUPPORT_COLUMNS = [
    "site",
    "monitor_start_utc",
    "monitor_end_utc",
    "local_calendar_date",
    "ambient_nox_ppm",
    "relative_nox_emission_index",
    "count_only_index",
    "primary_measurement_eligible",
    "target_observation_status",
    "predictor_window_start_utc",
    "predictor_window_end_utc",
    "method_config_sha256",
]


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, check=True, timeout=15
    )
    return result.stdout


def _argument_path(root: Path, value: Path, repository_alias: Path | None = None) -> Path:
    """Accept a repository-root alias without allowing internal symlink inputs."""
    if not value.is_absolute():
        return root / value
    aliases = [root]
    if repository_alias is not None:
        aliases.insert(0, repository_alias.absolute())
    for alias in aliases:
        if value.is_relative_to(alias):
            return root / value.relative_to(alias)
    for ancestor in value.parents:
        if ancestor.resolve() == root:
            return root / value.relative_to(ancestor)
    raise V4ContractError("path must be inside the selected repository")


def _safe_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    require_relative_path(relative, "path")
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise V4ContractError(f"symlink components are forbidden: {relative}")
    if must_exist and not current.is_file():
        raise V4ContractError(f"required regular file is unavailable: {relative}")
    if not current.resolve().is_relative_to(root):
        raise V4ContractError(f"path escapes repository: {relative}")
    return current


def _read_reference(root: Path, reference: dict[str, str], *, committed: bool = False) -> bytes:
    path = _safe_path(root, reference["path"])
    content, snapshot = stable_file_bytes(path, display_path=reference["path"])
    if snapshot.sha256 != reference["sha256"]:
        raise V4ContractError(f"input hash does not match declared bytes: {reference['path']}")
    if committed:
        try:
            stored = _git(root, "show", f"HEAD:{reference['path']}")
        except subprocess.CalledProcessError as error:
            raise V4ContractError(
                f"accepted evidence is not committed: {reference['path']}"
            ) from error
        if stored != content:
            raise V4ContractError(f"accepted evidence differs from HEAD: {reference['path']}")
    return content


def _csv_frame(content: bytes, role: str) -> pd.DataFrame:
    try:
        header = next(csv.reader(io.StringIO(content.decode("utf-8"))))
        if len(set(header)) != len(header):
            raise V4ContractError(f"{role} has duplicate CSV columns")
        return pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    except (StopIteration, UnicodeError, pd.errors.ParserError) as error:
        raise V4ContractError(f"{role} must be a valid UTF-8 CSV") from error


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()


def _primary_support_sha256(frame: pd.DataFrame) -> str:
    """Optional comparator availability cannot change primary support identity."""
    return hashlib.sha256(_csv_bytes(frame[PRIMARY_SUPPORT_COLUMNS])).hexdigest()


def _code_inventory(root: Path, *, committed: bool) -> dict[str, str]:
    inventory = {}
    for path in sorted((root / "src/vehicle_emission").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        content, snapshot = stable_file_bytes(_safe_path(root, relative), display_path=relative)
        if committed and _git(root, "show", f"HEAD:{relative}") != content:
            raise V4ContractError(f"execution code differs from HEAD: {relative}")
        inventory[relative] = snapshot.sha256
    if not inventory:
        raise V4ContractError("repository must contain the fixed execution package")
    executing_root = Path(__file__).resolve().parents[3]
    if root != executing_root:
        executing = {
            path.relative_to(executing_root)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted((executing_root / "src/vehicle_emission").rglob("*.py"))
        }
        if inventory != executing:
            raise V4ContractError(
                "repository code bytes differ from the fixed package actually executing"
            )
    return inventory


def _source_frame_checks(
    site: str, source: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> None:
    monitors = frames["monitor_observations"]
    if (
        "station_id" not in monitors
        or monitors["station_id"].isna().any()
        or not monitors["station_id"].astype(str).eq(source["monitor_station_id"]).all()
    ):
        raise V4ContractError(f"{site}: monitor observations must identify the recorded station")
    weather = frames["weather_observations"]
    if (
        "weather_source_id" not in weather
        or weather["weather_source_id"].isna().any()
        or not weather["weather_source_id"].eq(source["weather_source_id"]).all()
    ):
        raise V4ContractError(f"{site}: weather source identity differs from configuration")
    clips = frames["clip_intervals"]
    required = {"clip_id", "timing_evidence_id", "source_video_sha256", "source_video_path"}
    if not required.issubset(clips) or clips[list(required)].isna().any(axis=None):
        raise V4ContractError(
            f"{site}: clip intervals require source paths, hashes and timing evidence"
        )
    if not clips["source_video_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise V4ContractError(f"{site}: clip video hashes are invalid")
    if (
        clips["source_video_path"].astype(str).str.strip().eq("").any()
        or clips["timing_evidence_id"].astype(str).str.strip().eq("").any()
    ):
        raise V4ContractError(f"{site}: clip provenance identifiers must not be empty")
    seconds = frames["video_seconds"]
    if (
        "source_record_id" not in seconds
        or seconds["source_record_id"].isna().any()
        or seconds["source_record_id"].astype(str).str.strip().eq("").any()
    ):
        raise V4ContractError(f"{site}: each numeric feature row requires a source record identity")


def _destination(root: Path, output_directory: Path, mode: str) -> Path:
    path = output_directory if output_directory.is_absolute() else root / output_directory
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise V4ContractError("output must be within the repository") from error
    target = _safe_path(root, relative, must_exist=False)
    required_parent = "build" if mode == "preparation" else "results"
    if len(Path(relative).parts) < 2 or Path(relative).parts[0] != required_parent:
        raise V4ContractError(f"{mode} outputs require a fresh subdirectory of {required_parent}/")
    if target.exists():
        raise FileExistsError("output directory already exists; choose a new run path")
    if _git(root, "ls-files", "--", relative).strip():
        raise FileExistsError("output path contains committed artifacts, including deleted files")
    return target


def _accepted_evidence(
    root: Path, config: dict[str, Any], support: dict[str, Any]
) -> dict[str, Any]:
    references = config["evidence"]
    records = {
        role: json.loads(_read_reference(root, reference, committed=True))
        for role, reference in references.items()
    }
    method = records["method_lock"]
    exact_keys(
        method,
        {
            "schema_version",
            "record_type",
            "status",
            "scientific_config_sha256",
            "evaluation_design",
            "support",
            "interpretation",
            "design_evidence",
        },
        "method_lock",
    )
    if (
        method["schema_version"] != 4
        or method["record_type"] != "retrospective_method_lock"
        or method["status"] != "recorded"
        or method["scientific_config_sha256"] != scientific_config_sha256(config)
        or method["evaluation_design"] != EVALUATION_DESIGN
    ):
        raise V4ContractError("method lock must bind exact scientific choices and scope")
    if method["support"] != support or method["interpretation"] != "estimation_only":
        raise V4ContractError(
            "method support must match calculated eligibility, with an explicit interpretation"
        )
    from vehicle_emission.inference.v4_contracts import validate_reference

    validate_reference(method["design_evidence"], "design_evidence")
    _read_reference(root, method["design_evidence"], committed=True)
    source = records["source_review"]
    exact_keys(
        source,
        {
            "schema_version",
            "record_type",
            "status",
            "source_inventory_sha256",
            "station_role",
            "clip_timing_reviewed",
            "feature_records_traceable",
        },
        "source_review",
    )
    if (
        source["schema_version"] != 4
        or source["record_type"] != "numeric_export_source_review"
        or source["status"] != "verified"
        or source["source_inventory_sha256"] != canonical_sha256(config["sites"])
        or source["station_role"] != "spatially_separated_ambient_reference"
        or source["clip_timing_reviewed"] is not True
        or source["feature_records_traceable"] is not True
    ):
        raise V4ContractError(
            "source review must bind both source inventories and recording semantics"
        )
    measurement = records["measurement_receipt"]
    acceptance = records["measurement_acceptance"]
    _validate_measurement_evidence(root, config, measurement, acceptance)
    return {
        "references": references,
        "interpretation": method["interpretation"],
        "design_evidence": method["design_evidence"],
    }


def _validate_measurement_evidence(
    root: Path, config: dict[str, Any], measurement: dict[str, Any], acceptance: dict[str, Any]
) -> None:
    """Validate calculator receipt identities and an explicit evidence review."""
    # The calculation receipt is independently reproducible; its reviewed
    # acceptance must be recorded separately from numeric execution.
    exact_keys(
        acceptance,
        {
            "schema_version",
            "record_type",
            "status",
            "measurement_receipt_sha256",
            "scientific_config_sha256",
            "accepted_tasks",
            "calibration_ids",
            "criteria_sha256",
            "production_calibration_review",
        },
        "measurement_acceptance",
    )
    if (
        acceptance["schema_version"] != 4
        or acceptance["record_type"] != "measurement_acceptance"
        or acceptance["status"] != "accepted"
        or acceptance["measurement_receipt_sha256"]
        != config["evidence"]["measurement_receipt"]["sha256"]
        or acceptance["scientific_config_sha256"] != scientific_config_sha256(config)
    ):
        raise V4ContractError(
            "measurement acceptance must bind this receipt and scientific configuration"
        )
    if measurement.get("schema_version") != "measurement_validation_receipt_v4":
        raise V4ContractError("a calculated v4 measurement receipt is required")
    required_tasks = {"vehicle_classes", "entry_counts", "tracking", "speed"}
    if set(acceptance["accepted_tasks"]) != required_tasks:
        raise V4ContractError("primary emission comparisons require all four measurement tasks")
    from vehicle_emission.inference.v4_contracts import validate_reference
    from vehicle_emission.measurement.evaluate import validate_measurement_receipt

    validate_measurement_receipt(measurement, root)
    if set(measurement["measurement_tasks"]) != required_tasks:
        raise V4ContractError("measurement receipt must calculate all four tasks")
    identities = {value["site"]: value for value in measurement["site_identities"]}
    if set(identities) != set(SITES):
        raise V4ContractError("measurement receipt must cover both sites")
    exact_keys(acceptance["calibration_ids"], set(SITES), "calibration_ids")
    for site, value in identities.items():
        for key in ("detector_checkpoint_sha256", "tracker_config_sha256"):
            if value.get(key) != config["sites"][site][key]:
                raise V4ContractError(f"{site} measurement {key} does not match analysis")
        calculated_ids = {segment["calibration_id"] for segment in value["calibration_segments"]}
        if not calculated_ids or set(acceptance["calibration_ids"][site]) != calculated_ids:
            raise V4ContractError("accepted calibration IDs must match calculated references")
    assessed = measurement["acceptance"]
    if assessed["status"] != "passed" or acceptance["criteria_sha256"] != canonical_sha256(
        assessed["criteria"]
    ):
        raise V4ContractError(
            "measurement criteria must be explicit and passed, not merely present"
        )
    for site_result in assessed["site_results"]:
        if (
            site_result["status"] != "passed"
            or set(site_result["checks"]) != required_tasks
            or not all(value is True for value in site_result["checks"].values())
        ):
            raise V4ContractError("all source-linked measurement tasks must pass at both sites")
    reference = acceptance["production_calibration_review"]
    validate_reference(reference, "production_calibration_review")
    production = json.loads(_read_reference(root, reference, committed=True))
    exact_keys(
        production,
        {
            "schema_version",
            "record_type",
            "status",
            "video_seconds_sha256",
            "calibration_ids",
            "production_use_verified",
            "speed_samples_sha256",
        },
        "production_calibration_review",
    )
    expected_exports = {
        site: config["sites"][site]["sources"]["video_seconds"]["sha256"] for site in SITES
    }
    if (
        production["schema_version"] != 4
        or production["record_type"] != "production_calibration_review"
        or production["status"] != "verified"
        or production["production_use_verified"] is not True
        or production["video_seconds_sha256"] != expected_exports
        or production["calibration_ids"] != acceptance["calibration_ids"]
    ):
        raise V4ContractError(
            "speed references do not establish calibration use in these production exports"
        )
    exact_keys(production["speed_samples_sha256"], set(SITES), "speed_samples_sha256")


def validate_v4_result_bundle(
    directory: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
    calculated_primary: pd.DataFrame | None = None,
    calculated_secondary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Mandatory postcondition: verify every byte and the calculated row contract."""
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_type",
            "analysis_id",
            "analysis_scope",
            "mode",
            "evidence_status",
            "generated_at_utc",
            "configuration",
            "scientific_config_sha256",
            "code_commit",
            "code_sha256",
            "inputs",
            "evidence",
            "support",
            "artifacts",
            "result_rows",
            "secondary_row_count",
        },
        "result manifest",
    )
    if (
        manifest["schema_version"] != 4
        or manifest["manifest_type"] != "calculated_analysis_bundle_v4"
        or manifest["analysis_scope"] != SCOPE
        or manifest["mode"] not in {"preparation", "accepted_source"}
    ):
        raise V4ContractError("output must satisfy the v4 retrospective result contract")
    expected_status = (
        "provisional_components_only"
        if manifest["mode"] == "preparation"
        else "source_reproduced_unreviewed"
    )
    if manifest["evidence_status"] != expected_status:
        raise V4ContractError("result evidence status exceeds its execution mode")
    if expected_manifest is not None and manifest != expected_manifest:
        raise V4ContractError("serialized output manifest differs from calculated postconditions")
    names = set()
    for artifact in manifest["artifacts"]:
        exact_keys(artifact, {"path", "sha256", "size_bytes"}, "artifact")
        name = artifact["path"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in names
            or "_v4." not in name
        ):
            raise V4ContractError("v4 artifacts require unique local versioned filenames")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise V4ContractError("result artifact is absent or symlinked")
        content = path.read_bytes()
        if (
            len(content) != artifact["size_bytes"]
            or hashlib.sha256(content).hexdigest() != artifact["sha256"]
        ):
            raise V4ContractError("result artifact does not match its calculated manifest")
        names.add(name)
    if {path.name for path in directory.iterdir()} != names | {MANIFEST_NAME}:
        raise V4ContractError("result bundle has missing or unmanifested files")
    configuration_bytes = (directory / "analysis_config_v4.json").read_bytes()
    configuration = validate_v4_config(json.loads(configuration_bytes))
    if (
        manifest["configuration"]["sha256"] != hashlib.sha256(configuration_bytes).hexdigest()
        or manifest["scientific_config_sha256"] != scientific_config_sha256(configuration)
        or manifest["analysis_id"] != configuration["analysis_id"]
        or manifest["mode"] != configuration["mode"]
    ):
        raise V4ContractError("result manifest must bind the exact recorded configuration")
    expected_inputs = [
        {"site": site, "role": role, **reference}
        for site in SITES
        for role, reference in configuration["sites"][site]["sources"].items()
        if reference is not None
    ]
    if (
        manifest["inputs"] != expected_inputs
        or manifest["evidence"].get("references") != configuration["evidence"]
    ):
        raise V4ContractError("result inputs and evidence must match configured byte identities")
    expected_interpretation = (
        "preparation_only" if manifest["mode"] == "preparation" else "estimation_only"
    )
    if manifest["evidence"].get("interpretation") != expected_interpretation:
        raise V4ContractError("this version does not establish decision-quality precision")
    if (
        not isinstance(manifest["code_commit"], str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", manifest["code_commit"]) is None
        or not isinstance(manifest["code_sha256"], dict)
        or not manifest["code_sha256"]
    ):
        raise V4ContractError("result execution requires a code revision and byte inventory")
    for path, digest in manifest["code_sha256"].items():
        require_relative_path(path, "code path")
        require_sha256(digest, "code SHA-256")
    receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    if (
        receipt["config"] != configuration["inference"]
        or receipt["inputs"] != manifest["inputs"]
        or receipt["code_revision"] != manifest["code_commit"]
        or receipt["code_sha256"] != canonical_sha256(manifest["code_sha256"])
    ):
        raise V4ContractError("calculation receipt disagrees with recorded method and source bytes")
    expected_names = {
        "analysis_config_v4.json",
        RESULTS_NAME,
        RECEIPT_NAME,
        "fig_rq1_correlations_v4.png",
        "fig_rq3_paired_advantage_v4.png",
        "fig_rq2_wind_contrast_v4.png",
        "fig_cross_site_heterogeneity_v4.png",
    }
    hourly_frames = []
    candidate_dates = []
    for site in SITES:
        expected_names |= {
            f"{site}_{role}_v4.csv"
            for role in ("hourly_audit", "ambient_primary", "background_sensitivity")
        }
        eligible_path = directory / f"{site}_ambient_primary_v4.csv"
        eligible = pd.read_csv(eligible_path, float_precision="round_trip")
        hourly_frames.append(eligible)
        audit = pd.read_csv(directory / f"{site}_hourly_audit_v4.csv", float_precision="round_trip")
        candidate_dates.extend(audit["local_calendar_date"].tolist())
        support = {
            "candidate_hours": len(audit),
            "eligible_hours": len(eligible),
            "eligible_days": int(eligible["local_calendar_date"].nunique()),
            "eligible_sha256": _primary_support_sha256(eligible),
            "excluded_hours": len(audit) - len(eligible),
        }
        if manifest["support"][site] != support:
            raise V4ContractError("manifest support differs from calculated hourly artifacts")
        included = audit.loc[
            audit["primary_measurement_eligible"].eq(True),
            ["site", "monitor_start_utc", "monitor_end_utc"],
        ].reset_index(drop=True)
        if not included.equals(
            eligible[["site", "monitor_start_utc", "monitor_end_utc"]].reset_index(drop=True)
        ):
            raise V4ContractError(
                "eligible projection must include all and only audited eligible hours"
            )
    if configuration["baselines"]["groups"]:
        expected_names.add("secondary_baselines_v4.csv")
    if names != expected_names:
        raise V4ContractError(
            "v4 bundle must contain its full calculation and four figure artifacts"
        )
    for name in names:
        if name.endswith(".png") and not (directory / name).read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise V4ContractError("figure artifact is not a rendered PNG")
    rows = load_figure_inputs(directory / RESULTS_NAME, directory / RECEIPT_NAME)
    if len(rows) != 9 or len(manifest["result_rows"]) != 9:
        raise V4ContractError("primary v4 results must retain all nine site/contrast rows")
    frame = pd.read_csv(directory / RESULTS_NAME, float_precision="round_trip")
    serialized = json.loads(frame.to_json(orient="records", double_precision=15))
    if serialized != manifest["result_rows"]:
        raise V4ContractError("manifest result rows disagree with calculated CSV")
    for row in frame.itertuples():
        expected_evaluation = (
            "not_estimable"
            if row.classification == "not_estimable"
            else "descriptive" if row.site == "cross_site" else "exploratory"
        )
        if row.evaluation_status != expected_evaluation or row.classification != classify_interval(
            row.estimand, row.ci_low, row.ci_high
        ):
            raise V4ContractError(
                "row decisions or evaluation status exceed the calculated intervals"
            )
    for row in rows:
        if (
            row.site in SITES
            and row.estimand != "rq2_wind_contrast"
            and (
                row.n_hours != manifest["support"][row.site]["eligible_hours"]
                or row.n_days != manifest["support"][row.site]["eligible_days"]
            )
        ):
            raise V4ContractError("row support differs from audited eligible projection")
    secondary = directory / "secondary_baselines_v4.csv"
    if (len(pd.read_csv(secondary)) if secondary.is_file() else 0) != manifest[
        "secondary_row_count"
    ]:
        raise V4ContractError("secondary output population disagrees with manifest")
    if calculated_primary is None:
        paired_config = PairedInferenceConfig(**configuration["inference"])
        combined = pd.concat(hourly_frames, ignore_index=True)
        calculated_primary = compute_paired_inference(
            combined, paired_config, candidate_calendar_dates=tuple(candidate_dates)
        )
        calculated_primary["evaluation_status"] = [
            (
                "not_estimable"
                if row.classification == "not_estimable"
                else "descriptive" if row.site == "cross_site" else "exploratory"
            )
            for row in calculated_primary.itertuples()
        ]
        if configuration["baselines"]["groups"]:
            from vehicle_emission.inference.secondary_baselines import (
                BaselineComparisonGroup,
                compute_secondary_baseline_comparisons,
            )

            groups = tuple(
                BaselineComparisonGroup(
                    group_id=value["group_id"],
                    predictor_columns=tuple(value["predictor_columns"]),
                    definition_ids=tuple(value["definition_ids"]),
                    interval_family=value["interval_family"],
                    family_size=value["family_size"],
                )
                for value in configuration["baselines"]["groups"]
            )
            calculated_secondary = compute_secondary_baseline_comparisons(
                combined, paired_config, groups, candidate_calendar_dates=tuple(candidate_dates)
            )
    if _csv_bytes(calculated_primary) != (directory / RESULTS_NAME).read_bytes():
        raise V4ContractError(
            "primary output does not reproduce from the bundled eligible observations"
        )
    if (
        calculated_secondary is not None
        and _csv_bytes(calculated_secondary) != secondary.read_bytes()
    ):
        raise V4ContractError("secondary output does not reproduce paired comparisons and draws")
    return manifest


def run_v4_analysis(config_path: Path, *, repository_root: Path, output_directory: Path) -> Path:
    """Run fixed calculations and publish only a fully validated fresh bundle."""
    root = repository_root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise V4ContractError("repository_root must name the Git root")
    path = _argument_path(root, config_path, repository_root)
    relative = path.relative_to(root).as_posix()
    content, config_snapshot = stable_file_bytes(_safe_path(root, relative), display_path=relative)
    config = validate_v4_config(json.loads(content))
    accepted = config["mode"] == "accepted_source"
    if accepted:
        verified_clean_repository(root)
        _read_reference(root, {"path": relative, "sha256": config_snapshot.sha256}, committed=True)
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    code = _code_inventory(root, committed=accepted)
    scientific_hash = scientific_config_sha256(config)
    target = _destination(
        root, _argument_path(root, output_directory, repository_root), config["mode"]
    )
    inputs: list[dict[str, Any]] = []
    hourly: dict[str, Any] = {}
    support: dict[str, Any] = {}
    for site in SITES:
        source = config["sites"][site]
        frames = {}
        for role, reference in source["sources"].items():
            if reference is None:
                continue
            frames[role] = _csv_frame(_read_reference(root, reference), f"{site}.{role}")
            inputs.append({"site": site, "role": role, **reference})
        _source_frame_checks(site, source, frames)
        if accepted and "speed_samples" in frames:
            acceptance_record = json.loads(
                _read_reference(root, config["evidence"]["measurement_acceptance"], committed=True)
            )
            production_record = json.loads(
                _read_reference(
                    root, acceptance_record["production_calibration_review"], committed=True
                )
            )
            samples = frames["speed_samples"]
            validated_ids = acceptance_record["calibration_ids"].get(site, [])
            reviewed_hash = production_record.get("speed_samples_sha256", {}).get(site)
            if reviewed_hash != source["sources"]["speed_samples"]["sha256"]:
                samples["calibration_valid"] = False
            elif "calibration_id" in samples:
                if "calibration_valid" not in samples:
                    samples["calibration_valid"] = False
                else:
                    samples["calibration_valid"] = samples["calibration_valid"].eq(True) & samples[
                        "calibration_id"
                    ].isin(validated_ids)
        if config["inference"]["wind_mode"] == "recorded_strata":
            weather = frames["weather_observations"]
            if "wind_stratum" in weather:
                if "wind_definition_id" not in weather:
                    weather["wind_stratum"] = pd.NA
                else:
                    defined = (
                        weather["wind_stratum"].notna() & weather["wind_definition_id"].notna()
                    )
                    if (
                        not weather.loc[defined, "wind_definition_id"]
                        .eq(config["inference"]["wind_definition_id"])
                        .all()
                    ):
                        raise V4ContractError(
                            "recorded wind definition differs from configured strata"
                        )
                    weather.loc[weather["wind_definition_id"].isna(), "wind_stratum"] = pd.NA
        extraction = FrozenHourlyExtractionConfig(
            method_config_id=config["analysis_id"],
            method_config_sha256=scientific_hash,
            source_snapshot_id=source["source_snapshot_id"],
            status="self_attested_scaffold_only",
            analysis_scope=SCOPE,
            site=site,
            background_site=source["background_site"],
            **config["hourly_policy"],
        )
        hourly[site] = extract_hourly_measurements(
            config=extraction,
            candidate_hours=frames["candidate_hours"],
            monitor_observations=frames["monitor_observations"],
            video_seconds=frames["video_seconds"],
            clip_intervals=frames["clip_intervals"],
            weather_observations=frames["weather_observations"],
            background_observations=frames.get("background_observations"),
            speed_samples=frames.get("speed_samples"),
            secondary_increment_columns=config["baselines"]["secondary_increment_columns"],
        )
        table = hourly[site]
        support[site] = {
            "candidate_hours": len(table.audit),
            "eligible_hours": len(table.ambient_primary),
            "eligible_days": int(table.ambient_primary["local_calendar_date"].nunique()),
            "eligible_sha256": _primary_support_sha256(table.ambient_primary),
            "excluded_hours": len(table.audit) - len(table.ambient_primary),
        }
    evidence = (
        _accepted_evidence(root, config, support)
        if accepted
        else {"references": config["evidence"], "interpretation": "preparation_only"}
    )
    paired_config = PairedInferenceConfig(**config["inference"])
    combined = pd.concat([hourly[site].ambient_primary for site in SITES], ignore_index=True)
    candidate_dates = tuple(
        date for site in SITES for date in hourly[site].audit["local_calendar_date"]
    )
    results = compute_paired_inference(
        combined, paired_config, candidate_calendar_dates=candidate_dates
    )
    results["evaluation_status"] = [
        (
            "not_estimable"
            if row.classification == "not_estimable"
            else "descriptive" if row.site == "cross_site" else "exploratory"
        )
        for row in results.itertuples()
    ]
    secondary = None
    if config["baselines"]["groups"]:
        from vehicle_emission.inference.secondary_baselines import (
            BaselineComparisonGroup,
            compute_secondary_baseline_comparisons,
        )

        groups = tuple(
            BaselineComparisonGroup(
                group_id=value["group_id"],
                predictor_columns=tuple(value["predictor_columns"]),
                definition_ids=tuple(value["definition_ids"]),
                interval_family=value["interval_family"],
                family_size=value["family_size"],
            )
            for value in config["baselines"]["groups"]
        )
        secondary = compute_secondary_baseline_comparisons(
            combined, paired_config, groups, candidate_calendar_dates=candidate_dates
        )
    build = _safe_path(root, "build", must_exist=False)
    build.mkdir(exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish-lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("another writer owns this output path") from error
    stage = Path(tempfile.mkdtemp(prefix="v4-stage-", dir=build))
    try:
        os.close(descriptor)
        (stage / "analysis_config_v4.json").write_bytes(content)
        for site, tables in hourly.items():
            for role, frame in (
                ("hourly_audit", tables.audit),
                ("ambient_primary", tables.ambient_primary),
                ("background_sensitivity", tables.background_sensitivity),
            ):
                (stage / f"{site}_{role}_v4.csv").write_bytes(_csv_bytes(frame))
        result_bytes = _csv_bytes(results)
        (stage / RESULTS_NAME).write_bytes(result_bytes)
        receipt = {
            "schema_version": 1,
            "artifact_kind": "paired_inference_component",
            "results_csv": RESULTS_NAME,
            "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "row_count": len(results),
            "analysis_scope": SCOPE,
            "evidence_status": EVIDENCE_STATUS,
            "input_sha256": canonical_sha256(inputs),
            "inputs": inputs,
            "config_sha256": canonical_sha256(config["inference"]),
            "config": config["inference"],
            "code_revision": revision,
            "code_sha256": canonical_sha256(code),
            "calendar_gap_policy": "consecutive_calendar_with_explicit_empty_days",
            "observational_unit": "actual_hourly_ambient_monitor_observation",
            "resampling_unit": "joint_calendar_day_blocks_preserving_intra_day_rows",
        }
        (stage / RECEIPT_NAME).write_bytes(_json_bytes(receipt))
        if secondary is not None:
            (stage / "secondary_baselines_v4.csv").write_bytes(_csv_bytes(secondary))
        figures = build_figures(load_figure_inputs(stage / RESULTS_NAME, stage / RECEIPT_NAME))
        try:
            for name, figure in figures.items():
                output_name = name.replace("_components_v1.png", "_v4.png")
                figure.savefig(stage / output_name, format="png", dpi=180)
        finally:
            for figure in figures.values():
                plt.close(figure)
        artifacts = [
            {
                "path": item.name,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                "size_bytes": item.stat().st_size,
            }
            for item in sorted(stage.iterdir())
        ]
        manifest = {
            "schema_version": 4,
            "manifest_type": "calculated_analysis_bundle_v4",
            "analysis_id": config["analysis_id"],
            "analysis_scope": SCOPE,
            "mode": config["mode"],
            "evidence_status": "source_reproduced_unreviewed" if accepted else EVIDENCE_STATUS,
            "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "configuration": {"path": relative, "sha256": config_snapshot.sha256},
            "scientific_config_sha256": scientific_hash,
            "code_commit": revision,
            "code_sha256": code,
            "inputs": inputs,
            "evidence": evidence,
            "support": support,
            "artifacts": artifacts,
            "result_rows": json.loads(
                pd.read_csv(io.BytesIO(result_bytes), float_precision="round_trip").to_json(
                    orient="records", double_precision=15
                )
            ),
            "secondary_row_count": len(secondary) if secondary is not None else 0,
        }
        (stage / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        validate_v4_result_bundle(
            stage,
            expected_manifest=manifest,
            calculated_primary=results,
            calculated_secondary=secondary,
        )
        for item in inputs:
            _read_reference(root, {"path": item["path"], "sha256": item["sha256"]})
        _read_reference(
            root, {"path": relative, "sha256": config_snapshot.sha256}, committed=accepted
        )
        if accepted:
            if _accepted_evidence(root, config, support) != evidence:
                raise V4ContractError("recorded evidence changed during execution")
        else:
            for reference in config["evidence"].values():
                if reference is not None:
                    _read_reference(root, reference)
        if _git(root, "rev-parse", "HEAD").decode().strip() != revision:
            raise V4ContractError("repository revision changed during execution")
        if _code_inventory(root, committed=accepted) != code:
            raise V4ContractError("calculation code changed during execution")
        _destination(root, target, config["mode"])
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)
    return target / MANIFEST_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args(argv)
    try:
        path = run_v4_analysis(
            arguments.config,
            repository_root=arguments.repository_root,
            output_directory=arguments.output_directory,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"v4 analysis failed: {error}\n")
    print(f"Calculated v4 bundle: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
