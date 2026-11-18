"""Fail-closed manifest validation for the frozen Phase 5 analysis.

The accepted method and the formal terms used here live in the coordination
records.  This module validates recorded bytes, schema shape, and internal
consistency.  It cannot establish that a human approval is scientifically
correct, and it never treats a manifest's self-assertion as proof of approval.

All lightweight files that authorise an empirical run must be committed at
the current clean ``HEAD`` and must match their recorded SHA-256 digests.  The
empirical database is the sole out-of-band exception; callers must not hash it
until every lightweight gate has passed.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from vehicle_emission.inference.contracts import (
    EXPLORATORY_DECISION_CATEGORIES,
    LOCKED_RETROSPECTIVE_SCOPE,
    MONITOR_LABEL_SEMANTICS,
    RESULT_DECISION_CATEGORIES,
    VIDEO_COVERAGE_GAP_RULES,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

SITES = ("Sukiyabashi", "Shinjuku")
REQUIRED_DECISIONS = ("Q-012", "Q-016")
REQUIRED_MANIFEST_KINDS = (
    "method_config",
    "eligibility",
    "coverage",
    "model",
    "calibration",
    "station",
    "provenance",
)
ALLOWED_AUTHORISING_PREFIXES = ("coordination/", "data/manifests/", "data/calibration/")
ALLOWED_EVIDENCE_PREFIXES = (
    "coordination/",
    "data/manifests/",
    "data/calibration/",
    "data/models/",
)
EXECUTABLE_SOURCE_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dylib"}
DYNAMIC_CODE_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "import_module",
    "run_module",
    "run_path",
    "spec_from_file_location",
    "SourceFileLoader",
    "SourcelessFileLoader",
    "ExtensionFileLoader",
}
FORBIDDEN_INFERENCE_MODULES = {
    "vehicle_emission.inference.run_analysis",
    "vehicle_emission.inference.plot_lag",
    "vehicle_emission.inference.correlation",
}
METHOD_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "status",
        "manifest_id",
        "sites",
        "analysis_scope",
        "observational_unit",
        "primary_target",
        "target_interpolation",
        "monitor_interval",
        "predictor_window",
        "eligibility",
        "resampling",
        "multiplicity",
        "equivalence",
        "evaluation_design",
        "wind_strata",
        "support_rules",
        "cross_site_resampling",
        "predictor_aggregation",
        "engine",
    }
)
EVALUATION_DESIGN_FIELDS = frozenset(
    {
        "shinjuku_outcome_accessed_before_retrospective_lock",
        "method_revised_after_outcome_access",
        "no_further_outcome_guided_tuning_after_lock",
        "untouched_replication_claim_permitted",
    }
)
ELIGIBILITY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "status",
        "manifest_id",
        "sites",
        "prior_outcome_access_disclosed",
        "method_revised_after_outcome_access",
        "no_further_outcome_guided_tuning_after_lock",
        "outcome_neutral_claim_permitted",
        "row_key_fields",
        "rules_reference",
        "candidate_periods_reference",
    }
)
COVERAGE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "status",
        "manifest_id",
        "sites",
        "shinjuku_outcome_accessed_before_retrospective_lock",
        "coverage_source_reference",
        "site_summaries",
    }
)


class ManifestValidationError(ValueError):
    """Raised when a frozen-analysis contract fails closed."""


@dataclass(frozen=True)
class FileSnapshot:
    """Content and stat identity captured from one stable file descriptor."""

    path: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class LightweightValidationReceipt:
    """Receipt emitted before the empirical database may be read."""

    repository_root: str
    head_commit: str
    analysis_code_commit: str
    freeze_id: str
    freeze_snapshot: FileSnapshot
    decision_snapshots: tuple[FileSnapshot, ...]
    manifest_snapshots: tuple[FileSnapshot, ...]
    evidence_snapshots: tuple[FileSnapshot, ...]
    method_config: Mapping[str, Any]
    coverage_manifest: Mapping[str, Any]
    source_database_path: str
    source_database_sha256: str


@dataclass(frozen=True)
class ResultValidationContext:
    """Accepted, non-empirical constraints that a result must reproduce."""

    freeze_id: str
    freeze_sha256: str
    head_commit: str
    analysis_scope: str
    source_database_sha256: str
    manifest_sha256_by_kind: tuple[tuple[str, str], ...]
    interval_method: str
    resamples_requested: int
    minimum_valid_resamples: int
    minimum_clusters: int
    rq2_minimum_hours_per_stratum: int
    rq2_minimum_days_per_stratum: int
    adjusted_confidence_level: float
    rq2_adjusted_confidence_level: float
    equivalence_enabled: bool
    equivalence_lower_margin: float | None
    equivalence_upper_margin: float | None
    equivalence_confidence_level: float | None
    support_rules: tuple[tuple[str, int, int, float, str], ...]
    coverage_by_site: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ResultValidationReceipt:
    """Machine-readable summary of a validated result-manifest contract."""

    analysis_id: str
    artifact_path: str
    artifact_sha256: str
    row_count: int
    rq_site_keys: tuple[str, ...]


RESULT_ROW_FIELDS = (
    "rq",
    "site",
    "estimand",
    "evidence_status",
    "data_sufficiency_status",
    "not_estimable_reason",
    "estimate",
    "ci_lower",
    "ci_upper",
    "adjusted_confidence_level",
    "decision_category",
    "target_observation_count",
    "eligible_day_count",
    "effective_sample_size",
    "cluster_count",
    "component_observation_key_sha256",
    "cross_site_resample_index_sha256",
    "favourable_observation_count",
    "opposing_observation_count",
    "favourable_day_count",
    "opposing_day_count",
    "interval_method_requested",
    "interval_method_used",
    "fallback_used",
    "fallback_reason",
    "resamples_requested",
    "resamples_valid",
    "observation_key_sha256",
    "resample_index_sha256",
    "comparator_observation_key_sha256",
    "comparator_resample_index_sha256",
    "equivalence_tested",
    "equivalence_ci_lower",
    "equivalence_ci_upper",
    "equivalence_confidence_level",
    "equivalence_resample_index_sha256",
    "exclusions_sha256",
)


def snapshot_to_dict(snapshot: FileSnapshot) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot."""

    return asdict(snapshot)


def _run_git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=check,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return process.stdout


def verified_clean_repository(root: Path) -> tuple[Path, str]:
    """Return the canonical root and HEAD, rejecting a dirty worktree."""

    root = root.resolve()
    discovered = Path(_run_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if discovered != root:
        raise ManifestValidationError(
            f"repository_root must be the Git root ({discovered}), not {root}"
        )
    status_text = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if status_text.strip():
        raise ManifestValidationError(
            "repository has uncommitted or untracked files; frozen analysis requires "
            "a clean committed state"
        )
    flagged = [
        line
        for line in _run_git(root, "ls-files", "-v").splitlines()
        if line and not line.startswith("H ")
    ]
    if flagged:
        raise ManifestValidationError(
            "Git index has assume-unchanged, skip-worktree, or other nonstandard "
            f"flags: {flagged[:5]}"
        )
    tracked = {value for value in _run_git(root, "ls-files", "-z").split("\0") if value}
    for relative_root in ("src/vehicle_emission", "scripts"):
        source_root = root / relative_root
        if not source_root.exists():
            continue
        for current, directories, filenames in os.walk(source_root, followlinks=False):
            current_path = Path(current)
            for directory in tuple(directories):
                candidate = current_path / directory
                if candidate.is_symlink():
                    raise ManifestValidationError(
                        "symlinked source directories can alter execution: "
                        f"{candidate.relative_to(root).as_posix()}"
                    )
            for filename in filenames:
                source = current_path / filename
                relative = source.relative_to(root).as_posix()
                if source.is_symlink():
                    raise ManifestValidationError(
                        f"symlinked source files can alter execution: {relative}"
                    )
                if source.suffix == ".py" and relative not in tracked:
                    raise ManifestValidationError(
                        "untracked or ignored Python source can alter execution: " f"{relative}"
                    )
                if source.suffix == ".pyc" and "__pycache__" in source.parts:
                    source_stem = source.name.split(".", 1)[0]
                    matching_source = source.parent.parent / f"{source_stem}.py"
                    matching_relative = (
                        matching_source.relative_to(root).as_posix()
                        if matching_source.is_relative_to(root)
                        else ""
                    )
                    if matching_relative in tracked and matching_source.is_file():
                        continue
                if source.suffix in EXECUTABLE_SOURCE_SUFFIXES:
                    raise ManifestValidationError(
                        "bytecode or native executable code is forbidden in the project "
                        f"source tree: {relative}"
                    )
    head = _run_git(root, "rev-parse", "HEAD").strip()
    if GIT_OBJECT_PATTERN.fullmatch(head) is None:
        raise ManifestValidationError("Git HEAD is not a full object identifier")
    return root, head


def _validate_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty repository-relative path")
    if "\\" in value:
        raise ManifestValidationError(f"{field} must use POSIX separators")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ManifestValidationError(f"{field} must not be absolute or contain traversal")
    canonical = pure.as_posix()
    if canonical != value:
        raise ManifestValidationError(f"{field} must be canonical: {canonical}")
    return canonical


def _validate_authorising_path(value: object, *, field: str) -> str:
    relative = _validate_relative_path(value, field=field)
    if not relative.endswith(".json"):
        raise ManifestValidationError(f"{field} must name a JSON file")
    if not relative.startswith(ALLOWED_AUTHORISING_PREFIXES):
        raise ManifestValidationError(
            f"{field} must be under an approved coordination/data manifest directory"
        )
    return relative


def _validate_evidence_reference(reference: object, *, field: str) -> tuple[str, str]:
    if not isinstance(reference, dict):
        raise ManifestValidationError(f"{field} must be an object")
    _require_exact_keys(reference, {"path", "sha256"}, context=field)
    relative = _validate_relative_path(reference["path"], field=f"{field}.path")
    if not relative.startswith(ALLOWED_EVIDENCE_PREFIXES):
        raise ManifestValidationError(f"{field}.path is outside the accepted evidence directories")
    digest = _require_sha(reference["sha256"], field=f"{field}.sha256")
    return relative, digest


def _reject_symlink_components(root: Path, relative: str, *, require_file: bool) -> Path:
    """Resolve a path inside ``root`` without accepting any symlink component."""

    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ManifestValidationError(f"required path is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ManifestValidationError(f"symlink components are forbidden: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ManifestValidationError(f"non-directory path component in {relative}")
    if require_file and not stat.S_ISREG(current.lstat().st_mode):
        raise ManifestValidationError(f"required path is not a regular file: {relative}")
    resolved = current.resolve()
    if not resolved.is_relative_to(root):
        raise ManifestValidationError(f"path escapes repository root: {relative}")
    return resolved


def _open_no_follow(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    return os.fdopen(descriptor, "rb", closefd=True)


def stable_file_bytes(path: Path, *, display_path: str) -> tuple[bytes, FileSnapshot]:
    """Read one regular file once and reject replacement/modification races."""

    with _open_no_follow(path) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ManifestValidationError(f"required path is not a regular file: {display_path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ManifestValidationError(f"file changed while being read: {display_path}")
    return b"".join(chunks), FileSnapshot(
        path=display_path,
        sha256=digest.hexdigest(),
        size=after.st_size,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
    )


def stable_file_snapshot(path: Path, *, display_path: str) -> FileSnapshot:
    """Hash a file from one stable descriptor without retaining its bytes."""

    _, snapshot = stable_file_bytes(path, display_path=display_path)
    return snapshot


def _load_json_bytes(data: bytes, *, display_path: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(f"invalid UTF-8 JSON: {display_path}") from error
    if not isinstance(value, dict):
        raise ManifestValidationError(f"JSON root must be an object: {display_path}")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, context: str) -> None:
    actual = set(value)
    missing = sorted(keys - actual)
    extra = sorted(actual - keys)
    if missing or extra:
        raise ManifestValidationError(
            f"{context} fields do not match schema; missing={missing}, extra={extra}"
        )


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or UTC_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(f"{field} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ManifestValidationError(f"{field} is not a valid UTC timestamp") from error
    return parsed.replace(tzinfo=UTC)


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: object, expected: bool, *, field: str) -> None:
    if value is not expected:
        raise ManifestValidationError(f"{field} must be {expected}")


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ManifestValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestValidationError(f"{field} must be finite")
    return result


def _require_sites(value: object, *, field: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2 or set(value) != set(SITES):
        raise ManifestValidationError(f"{field} must contain exactly {list(SITES)}")
    if len(set(value)) != 2:
        raise ManifestValidationError(f"{field} contains duplicate sites")
    return SITES


def _validate_reference(reference: object, *, field: str) -> tuple[str, str]:
    if not isinstance(reference, dict):
        raise ManifestValidationError(f"{field} must be an object")
    _require_exact_keys(reference, {"path", "sha256"}, context=field)
    path = _validate_authorising_path(reference["path"], field=f"{field}.path")
    digest = _require_sha(reference["sha256"], field=f"{field}.sha256")
    return path, digest


def _validate_freeze_shape(freeze: Mapping[str, Any]) -> None:
    _require_exact_keys(
        freeze,
        {
            "schema_version",
            "status",
            "freeze_id",
            "approved_by",
            "accepted_at_utc",
            "analysis_code_commit",
            "required_sites",
            "source_database",
            "decision_records",
            "manifests",
        },
        context="analysis freeze",
    )
    if freeze["schema_version"] != 2 or freeze["status"] != "accepted":
        raise ManifestValidationError(
            "analysis freeze must have schema_version=2 and status='accepted'"
        )
    _require_string(freeze["freeze_id"], field="freeze_id")
    _require_string(freeze["approved_by"], field="approved_by")
    _parse_utc(freeze["accepted_at_utc"], field="accepted_at_utc")
    code_commit = freeze["analysis_code_commit"]
    if not isinstance(code_commit, str) or GIT_OBJECT_PATTERN.fullmatch(code_commit) is None:
        raise ManifestValidationError("analysis_code_commit must be a full Git object ID")
    _require_sites(freeze["required_sites"], field="required_sites")

    source = freeze["source_database"]
    if not isinstance(source, dict):
        raise ManifestValidationError("source_database must be an object")
    _require_exact_keys(source, {"path", "sha256"}, context="source_database")
    db_path = _validate_relative_path(source["path"], field="source_database.path")
    if not db_path.startswith("data/") or Path(db_path).suffix not in {
        ".db",
        ".sqlite",
        ".sqlite3",
    }:
        raise ManifestValidationError(
            "source_database.path must name an out-of-band database under data/"
        )
    _require_sha(source["sha256"], field="source_database.sha256")

    records = freeze["decision_records"]
    if not isinstance(records, list) or len(records) != len(REQUIRED_DECISIONS):
        raise ManifestValidationError(
            f"decision_records must contain exactly {list(REQUIRED_DECISIONS)}"
        )
    identifiers: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ManifestValidationError(f"decision_records[{index}] must be an object")
        _require_exact_keys(
            record,
            {"question_id", "path", "sha256"},
            context=f"decision_records[{index}]",
        )
        identifiers.append(_require_string(record["question_id"], field="question_id"))
        _validate_reference(
            {"path": record["path"], "sha256": record["sha256"]},
            field=f"decision_records[{index}]",
        )
    if set(identifiers) != set(REQUIRED_DECISIONS) or len(set(identifiers)) != len(identifiers):
        raise ManifestValidationError(
            f"decision_records must identify exactly {list(REQUIRED_DECISIONS)}"
        )

    manifests = freeze["manifests"]
    if not isinstance(manifests, dict) or set(manifests) != set(REQUIRED_MANIFEST_KINDS):
        raise ManifestValidationError(
            f"manifests must contain exactly {list(REQUIRED_MANIFEST_KINDS)}"
        )
    for kind in REQUIRED_MANIFEST_KINDS:
        _validate_reference(manifests[kind], field=f"manifests.{kind}")


def _validate_decision_record(
    record: Mapping[str, Any], *, expected_question: str, freeze_time: datetime
) -> tuple[str, str]:
    _require_exact_keys(
        record,
        {
            "schema_version",
            "question_id",
            "status",
            "accepted_by",
            "accepted_at_utc",
            "decision",
            "approved_method_config_sha256",
            "approved_coverage_manifest_sha256",
        },
        context=f"decision {expected_question}",
    )
    if record["schema_version"] != 1 or record["question_id"] != expected_question:
        raise ManifestValidationError(f"decision record does not identify {expected_question}")
    if record["status"] != "accepted":
        raise ManifestValidationError(f"decision {expected_question} is not accepted")
    _require_string(record["accepted_by"], field=f"{expected_question}.accepted_by")
    accepted = _parse_utc(record["accepted_at_utc"], field=f"{expected_question}.accepted_at_utc")
    if accepted >= freeze_time:
        raise ManifestValidationError(
            f"analysis freeze must be accepted after decision {expected_question}"
        )
    _require_string(record["decision"], field=f"{expected_question}.decision")
    method_hash = _require_sha(
        record["approved_method_config_sha256"],
        field=f"{expected_question}.approved_method_config_sha256",
    )
    coverage_hash = _require_sha(
        record["approved_coverage_manifest_sha256"],
        field=f"{expected_question}.approved_coverage_manifest_sha256",
    )
    return method_hash, coverage_hash


def _validate_manifest_common(
    manifest: Mapping[str, Any], *, kind: str, statuses: set[str]
) -> None:
    if manifest.get("schema_version") != 2 or manifest.get("manifest_type") != kind:
        raise ManifestValidationError(f"{kind} manifest identity is invalid")
    if manifest.get("status") not in statuses:
        raise ManifestValidationError(f"{kind} manifest has an unaccepted status")
    _require_string(manifest.get("manifest_id"), field=f"{kind}.manifest_id")
    _require_sites(manifest.get("sites"), field=f"{kind}.sites")


def _validate_method_config(config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        set(METHOD_CONFIG_FIELDS),
        context="method_config",
    )
    _validate_manifest_common(config, kind="method_config", statuses={"accepted"})
    if config["analysis_scope"] != LOCKED_RETROSPECTIVE_SCOPE:
        raise ManifestValidationError(
            "analysis_scope must disclose a locked retrospective two-site evaluation"
        )
    if config["observational_unit"] != "actual_hourly_ambient_monitor_observation":
        raise ManifestValidationError("method must use actual hourly monitor observations")
    if config["primary_target"] != "ambient_nox_reference_observation":
        raise ManifestValidationError("ambient NOx reference observation must be primary")
    _require_bool(config["target_interpolation"], False, field="target_interpolation")

    interval = config["monitor_interval"]
    if not isinstance(interval, dict):
        raise ManifestValidationError("monitor_interval must be an object")
    _require_exact_keys(
        interval,
        {"source_timezone", "internal_timezone", "boundaries", "label_semantics"},
        context="monitor_interval",
    )
    if interval["source_timezone"] != "Asia/Tokyo" or interval["internal_timezone"] != "UTC":
        raise ManifestValidationError("monitor interval must convert Asia/Tokyo to UTC")
    if interval["boundaries"] != "half_open":
        raise ManifestValidationError("monitor intervals must be half-open")
    if interval["label_semantics"] not in MONITOR_LABEL_SEMANTICS:
        raise ManifestValidationError("monitor label semantics are not approved")

    window = config["predictor_window"]
    if not isinstance(window, dict):
        raise ManifestValidationError("predictor_window must be an object")
    _require_exact_keys(
        window,
        {
            "duration_minutes",
            "offset_hours",
            "sign_convention",
            "lag_mode",
            "candidate_offsets_hours",
            "selection_functional",
            "selection_repeated_in_every_resample",
        },
        context="predictor_window",
    )
    if (
        window["duration_minutes"] != 60
        or window["sign_convention"] != "predictor_precedes_receptor"
    ):
        raise ManifestValidationError("predictor window must be a 60-minute preceding window")
    _require_int(window["offset_hours"], field="offset_hours", minimum=0)
    offsets = window["candidate_offsets_hours"]
    if not isinstance(offsets, list) or not offsets:
        raise ManifestValidationError("candidate_offsets_hours must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in offsets):
        raise ManifestValidationError(
            "candidate offsets must be non-negative whole hours under the "
            "predictor-precedes-receptor convention"
        )
    if len(set(offsets)) != len(offsets):
        raise ManifestValidationError("candidate offsets must be unique")
    lag_mode = window["lag_mode"]
    if lag_mode == "fixed_whole_hour":
        if offsets != [window["offset_hours"]]:
            raise ManifestValidationError("fixed lag must list only the frozen offset")
        if (
            window["selection_functional"] != "none"
            or window["selection_repeated_in_every_resample"] is not False
        ):
            raise ManifestValidationError("fixed lag cannot claim a selection functional")
    elif lag_mode == "selection_adjusted_whole_hour":
        if len(offsets) < 2:
            raise ManifestValidationError("lag selection requires at least two offsets")
        if window["selection_functional"] != "maximum_spearman_with_simultaneous_interval":
            raise ManifestValidationError("lag selection functional is not selection-aware")
        _require_bool(
            window["selection_repeated_in_every_resample"],
            True,
            field="selection_repeated_in_every_resample",
        )
    else:
        raise ManifestValidationError("unsupported lag_mode")

    eligibility = config["eligibility"]
    if not isinstance(eligibility, dict):
        raise ManifestValidationError("eligibility must be an object")
    _require_exact_keys(
        eligibility,
        {
            "rain_exclusion",
            "rain_rule_id",
            "rain_support",
            "rain_threshold_mm",
            "trace_precipitation_handling",
            "coverage_rule_id",
            "coverage_mode",
            "minimum_coverage_fraction",
            "partial_hour_normalisation",
            "gap_rule",
            "rq2_minimum_hours_per_stratum",
            "rq2_minimum_days_per_stratum",
        },
        context="eligibility",
    )
    _require_bool(eligibility["rain_exclusion"], True, field="rain_exclusion")
    _require_string(eligibility["rain_rule_id"], field="rain_rule_id")
    if eligibility["rain_support"] != "same_half_open_monitor_interval":
        raise ManifestValidationError("rain support must match the monitor interval")
    rain_threshold = _require_number(eligibility["rain_threshold_mm"], field="rain_threshold_mm")
    if rain_threshold < 0.0:
        raise ManifestValidationError("rain_threshold_mm must be non-negative")
    if eligibility["trace_precipitation_handling"] not in {
        "exclude_as_rain",
        "retain_as_dry",
    }:
        raise ManifestValidationError("trace precipitation handling must be explicit")
    _require_string(eligibility["coverage_rule_id"], field="coverage_rule_id")
    coverage = _require_number(
        eligibility["minimum_coverage_fraction"], field="minimum_coverage_fraction"
    )
    if not 0.0 < coverage <= 1.0:
        raise ManifestValidationError("minimum_coverage_fraction must be in (0, 1]")
    if eligibility["coverage_mode"] not in {"complete_only", "threshold_normalize"}:
        raise ManifestValidationError("unsupported coverage_mode")
    if eligibility["coverage_mode"] == "complete_only" and (
        coverage != 1.0 or eligibility["partial_hour_normalisation"] != "none"
    ):
        raise ManifestValidationError(
            "complete-only coverage must require 1.0 and no normalisation"
        )
    if (
        eligibility["coverage_mode"] == "threshold_normalize"
        and eligibility["partial_hour_normalisation"]
        != "same_covered_seconds_scale_for_all_predictors"
    ):
        raise ManifestValidationError("partial-hour normalisation must preserve paired coverage")
    if eligibility["gap_rule"] not in VIDEO_COVERAGE_GAP_RULES:
        raise ManifestValidationError("eligibility gap_rule is not executable")
    _require_int(
        eligibility["rq2_minimum_hours_per_stratum"],
        field="rq2_minimum_hours_per_stratum",
        minimum=3,
    )
    _require_int(
        eligibility["rq2_minimum_days_per_stratum"],
        field="rq2_minimum_days_per_stratum",
        minimum=4,
    )

    resampling = config["resampling"]
    if not isinstance(resampling, dict):
        raise ManifestValidationError("resampling must be an object")
    _require_exact_keys(
        resampling,
        {
            "unit",
            "interval_method",
            "resamples",
            "minimum_valid_resamples",
            "seed",
            "fallback_policy",
            "minimum_clusters",
            "block_length_days",
        },
        context="resampling",
    )
    if resampling["unit"] not in {"local_calendar_day", "moving_block_of_local_days"}:
        raise ManifestValidationError("resampling unit must preserve day dependence")
    if resampling["interval_method"] not in {"cluster_percentile", "cluster_bca"}:
        raise ManifestValidationError("unsupported dependence-aware interval method")
    requested_resamples = _require_int(resampling["resamples"], field="resamples", minimum=1999)
    minimum_valid = _require_int(
        resampling["minimum_valid_resamples"],
        field="minimum_valid_resamples",
        minimum=19,
    )
    if minimum_valid > requested_resamples:
        raise ManifestValidationError("minimum_valid_resamples cannot exceed requested resamples")
    if minimum_valid < math.ceil(0.95 * requested_resamples):
        raise ManifestValidationError(
            "minimum_valid_resamples must retain at least 95% of requested resamples"
        )
    _require_int(resampling["seed"], field="seed", minimum=0)
    if resampling["fallback_policy"] != "error":
        raise ManifestValidationError("primary inference cannot silently or automatically fallback")
    _require_int(resampling["minimum_clusters"], field="minimum_clusters", minimum=4)
    if resampling["unit"] == "moving_block_of_local_days":
        _require_int(resampling["block_length_days"], field="block_length_days", minimum=2)
    elif resampling["block_length_days"] is not None:
        raise ManifestValidationError("day-cluster resampling must use null block_length_days")

    multiplicity = config["multiplicity"]
    if not isinstance(multiplicity, dict):
        raise ManifestValidationError("multiplicity must be an object")
    _require_exact_keys(
        multiplicity,
        {
            "family_id",
            "procedure",
            "family_size",
            "family_members",
            "per_interval_confidence_level",
            "conditional_rq2_family",
        },
        context="multiplicity",
    )
    _require_string(multiplicity["family_id"], field="multiplicity.family_id")
    allowed_multiplicity_procedures = {"bonferroni", "holm", "cluster_max_t"}
    if multiplicity["procedure"] not in allowed_multiplicity_procedures:
        raise ManifestValidationError("unsupported multiplicity procedure")
    primary_family_members = multiplicity["family_members"]
    expected_primary_members = {f"{rq}:{site}" for rq in ("RQ1", "RQ3") for site in SITES}
    if (
        not isinstance(primary_family_members, list)
        or len(primary_family_members) != len(set(primary_family_members))
        or set(primary_family_members) != expected_primary_members
    ):
        raise ManifestValidationError(
            "primary multiplicity family must enumerate both-site RQ1 and RQ3 claims"
        )
    family_size = _require_int(multiplicity["family_size"], field="family_size", minimum=1)
    if family_size != len(primary_family_members):
        raise ManifestValidationError("family_size does not match enumerated family_members")
    interval_confidence = _require_number(
        multiplicity["per_interval_confidence_level"],
        field="multiplicity.per_interval_confidence_level",
    )
    if not 0.5 < interval_confidence < 1.0:
        raise ManifestValidationError(
            "multiplicity.per_interval_confidence_level must be in (0.5, 1)"
        )
    conditional_rq2 = multiplicity["conditional_rq2_family"]
    if not isinstance(conditional_rq2, dict):
        raise ManifestValidationError("conditional_rq2_family must be an object")
    _require_exact_keys(
        conditional_rq2,
        {
            "family_id",
            "activation_rule",
            "family_members",
            "procedure",
            "per_interval_confidence_level",
        },
        context="conditional_rq2_family",
    )
    _require_string(
        conditional_rq2["family_id"],
        field="conditional_rq2_family.family_id",
    )
    if conditional_rq2["activation_rule"] != "evaluate_only_site_rows_passing_frozen_support":
        raise ManifestValidationError("conditional RQ2 activation rule is not explicit")
    rq2_members = conditional_rq2["family_members"]
    if (
        not isinstance(rq2_members, list)
        or len(rq2_members) != len(set(rq2_members))
        or set(rq2_members) != {f"RQ2:{site}" for site in SITES}
    ):
        raise ManifestValidationError(
            "conditional RQ2 family must enumerate both site-specific RQ2 claims"
        )
    if conditional_rq2["procedure"] not in allowed_multiplicity_procedures:
        raise ManifestValidationError("unsupported conditional RQ2 multiplicity procedure")
    rq2_interval_confidence = _require_number(
        conditional_rq2["per_interval_confidence_level"],
        field="conditional_rq2_family.per_interval_confidence_level",
    )
    if not 0.5 < rq2_interval_confidence < 1.0:
        raise ManifestValidationError(
            "conditional RQ2 per-interval confidence level must be in (0.5, 1)"
        )

    equivalence = config["equivalence"]
    if not isinstance(equivalence, dict):
        raise ManifestValidationError("equivalence must be an object")
    _require_exact_keys(
        equivalence,
        {
            "enabled",
            "procedure",
            "lower_margin",
            "upper_margin",
            "interval_confidence_level",
        },
        context="equivalence",
    )
    if not isinstance(equivalence["enabled"], bool):
        raise ManifestValidationError("equivalence.enabled must be boolean")
    if equivalence["enabled"]:
        if equivalence["procedure"] != "dependence_aware_cluster_tost":
            raise ManifestValidationError("enabled equivalence requires cluster TOST")
        if equivalence["lower_margin"] != -0.1 or equivalence["upper_margin"] != 0.1:
            raise ManifestValidationError("RQ3 equivalence margins must be [-0.1, 0.1]")
        equivalence_confidence = _require_number(
            equivalence["interval_confidence_level"],
            field="equivalence.interval_confidence_level",
        )
        if not 0.5 < equivalence_confidence < 1.0:
            raise ManifestValidationError(
                "equivalence.interval_confidence_level must be in (0.5, 1)"
            )
    elif (
        equivalence["procedure"] != "none"
        or equivalence["lower_margin"] is not None
        or equivalence["upper_margin"] is not None
        or equivalence["interval_confidence_level"] is not None
    ):
        raise ManifestValidationError(
            "disabled equivalence must use procedure='none' and null parameters"
        )

    evaluation_design = config["evaluation_design"]
    if not isinstance(evaluation_design, dict):
        raise ManifestValidationError("evaluation_design must be an object")
    _require_exact_keys(
        evaluation_design,
        set(EVALUATION_DESIGN_FIELDS),
        context="evaluation_design",
    )
    _require_bool(
        evaluation_design["shinjuku_outcome_accessed_before_retrospective_lock"],
        True,
        field="evaluation_design.shinjuku_outcome_accessed_before_retrospective_lock",
    )
    _require_bool(
        evaluation_design["method_revised_after_outcome_access"],
        True,
        field="evaluation_design.method_revised_after_outcome_access",
    )
    _require_bool(
        evaluation_design["no_further_outcome_guided_tuning_after_lock"],
        True,
        field="evaluation_design.no_further_outcome_guided_tuning_after_lock",
    )
    _require_bool(
        evaluation_design["untouched_replication_claim_permitted"],
        False,
        field="evaluation_design.untouched_replication_claim_permitted",
    )

    wind = config["wind_strata"]
    if not isinstance(wind, dict):
        raise ManifestValidationError("wind_strata must be an object")
    _require_exact_keys(
        wind,
        {
            "rule_id",
            "direction_source",
            "meteorological_from_conversion",
            "calm_wind_threshold_mps",
            "neutral_handling",
            "site_geometry_reference",
        },
        context="wind_strata",
    )
    _require_string(wind["rule_id"], field="wind_strata.rule_id")
    _require_string(wind["direction_source"], field="wind_strata.direction_source")
    _require_bool(
        wind["meteorological_from_conversion"],
        True,
        field="wind_strata.meteorological_from_conversion",
    )
    calm = _require_number(
        wind["calm_wind_threshold_mps"], field="wind_strata.calm_wind_threshold_mps"
    )
    if calm < 0.0:
        raise ManifestValidationError("calm-wind threshold must be non-negative")
    if wind["neutral_handling"] not in {"exclude_from_rq2", "separate_descriptive"}:
        raise ManifestValidationError("neutral wind handling must be explicit")
    _validate_evidence_reference(
        wind["site_geometry_reference"], field="wind_strata.site_geometry_reference"
    )

    support_rules = config["support_rules"]
    if not isinstance(support_rules, list) or len(support_rules) != 7:
        raise ManifestValidationError("support_rules must define the seven RQ/site rows")
    support_keys: set[tuple[str, str]] = set()
    for index, rule in enumerate(support_rules):
        if not isinstance(rule, dict):
            raise ManifestValidationError("support rule must be an object")
        _require_exact_keys(
            rule,
            {
                "rq",
                "site",
                "minimum_hours",
                "minimum_days",
                "maximum_interval_width",
                "classification_on_failure",
            },
            context=f"support_rules[{index}]",
        )
        rq, site = str(rule["rq"]), str(rule["site"])
        if rq == "H_E":
            if site != "cross_site":
                raise ManifestValidationError("H_E support rule must be cross_site")
        elif rq not in {"RQ1", "RQ2", "RQ3"} or site not in SITES:
            raise ManifestValidationError("invalid support-rule RQ/site")
        if (rq, site) in support_keys:
            raise ManifestValidationError("duplicate support-rule RQ/site")
        support_keys.add((rq, site))
        _require_int(rule["minimum_hours"], field="minimum_hours", minimum=3)
        _require_int(rule["minimum_days"], field="minimum_days", minimum=4)
        maximum_width = _require_number(
            rule["maximum_interval_width"], field="maximum_interval_width"
        )
        if not 0.0 < maximum_width <= 4.0:
            raise ManifestValidationError("maximum interval width is out of range")
        if rule["classification_on_failure"] not in {
            "exploratory",
            "not_estimable",
        }:
            raise ManifestValidationError("support failure classification is invalid")
        if rq == "H_E" and rule["classification_on_failure"] != "not_estimable":
            raise ManifestValidationError("H_E support failure must be classified not_estimable")
    required_support_keys = {
        (rq_name, site_name) for rq_name in ("RQ1", "RQ2", "RQ3") for site_name in SITES
    } | {("H_E", "cross_site")}
    if support_keys != required_support_keys:
        raise ManifestValidationError("support rules do not cover all seven result rows")

    cross_site = config["cross_site_resampling"]
    if not isinstance(cross_site, dict):
        raise ManifestValidationError("cross_site_resampling must be an object")
    _require_exact_keys(
        cross_site,
        {
            "matched_date_rule",
            "unmatched_date_rule",
            "shared_date_covariance_preserved",
            "heterogeneity_recomputed_each_resample",
        },
        context="cross_site_resampling",
    )
    if cross_site["matched_date_rule"] != "joint_draws_on_union_calendar_with_presence_masks":
        raise ManifestValidationError("matched-date resampling rule is not approved")
    if cross_site["unmatched_date_rule"] != "independent_only_when_declared_site_only":
        raise ManifestValidationError("unmatched-date resampling rule is not approved")
    _require_bool(
        cross_site["shared_date_covariance_preserved"],
        True,
        field="shared_date_covariance_preserved",
    )
    _require_bool(
        cross_site["heterogeneity_recomputed_each_resample"],
        True,
        field="heterogeneity_recomputed_each_resample",
    )

    aggregation = config["predictor_aggregation"]
    if not isinstance(aggregation, dict):
        raise ManifestValidationError("predictor_aggregation must be an object")
    _require_exact_keys(
        aggregation,
        {
            "shared_valid_seconds",
            "unique_count_rule",
            "emission_index_rule",
            "mean_speed_rule",
        },
        context="predictor_aggregation",
    )
    _require_bool(aggregation["shared_valid_seconds"], True, field="shared_valid_seconds")
    if aggregation["unique_count_rule"] != "unique_entering_vehicle_once":
        raise ManifestValidationError("unique-count aggregation rule is invalid")
    if aggregation["emission_index_rule"] != "sum_native_source_index_over_valid_seconds":
        raise ManifestValidationError("emission-index aggregation rule is invalid")
    if aggregation["mean_speed_rule"] != "time_weighted_valid_calibrated_samples":
        raise ManifestValidationError("mean-speed aggregation must be time weighted")

    engine = config["engine"]
    if not isinstance(engine, dict):
        raise ManifestValidationError("engine must be an object")
    _require_exact_keys(
        engine,
        {"engine_id", "module_path", "module_sha256", "legacy_iid_or_interpolated_path"},
        context="engine",
    )
    _require_string(engine["engine_id"], field="engine.engine_id")
    engine_path = _validate_relative_path(engine["module_path"], field="engine.module_path")
    if not engine_path.startswith("src/vehicle_emission/inference/") or not engine_path.endswith(
        ".py"
    ):
        raise ManifestValidationError("engine module must be a project inference module")
    _require_sha(engine["module_sha256"], field="engine.module_sha256")
    _require_bool(
        engine["legacy_iid_or_interpolated_path"],
        False,
        field="legacy_iid_or_interpolated_path",
    )


def _validate_eligibility(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        set(ELIGIBILITY_MANIFEST_FIELDS),
        context="eligibility manifest",
    )
    _validate_manifest_common(manifest, kind="eligibility", statuses={"frozen_retrospective"})
    _require_bool(
        manifest["prior_outcome_access_disclosed"],
        True,
        field="eligibility.prior_outcome_access_disclosed",
    )
    _require_bool(
        manifest["method_revised_after_outcome_access"],
        True,
        field="eligibility.method_revised_after_outcome_access",
    )
    _require_bool(
        manifest["no_further_outcome_guided_tuning_after_lock"],
        True,
        field="eligibility.no_further_outcome_guided_tuning_after_lock",
    )
    _require_bool(
        manifest["outcome_neutral_claim_permitted"],
        False,
        field="eligibility.outcome_neutral_claim_permitted",
    )
    if manifest["row_key_fields"] != ["site", "monitor_start_utc", "monitor_end_utc"]:
        raise ManifestValidationError("eligibility row keys must identify actual site-hours")
    _validate_evidence_reference(manifest["rules_reference"], field="eligibility.rules_reference")
    _validate_evidence_reference(
        manifest["candidate_periods_reference"],
        field="eligibility.candidate_periods_reference",
    )


def _validate_coverage(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        set(COVERAGE_MANIFEST_FIELDS),
        context="coverage manifest",
    )
    _validate_manifest_common(manifest, kind="coverage", statuses={"frozen_retrospective"})
    _require_bool(
        manifest["shinjuku_outcome_accessed_before_retrospective_lock"],
        True,
        field="coverage.shinjuku_outcome_accessed_before_retrospective_lock",
    )
    _validate_evidence_reference(
        manifest["coverage_source_reference"],
        field="coverage.coverage_source_reference",
    )
    summaries = manifest["site_summaries"]
    if not isinstance(summaries, list) or len(summaries) != 2:
        raise ManifestValidationError("coverage.site_summaries must contain both sites")
    seen: set[str] = set()
    fields = {
        "site",
        "candidate_days",
        "candidate_hours",
        "eligible_dry_hours",
        "missing_monitor_hours",
        "missing_video_hours",
        "favourable_wind_hours",
        "opposing_wind_hours",
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ManifestValidationError("coverage site summary must be an object")
        _require_exact_keys(summary, fields, context="coverage site summary")
        site = summary["site"]
        if site not in SITES or site in seen:
            raise ManifestValidationError("coverage site summaries have invalid/duplicate sites")
        seen.add(site)
        for key in fields - {"site"}:
            _require_int(summary[key], field=f"coverage.{site}.{key}", minimum=0)
        if summary["eligible_dry_hours"] > summary["candidate_hours"]:
            raise ManifestValidationError("eligible dry hours exceed candidate hours")
        if summary["candidate_days"] > summary["candidate_hours"]:
            raise ManifestValidationError("candidate days exceed candidate hours")
        if summary["missing_monitor_hours"] > summary["candidate_hours"]:
            raise ManifestValidationError("missing monitor hours exceed candidate hours")
        if summary["missing_video_hours"] > summary["candidate_hours"]:
            raise ManifestValidationError("missing video hours exceed candidate hours")
        if (
            summary["favourable_wind_hours"] + summary["opposing_wind_hours"]
            > summary["eligible_dry_hours"]
        ):
            raise ManifestValidationError(
                "mutually exclusive wind-stratum hours exceed eligible dry hours"
            )
    if seen != set(SITES):
        raise ManifestValidationError("coverage manifest is missing a site")


def _validate_simple_manifest(manifest: Mapping[str, Any], *, kind: str) -> None:
    common = {"schema_version", "manifest_type", "status", "manifest_id", "sites"}
    fields_by_kind = {
        "model": {
            "detector_id",
            "detector_artifact_reference",
            "tracker_id",
            "emission_model_id",
            "feature_schema_reference",
        },
        "calibration": {
            "calibration_artifact_reference",
            "speed_validation_manifest_reference",
            "acceptance_rule_id",
            "passed_sites",
        },
        "station": {"station_mapping_reference", "timestamp_semantics_id", "rain_rule_id"},
        "provenance": {
            "database_snapshot_id",
            "source_database_sha256",
            "feature_manifest_reference",
            "video_manifest_reference",
            "station_manifest_sha256",
        },
    }
    required = common | fields_by_kind[kind]
    _require_exact_keys(manifest, required, context=f"{kind} manifest")
    status = "accepted" if kind in {"calibration", "station"} else "frozen"
    _validate_manifest_common(manifest, kind=kind, statuses={status})
    for field in fields_by_kind[kind]:
        value = manifest[field]
        if field.endswith("_reference"):
            _validate_evidence_reference(value, field=f"{kind}.{field}")
        elif field.endswith("_sha256"):
            _require_sha(value, field=f"{kind}.{field}")
        elif field == "passed_sites":
            _require_sites(value, field="calibration.passed_sites")
        else:
            _require_string(value, field=f"{kind}.{field}")


def _tracked_json_at_head(
    root: Path,
    relative: str,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], FileSnapshot]:
    path = _reject_symlink_components(root, relative, require_file=True)
    data, snapshot = stable_file_bytes(path, display_path=relative)
    if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
        raise ManifestValidationError(f"SHA-256 mismatch for {relative}")
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except subprocess.CalledProcessError as error:
        raise ManifestValidationError(f"authorising file is not tracked: {relative}") from error
    if not tracked.stdout:
        raise ManifestValidationError(f"authorising file is not tracked: {relative}")
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
        timeout=15,
    ).stdout
    if data.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
        raise ManifestValidationError(f"authorising file does not match HEAD: {relative}")
    return _load_json_bytes(data, display_path=relative), snapshot


def _tracked_evidence_at_head(
    root: Path,
    reference: object,
    *,
    field: str,
) -> FileSnapshot:
    relative, expected = _validate_evidence_reference(reference, field=field)
    path = _reject_symlink_components(root, relative, require_file=True)
    data, snapshot = stable_file_bytes(path, display_path=relative)
    if snapshot.sha256 != expected:
        raise ManifestValidationError(f"SHA-256 mismatch for {relative}")
    try:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ManifestValidationError(f"evidence file is not tracked: {relative}") from error
    if committed.replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n"):
        raise ManifestValidationError(f"evidence file does not match HEAD: {relative}")
    return snapshot


def _iter_evidence_references(
    manifests: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[str, object]]:
    method = manifests["method_config"]
    yield "method_config.wind_strata.site_geometry_reference", method["wind_strata"][
        "site_geometry_reference"
    ]
    eligibility = manifests["eligibility"]
    for field in ("rules_reference", "candidate_periods_reference"):
        yield f"eligibility.{field}", eligibility[field]
    yield "coverage.coverage_source_reference", manifests["coverage"]["coverage_source_reference"]
    for kind in ("model", "calibration", "station", "provenance"):
        for field, value in manifests[kind].items():
            if field.endswith("_reference"):
                yield f"{kind}.{field}", value


def _project_module_path(root: Path, module: str) -> Path | None:
    if not module.startswith("vehicle_emission"):
        return None
    relative = Path("src") / Path(*module.split("."))
    for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
        candidate_relative = candidate.as_posix()
        try:
            return _reject_symlink_components(root, candidate_relative, require_file=True)
        except ManifestValidationError as error:
            if "required path is missing" not in str(error):
                raise
    return None


def _tracked_source_at_head(root: Path, path: Path) -> bytes:
    relative = path.relative_to(root).as_posix()
    data, _ = stable_file_bytes(path, display_path=relative)
    try:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ManifestValidationError(
            f"engine dependency is not tracked at HEAD: {relative}"
        ) from error
    if data.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
        raise ManifestValidationError(f"engine dependency does not match HEAD: {relative}")
    return data


def _validate_engine_dependency_graph(root: Path, engine_relative: str) -> None:
    """Reject direct/transitive references to the historical inference path."""

    pending = [root / engine_relative]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        relative = path.relative_to(root).as_posix()
        data = _tracked_source_at_head(root, path)
        try:
            source = data.decode("utf-8")
            tree = ast.parse(source, filename=relative)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise ManifestValidationError(f"cannot parse engine dependency: {relative}") from error
        if any(forbidden in source for forbidden in FORBIDDEN_INFERENCE_MODULES):
            raise ManifestValidationError(
                f"engine dependency references a forbidden legacy inference module: {relative}"
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name: str | None = None
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            if called_name in DYNAMIC_CODE_CALLS:
                raise ManifestValidationError(
                    "engine dependency uses dynamic code or import loading, so its static "
                    f"dependency closure cannot be verified: {relative}"
                )
        imports: set[tuple[str, bool]] = set()
        module_parts = list(Path(relative).with_suffix("").parts)
        if module_parts[0] == "src":
            module_parts = module_parts[1:]
        is_package = module_parts[-1] == "__init__"
        package_parts = module_parts[:-1] if not is_package else module_parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update((alias.name, True) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    # The project uses namespace-package directories with no
                    # __init__.py. Resolve imported submodules when they
                    # exist; an imported name may instead be a symbol.
                    imports.add((node.module, False))
                    imports.update(
                        (f"{node.module}.{alias.name}", False)
                        for alias in node.names
                        if alias.name != "*"
                    )
                elif node.level > 0:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 1:
                        raise ManifestValidationError(
                            f"engine dependency has an invalid relative import: {relative}"
                        )
                    base = package_parts[:keep]
                    if node.module:
                        imported_base = ".".join([*base, node.module])
                        imports.add((imported_base, True))
                        imports.update(
                            (f"{imported_base}.{alias.name}", False)
                            for alias in node.names
                            if alias.name != "*"
                        )
                    else:
                        imports.update(
                            (".".join([*base, alias.name]), True)
                            for alias in node.names
                            if alias.name != "*"
                        )
        forbidden_imports = sorted(
            module for module, _ in imports if module in FORBIDDEN_INFERENCE_MODULES
        )
        if forbidden_imports:
            raise ManifestValidationError(
                f"engine dependency imports forbidden legacy module {forbidden_imports[0]}"
            )
        for module, resolution_required in imports:
            dependency = _project_module_path(root, module)
            if dependency is None:
                if module.startswith("vehicle_emission") and resolution_required:
                    raise ManifestValidationError(
                        f"unresolved project dependency is forbidden: {module}"
                    )
                continue
            pending.append(dependency)


def validate_lightweight_gates(
    freeze_path: Path,
    *,
    repository_root: Path,
) -> LightweightValidationReceipt:
    """Validate every committed gate before any empirical database read.

    The function intentionally performs no filesystem operation on the source
    database. It returns the lexical path and expected digest for the later,
    fixed database-verification gate.
    """

    root, head = verified_clean_repository(repository_root)
    if freeze_path.is_absolute():
        try:
            freeze_relative = freeze_path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ManifestValidationError(
                "freeze manifest must be inside repository_root"
            ) from error
    else:
        freeze_relative = freeze_path.as_posix()
    freeze_relative = _validate_authorising_path(freeze_relative, field="freeze_path")
    freeze, freeze_snapshot = _tracked_json_at_head(root, freeze_relative, None)
    _validate_freeze_shape(freeze)

    code_commit = str(freeze["analysis_code_commit"])
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", code_commit, head],
        check=False,
        capture_output=True,
        timeout=15,
    )
    if ancestor.returncode != 0:
        raise ManifestValidationError("analysis_code_commit is not an ancestor of HEAD")

    freeze_time = _parse_utc(freeze["accepted_at_utc"], field="accepted_at_utc")
    decision_snapshots: list[FileSnapshot] = []
    decision_approvals: list[tuple[str, str]] = []
    authorising_paths = {freeze_relative}
    for reference in freeze["decision_records"]:
        question = str(reference["question_id"])
        relative, expected = _validate_reference(
            {"path": reference["path"], "sha256": reference["sha256"]},
            field=f"decision {question}",
        )
        decision, snapshot = _tracked_json_at_head(root, relative, expected)
        decision_approvals.append(
            _validate_decision_record(
                decision,
                expected_question=question,
                freeze_time=freeze_time,
            )
        )
        authorising_paths.add(relative)
        decision_snapshots.append(snapshot)

    manifests: dict[str, dict[str, Any]] = {}
    manifest_snapshots: list[FileSnapshot] = []
    for kind in REQUIRED_MANIFEST_KINDS:
        relative, expected = _validate_reference(
            freeze["manifests"][kind], field=f"manifests.{kind}"
        )
        manifest, snapshot = _tracked_json_at_head(root, relative, expected)
        manifests[kind] = manifest
        manifest_snapshots.append(snapshot)
        authorising_paths.add(relative)

    expected_approval = (
        str(freeze["manifests"]["method_config"]["sha256"]),
        str(freeze["manifests"]["coverage"]["sha256"]),
    )
    if any(approval != expected_approval for approval in decision_approvals):
        raise ManifestValidationError(
            "Q-012/Q-016 decision records do not approve the exact method and "
            "coverage manifest bytes named by the freeze"
        )

    _validate_method_config(manifests["method_config"])
    _validate_eligibility(manifests["eligibility"])
    _validate_coverage(manifests["coverage"])
    for kind in ("model", "calibration", "station", "provenance"):
        _validate_simple_manifest(manifests[kind], kind=kind)

    evidence_snapshots: list[FileSnapshot] = []
    for field, reference in _iter_evidence_references(manifests):
        snapshot = _tracked_evidence_at_head(root, reference, field=field)
        if snapshot.path in authorising_paths:
            raise ManifestValidationError(f"duplicate evidence/authorising path: {snapshot.path}")
        authorising_paths.add(snapshot.path)
        evidence_snapshots.append(snapshot)

    manifest_ids = [str(manifests[kind]["manifest_id"]) for kind in REQUIRED_MANIFEST_KINDS]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ManifestValidationError("authorising manifest IDs must be unique")
    source_hash = str(freeze["source_database"]["sha256"])
    if manifests["provenance"]["source_database_sha256"] != source_hash:
        raise ManifestValidationError("provenance and freeze database hashes disagree")
    station_snapshot = manifest_snapshots[REQUIRED_MANIFEST_KINDS.index("station")]
    if manifests["provenance"]["station_manifest_sha256"] != station_snapshot.sha256:
        raise ManifestValidationError(
            "provenance station-manifest hash does not match the accepted station bytes"
        )
    if (
        manifests["station"]["timestamp_semantics_id"]
        != manifests["method_config"]["monitor_interval"]["label_semantics"]
    ):
        raise ManifestValidationError("station and method timestamp semantics disagree")
    if (
        manifests["station"]["rain_rule_id"]
        != manifests["method_config"]["eligibility"]["rain_rule_id"]
    ):
        raise ManifestValidationError("station and method rain rules disagree")

    engine = manifests["method_config"]["engine"]
    engine_relative = _validate_relative_path(engine["module_path"], field="engine.module_path")
    engine_path = _reject_symlink_components(root, engine_relative, require_file=True)
    engine_data, engine_snapshot = stable_file_bytes(engine_path, display_path=engine_relative)
    if engine_snapshot.sha256 != engine["module_sha256"]:
        raise ManifestValidationError("registered engine module hash mismatch")
    committed_engine = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{engine_relative}"],
        check=True,
        capture_output=True,
        timeout=15,
    ).stdout
    if engine_data.replace(b"\r\n", b"\n") != committed_engine.replace(b"\r\n", b"\n"):
        raise ManifestValidationError("registered engine module does not match HEAD")
    _validate_engine_dependency_graph(root, engine_relative)

    changed = {
        line.strip()
        for line in _run_git(
            root, "diff", "--name-only", f"{code_commit}..{head}", "--"
        ).splitlines()
        if line.strip()
    }
    unexpected = sorted(changed - authorising_paths)
    if unexpected:
        raise ManifestValidationError(
            "code or unapproved files changed after analysis_code_commit: " f"{unexpected}"
        )

    return LightweightValidationReceipt(
        repository_root=str(root),
        head_commit=head,
        analysis_code_commit=code_commit,
        freeze_id=str(freeze["freeze_id"]),
        freeze_snapshot=freeze_snapshot,
        decision_snapshots=tuple(decision_snapshots),
        manifest_snapshots=tuple(manifest_snapshots),
        evidence_snapshots=tuple(evidence_snapshots),
        method_config=manifests["method_config"],
        coverage_manifest=manifests["coverage"],
        source_database_path=str(freeze["source_database"]["path"]),
        source_database_sha256=source_hash,
    )


def verify_external_database(
    receipt: LightweightValidationReceipt,
) -> FileSnapshot:
    """Hash the out-of-band database only after lightweight validation."""

    root = Path(receipt.repository_root)
    relative = _validate_relative_path(receipt.source_database_path, field="source_database_path")
    path = _reject_symlink_components(root, relative, require_file=True)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if os.path.lexists(sidecar):
            raise ManifestValidationError(
                "source database has an unbound SQLite sidecar; checkpoint it to a "
                f"self-contained immutable snapshot first: {sidecar.name}"
            )
    snapshot = stable_file_snapshot(path, display_path=relative)
    if snapshot.sha256 != receipt.source_database_sha256:
        raise ManifestValidationError("source database hash does not match accepted freeze")
    return snapshot


def build_result_validation_context(
    receipt: LightweightValidationReceipt,
    database_snapshot: FileSnapshot,
) -> ResultValidationContext:
    """Bind result validation to the exact accepted preflight bytes."""

    if database_snapshot.sha256 != receipt.source_database_sha256:
        raise ManifestValidationError(
            "database snapshot does not match the accepted lightweight receipt"
        )
    if len(receipt.manifest_snapshots) != len(REQUIRED_MANIFEST_KINDS):
        raise ManifestValidationError("lightweight receipt has incomplete manifest snapshots")
    method = receipt.method_config
    resampling = method["resampling"]
    eligibility = method["eligibility"]
    multiplicity = method["multiplicity"]
    equivalence = method["equivalence"]
    support_rules = method["support_rules"]
    site_summaries = receipt.coverage_manifest["site_summaries"]
    assert isinstance(resampling, Mapping)
    assert isinstance(eligibility, Mapping)
    assert isinstance(multiplicity, Mapping)
    assert isinstance(equivalence, Mapping)
    assert isinstance(support_rules, list)
    assert isinstance(site_summaries, list)
    return ResultValidationContext(
        freeze_id=receipt.freeze_id,
        freeze_sha256=receipt.freeze_snapshot.sha256,
        head_commit=receipt.head_commit,
        analysis_scope=str(method["analysis_scope"]),
        source_database_sha256=database_snapshot.sha256,
        manifest_sha256_by_kind=tuple(
            (kind, snapshot.sha256)
            for kind, snapshot in zip(
                REQUIRED_MANIFEST_KINDS,
                receipt.manifest_snapshots,
                strict=True,
            )
        ),
        interval_method=str(resampling["interval_method"]),
        resamples_requested=int(resampling["resamples"]),
        minimum_valid_resamples=int(resampling["minimum_valid_resamples"]),
        minimum_clusters=int(resampling["minimum_clusters"]),
        rq2_minimum_hours_per_stratum=int(eligibility["rq2_minimum_hours_per_stratum"]),
        rq2_minimum_days_per_stratum=int(eligibility["rq2_minimum_days_per_stratum"]),
        adjusted_confidence_level=float(multiplicity["per_interval_confidence_level"]),
        rq2_adjusted_confidence_level=float(
            multiplicity["conditional_rq2_family"]["per_interval_confidence_level"]
        ),
        equivalence_enabled=bool(equivalence["enabled"]),
        equivalence_lower_margin=(
            float(equivalence["lower_margin"]) if equivalence["enabled"] else None
        ),
        equivalence_upper_margin=(
            float(equivalence["upper_margin"]) if equivalence["enabled"] else None
        ),
        equivalence_confidence_level=(
            float(equivalence["interval_confidence_level"]) if equivalence["enabled"] else None
        ),
        support_rules=tuple(
            (
                f"{rule['rq']}:{rule['site']}",
                int(rule["minimum_hours"]),
                int(rule["minimum_days"]),
                float(rule["maximum_interval_width"]),
                str(rule["classification_on_failure"]),
            )
            for rule in support_rules
            if isinstance(rule, Mapping)
        ),
        coverage_by_site=tuple(
            (
                str(summary["site"]),
                int(summary["eligible_dry_hours"]),
                int(summary["candidate_days"]),
            )
            for summary in site_summaries
            if isinstance(summary, Mapping)
        ),
    )


def _canonical_csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError("result CSV cannot contain non-finite numbers")
        return format(value, ".17g")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, str):
        return value
    raise ManifestValidationError(f"result CSV contains unsupported value type: {type(value)}")


def serialise_result_rows_csv(rows: object) -> bytes:
    """Return the only accepted byte representation of primary result rows."""

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ManifestValidationError("result rows must be a list of objects")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(RESULT_ROW_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        _require_exact_keys(row, set(RESULT_ROW_FIELDS), context="result row")
        writer.writerow({field: _canonical_csv_cell(row[field]) for field in RESULT_ROW_FIELDS})
    return stream.getvalue().encode("utf-8")


def _validate_result_row(
    row: Mapping[str, Any], *, context: ResultValidationContext
) -> tuple[str, str]:
    _require_exact_keys(row, set(RESULT_ROW_FIELDS), context="result row")
    rq = row["rq"]
    site = row["site"]
    expected_estimands = {
        "RQ1": "site_specific_emission_index_association",
        "RQ2": "wind_moderation_contrast",
        "RQ3": "association_advantage_over_counts",
        "H_E": "cross_site_association_heterogeneity",
    }
    if rq not in expected_estimands or row["estimand"] != expected_estimands[rq]:
        raise ManifestValidationError("result row has an invalid RQ/estimand pairing")
    expected_site = "cross_site" if rq == "H_E" else None
    if expected_site is not None:
        if site != expected_site:
            raise ManifestValidationError("H_E must be a cross_site row")
    elif site not in SITES:
        raise ManifestValidationError("site-specific RQ row has an invalid site")

    status = row["evidence_status"]
    if status == "confirmatory":
        raise ManifestValidationError(
            "locked retrospective results cannot be confirmatory; site rows must be "
            "exploratory or not estimable"
        )
    if status not in {"descriptive", "exploratory", "not_estimable"}:
        raise ManifestValidationError("invalid evidence_status")
    if row["decision_category"] not in RESULT_DECISION_CATEGORIES:
        raise ManifestValidationError("invalid retrospective decision_category")
    if status == "exploratory" and row["decision_category"] not in EXPLORATORY_DECISION_CATEGORIES:
        raise ManifestValidationError(
            "exploratory rows must use an explicitly exploratory decision category"
        )
    if status == "descriptive" and rq != "H_E":
        raise ManifestValidationError("only H_E may use descriptive evidence status")
    if rq == "H_E" and status not in {"descriptive", "not_estimable"}:
        raise ManifestValidationError("H_E must be descriptive or not estimable")
    if (
        context.analysis_scope == LOCKED_RETROSPECTIVE_SCOPE
        and rq != "H_E"
        and status not in {"exploratory", "not_estimable"}
    ):
        raise ManifestValidationError(
            "locked retrospective site rows must be exploratory or not estimable"
        )
    sufficient = row["data_sufficiency_status"]
    if sufficient not in {"passed", "failed"}:
        raise ManifestValidationError("invalid data_sufficiency_status")
    reason = row["not_estimable_reason"]
    if status == "not_estimable":
        _require_string(reason, field="not_estimable_reason")
    elif reason is not None:
        raise ManifestValidationError("not_estimable_reason must be null for an estimable row")

    for field in (
        "target_observation_count",
        "eligible_day_count",
        "cluster_count",
        "resamples_requested",
        "resamples_valid",
    ):
        _require_int(row[field], field=field, minimum=0)
    effective = _require_number(row["effective_sample_size"], field="effective_sample_size")
    if effective < 0:
        raise ManifestValidationError("effective_sample_size must be non-negative")
    if row["resamples_valid"] > row["resamples_requested"]:
        raise ManifestValidationError("valid resamples cannot exceed requested resamples")
    if row["resamples_requested"] != context.resamples_requested:
        raise ManifestValidationError("resamples_requested does not match the accepted method")
    for field in (
        "favourable_observation_count",
        "opposing_observation_count",
        "favourable_day_count",
        "opposing_day_count",
    ):
        if rq == "RQ2":
            _require_int(row[field], field=field, minimum=0)
        elif row[field] is not None:
            raise ManifestValidationError(f"{field} is only valid for RQ2")

    if row["eligible_day_count"] > row["target_observation_count"]:
        raise ManifestValidationError("eligible days cannot exceed target observations")
    if row["cluster_count"] > row["eligible_day_count"]:
        raise ManifestValidationError("cluster count cannot exceed eligible days")
    if effective > row["target_observation_count"]:
        raise ManifestValidationError("effective sample size cannot exceed target observations")
    coverage_limits = dict(
        (site_name, (hours, days)) for site_name, hours, days in context.coverage_by_site
    )
    if site in SITES:
        if site not in coverage_limits:
            raise ManifestValidationError(f"accepted coverage is missing {site}")
        maximum_hours, maximum_days = coverage_limits[str(site)]
        if row["target_observation_count"] > maximum_hours:
            raise ManifestValidationError("result observations exceed accepted coverage")
        if row["eligible_day_count"] > maximum_days:
            raise ManifestValidationError("result days exceed accepted coverage")
    if rq == "RQ2":
        favourable_hours = int(row["favourable_observation_count"])
        opposing_hours = int(row["opposing_observation_count"])
        favourable_days = int(row["favourable_day_count"])
        opposing_days = int(row["opposing_day_count"])
        if favourable_hours + opposing_hours > row["target_observation_count"]:
            raise ManifestValidationError("wind-stratum hours exceed target observations")
        if favourable_days > row["eligible_day_count"] or opposing_days > row["eligible_day_count"]:
            raise ManifestValidationError("wind-stratum days exceed eligible days")

    support_rule_by_key = {
        key: (minimum_hours, minimum_days, maximum_width, failure_classification)
        for (
            key,
            minimum_hours,
            minimum_days,
            maximum_width,
            failure_classification,
        ) in context.support_rules
    }
    support_key = f"{rq}:{site}"
    if support_key not in support_rule_by_key:
        raise ManifestValidationError(f"accepted support rule is missing {support_key}")
    (
        minimum_hours,
        minimum_days,
        maximum_interval_width,
        failure_classification,
    ) = support_rule_by_key[support_key]
    if rq == "H_E" and failure_classification != "not_estimable":
        raise ManifestValidationError("H_E support failure must be classified not_estimable")

    if status == "not_estimable":
        if sufficient != "failed" or row["decision_category"] != "not_estimable":
            raise ManifestValidationError("not-estimable rows must fail support and say so")
        if any(
            row[field] is not None
            for field in (
                "estimate",
                "ci_lower",
                "ci_upper",
                "adjusted_confidence_level",
            )
        ):
            raise ManifestValidationError("not-estimable rows cannot report an estimate or CI")
        if row["resamples_valid"] != 0:
            raise ManifestValidationError("not-estimable rows must record zero valid resamples")
    else:
        estimate = _require_number(row["estimate"], field="estimate")
        lower = _require_number(row["ci_lower"], field="ci_lower")
        upper = _require_number(row["ci_upper"], field="ci_upper")
        limit = 1.0 if rq == "RQ1" else 2.0
        if not -limit <= lower <= estimate <= upper <= limit:
            raise ManifestValidationError("estimate and confidence interval are out of order/range")
        confidence = _require_number(
            row["adjusted_confidence_level"], field="adjusted_confidence_level"
        )
        expected_confidence = (
            context.rq2_adjusted_confidence_level
            if rq == "RQ2"
            else context.adjusted_confidence_level
        )
        if not math.isclose(
            confidence,
            expected_confidence,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ManifestValidationError(
                "adjusted_confidence_level does not match the frozen multiplicity method"
            )
        support_failures: list[str] = []
        if row["target_observation_count"] < minimum_hours:
            support_failures.append("minimum hourly support")
        if row["eligible_day_count"] < minimum_days:
            support_failures.append("minimum day support")
        if row["eligible_day_count"] < context.minimum_clusters:
            support_failures.append("minimum eligible day-cluster support")
        if row["cluster_count"] < context.minimum_clusters:
            support_failures.append("minimum realised day-cluster support")
        if row["resamples_valid"] < context.minimum_valid_resamples:
            support_failures.append("frozen resample support")
        if effective <= 0:
            support_failures.append("positive effective support")
        if upper - lower > maximum_interval_width:
            support_failures.append("frozen data-sufficiency width")
        if rq == "RQ2":
            for field, minimum in (
                (
                    "favourable_observation_count",
                    context.rq2_minimum_hours_per_stratum,
                ),
                (
                    "opposing_observation_count",
                    context.rq2_minimum_hours_per_stratum,
                ),
                (
                    "favourable_day_count",
                    context.rq2_minimum_days_per_stratum,
                ),
                (
                    "opposing_day_count",
                    context.rq2_minimum_days_per_stratum,
                ),
            ):
                if int(row[field]) < minimum:
                    support_failures.append(f"minimum support for {field}")

        expected_sufficiency = "failed" if support_failures else "passed"
        if sufficient != expected_sufficiency:
            details = ", ".join(support_failures) or "all frozen support rules passed"
            raise ManifestValidationError(
                "data_sufficiency_status contradicts the frozen support rule: " + details
            )
        if support_failures:
            if failure_classification == "not_estimable":
                raise ManifestValidationError(
                    "failed frozen support is classified not_estimable, not as an "
                    "estimable result"
                )
            if status != "exploratory":
                raise ManifestValidationError(
                    "failed frozen support may only be reported as exploratory under "
                    "the accepted support rule"
                )
        if support_failures:
            if row["decision_category"] != "exploratory_no_confirmatory_decision":
                raise ManifestValidationError(
                    "failed-support exploratory rows cannot make a threshold classification"
                )
        elif rq == "RQ1":
            if lower > 0.4:
                expected = "exploratory_practically_strong"
            elif lower > 0.0:
                expected = "exploratory_positive"
            elif upper <= 0.0:
                expected = "exploratory_non_positive"
            else:
                expected = "exploratory_inconclusive"
            if row["decision_category"] != expected:
                raise ManifestValidationError(
                    "exploratory RQ1 decision category contradicts its adjusted CI"
                )
        elif rq == "RQ2":
            if lower > 0.15:
                expected = "exploratory_meaningful_moderation"
            elif upper < 0.15:
                expected = "exploratory_rules_out_proposed_magnitude"
            else:
                expected = "exploratory_inconclusive"
            if row["decision_category"] != expected:
                raise ManifestValidationError(
                    "exploratory RQ2 decision category contradicts its adjusted CI"
                )
        elif rq == "H_E" and row["decision_category"] != "descriptive_only":
            raise ManifestValidationError("H_E cannot replace site-specific decisions")

    requested = row["interval_method_requested"]
    used = row["interval_method_used"]
    if requested != context.interval_method:
        raise ManifestValidationError(
            "requested interval method does not match the accepted method"
        )
    if used not in {"cluster_percentile", "cluster_bca", "none_not_estimable"}:
        raise ManifestValidationError("invalid used interval method")
    if not isinstance(row["fallback_used"], bool):
        raise ManifestValidationError("fallback_used must be boolean")
    if status == "not_estimable":
        if used != "none_not_estimable" or row["fallback_used"]:
            raise ManifestValidationError(
                "not-estimable rows must use none_not_estimable without fallback"
            )
    elif row["fallback_used"]:
        if (
            status != "exploratory"
            or requested != "cluster_bca"
            or used != "cluster_percentile"
            or not isinstance(row["fallback_reason"], str)
            or not row["fallback_reason"].strip()
        ):
            raise ManifestValidationError("fallback results must be explicit and exploratory")
    else:
        if row["fallback_reason"] is not None:
            raise ManifestValidationError("fallback_reason must be null when no fallback occurred")
        if status != "not_estimable" and used != requested:
            raise ManifestValidationError(
                "interval method used differs from requested without an explicit fallback"
            )
    if status == "descriptive" and row["fallback_used"]:
        raise ManifestValidationError("primary evidence cannot use an interval fallback")

    digest_fields = ("exclusions_sha256",)
    for field in digest_fields:
        _require_sha(row[field], field=field)
    for field in ("observation_key_sha256", "resample_index_sha256"):
        if status == "not_estimable":
            if row[field] is not None:
                raise ManifestValidationError(f"{field} must be null when not estimable")
        else:
            _require_sha(row[field], field=field)

    if rq == "RQ3" and status != "not_estimable":
        for field in (
            "comparator_observation_key_sha256",
            "comparator_resample_index_sha256",
        ):
            _require_sha(row[field], field=field)
        if row["comparator_observation_key_sha256"] != row["observation_key_sha256"]:
            raise ManifestValidationError("RQ3 comparators do not use the same observations")
        if row["comparator_resample_index_sha256"] != row["resample_index_sha256"]:
            raise ManifestValidationError("RQ3 comparators do not use the same resamples")
    elif (
        row["comparator_observation_key_sha256"] is not None
        or row["comparator_resample_index_sha256"] is not None
    ):
        raise ManifestValidationError("comparator hashes are only valid for estimable RQ3 rows")

    if rq == "H_E" and status != "not_estimable":
        components = row["component_observation_key_sha256"]
        if not isinstance(components, dict) or set(components) != set(SITES):
            raise ManifestValidationError("H_E must identify both site observation-key hashes")
        for site_name in SITES:
            _require_sha(components[site_name], field=f"H_E.components.{site_name}")
        cross_site_hash = _require_sha(
            row["cross_site_resample_index_sha256"],
            field="H_E.cross_site_resample_index_sha256",
        )
        if cross_site_hash != row["resample_index_sha256"]:
            raise ManifestValidationError("H_E cross-site draw hash does not match its interval")
    elif (
        row["component_observation_key_sha256"] is not None
        or row["cross_site_resample_index_sha256"] is not None
    ):
        raise ManifestValidationError("cross-site hashes are only valid for estimable H_E")

    if not isinstance(row["equivalence_tested"], bool):
        raise ManifestValidationError("equivalence_tested must be boolean")
    equivalence_supported = False
    if (
        row["decision_category"] == "exploratory_practical_equivalence"
        and not row["equivalence_tested"]
    ):
        raise ManifestValidationError(
            "exploratory practical equivalence requires a predeclared test"
        )
    if row["equivalence_tested"]:
        if (
            rq != "RQ3"
            or status != "exploratory"
            or sufficient != "passed"
            or not context.equivalence_enabled
        ):
            raise ManifestValidationError(
                "equivalence is only valid for support-passing exploratory RQ3"
            )
        if (
            context.equivalence_confidence_level is None
            or context.equivalence_lower_margin is None
            or context.equivalence_upper_margin is None
        ):
            raise ManifestValidationError(
                "enabled equivalence context is missing its frozen parameters"
            )
        eq_lower = _require_number(row["equivalence_ci_lower"], field="equivalence_ci_lower")
        eq_upper = _require_number(row["equivalence_ci_upper"], field="equivalence_ci_upper")
        if eq_lower > eq_upper:
            raise ManifestValidationError("equivalence interval is out of order")
        point_estimate = _require_number(row["estimate"], field="estimate")
        primary_lower = _require_number(row["ci_lower"], field="ci_lower")
        primary_upper = _require_number(row["ci_upper"], field="ci_upper")
        if not eq_lower <= point_estimate <= eq_upper:
            raise ManifestValidationError(
                "equivalence interval does not contain the RQ3 point estimate"
            )
        if primary_lower >= float(context.equivalence_upper_margin) or primary_upper <= float(
            context.equivalence_lower_margin
        ):
            raise ManifestValidationError(
                "equivalence evidence contradicts the primary RQ3 interval"
            )
        eq_confidence = _require_number(
            row["equivalence_confidence_level"],
            field="equivalence_confidence_level",
        )
        if not math.isclose(
            eq_confidence,
            float(context.equivalence_confidence_level),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ManifestValidationError(
                "equivalence confidence level does not match the accepted method"
            )
        if row["decision_category"] == "exploratory_practical_equivalence" and not (
            eq_lower > float(context.equivalence_lower_margin)
            and eq_upper < float(context.equivalence_upper_margin)
        ):
            raise ManifestValidationError(
                "exploratory practical equivalence is not supported by its interval"
            )
        equivalence_supported = eq_lower > float(
            context.equivalence_lower_margin
        ) and eq_upper < float(context.equivalence_upper_margin)
        equivalence_resample_hash = _require_sha(
            row["equivalence_resample_index_sha256"],
            field="equivalence_resample_index_sha256",
        )
        if equivalence_resample_hash != row["resample_index_sha256"]:
            raise ManifestValidationError(
                "equivalence and primary intervals do not use the same resamples"
            )
    elif any(
        row[field] is not None
        for field in (
            "equivalence_ci_lower",
            "equivalence_ci_upper",
            "equivalence_confidence_level",
            "equivalence_resample_index_sha256",
        )
    ):
        raise ManifestValidationError("equivalence CI requires equivalence_tested=true")

    if rq == "RQ3" and status == "exploratory" and sufficient == "passed":
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        lower_margin = float(context.equivalence_lower_margin or -0.1)
        upper_margin = float(context.equivalence_upper_margin or 0.1)
        if equivalence_supported:
            expected = "exploratory_practical_equivalence"
        elif lower > upper_margin:
            expected = "exploratory_meaningful_added_value"
        elif upper < lower_margin:
            expected = "exploratory_meaningful_count_advantage"
        else:
            expected = "exploratory_inconclusive"
        if row["decision_category"] != expected:
            raise ManifestValidationError(
                "exploratory RQ3 decision category contradicts its adjusted CI"
            )
    return str(rq), str(site)


def validate_result_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    context: ResultValidationContext,
) -> ResultValidationReceipt:
    """Validate a result against its accepted freeze and canonical CSV bytes."""

    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_type",
            "status",
            "analysis_id",
            "analysis_scope",
            "freeze_id",
            "analysis_freeze_sha256",
            "artifact_path",
            "artifact_sha256",
            "generated_at_utc",
            "code_commit",
            "source_database_sha256",
            "method_config_sha256",
            "eligibility_manifest_sha256",
            "coverage_manifest_sha256",
            "model_manifest_sha256",
            "calibration_manifest_sha256",
            "station_manifest_sha256",
            "provenance_manifest_sha256",
            "primary_target",
            "observational_unit",
            "rows",
        },
        context="result manifest",
    )
    if manifest["schema_version"] != 2 or manifest["manifest_type"] != "phase5_result":
        raise ManifestValidationError("result manifest identity is invalid")
    if manifest["status"] != "generated_unreviewed":
        raise ManifestValidationError(
            "this validator accepts only generated_unreviewed output; reviewed status "
            "requires a separate committed review-attestation workflow"
        )
    analysis_id = _require_string(manifest["analysis_id"], field="analysis_id")
    if context.analysis_scope != LOCKED_RETROSPECTIVE_SCOPE:
        raise ManifestValidationError(
            "result validation requires locked_retrospective_two_site preflight scope"
        )
    if manifest["analysis_scope"] != LOCKED_RETROSPECTIVE_SCOPE:
        raise ManifestValidationError(
            "result manifest must record locked_retrospective_two_site scope"
        )
    if manifest["analysis_scope"] != context.analysis_scope:
        raise ManifestValidationError("result analysis_scope does not match the accepted preflight")
    if manifest["freeze_id"] != context.freeze_id:
        raise ManifestValidationError("result freeze_id does not match the accepted preflight")
    if manifest["analysis_freeze_sha256"] != context.freeze_sha256:
        raise ManifestValidationError(
            "result analysis_freeze_sha256 does not match the accepted preflight"
        )
    artifact_path = _validate_relative_path(manifest["artifact_path"], field="artifact_path")
    if not artifact_path.startswith("results/") or not artifact_path.endswith("_v2.csv"):
        raise ManifestValidationError("result artifact must be an immutable results/*_v2.csv path")
    artifact_sha = _require_sha(manifest["artifact_sha256"], field="artifact_sha256")
    _parse_utc(manifest["generated_at_utc"], field="generated_at_utc")
    code_commit = manifest["code_commit"]
    if not isinstance(code_commit, str) or GIT_OBJECT_PATTERN.fullmatch(code_commit) is None:
        raise ManifestValidationError("code_commit must be a full Git object ID")
    if code_commit != context.head_commit:
        raise ManifestValidationError("result code_commit does not match the preflight HEAD")
    for field in (
        "source_database_sha256",
        "method_config_sha256",
        "eligibility_manifest_sha256",
        "coverage_manifest_sha256",
        "model_manifest_sha256",
        "calibration_manifest_sha256",
        "station_manifest_sha256",
        "provenance_manifest_sha256",
    ):
        _require_sha(manifest[field], field=field)
    if manifest["source_database_sha256"] != context.source_database_sha256:
        raise ManifestValidationError("result source database does not match the preflight")
    expected_manifest_hashes = dict(context.manifest_sha256_by_kind)
    result_manifest_hashes = {
        "method_config": manifest["method_config_sha256"],
        "eligibility": manifest["eligibility_manifest_sha256"],
        "coverage": manifest["coverage_manifest_sha256"],
        "model": manifest["model_manifest_sha256"],
        "calibration": manifest["calibration_manifest_sha256"],
        "station": manifest["station_manifest_sha256"],
        "provenance": manifest["provenance_manifest_sha256"],
    }
    if result_manifest_hashes != expected_manifest_hashes:
        raise ManifestValidationError(
            "result manifest hashes do not match the accepted preflight manifests"
        )
    if manifest["primary_target"] != "ambient_nox_reference_observation":
        raise ManifestValidationError("primary result must use ambient NOx reference observations")
    if manifest["observational_unit"] != "actual_hourly_ambient_monitor_observation":
        raise ManifestValidationError("result must use actual hourly monitor observations")
    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != 7:
        raise ManifestValidationError("primary result manifest must contain exactly seven RQ rows")
    keys = [_validate_result_row(row, context=context) for row in rows if isinstance(row, dict)]
    if len(keys) != len(rows):
        raise ManifestValidationError("every result row must be an object")
    required_keys = {
        ("RQ1", "Sukiyabashi"),
        ("RQ1", "Shinjuku"),
        ("RQ2", "Sukiyabashi"),
        ("RQ2", "Shinjuku"),
        ("RQ3", "Sukiyabashi"),
        ("RQ3", "Shinjuku"),
        ("H_E", "cross_site"),
    }
    if set(keys) != required_keys or len(set(keys)) != len(keys):
        raise ManifestValidationError("result rows must cover each required RQ/site exactly once")
    canonical_keys = [
        ("RQ1", "Sukiyabashi"),
        ("RQ1", "Shinjuku"),
        ("RQ2", "Sukiyabashi"),
        ("RQ2", "Shinjuku"),
        ("RQ3", "Sukiyabashi"),
        ("RQ3", "Shinjuku"),
        ("H_E", "cross_site"),
    ]
    if keys != canonical_keys:
        raise ManifestValidationError("result rows are not in the canonical RQ/site order")
    row_by_key = {(str(row["rq"]), str(row["site"])): row for row in rows if isinstance(row, dict)}
    heterogeneity = row_by_key[("H_E", "cross_site")]
    rq1_rows = {site_name: row_by_key[("RQ1", site_name)] for site_name in SITES}
    if any(row["evidence_status"] == "not_estimable" for row in rq1_rows.values()):
        if heterogeneity["evidence_status"] != "not_estimable":
            raise ManifestValidationError(
                "H_E must be not estimable when either site-specific RQ1 is not estimable"
            )
    elif heterogeneity["evidence_status"] != "not_estimable":
        expected_components = {
            site_name: rq1_rows[site_name]["observation_key_sha256"] for site_name in SITES
        }
        if heterogeneity["component_observation_key_sha256"] != expected_components:
            raise ManifestValidationError(
                "H_E component hashes do not match the two site-specific RQ1 rows"
            )
        expected_difference = float(rq1_rows["Shinjuku"]["estimate"]) - float(
            rq1_rows["Sukiyabashi"]["estimate"]
        )
        if not math.isclose(
            float(heterogeneity["estimate"]),
            expected_difference,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ManifestValidationError("H_E estimate is not Shinjuku RQ1 minus Sukiyabashi RQ1")
        conservative_support_fields = (
            "target_observation_count",
            "eligible_day_count",
            "effective_sample_size",
            "cluster_count",
            "resamples_valid",
        )
        for field in conservative_support_fields:
            expected_support = min(rq1_rows[site_name][field] for site_name in SITES)
            if heterogeneity[field] != expected_support:
                raise ManifestValidationError(
                    f"H_E {field} must equal the conservative minimum of its two RQ1 rows"
                )

    shared_support_fields = (
        "target_observation_count",
        "eligible_day_count",
        "effective_sample_size",
        "cluster_count",
        "observation_key_sha256",
        "resample_index_sha256",
        "exclusions_sha256",
        "resamples_requested",
        "resamples_valid",
    )
    for site_name in SITES:
        rq1 = row_by_key[("RQ1", site_name)]
        rq3 = row_by_key[("RQ3", site_name)]
        if any(rq1[field] != rq3[field] for field in shared_support_fields):
            raise ManifestValidationError(
                f"{site_name} RQ1/RQ3 do not use identical support and resamples"
            )

    root = repository_root.resolve()
    artifact = _reject_symlink_components(root, artifact_path, require_file=True)
    artifact_bytes, snapshot = stable_file_bytes(artifact, display_path=artifact_path)
    if snapshot.sha256 != artifact_sha:
        raise ManifestValidationError("result artifact hash does not match manifest")
    expected_artifact = serialise_result_rows_csv(rows)
    if artifact_bytes != expected_artifact:
        raise ManifestValidationError(
            "result artifact bytes do not canonically encode the manifest rows"
        )
    return ResultValidationReceipt(
        analysis_id=analysis_id,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        row_count=len(rows),
        rq_site_keys=tuple(f"{rq}:{site}" for rq, site in sorted(keys)),
    )
