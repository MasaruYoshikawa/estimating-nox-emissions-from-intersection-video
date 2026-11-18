from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_emission.detection.lineage import (
    CLASS_NAMES,
    DATASET_CONFIG,
    TRAINING_PARAMETERS,
    TRAINING_PROVENANCE_SCHEMA,
    detector_code_identity,
    detector_library_versions,
    validate_code_identity,
    validate_library_versions,
)

CLASS_MAP = {
    0: "bus",
    1: "car",
    2: "heavy_truck",
    3: "light_truck",
}
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "path",
        "split_id",
        "site",
        "source_video_id",
        "source_video_sha256",
        "source_frame_index",
        "image_sha256",
        "label_sha256",
    }
)
REQUIRED_SPLITS = frozenset({"train", "validation", "test"})
REQUIRED_SITES = frozenset({"Sukiyabashi", "Shinjuku"})
SPLIT_DIRECTORY_NAMES = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EXPECTED_MODEL_NAMES = {
    0: "bus",
    1: "car",
    2: "heavy truck",
    3: "light truck",
}
GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EVALUATION_CONFIG_SCHEMA = "detector_evaluation_config_v3"
EVALUATION_RECEIPT_SCHEMA = "detector_evaluation_receipt_v3"
EVALUATION_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_provenance_path",
        "training_provenance_sha256",
        "output_path",
        "receipt_path",
        "inference_confidence_threshold",
        "inference_nms_iou",
        "matching_iou",
        "bootstrap_seed",
        "bootstrap_resamples",
        "test_assets_sha256",
    }
)


class SplitManifestError(ValueError):
    """Raised when the detector split manifest is incomplete or leaks source videos."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a required local file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of an in-memory immutable artefact."""

    return hashlib.sha256(content).hexdigest()


def repository_revision(repository_root: Path, *, tracked_only: bool = False) -> str:
    """Require clean production inputs, allowing generated outputs only after training."""

    process = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    revision = process.stdout.strip()
    if GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError("Git did not return a full repository revision")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain",
            "--untracked-files=no" if tracked_only else "--untracked-files=normal",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "Detector production execution requires a clean Git working tree; "
            "commit or remove all tracked and untracked changes first"
        )
    return revision


def require_tracked_file(path: Path, repository_root: Path) -> None:
    """Require an immutable configuration file to be present in the current commit."""

    root = repository_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Required tracked file is outside repository root: {resolved}")
    relative = resolved.relative_to(root).as_posix()
    process = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Detector evaluation config must be committed before production use: {relative}"
        )


def _require_sha256(value: Any, field_name: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")
    return digest


def _repository_path(value: Any, repository_root: Path, field_name: str) -> Path:
    """Resolve a canonical repository-relative path from a frozen config."""

    text = str(value)
    if "\\" in text:
        raise ValueError(f"{field_name} must use canonical forward slashes")
    segments = text.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{field_name} is not a canonical relative path: {text}")
    candidate = Path(*segments)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative: {text}")
    resolved = (repository_root.resolve() / candidate).resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{field_name} escapes repository root: {text}")
    return resolved


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic JSON artefact without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as file_handle:
        file_handle.write(content)


def validate_model_class_names(model: Any) -> None:
    """Fail when an inspectable checkpoint does not expose the exact four classes."""

    names = getattr(model, "names", None)
    if names is None and getattr(model, "model", None) is not None:
        names = getattr(model.model, "names", None)
    if names is None:
        raise SplitManifestError("Detector checkpoint does not expose its class mapping")
    if isinstance(names, Sequence) and not isinstance(names, str | bytes):
        observed = {index: str(value) for index, value in enumerate(names)}
    elif isinstance(names, dict):
        try:
            observed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise SplitManifestError("Detector checkpoint class IDs must be integers") from exc
    else:
        raise SplitManifestError("Detector checkpoint class names use an unsupported format")
    normalised = {
        class_id: class_name.strip().lower().replace("_", " ")
        for class_id, class_name in observed.items()
    }
    if normalised != EXPECTED_MODEL_NAMES:
        raise SplitManifestError(
            f"Detector checkpoint classes do not match the project contract: {normalised!r}"
        )


def xywhn_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return (x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0)


def calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    first = xywhn_to_xyxy(box1)
    second = xywhn_to_xyxy(box2)

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


def match_image_predictions(
    pred_boxes: Sequence[Sequence[float]],
    pred_cls: Sequence[int],
    pred_confs: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    gt_cls: Sequence[int],
    iou_threshold: float = 0.5,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Match predictions to same-class annotations and return TP, FP, and FN counts."""

    if not (len(pred_boxes) == len(pred_cls) == len(pred_confs)):
        raise ValueError("Prediction boxes, classes, and confidences must have equal lengths")
    if len(gt_boxes) != len(gt_cls):
        raise ValueError("Ground-truth boxes and classes must have equal lengths")
    unknown_predictions = sorted(set(pred_cls) - set(CLASS_MAP))
    if unknown_predictions:
        raise ValueError(f"Detector produced unknown class IDs: {unknown_predictions}")
    unknown_ground_truth = sorted(set(gt_cls) - set(CLASS_MAP))
    if unknown_ground_truth:
        raise ValueError(f"Ground truth contains unknown class IDs: {unknown_ground_truth}")

    true_positives = {class_id: 0 for class_id in CLASS_MAP}
    false_positives = {class_id: 0 for class_id in CLASS_MAP}
    false_negatives = {class_id: 0 for class_id in CLASS_MAP}

    for class_id in CLASS_MAP:
        prediction_indices = [
            index for index, predicted_class in enumerate(pred_cls) if predicted_class == class_id
        ]
        ground_truth_indices = [
            index
            for index, ground_truth_class in enumerate(gt_cls)
            if ground_truth_class == class_id
        ]
        prediction_indices.sort(key=lambda index: pred_confs[index], reverse=True)

        matched_ground_truth: set[int] = set()
        for prediction_index in prediction_indices:
            best_iou = 0.0
            best_ground_truth_index = -1
            for ground_truth_index in ground_truth_indices:
                if ground_truth_index in matched_ground_truth:
                    continue
                overlap = calculate_iou(pred_boxes[prediction_index], gt_boxes[ground_truth_index])
                if overlap > best_iou:
                    best_iou = overlap
                    best_ground_truth_index = ground_truth_index

            if best_iou >= iou_threshold:
                true_positives[class_id] += 1
                matched_ground_truth.add(best_ground_truth_index)
            else:
                false_positives[class_id] += 1

        false_negatives[class_id] = len(ground_truth_indices) - len(matched_ground_truth)

    return true_positives, false_positives, false_negatives


def label_path_for_image(image_path: Path) -> Path:
    """Return the sibling YOLO label path for an image under ``<split>/images``."""

    if image_path.parent.name != "images":
        raise SplitManifestError(
            f"Image path must be under a split's images directory: {image_path}"
        )
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def validate_split_manifest(
    manifest: pd.DataFrame,
    repository_root: Path,
    *,
    require_assets: bool = True,
    required_sites: frozenset[str] = REQUIRED_SITES,
) -> pd.DataFrame:
    """Validate split metadata, source-video isolation, site coverage, and local assets."""

    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise SplitManifestError(
            f"Detector split manifest is missing required columns: {', '.join(missing_columns)}"
        )
    if manifest.empty:
        raise SplitManifestError("Detector split manifest is empty")

    validated = manifest.copy()
    for column in REQUIRED_MANIFEST_COLUMNS:
        values = validated[column].astype("string")
        missing_rows = values.isna() | values.str.strip().eq("")
        if missing_rows.any():
            row_numbers = ", ".join(str(index + 2) for index in validated.index[missing_rows])
            raise SplitManifestError(
                f"Detector split manifest has empty {column!r} values on CSV rows {row_numbers}"
            )
        validated[column] = values.str.strip()

    observed_splits = frozenset(validated["split_id"])
    unsupported_splits = sorted(observed_splits - REQUIRED_SPLITS)
    if unsupported_splits:
        raise SplitManifestError(
            f"Unsupported detector split_id values: {', '.join(unsupported_splits)}"
        )
    missing_splits = sorted(REQUIRED_SPLITS - observed_splits)
    if missing_splits:
        raise SplitManifestError(
            f"Detector split manifest is missing required splits: {', '.join(missing_splits)}"
        )

    frame_indices = pd.to_numeric(validated["source_frame_index"], errors="coerce")
    invalid_frame_indices = frame_indices.isna() | frame_indices.lt(0) | frame_indices.mod(1).ne(0)
    if invalid_frame_indices.any():
        row_numbers = ", ".join(str(index + 2) for index in validated.index[invalid_frame_indices])
        raise SplitManifestError(
            "source_frame_index must be a non-negative integer on CSV rows " + row_numbers
        )
    validated["source_frame_index"] = frame_indices.astype("int64")

    for hash_column in ("source_video_sha256", "image_sha256", "label_sha256"):
        invalid_hashes = ~validated[hash_column].map(
            lambda value: SHA256_PATTERN.fullmatch(str(value)) is not None
        )
        if invalid_hashes.any():
            row_numbers = ", ".join(str(index + 2) for index in validated.index[invalid_hashes])
            raise SplitManifestError(
                f"{hash_column} must contain lowercase SHA-256 digests on CSV rows " + row_numbers
            )

    duplicated_paths = validated.loc[validated["path"].duplicated(keep=False), "path"].unique()
    if len(duplicated_paths):
        raise SplitManifestError(
            f"Image paths must be unique across splits: {', '.join(sorted(duplicated_paths))}"
        )

    source_split_counts = validated.groupby("source_video_id")["split_id"].nunique()
    leaking_sources = sorted(source_split_counts[source_split_counts > 1].index.astype(str))
    if leaking_sources:
        raise SplitManifestError(
            "Source-video leakage across train/validation/test: " + ", ".join(leaking_sources)
        )

    source_site_counts = validated.groupby("source_video_id")["site"].nunique()
    ambiguous_sources = sorted(source_site_counts[source_site_counts > 1].index.astype(str))
    if ambiguous_sources:
        raise SplitManifestError(
            "Source videos map to multiple sites: " + ", ".join(ambiguous_sources)
        )

    source_hash_counts = validated.groupby("source_video_id")["source_video_sha256"].nunique()
    conflicting_source_ids = sorted(source_hash_counts[source_hash_counts > 1].index.astype(str))
    if conflicting_source_ids:
        raise SplitManifestError(
            "Source-video IDs map to multiple source hashes: " + ", ".join(conflicting_source_ids)
        )
    hash_id_counts = validated.groupby("source_video_sha256")["source_video_id"].nunique()
    aliased_source_hashes = sorted(hash_id_counts[hash_id_counts > 1].index.astype(str))
    if aliased_source_hashes:
        raise SplitManifestError(
            "Source-video hashes map to multiple source IDs: " + ", ".join(aliased_source_hashes)
        )
    duplicated_source_frames = validated.duplicated(
        subset=["source_video_sha256", "source_frame_index"], keep=False
    )
    if duplicated_source_frames.any():
        row_numbers = ", ".join(
            str(index + 2) for index in validated.index[duplicated_source_frames]
        )
        raise SplitManifestError(
            "A source-video frame may appear only once; duplicate CSV rows " + row_numbers
        )
    image_hash_split_counts = validated.groupby("image_sha256")["split_id"].nunique()
    cross_split_image_hashes = sorted(
        image_hash_split_counts[image_hash_split_counts > 1].index.astype(str)
    )
    if cross_split_image_hashes:
        raise SplitManifestError(
            "Identical image bytes cannot cross detector splits: "
            + ", ".join(cross_split_image_hashes)
        )

    observed_sites = frozenset(validated["site"])
    unsupported_sites = sorted(observed_sites - required_sites)
    if unsupported_sites:
        raise SplitManifestError(
            "Detector split manifest contains unsupported sites: " + ", ".join(unsupported_sites)
        )

    for split_id in sorted(REQUIRED_SPLITS):
        observed_split_sites = frozenset(validated.loc[validated["split_id"] == split_id, "site"])
        missing_split_sites = sorted(required_sites - observed_split_sites)
        if missing_split_sites:
            raise SplitManifestError(
                f"Detector {split_id} split is missing required sites: "
                + ", ".join(missing_split_sites)
            )

    root = repository_root.resolve()
    missing_assets: list[str] = []
    resolved_images: dict[Path, str] = {}
    resolved_labels: dict[Path, str] = {}
    for _, row in validated.iterrows():
        relative_path = str(row["path"])
        if "\\" in relative_path:
            raise SplitManifestError(
                f"Manifest image paths must use canonical forward slashes: {relative_path}"
            )
        segments = relative_path.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise SplitManifestError(f"Manifest image path is not canonical: {relative_path}")
        candidate = Path(*segments)
        if candidate.is_absolute():
            raise SplitManifestError(f"Manifest image paths must be relative: {relative_path}")
        image_path = (root / candidate).resolve()
        if not image_path.is_relative_to(root):
            raise SplitManifestError(
                f"Manifest image path escapes repository root: {relative_path}"
            )
        expected_split_directory = SPLIT_DIRECTORY_NAMES[str(row["split_id"])]
        if (
            image_path.parent.name != "images"
            or image_path.parent.parent.name != expected_split_directory
        ):
            raise SplitManifestError(
                f"Manifest split_id {row['split_id']!r} does not match image path: {relative_path}"
            )
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise SplitManifestError(f"Unsupported detector image suffix: {relative_path}")
        previous_image = resolved_images.get(image_path)
        if previous_image is not None:
            raise SplitManifestError(
                "Manifest image paths resolve to the same asset: "
                f"{previous_image}, {relative_path}"
            )
        resolved_images[image_path] = relative_path
        label_path = label_path_for_image(image_path)
        resolved_label = label_path.resolve()
        if not resolved_label.is_relative_to(root):
            raise SplitManifestError(
                f"Manifest label path escapes repository root: {relative_path}"
            )
        previous_label = resolved_labels.get(resolved_label)
        if previous_label is not None:
            raise SplitManifestError(
                "Manifest image paths resolve to the same label asset: "
                f"{previous_label}, {relative_path}"
            )
        resolved_labels[resolved_label] = relative_path
        if require_assets and not image_path.is_file():
            missing_assets.append(str(candidate))
        if require_assets and not resolved_label.is_file():
            missing_assets.append(str(resolved_label.relative_to(root)))
        if require_assets and image_path.is_file():
            observed_image_hash = sha256_file(image_path)
            if observed_image_hash != row["image_sha256"]:
                raise SplitManifestError(
                    f"Image SHA-256 mismatch for {relative_path}: "
                    f"expected {row['image_sha256']}, observed {observed_image_hash}"
                )
        if require_assets and resolved_label.is_file():
            observed_label_hash = sha256_file(resolved_label)
            if observed_label_hash != row["label_sha256"]:
                raise SplitManifestError(
                    f"Label SHA-256 mismatch for {relative_path}: "
                    f"expected {row['label_sha256']}, observed {observed_label_hash}"
                )
            read_ground_truth(resolved_label)

    if missing_assets:
        preview = ", ".join(missing_assets[:5])
        remainder = len(missing_assets) - min(len(missing_assets), 5)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise FileNotFoundError(f"Detector evaluation assets are missing: {preview}{suffix}")

    return validated


def load_split_manifest(manifest_path: Path, repository_root: Path) -> pd.DataFrame:
    """Read and validate an existing detector split manifest without creating one implicitly."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Detector split manifest is missing: {manifest_path}")
    return validate_split_manifest(pd.read_csv(manifest_path), repository_root)


def manifest_asset_digest(manifest: pd.DataFrame, repository_root: Path) -> str:
    """Bind ordered image/label hashes and source provenance into one digest."""

    root = repository_root.resolve()
    records: list[dict[str, Any]] = []
    ordered = manifest.sort_values(
        ["split_id", "site", "source_video_id", "source_frame_index", "path"]
    )
    for _, row in ordered.iterrows():
        image_path = (root / str(row["path"])).resolve()
        label_path = label_path_for_image(image_path)
        records.append(
            {
                "image_path": image_path.relative_to(root).as_posix(),
                "image_sha256": str(row["image_sha256"]),
                "label_path": label_path.relative_to(root).as_posix(),
                "label_sha256": str(row["label_sha256"]),
                "source_frame_index": int(row["source_frame_index"]),
                "source_video_id": str(row["source_video_id"]),
                "source_video_sha256": str(row["source_video_sha256"]),
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


def validate_manifest_asset_bytes(manifest: pd.DataFrame, repository_root: Path) -> None:
    """Recheck bound image and label bytes without reapplying split-level requirements."""

    root = repository_root.resolve()
    for _, row in manifest.iterrows():
        image_path = (root / str(row["path"])).resolve()
        label_path = label_path_for_image(image_path)
        for asset_name, path, expected_hash in (
            ("image", image_path, str(row["image_sha256"])),
            ("label", label_path, str(row["label_sha256"])),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Detector {asset_name} disappeared during execution: {path}"
                )
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Detector {asset_name} changed during execution: {path}; "
                    f"expected {expected_hash}, observed {observed_hash}"
                )


def load_evaluation_config(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Load and validate the single immutable detector evaluation contract."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Detector evaluation config is missing: {config_path}")
    raw_config = config_path.read_bytes()
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Detector evaluation config is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("Detector evaluation config must be a JSON object")
    missing = sorted(EVALUATION_CONFIG_KEYS - set(config))
    extra = sorted(set(config) - EVALUATION_CONFIG_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unsupported: " + ", ".join(extra))
        raise ValueError("Detector evaluation config fields are invalid; " + "; ".join(details))
    if config["schema_version"] != EVALUATION_CONFIG_SCHEMA:
        raise ValueError(
            f"schema_version must be {EVALUATION_CONFIG_SCHEMA!r}, "
            f"not {config['schema_version']!r}"
        )

    resolved = dict(config)
    for field_name in (
        "manifest_path",
        "checkpoint_path",
        "training_provenance_path",
        "output_path",
        "receipt_path",
    ):
        resolved[field_name] = _repository_path(config[field_name], repository_root, field_name)
    for field_name in (
        "manifest_sha256",
        "checkpoint_sha256",
        "training_provenance_sha256",
        "test_assets_sha256",
    ):
        resolved[field_name] = _require_sha256(config[field_name], field_name)
    for field_name in (
        "inference_confidence_threshold",
        "inference_nms_iou",
        "matching_iou",
    ):
        value = config[field_name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field_name} must be numeric")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1]")
        if field_name == "matching_iou" and value == 0.0:
            raise ValueError("matching_iou must be greater than zero")
        resolved[field_name] = value
    seed = config["bootstrap_seed"]
    resamples = config["bootstrap_resamples"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    resolved["config_sha256"] = sha256_bytes(raw_config)

    path_values = {
        resolved[field_name]
        for field_name in (
            "manifest_path",
            "checkpoint_path",
            "training_provenance_path",
            "output_path",
            "receipt_path",
        )
    }
    if len(path_values) != 5:
        raise ValueError("Evaluation config paths must identify five distinct artefacts")
    if resolved["output_path"].suffix.lower() != ".csv":
        raise ValueError("output_path must name a CSV file")
    if resolved["receipt_path"].suffix.lower() != ".json":
        raise ValueError("receipt_path must name a JSON file")
    return resolved


def _verify_file_digest(path: Path, expected_digest: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Detector {description} is missing: {path}")
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            f"Detector {description} SHA-256 mismatch: expected {expected_digest}, "
            f"observed {observed_digest}"
        )


def validate_training_provenance(
    provenance_path: Path,
    *,
    repository_root: Path,
    manifest_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    library_versions: dict[str, str] | None = None,
    injected_fixture: bool = False,
) -> None:
    """Verify complete v4 lineage; historical v3 receipts cannot establish these identities."""

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Detector training provenance is not valid JSON") from exc
    if not isinstance(provenance, dict):
        raise ValueError("Detector training provenance must be a JSON object")
    if provenance.get("schema_version") != TRAINING_PROVENANCE_SCHEMA:
        raise ValueError("Detector training provenance has an unsupported schema")
    required = {
        "schema_version",
        "code_revision",
        "code_files_sha256",
        "execution_mode",
        "manifest_path",
        "manifest_sha256",
        "training_assets_sha256",
        "base_checkpoint",
        "output_checkpoints",
        "class_names",
        "training_parameters",
        "dataset_config",
        "ultralytics_version",
        "library_versions",
    }
    if set(provenance) != required:
        raise ValueError("Detector training provenance fields are incomplete or unsupported")
    mode = "injected_fixture" if injected_fixture else "production"
    if provenance["execution_mode"] != mode:
        raise ValueError("Detector training execution_mode is incompatible with this evaluation")
    revision = provenance["code_revision"]
    if not isinstance(revision, str) or GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("Detector training code_revision must be a full Git object identifier")
    validate_code_identity(
        provenance["code_files_sha256"],
        repository_root,
        revision,
        injected_fixture=injected_fixture,
    )
    versions = library_versions if library_versions is not None else detector_library_versions()
    validate_library_versions(versions, repository_root)
    validate_library_versions(
        provenance["library_versions"],
        repository_root,
        revision=None if injected_fixture else revision,
    )
    if provenance["ultralytics_version"] != provenance["library_versions"]["ultralytics"]:
        raise ValueError("Detector training ultralytics_version does not match library_versions")
    if provenance.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Detector training provenance does not match the evaluation manifest")
    _require_sha256(provenance["manifest_sha256"], "training manifest_sha256")
    manifest_path = _repository_path(provenance["manifest_path"], repository_root, "manifest_path")
    _verify_file_digest(manifest_path, manifest_sha256, "training manifest")
    manifest = load_split_manifest(manifest_path, repository_root)
    training_manifest = manifest.loc[manifest["split_id"].isin(["train", "validation"])]
    assets_digest = _require_sha256(provenance["training_assets_sha256"], "training_assets_sha256")
    if manifest_asset_digest(training_manifest, repository_root) != assets_digest:
        raise ValueError("Detector training source assets do not match their recorded digest")
    if provenance.get("class_names") != CLASS_NAMES:
        raise ValueError("Detector training provenance has an incompatible class mapping")
    if (
        provenance["dataset_config"] != DATASET_CONFIG
        or type(provenance["dataset_config"].get("nc")) is not int
    ):
        raise ValueError("Detector training dataset_config does not match the frozen configuration")
    parameters = provenance["training_parameters"]
    required_parameters = {*TRAINING_PARAMETERS, "seed", "project", "name"}
    if not isinstance(parameters, dict) or set(parameters) != required_parameters:
        raise ValueError("Detector training_parameters fields are incomplete or unsupported")
    for name, value in TRAINING_PARAMETERS.items():
        if type(parameters[name]) is not type(value) or parameters[name] != value:
            raise ValueError(
                f"Detector training parameter does not match the frozen setting: {name}"
            )
    seed = parameters["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Detector training seed must be a non-negative integer")
    if parameters["project"] != "runs" or parameters["name"] != "rtdetr_finetune_v3":
        raise ValueError(
            "Detector training output configuration does not match the frozen settings"
        )

    def verified_checkpoint(record: Any, description: str) -> Path:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(
                f"Detector training {description} fields are incomplete or unsupported"
            )
        path = _repository_path(record["path"], repository_root, description)
        digest = _require_sha256(record["sha256"], description)
        _verify_file_digest(path, digest, description)
        return path

    base_path = verified_checkpoint(provenance["base_checkpoint"], "base checkpoint")
    outputs = provenance.get("output_checkpoints")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("best"), dict):
        raise ValueError("Detector training provenance does not identify a best checkpoint")
    best = outputs["best"]
    expected_path = checkpoint_path.resolve().relative_to(repository_root.resolve()).as_posix()
    if best.get("path") != expected_path or best.get("sha256") != checkpoint_sha256:
        raise ValueError("Detector training provenance does not bind the evaluated checkpoint")
    if set(outputs) - {"best", "last"}:
        raise ValueError("Detector training output_checkpoints has unsupported roles")
    paths = [base_path, manifest_path]
    for role, record in outputs.items():
        paths.append(verified_checkpoint(record, f"{role} checkpoint"))
    if len(set(paths)) != len(paths):
        raise ValueError("Detector training checkpoints and manifest must identify distinct assets")


def read_ground_truth(label_path: Path) -> tuple[list[list[float]], list[int]]:
    """Read valid project YOLO boxes/polygons before either training or evaluation."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    with label_path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO annotation at {label_path}:{line_number}")
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"Malformed YOLO annotation at {label_path}:{line_number}"
                ) from exc
            if class_id not in CLASS_MAP:
                raise ValueError(
                    f"Unsupported project class_id {class_id} at {label_path}:{line_number}"
                )
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(
                    f"YOLO coordinates must be finite and in [0, 1] at {label_path}:{line_number}"
                )
            if len(coordinates) == 4:
                box = coordinates
            elif len(coordinates) > 4 and len(coordinates) % 2 == 0:
                x_values = coordinates[0::2]
                y_values = coordinates[1::2]
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                width = x_max - x_min
                height = y_max - y_min
                box = [x_min + width / 2.0, y_min + height / 2.0, width, height]
                vertices = list(zip(x_values, y_values, strict=True))
                area = sum(
                    first[0] * second[1] - second[0] * first[1]
                    for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=True)
                )
                if area == 0.0:
                    raise ValueError(f"Degenerate YOLO polygon at {label_path}:{line_number}")
            else:
                raise ValueError(f"Malformed YOLO annotation at {label_path}:{line_number}")
            x_min, y_min, x_max, y_max = xywhn_to_xyxy(box)
            if box[2] <= 0.0 or box[3] <= 0.0:
                raise ValueError(f"Degenerate YOLO box at {label_path}:{line_number}")
            if x_min < 0.0 or y_min < 0.0 or x_max > 1.0 or y_max > 1.0:
                raise ValueError(f"YOLO box is out of bounds at {label_path}:{line_number}")
            classes.append(class_id)
            boxes.append(box)
    return boxes, classes


def summarize_detection_results(
    image_results: Sequence[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Return per-site/class metrics with source-video-cluster bootstrap intervals."""

    if not image_results:
        raise ValueError("No detector test images were evaluated")
    if resamples <= 0:
        raise ValueError("resamples must be a positive integer")

    rows: list[dict[str, Any]] = []
    random_generator = random.Random(seed)
    sites = sorted({str(result["site"]) for result in image_results})
    for site in sites:
        site_results = [result for result in image_results if result["site"] == site]
        source_groups: dict[str, list[dict[str, Any]]] = {}
        for result in site_results:
            source_video_id = str(result.get("source_video_id", "")).strip()
            if not source_video_id:
                raise ValueError("Every detector result must record source_video_id")
            source_groups.setdefault(source_video_id, []).append(result)
        source_video_ids = sorted(source_groups)
        bootstrap_f1: dict[int, list[float]] = {class_id: [] for class_id in CLASS_MAP}
        for _ in range(resamples):
            selected_sources = [random_generator.choice(source_video_ids) for _ in source_video_ids]
            resampled_results = [
                result
                for source_video_id in selected_sources
                for result in source_groups[source_video_id]
            ]
            for class_id in CLASS_MAP:
                support = sum(result["gt_count"][class_id] for result in resampled_results)
                if support == 0:
                    bootstrap_f1[class_id].append(math.nan)
                    continue
                true_positives = sum(result["tp"][class_id] for result in resampled_results)
                false_positives = sum(result["fp"][class_id] for result in resampled_results)
                false_negatives = sum(result["fn"][class_id] for result in resampled_results)
                precision = (
                    true_positives / (true_positives + false_positives)
                    if true_positives + false_positives
                    else 0.0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if true_positives + false_negatives
                    else 0.0
                )
                f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                bootstrap_f1[class_id].append(f1)

        for class_id, class_name in CLASS_MAP.items():
            true_positives = sum(result["tp"][class_id] for result in site_results)
            false_positives = sum(result["fp"][class_id] for result in site_results)
            false_negatives = sum(result["fn"][class_id] for result in site_results)
            support = sum(result["gt_count"][class_id] for result in site_results)
            class_source_video_support = sum(
                any(result["gt_count"][class_id] > 0 for result in group_results)
                for group_results in source_groups.values()
            )
            valid_bootstrap = sorted(
                value for value in bootstrap_f1[class_id] if math.isfinite(value)
            )
            if support == 0:
                precision = recall = f1 = ci_low = ci_high = math.nan
                metric_status = "not_estimable_zero_support"
            else:
                precision = (
                    true_positives / (true_positives + false_positives)
                    if true_positives + false_positives
                    else 0.0
                )
                recall = true_positives / (true_positives + false_negatives)
                f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                if class_source_video_support < 2:
                    ci_low = ci_high = math.nan
                    metric_status = "interval_not_estimable_insufficient_source_clusters"
                elif valid_bootstrap:
                    lower_index = int((len(valid_bootstrap) - 1) * 0.025)
                    upper_index = int((len(valid_bootstrap) - 1) * 0.975)
                    ci_low = valid_bootstrap[lower_index]
                    ci_high = valid_bootstrap[upper_index]
                    metric_status = "estimated"
                else:
                    ci_low = ci_high = math.nan
                    metric_status = "interval_not_estimable"
            rows.append(
                {
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": true_positives,
                    "fp": false_positives,
                    "fn": false_negatives,
                    "support": support,
                    "source_video_support": len(source_video_ids),
                    "class_source_video_support": class_source_video_support,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "valid_bootstrap_resamples": len(valid_bootstrap),
                    "bootstrap_unit": "source_video_id",
                    "seed": seed,
                    "resamples": resamples,
                    "metric_status": metric_status,
                    "split_id": "test",
                    "site": site,
                }
            )
    return pd.DataFrame(rows)


def _write_immutable_outputs(outputs: dict[Path, bytes]) -> None:
    """Create a small output set exclusively and roll it back on write failure."""

    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Detector output already exists and is immutable: {path}. "
                "Create and commit a new evaluation config for a new version."
            )
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as file_handle:
                file_handle.write(content)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def run_evaluation(
    config_path: Path | str,
    *,
    repository_root: Path | None = None,
    model_factory: Callable[[str], Any] | None = None,
    code_revision: str | None = None,
    library_versions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Execute exactly one precommitted detector evaluation configuration."""

    root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    injected_fixture = model_factory is not None
    if not injected_fixture and (code_revision is not None or library_versions is not None):
        raise ValueError("Production evaluation cannot override code or library identity")
    selected_config_path = Path(config_path)
    if not selected_config_path.is_absolute():
        selected_config_path = root / selected_config_path
    selected_config_path = selected_config_path.resolve()
    if not selected_config_path.is_relative_to(root):
        raise ValueError("Detector evaluation config must remain under repository root")

    if code_revision is None:
        resolved_revision = repository_revision(root)
        require_tracked_file(selected_config_path, root)
    else:
        resolved_revision = code_revision
    if GIT_REVISION_PATTERN.fullmatch(resolved_revision) is None:
        raise ValueError("code_revision must be a full Git object identifier")
    source_identity = detector_code_identity(root)
    validate_code_identity(
        source_identity,
        root,
        resolved_revision,
        injected_fixture=injected_fixture,
        require_current=True,
    )
    versions = (
        dict(library_versions) if library_versions is not None else detector_library_versions()
    )
    validate_library_versions(versions, root)

    config = load_evaluation_config(selected_config_path, root)
    manifest_path = config["manifest_path"]
    model_path = config["checkpoint_path"]
    training_provenance_path = config["training_provenance_path"]
    output_path = config["output_path"]
    receipt_path = config["receipt_path"]
    if output_path.exists() or receipt_path.exists():
        existing = output_path if output_path.exists() else receipt_path
        raise FileExistsError(
            f"Detector output already exists and is immutable: {existing}. "
            "Create and commit a new evaluation config for a new version."
        )

    _verify_file_digest(manifest_path, config["manifest_sha256"], "manifest")
    _verify_file_digest(model_path, config["checkpoint_sha256"], "checkpoint")
    _verify_file_digest(
        training_provenance_path,
        config["training_provenance_sha256"],
        "training provenance",
    )
    validate_training_provenance(
        training_provenance_path,
        repository_root=root,
        manifest_sha256=config["manifest_sha256"],
        checkpoint_path=model_path,
        checkpoint_sha256=config["checkpoint_sha256"],
        library_versions=versions,
        injected_fixture=injected_fixture,
    )
    manifest = load_split_manifest(manifest_path, root)
    test_manifest = manifest.loc[manifest["split_id"] == "test"].copy()
    observed_test_assets_digest = manifest_asset_digest(test_manifest, root)
    if observed_test_assets_digest != config["test_assets_sha256"]:
        raise ValueError(
            "Detector test asset digest does not match the immutable evaluation config: "
            f"expected {config['test_assets_sha256']}, observed {observed_test_assets_digest}"
        )

    if model_factory is None:
        from ultralytics import RTDETR

        model_factory = RTDETR
    model = model_factory(str(model_path))
    validate_model_class_names(model)

    image_results: list[dict[str, Any]] = []
    for _, row in test_manifest.iterrows():
        image_path = (root / str(row["path"])).resolve()
        label_path = label_path_for_image(image_path)
        ground_truth_boxes, ground_truth_classes = read_ground_truth(label_path)

        predictions = model.predict(
            str(image_path),
            conf=config["inference_confidence_threshold"],
            iou=config["inference_nms_iou"],
            verbose=False,
        )
        predicted_boxes: list[list[float]] = []
        predicted_classes: list[int] = []
        predicted_confidences: list[float] = []
        if predictions and predictions[0].boxes is not None:
            boxes = predictions[0].boxes
            predicted_classes = boxes.cls.cpu().numpy().astype(int).tolist()
            predicted_boxes = boxes.xywhn.cpu().numpy().tolist()
            predicted_confidences = boxes.conf.cpu().numpy().tolist()

        true_positives, false_positives, false_negatives = match_image_predictions(
            predicted_boxes,
            predicted_classes,
            predicted_confidences,
            ground_truth_boxes,
            ground_truth_classes,
            iou_threshold=config["matching_iou"],
        )
        image_results.append(
            {
                "site": row["site"],
                "source_video_id": row["source_video_id"],
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "gt_count": {
                    class_id: ground_truth_classes.count(class_id) for class_id in CLASS_MAP
                },
            }
        )

    metrics = summarize_detection_results(
        image_results,
        resamples=config["bootstrap_resamples"],
        seed=config["bootstrap_seed"],
    )
    metrics["matching_iou"] = config["matching_iou"]
    metrics["inference_confidence_threshold"] = config["inference_confidence_threshold"]
    metrics["inference_nms_iou"] = config["inference_nms_iou"]
    metrics["interval_method"] = "source_video_cluster_percentile"
    metrics["confidence_level"] = 0.95
    metrics["manifest_sha256"] = config["manifest_sha256"]
    metrics["checkpoint_sha256"] = config["checkpoint_sha256"]
    metrics["test_assets_sha256"] = config["test_assets_sha256"]
    metrics["evaluation_config_sha256"] = config["config_sha256"]
    metrics["code_revision"] = resolved_revision

    _verify_file_digest(selected_config_path, config["config_sha256"], "evaluation config")
    _verify_file_digest(manifest_path, config["manifest_sha256"], "manifest")
    _verify_file_digest(model_path, config["checkpoint_sha256"], "checkpoint")
    _verify_file_digest(
        training_provenance_path,
        config["training_provenance_sha256"],
        "training provenance",
    )
    validate_manifest_asset_bytes(manifest, root)
    validate_training_provenance(
        training_provenance_path,
        repository_root=root,
        manifest_sha256=config["manifest_sha256"],
        checkpoint_path=model_path,
        checkpoint_sha256=config["checkpoint_sha256"],
        library_versions=versions,
        injected_fixture=injected_fixture,
    )
    if code_revision is None:
        final_revision = repository_revision(root)
        if final_revision != resolved_revision:
            raise RuntimeError("Git revision changed during detector evaluation")
    if detector_code_identity(root) != source_identity:
        raise RuntimeError("Detector source code changed during evaluation")
    if library_versions is None and detector_library_versions() != versions:
        raise RuntimeError("Detector library versions changed during evaluation")

    metrics_bytes = metrics.to_csv(index=False, float_format="%.17g").encode("utf-8")
    receipt = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "code_revision": resolved_revision,
        "code_files_sha256": source_identity,
        "library_versions": versions,
        "execution_mode": "injected_fixture" if injected_fixture else "production",
        "evaluation_config_path": selected_config_path.relative_to(root).as_posix(),
        "evaluation_config_sha256": config["config_sha256"],
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": config["manifest_sha256"],
        "checkpoint_path": model_path.relative_to(root).as_posix(),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "training_provenance_path": training_provenance_path.relative_to(root).as_posix(),
        "training_provenance_sha256": config["training_provenance_sha256"],
        "test_assets_sha256": config["test_assets_sha256"],
        "metrics_output_path": output_path.relative_to(root).as_posix(),
        "metrics_output_sha256": sha256_bytes(metrics_bytes),
        "metric_rows": len(metrics),
    }
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    _write_immutable_outputs(
        {
            output_path: metrics_bytes,
            receipt_path: receipt_bytes,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one immutable RT-DETR evaluation configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a committed detector_evaluation_config_v3 JSON file",
    )
    return parser.parse_args()


def main() -> None:
    metrics = run_evaluation(parse_args().config)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
