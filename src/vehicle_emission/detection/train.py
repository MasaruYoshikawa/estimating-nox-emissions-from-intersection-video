from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

from vehicle_emission.detection.evaluate import (
    GIT_REVISION_PATTERN,
    SHA256_PATTERN,
    SPLIT_DIRECTORY_NAMES,
    SUPPORTED_IMAGE_SUFFIXES,
    label_path_for_image,
    load_split_manifest,
    manifest_asset_digest,
    repository_revision,
    require_tracked_file,
    sha256_file,
    validate_manifest_asset_bytes,
    write_immutable_json,
)
from vehicle_emission.detection.lineage import (
    CLASS_NAMES,
    TRAINING_PARAMETERS,
    TRAINING_PROVENANCE_SCHEMA,
    detector_code_identity,
    detector_library_versions,
    validate_code_identity,
    validate_library_versions,
)


def build_training_dataset_config(dataset_root: Path) -> dict[str, Any]:
    """Build a config containing only the frozen training and validation splits."""

    train_images = dataset_root / "train" / "images"
    validation_images = dataset_root / "valid" / "images"
    missing = [path for path in (train_images, validation_images) if not path.is_dir()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required detector split directories are missing: {missing_text}")

    return {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }


def _assets(directory: Path, suffixes: frozenset[str]) -> set[Path]:
    if not directory.is_dir():
        return set()
    return {
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def validate_training_asset_coverage(
    manifest: pd.DataFrame,
    repository_root: Path,
    dataset_root: Path,
) -> None:
    """Require the source train/validation directories to exactly match the manifest."""

    root = repository_root.resolve()
    dataset = dataset_root.resolve()
    if not dataset.is_relative_to(root):
        raise ValueError(f"Detector dataset must remain under repository root: {dataset}")

    for split_id in ("train", "validation"):
        source_split = dataset / SPLIT_DIRECTORY_NAMES[split_id]
        expected_rows = manifest.loc[manifest["split_id"] == split_id]
        expected_images = {(root / str(path)).resolve() for path in expected_rows["path"]}
        expected_labels = {label_path_for_image(path) for path in expected_images}
        actual_images = _assets(source_split / "images", SUPPORTED_IMAGE_SUFFIXES)
        actual_labels = _assets(source_split / "labels", frozenset({".txt"}))

        missing_images = expected_images - actual_images
        extra_images = actual_images - expected_images
        missing_labels = expected_labels - actual_labels
        extra_labels = actual_labels - expected_labels
        if missing_images or extra_images or missing_labels or extra_labels:
            details = []
            for name, paths in (
                ("missing images", missing_images),
                ("extra images", extra_images),
                ("missing labels", missing_labels),
                ("extra labels", extra_labels),
            ):
                if paths:
                    preview = ", ".join(str(path.relative_to(root)) for path in sorted(paths)[:3])
                    details.append(f"{name}: {preview}")
            raise ValueError(
                f"Detector {split_id} assets do not exactly match the validated manifest; "
                + "; ".join(details)
            )


def prepare_training_dataset(
    manifest: pd.DataFrame,
    repository_root: Path,
    dataset_root: Path,
    staged_root: Path,
) -> None:
    """Copy exactly the manifested train/validation assets into a clean staging tree."""

    validate_training_asset_coverage(manifest, repository_root, dataset_root)
    root = repository_root.resolve()
    staged = staged_root.resolve()
    if staged.exists():
        raise FileExistsError(f"Training staging directory already exists: {staged}")

    for split_id in ("train", "validation"):
        destination_split = staged / SPLIT_DIRECTORY_NAMES[split_id]
        (destination_split / "images").mkdir(parents=True)
        (destination_split / "labels").mkdir(parents=True)
        split_rows = manifest.loc[manifest["split_id"] == split_id]
        for _, row in split_rows.iterrows():
            source_image = (root / str(row["path"])).resolve()
            source_label = label_path_for_image(source_image)
            destination_image = destination_split / "images" / source_image.name
            destination_label = destination_split / "labels" / source_label.name
            if destination_image.exists() or destination_label.exists():
                raise ValueError(
                    "Manifest assets collide after staging by basename: " f"{source_image.name}"
                )
            shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)
            if sha256_file(destination_image) != str(row["image_sha256"]):
                raise RuntimeError(f"Staged detector image hash mismatch: {source_image}")
            if sha256_file(destination_label) != str(row["label_sha256"]):
                raise RuntimeError(f"Staged detector label hash mismatch: {source_label}")


def resolve_training_checkpoint(
    *,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
) -> Path:
    """Require a local base checkpoint whose bytes match an explicit digest."""

    checkpoint = base_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training base checkpoint is missing: {checkpoint}")
    if SHA256_PATTERN.fullmatch(base_checkpoint_sha256) is None:
        raise ValueError(
            "The training base checkpoint requires a lowercase 64-character SHA-256 digest"
        )
    actual_digest = sha256_file(checkpoint)
    if actual_digest != base_checkpoint_sha256:
        raise ValueError(
            "Training base checkpoint SHA-256 does not match: "
            f"expected {base_checkpoint_sha256}, observed {actual_digest}"
        )
    return checkpoint


def _repository_relative(path: Path, repository_root: Path, description: str) -> str:
    resolved = path.resolve()
    root = repository_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{description} must remain under repository root: {resolved}")
    return resolved.relative_to(root).as_posix()


def _trained_checkpoints(model: Any, repository_root: Path) -> dict[str, Path]:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("RT-DETR did not expose its trainer checkpoint paths")
    checkpoints: dict[str, Path] = {}
    for role in ("best", "last"):
        value = getattr(trainer, role, None)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repository_root / path
        path = path.resolve()
        if path.is_file():
            _repository_relative(path, repository_root, f"Trained {role} checkpoint")
            checkpoints[role] = path
    if "best" not in checkpoints:
        raise RuntimeError("RT-DETR training did not produce a readable best checkpoint")
    return checkpoints


def train_model(
    *,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    provenance_output: Path,
    manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    seed: int = 42,
    repository_root: Path | None = None,
    model_factory: Any | None = None,
    code_revision: str | None = None,
    ultralytics_version: str | None = None,
    library_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train once from frozen inputs and write an immutable provenance receipt."""

    project_root = (repository_root or Path(__file__).resolve().parents[3]).resolve()
    injected_fixture = model_factory is not None
    if not injected_fixture and any(
        value is not None for value in (code_revision, ultralytics_version, library_versions)
    ):
        raise ValueError("Production training cannot override code or library identity")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Training seed must be a non-negative integer")
    selected_manifest = manifest_path or (
        project_root / "data" / "manifests" / "detection_split_v3.csv"
    )
    if not selected_manifest.is_absolute():
        selected_manifest = project_root / selected_manifest
    selected_dataset = dataset_root or (project_root / "data" / "manual_annotations")
    if not selected_dataset.is_absolute():
        selected_dataset = project_root / selected_dataset
    selected_base_checkpoint = base_checkpoint
    if not selected_base_checkpoint.is_absolute():
        selected_base_checkpoint = project_root / selected_base_checkpoint
    selected_provenance = provenance_output
    if not selected_provenance.is_absolute():
        selected_provenance = project_root / selected_provenance
    _repository_relative(selected_manifest, project_root, "Training manifest")
    _repository_relative(selected_dataset, project_root, "Training dataset")
    _repository_relative(selected_base_checkpoint, project_root, "Training base checkpoint")
    _repository_relative(selected_provenance, project_root, "Training provenance output")
    if selected_provenance.exists():
        raise FileExistsError(f"Training provenance output is immutable: {selected_provenance}")

    if code_revision is None:
        resolved_revision = repository_revision(project_root)
        require_tracked_file(selected_manifest, project_root)
    else:
        resolved_revision = code_revision
    if GIT_REVISION_PATTERN.fullmatch(resolved_revision) is None:
        raise ValueError("code_revision must be a full Git object identifier")
    source_identity = detector_code_identity(project_root)
    validate_code_identity(
        source_identity,
        project_root,
        resolved_revision,
        injected_fixture=injected_fixture,
        require_current=True,
    )
    versions = (
        dict(library_versions) if library_versions is not None else detector_library_versions()
    )
    validate_library_versions(versions, project_root)
    if ultralytics_version is not None and ultralytics_version != versions["ultralytics"]:
        raise ValueError("ultralytics_version does not match library_versions")

    if not selected_manifest.is_file():
        raise FileNotFoundError(f"Detector split manifest is missing: {selected_manifest}")
    manifest_sha256 = sha256_file(selected_manifest)
    manifest = load_split_manifest(selected_manifest, project_root)
    validate_training_asset_coverage(manifest, project_root, selected_dataset)
    training_manifest = manifest.loc[manifest["split_id"].isin(["train", "validation"])].copy()
    training_assets_sha256 = manifest_asset_digest(training_manifest, project_root)
    checkpoint = resolve_training_checkpoint(
        base_checkpoint=selected_base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )

    print("1. Preparing the manifest-bound training dataset...")
    data_directory = project_root / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="detector_training_", dir=data_directory) as temp:
        staged_root = Path(temp) / "dataset"
        prepare_training_dataset(manifest, project_root, selected_dataset, staged_root)

        print("2. Generating the four-class training configuration...")
        yaml_content = build_training_dataset_config(staged_root)
        yaml_path = staged_root / "detector_train.yaml"
        with yaml_path.open("w", encoding="utf-8") as file_handle:
            yaml.safe_dump(yaml_content, file_handle, sort_keys=False)

        print("3. Starting RT-DETR fine-tuning...")
        if model_factory is None:
            import ultralytics
            from ultralytics import RTDETR

            model_factory = RTDETR
            if str(ultralytics.__version__) != versions["ultralytics"]:
                raise RuntimeError(
                    "Imported Ultralytics does not match its recorded installed version"
                )

        model = model_factory(str(checkpoint))
        model.train(
            data=str(yaml_path),
            epochs=TRAINING_PARAMETERS["epochs"],
            imgsz=TRAINING_PARAMETERS["imgsz"],
            batch=TRAINING_PARAMETERS["batch"],
            seed=seed,
            deterministic=TRAINING_PARAMETERS["deterministic"],
            project=str(project_root / "runs"),
            name="rtdetr_finetune_v3",
        )
        if yaml.safe_load(yaml_path.read_text(encoding="utf-8")) != yaml_content:
            raise RuntimeError("Detector staged dataset configuration changed during training")
        for _, row in training_manifest.iterrows():
            image = (
                staged_root
                / SPLIT_DIRECTORY_NAMES[str(row["split_id"])]
                / "images"
                / Path(str(row["path"])).name
            )
            label = label_path_for_image(image)
            if (
                sha256_file(image) != row["image_sha256"]
                or sha256_file(label) != row["label_sha256"]
            ):
                raise RuntimeError("Detector staged source assets changed during training")

    if sha256_file(selected_manifest) != manifest_sha256:
        raise RuntimeError("Detector training manifest changed during training")
    if sha256_file(checkpoint) != base_checkpoint_sha256:
        raise RuntimeError("Detector base checkpoint changed during training")
    validate_manifest_asset_bytes(training_manifest, project_root)
    validate_training_asset_coverage(manifest, project_root, selected_dataset)
    if detector_code_identity(project_root) != source_identity:
        raise RuntimeError("Detector source code changed during training")
    if (
        code_revision is None
        and repository_revision(project_root, tracked_only=True) != resolved_revision
    ):
        raise RuntimeError("Git revision changed during detector training")
    if library_versions is None and detector_library_versions() != versions:
        raise RuntimeError("Detector library versions changed during training")
    trained_checkpoints = _trained_checkpoints(model, project_root)
    checkpoint_paths = [checkpoint, selected_manifest.resolve(), *trained_checkpoints.values()]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise RuntimeError("Detector trained checkpoints must identify distinct output assets")
    output_checkpoints = {
        role: {
            "path": _repository_relative(path, project_root, f"Trained {role} checkpoint"),
            "sha256": sha256_file(path),
        }
        for role, path in trained_checkpoints.items()
    }
    provenance = {
        "schema_version": TRAINING_PROVENANCE_SCHEMA,
        "code_revision": resolved_revision,
        "code_files_sha256": source_identity,
        "library_versions": versions,
        "execution_mode": "injected_fixture" if injected_fixture else "production",
        "manifest_path": _repository_relative(selected_manifest, project_root, "Training manifest"),
        "manifest_sha256": manifest_sha256,
        "training_assets_sha256": training_assets_sha256,
        "base_checkpoint": {
            "path": _repository_relative(checkpoint, project_root, "Training base checkpoint"),
            "sha256": base_checkpoint_sha256,
        },
        "output_checkpoints": output_checkpoints,
        "class_names": CLASS_NAMES,
        "training_parameters": {
            **TRAINING_PARAMETERS,
            "seed": seed,
            "project": "runs",
            "name": "rtdetr_finetune_v3",
        },
        "dataset_config": {
            "train": yaml_content["train"],
            "val": yaml_content["val"],
            "nc": yaml_content["nc"],
            "names": yaml_content["names"],
        },
        "ultralytics_version": versions["ultralytics"],
    }
    write_immutable_json(selected_provenance, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the four-class RT-DETR detector from a validated split manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/detection_split_v3.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/manual_annotations"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Local base checkpoint with independently recorded provenance",
    )
    parser.add_argument(
        "--base-checkpoint-sha256",
        required=True,
        help="Expected lowercase SHA-256 of --base-checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        required=True,
        help="New immutable detector_training_provenance_v4 JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        base_checkpoint=args.base_checkpoint,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
        provenance_output=args.provenance_output,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
