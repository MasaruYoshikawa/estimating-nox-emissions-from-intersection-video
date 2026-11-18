from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from vehicle_emission.detection.evaluate import (
    SplitManifestError,
    manifest_asset_digest,
    match_image_predictions,
    read_ground_truth,
    repository_revision,
    run_evaluation,
    sha256_bytes,
    sha256_file,
    summarize_detection_results,
    validate_model_class_names,
    validate_split_manifest,
    validate_training_provenance,
)
from vehicle_emission.detection.evaluate import (
    parse_args as parse_evaluation_args,
)
from vehicle_emission.detection.lineage import (
    CODE_PATHS,
    DATASET_CONFIG,
    TRAINING_PARAMETERS,
    TRAINING_PROVENANCE_SCHEMA,
    detector_code_identity,
    validate_code_identity,
)
from vehicle_emission.detection.train import (
    build_training_dataset_config,
    prepare_training_dataset,
    resolve_training_checkpoint,
    train_model,
    validate_training_asset_coverage,
)

SPLIT_DIRECTORIES = {"train": "train", "validation": "valid", "test": "test"}


def _code_fixture(root: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative_path in CODE_PATHS:
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative_path, destination)


def _library_fixture() -> dict[str, str]:
    return {
        "python": "3.11.12",
        "numpy": "2.1.3",
        "pandas": "2.2.3",
        "PyYAML": "6.0.2",
        "pillow": "11.0.0",
        "opencv-python": "4.10.0.84",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "ultralytics": "8.3.220",
    }


def _manifest_row(
    root: Path,
    *,
    split_id: str,
    site: str,
    source_video_id: str,
    frame_id: int = 1,
) -> dict[str, str]:
    split_directory = SPLIT_DIRECTORIES[split_id]
    image_path = (
        root
        / "annotations"
        / split_directory
        / "images"
        / f"{source_video_id}_frame_{frame_id:06d}.jpg"
    )
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_content = f"synthetic detector fixture {source_video_id} {frame_id}".encode()
    label_content = b"0 0.5 0.5 0.2 0.2\n"
    image_path.write_bytes(image_content)
    label_path.write_bytes(label_content)
    return {
        "path": image_path.relative_to(root).as_posix(),
        "split_id": split_id,
        "site": site,
        "source_video_id": source_video_id,
        "source_video_sha256": sha256_bytes(f"source:{source_video_id}".encode()),
        "source_frame_index": str(frame_id),
        "image_sha256": sha256_bytes(image_content),
        "label_sha256": sha256_bytes(label_content),
    }


def _valid_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for split_id in ("train", "validation", "test"):
        for site in ("Sukiyabashi", "Shinjuku"):
            rows.append(
                _manifest_row(
                    root,
                    split_id=split_id,
                    site=site,
                    source_video_id=f"{site.lower()}-{split_id}",
                )
            )
    return pd.DataFrame(rows)


def _image_result(
    *,
    site: str,
    source_video_id: str,
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    support: int = 0,
) -> dict[str, Any]:
    empty_counts = {class_id: 0 for class_id in range(4)}
    true_positives = empty_counts.copy()
    false_positives = empty_counts.copy()
    false_negatives = empty_counts.copy()
    ground_truth_count = empty_counts.copy()
    true_positives[0] = tp
    false_positives[0] = fp
    false_negatives[0] = fn
    ground_truth_count[0] = support
    return {
        "site": site,
        "source_video_id": source_video_id,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "gt_count": ground_truth_count,
    }


def test_training_config_never_exposes_test_as_train_or_validation(tmp_path: Path) -> None:
    for split_directory in ("train", "valid", "test"):
        (tmp_path / split_directory / "images").mkdir(parents=True)

    config = build_training_dataset_config(tmp_path)

    assert config["train"] == "train/images"
    assert config["val"] == "valid/images"
    assert "test" not in config
    assert "test/images" not in config.values()


def test_valid_manifest_has_isolated_splits_and_both_test_sites(tmp_path: Path) -> None:
    validated = validate_split_manifest(_valid_manifest(tmp_path), tmp_path)

    assert set(validated["split_id"]) == {"train", "validation", "test"}
    assert set(validated.loc[validated["split_id"] == "test", "site"]) == {
        "Sukiyabashi",
        "Shinjuku",
    }


def test_source_video_cannot_cross_splits(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    validation_index = manifest.index[manifest["split_id"] == "validation"][0]
    train_source = manifest.loc[manifest["split_id"] == "train", "source_video_id"].iloc[0]
    manifest.loc[validation_index, "source_video_id"] = train_source

    with pytest.raises(SplitManifestError, match="Source-video leakage"):
        validate_split_manifest(manifest, tmp_path)


@pytest.mark.parametrize(
    "missing_column",
    [
        "split_id",
        "site",
        "source_video_id",
        "source_video_sha256",
        "source_frame_index",
        "image_sha256",
        "label_sha256",
    ],
)
def test_required_manifest_metadata_is_fail_closed(tmp_path: Path, missing_column: str) -> None:
    manifest = _valid_manifest(tmp_path).drop(columns=[missing_column])

    with pytest.raises(SplitManifestError, match=missing_column):
        validate_split_manifest(manifest, tmp_path)


def test_all_three_splits_are_required(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest = manifest.loc[manifest["split_id"] != "validation"]

    with pytest.raises(SplitManifestError, match="missing required splits: validation"):
        validate_split_manifest(manifest, tmp_path)


@pytest.mark.parametrize("split_id", ["train", "validation", "test"])
def test_both_sites_are_required_in_every_split(tmp_path: Path, split_id: str) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest = manifest.loc[
        ~((manifest["split_id"] == split_id) & (manifest["site"] == "Shinjuku"))
    ]

    with pytest.raises(
        SplitManifestError,
        match=rf"Detector {split_id} split is missing required sites: Shinjuku",
    ):
        validate_split_manifest(manifest, tmp_path)


def test_manifest_split_id_must_match_physical_directory(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    row_index = manifest.index[manifest["split_id"] == "validation"][0]
    manifest.loc[row_index, "path"] = str(manifest.loc[row_index, "path"]).replace(
        "/valid/", "/train/"
    )

    with pytest.raises(SplitManifestError, match="does not match image path"):
        validate_split_manifest(manifest, tmp_path, require_assets=False)


@pytest.mark.parametrize(
    "path_value",
    [
        r"annotations\train\images\frame.jpg",
        "annotations/train/images/../images/frame.jpg",
        "./annotations/train/images/frame.jpg",
        "annotations//train/images/frame.jpg",
    ],
)
def test_manifest_rejects_noncanonical_path_aliases(
    tmp_path: Path,
    path_value: str,
) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest.loc[0, "path"] = path_value

    with pytest.raises(SplitManifestError, match="canonical"):
        validate_split_manifest(manifest, tmp_path, require_assets=False)


def test_manifest_rejects_assets_that_resolve_to_same_file(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    original = tmp_path / str(manifest.loc[0, "path"])
    alias = original.with_name("alias.jpg")
    try:
        alias.symlink_to(original.name)
    except OSError as error:
        pytest.skip(f"Symlinks not supported: {error}")
    duplicate = manifest.loc[0].copy()
    duplicate["path"] = alias.relative_to(tmp_path).as_posix()
    duplicate["source_frame_index"] = 2
    manifest = pd.concat([manifest, duplicate.to_frame().T], ignore_index=True)

    with pytest.raises(SplitManifestError, match="resolve to the same asset"):
        validate_split_manifest(manifest, tmp_path)


def test_source_video_id_and_hash_have_one_to_one_identity(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest.loc[1, "source_video_sha256"] = manifest.loc[0, "source_video_sha256"]

    with pytest.raises(SplitManifestError, match="hashes map to multiple source IDs"):
        validate_split_manifest(manifest, tmp_path)


def test_identical_image_bytes_cannot_cross_splits(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    train_hash = manifest.loc[manifest["split_id"] == "train", "image_sha256"].iloc[0]
    validation_index = manifest.index[manifest["split_id"] == "validation"][0]
    manifest.loc[validation_index, "image_sha256"] = train_hash

    with pytest.raises(SplitManifestError, match="Identical image bytes cannot cross"):
        validate_split_manifest(manifest, tmp_path, require_assets=False)


def test_manifest_hashes_are_checked_against_asset_bytes(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    image_path = tmp_path / str(manifest.loc[0, "path"])
    image_path.write_bytes(b"changed after manifest creation")

    with pytest.raises(SplitManifestError, match="Image SHA-256 mismatch"):
        validate_split_manifest(manifest, tmp_path)


def test_missing_image_or_label_fails_before_model_loading(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    missing_image = tmp_path / str(manifest.loc[0, "path"])
    missing_image.unlink()

    with pytest.raises(FileNotFoundError, match="Detector evaluation assets are missing"):
        validate_split_manifest(manifest, tmp_path)


def test_training_asset_coverage_rejects_unmanifested_files(tmp_path: Path) -> None:
    manifest = validate_split_manifest(_valid_manifest(tmp_path), tmp_path)
    extra_image = tmp_path / "annotations" / "train" / "images" / "extra.jpg"
    extra_label = tmp_path / "annotations" / "train" / "labels" / "extra.txt"
    extra_image.write_bytes(b"unmanifested")
    extra_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra images"):
        validate_training_asset_coverage(manifest, tmp_path, tmp_path / "annotations")


def test_training_stage_contains_only_manifested_train_and_validation_assets(
    tmp_path: Path,
) -> None:
    manifest = validate_split_manifest(_valid_manifest(tmp_path), tmp_path)
    staged_root = tmp_path / "staged"

    prepare_training_dataset(
        manifest,
        tmp_path,
        tmp_path / "annotations",
        staged_root,
    )

    assert len(list((staged_root / "train" / "images").glob("*.jpg"))) == 2
    assert len(list((staged_root / "valid" / "images").glob("*.jpg"))) == 2
    assert not (staged_root / "test").exists()


def test_training_checkpoint_must_be_local_and_digest_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"recorded pristine base checkpoint")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        resolve_training_checkpoint(
            base_checkpoint=checkpoint,
            base_checkpoint_sha256="0" * 64,
        )
    assert (
        resolve_training_checkpoint(
            base_checkpoint=checkpoint,
            base_checkpoint_sha256=sha256_file(checkpoint),
        )
        == checkpoint.resolve()
    )


def test_unknown_predicted_class_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown class IDs"):
        match_image_predictions(
            pred_boxes=[[0.5, 0.5, 0.2, 0.2]],
            pred_cls=[80],
            pred_confs=[0.9],
            gt_boxes=[],
            gt_cls=[],
        )


def test_checkpoint_class_names_must_match_four_class_contract() -> None:
    valid_model = SimpleNamespace(names={0: "bus", 1: "car", 2: "heavy_truck", 3: "light_truck"})
    validate_model_class_names(valid_model)

    invalid_model = SimpleNamespace(names={0: "person", 1: "car"})
    with pytest.raises(SplitManifestError, match="checkpoint classes"):
        validate_model_class_names(invalid_model)

    with pytest.raises(SplitManifestError, match="does not expose its class mapping"):
        validate_model_class_names(SimpleNamespace())


def test_metrics_are_reported_for_each_manifest_site() -> None:
    results = [
        _image_result(
            site="Sukiyabashi",
            source_video_id="sukiyabashi-test",
            tp=2,
            support=2,
        ),
        _image_result(
            site="Shinjuku",
            source_video_id="shinjuku-test",
            tp=1,
            support=1,
        ),
    ]

    metrics = summarize_detection_results(results, resamples=20, seed=7)

    assert set(metrics["site"]) == {"Sukiyabashi", "Shinjuku"}
    assert len(metrics) == 8
    bus_rows = metrics.loc[metrics["class_name"] == "bus"].set_index("site")
    assert bus_rows.loc["Sukiyabashi", "support"] == 2
    assert bus_rows.loc["Shinjuku", "support"] == 1
    assert set(bus_rows["metric_status"]) == {"interval_not_estimable_insufficient_source_clusters"}
    assert bus_rows["ci_low"].isna().all()


def test_bootstrap_resamples_source_videos_not_individual_frames() -> None:
    results = [
        _image_result(
            site="Sukiyabashi",
            source_video_id="source-a",
            tp=1,
            support=1,
        )
        for _ in range(20)
    ]
    results.append(
        _image_result(
            site="Sukiyabashi",
            source_video_id="source-b",
            fn=1,
            support=1,
        )
    )

    metrics = summarize_detection_results(results, resamples=400, seed=19)
    bus = metrics.loc[metrics["class_name"] == "bus"].iloc[0]

    assert bus["bootstrap_unit"] == "source_video_id"
    assert bus["source_video_support"] == 2
    assert bus["class_source_video_support"] == 2
    assert bus["ci_low"] == 0.0
    assert bus["ci_high"] == 1.0
    assert bus["metric_status"] == "estimated"


def test_f1_is_not_rounded_up_to_the_acceptance_threshold() -> None:
    results = [
        _image_result(
            site="Sukiyabashi",
            source_video_id="source-a",
            tp=2250,
            fp=250,
            fn=250,
            support=2500,
        ),
        _image_result(
            site="Sukiyabashi",
            source_video_id="source-b",
            tp=2249,
            fp=250,
            fn=250,
            support=2499,
        ),
    ]

    metrics = summarize_detection_results(results, resamples=20, seed=7)
    f1 = metrics.loc[metrics["class_name"] == "bus", "f1"].iloc[0]

    assert 0.89995 < f1 < 0.9
    assert f1 != round(f1, 4)


def test_zero_support_class_site_metrics_are_not_estimable() -> None:
    result = _image_result(
        site="Shinjuku",
        source_video_id="source-a",
        support=1,
        fn=1,
    )
    result["fp"][1] = 3

    metrics = summarize_detection_results([result], resamples=20, seed=7)
    car = metrics.loc[metrics["class_name"] == "car"].iloc[0]

    assert car["metric_status"] == "not_estimable_zero_support"
    assert pd.isna(car["precision"])
    assert pd.isna(car["recall"])
    assert pd.isna(car["f1"])
    assert pd.isna(car["ci_low"])
    assert pd.isna(car["ci_high"])


def _evaluation_config_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    _code_fixture(tmp_path)
    manifest_path = tmp_path / "detection_split_v3.csv"
    manifest = _valid_manifest(tmp_path)
    manifest.to_csv(manifest_path, index=False)
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint fixture")
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"verified base checkpoint fixture")
    training_provenance_path = tmp_path / "detector_training_provenance_v3.json"
    training_provenance = {
        "schema_version": TRAINING_PROVENANCE_SCHEMA,
        "code_revision": "a" * 40,
        "code_files_sha256": detector_code_identity(tmp_path),
        "execution_mode": "injected_fixture",
        "library_versions": _library_fixture(),
        "ultralytics_version": "8.3.220",
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "training_assets_sha256": manifest_asset_digest(
            manifest.loc[manifest["split_id"].isin(["train", "validation"])], tmp_path
        ),
        "base_checkpoint": {"path": "base.pt", "sha256": sha256_file(base_checkpoint)},
        "dataset_config": DATASET_CONFIG,
        "training_parameters": {
            **TRAINING_PARAMETERS,
            "seed": 42,
            "project": "runs",
            "name": "rtdetr_finetune_v3",
        },
        "class_names": ["bus", "car", "heavy truck", "light truck"],
        "output_checkpoints": {
            "best": {
                "path": checkpoint_path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(checkpoint_path),
            }
        },
    }
    training_provenance_path.write_text(
        json.dumps(training_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "phase3_detection_metrics_v3.csv"
    receipt_path = tmp_path / "phase3_detection_receipt_v3.json"
    config_path = tmp_path / "detection_evaluation_config_v3.json"
    validated = validate_split_manifest(manifest, tmp_path)
    test_manifest = validated.loc[validated["split_id"] == "test"]
    config = {
        "schema_version": "detector_evaluation_config_v3",
        "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": checkpoint_path.relative_to(tmp_path).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_provenance_path": training_provenance_path.relative_to(tmp_path).as_posix(),
        "training_provenance_sha256": sha256_file(training_provenance_path),
        "output_path": output_path.relative_to(tmp_path).as_posix(),
        "receipt_path": receipt_path.relative_to(tmp_path).as_posix(),
        "inference_confidence_threshold": 0.2,
        "inference_nms_iou": 0.3,
        "matching_iou": 0.4,
        "bootstrap_seed": 17,
        "bootstrap_resamples": 20,
        "test_assets_sha256": manifest_asset_digest(test_manifest, tmp_path),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path, config


def test_evaluation_uses_only_frozen_config_and_writes_provenance_receipt(
    tmp_path: Path,
) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    prediction_calls: list[dict[str, Any]] = []

    class FakeModel:
        names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}

        def predict(self, image_path: str, **kwargs: Any) -> list[SimpleNamespace]:
            prediction_calls.append({"image_path": image_path, **kwargs})
            return [SimpleNamespace(boxes=None)]

    metrics = run_evaluation(
        config_path,
        repository_root=tmp_path,
        model_factory=lambda _: FakeModel(),
        code_revision="b" * 40,
        library_versions=_library_fixture(),
    )

    assert len(prediction_calls) == 2
    assert {call["conf"] for call in prediction_calls} == {0.2}
    assert {call["iou"] for call in prediction_calls} == {0.3}
    assert set(metrics["matching_iou"]) == {0.4}
    assert set(metrics["inference_confidence_threshold"]) == {0.2}
    assert set(metrics["inference_nms_iou"]) == {0.3}
    assert set(metrics["manifest_sha256"]) == {config["manifest_sha256"]}
    assert set(metrics["checkpoint_sha256"]) == {config["checkpoint_sha256"]}
    assert set(metrics["test_assets_sha256"]) == {config["test_assets_sha256"]}
    assert set(metrics["evaluation_config_sha256"]) == {sha256_file(config_path)}
    assert set(metrics["code_revision"]) == {"b" * 40}
    assert set(metrics["interval_method"]) == {"source_video_cluster_percentile"}
    assert set(metrics["confidence_level"]) == {0.95}
    receipt_path = tmp_path / config["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "detector_evaluation_receipt_v3"
    assert receipt["evaluation_config_sha256"] == sha256_file(config_path)
    assert receipt["training_provenance_sha256"] == config["training_provenance_sha256"]
    assert receipt["metrics_output_sha256"] == sha256_file(tmp_path / config["output_path"])

    with pytest.raises(FileExistsError, match="immutable"):
        run_evaluation(
            config_path,
            repository_root=tmp_path,
            model_factory=lambda _: FakeModel(),
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )


def test_evaluation_rejects_config_hash_mismatch_before_model_loading(tmp_path: Path) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    config["manifest_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    model_loaded = False

    def model_factory(_: str) -> None:
        nonlocal model_loaded
        model_loaded = True

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        run_evaluation(
            config_path,
            repository_root=tmp_path,
            model_factory=model_factory,
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )
    assert not model_loaded


def test_evaluation_rejects_zero_matching_iou_before_model_loading(tmp_path: Path) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    config["matching_iou"] = 0.0
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    model_loaded = False

    def model_factory(_: str) -> None:
        nonlocal model_loaded
        model_loaded = True

    with pytest.raises(ValueError, match="matching_iou must be greater than zero"):
        run_evaluation(
            config_path,
            repository_root=tmp_path,
            model_factory=model_factory,
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )
    assert not model_loaded


def test_evaluation_rejects_unbound_training_lineage_before_model_loading(
    tmp_path: Path,
) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    provenance_path = tmp_path / config["training_provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output_checkpoints"]["best"]["sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    config["training_provenance_sha256"] = sha256_file(provenance_path)
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    model_loaded = False

    def model_factory(_: str) -> None:
        nonlocal model_loaded
        model_loaded = True

    with pytest.raises(ValueError, match="does not bind the evaluated checkpoint"):
        run_evaluation(
            config_path,
            repository_root=tmp_path,
            model_factory=model_factory,
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )
    assert not model_loaded


def test_production_revision_rejects_dirty_git_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Detector Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "fixture"],
        check=True,
    )

    assert len(repository_revision(tmp_path)) == 40
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git working tree"):
        repository_revision(tmp_path)


def test_evaluation_cli_rejects_method_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--config", "config.json", "--match-iou", "0.7"],
    )

    with pytest.raises(SystemExit):
        parse_evaluation_args()


def test_training_writes_immutable_checkpoint_lineage(tmp_path: Path) -> None:
    _code_fixture(tmp_path)
    manifest_path = tmp_path / "detection_split_v3.csv"
    _valid_manifest(tmp_path).to_csv(manifest_path, index=False)
    base_checkpoint = tmp_path / "models" / "rtdetr-l.pt"
    base_checkpoint.parent.mkdir()
    base_checkpoint.write_bytes(b"verified local base")
    provenance_path = tmp_path / "runs" / "training_provenance_v3.json"
    captured_training_args: dict[str, Any] = {}

    class FakeTrainingModel:
        def __init__(self) -> None:
            self.trainer: SimpleNamespace | None = None

        def train(self, **kwargs: Any) -> None:
            captured_training_args.update(kwargs)
            weights = tmp_path / "runs" / "fake" / "weights"
            weights.mkdir(parents=True)
            best = weights / "best.pt"
            last = weights / "last.pt"
            best.write_bytes(b"trained best")
            last.write_bytes(b"trained last")
            self.trainer = SimpleNamespace(best=best, last=last)

    provenance = train_model(
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=sha256_file(base_checkpoint),
        provenance_output=provenance_path,
        manifest_path=manifest_path,
        dataset_root=tmp_path / "annotations",
        seed=23,
        repository_root=tmp_path,
        model_factory=lambda _: FakeTrainingModel(),
        code_revision="c" * 40,
        ultralytics_version="8.3.220",
        library_versions=_library_fixture(),
    )

    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert provenance["manifest_sha256"] == sha256_file(manifest_path)
    assert provenance["base_checkpoint"]["sha256"] == sha256_file(base_checkpoint)
    assert provenance["training_parameters"]["seed"] == 23
    assert provenance["output_checkpoints"]["best"]["sha256"] == sha256_file(
        tmp_path / provenance["output_checkpoints"]["best"]["path"]
    )
    assert captured_training_args["seed"] == 23
    assert captured_training_args["deterministic"] is True
    assert provenance["schema_version"] == "detector_training_provenance_v4"
    assert provenance["execution_mode"] == "injected_fixture"
    assert provenance["code_files_sha256"] == detector_code_identity(tmp_path)
    assert provenance["library_versions"] == _library_fixture()

    with pytest.raises(FileExistsError, match="immutable"):
        train_model(
            base_checkpoint=base_checkpoint,
            base_checkpoint_sha256=sha256_file(base_checkpoint),
            provenance_output=provenance_path,
            manifest_path=manifest_path,
            dataset_root=tmp_path / "annotations",
            repository_root=tmp_path,
            model_factory=lambda _: FakeTrainingModel(),
            code_revision="c" * 40,
        )


def _rewrite_lineage(root: Path, config_path: Path, config: dict[str, Any], lineage: Any) -> None:
    path = root / config["training_provenance_path"]
    path.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")
    config["training_provenance_sha256"] = sha256_file(path)
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")


def _assert_lineage_rejected(root: Path, config_path: Path, message: str) -> None:
    def unexpected_model(_: str) -> None:
        pytest.fail("Invalid lineage or annotations reached model loading")

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run_evaluation(
            config_path,
            repository_root=root,
            model_factory=unexpected_model,
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_incomplete_training_lineage_is_rejected_before_loading(tmp_path: Path, field: str) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    del lineage[field]
    _rewrite_lineage(tmp_path, config_path, config, lineage)
    _assert_lineage_rejected(tmp_path, config_path, "fields are incomplete")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "detector_training_provenance_v3", "unsupported schema"),
        ("code_revision", "abc123", "full Git"),
        ("code_files_sha256", {}, "code_files_sha256 fields"),
        ("execution_mode", "production", "execution_mode"),
        ("manifest_path", "absent.csv", "training manifest is missing"),
        ("manifest_sha256", "0" * 64, "evaluation manifest"),
        ("training_assets_sha256", "0" * 64, "source assets"),
        ("base_checkpoint", {}, "base checkpoint fields"),
        ("output_checkpoints", {}, "best checkpoint"),
        ("class_names", ["car"], "class mapping"),
        ("dataset_config", {"train": "test/images"}, "dataset_config"),
        ("training_parameters", {}, "training_parameters fields"),
        ("library_versions", {}, "library_versions fields"),
        ("ultralytics_version", "8.0.0", "ultralytics_version"),
    ],
)
def test_forged_training_lineage_is_rejected(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    lineage[field] = value
    _rewrite_lineage(tmp_path, config_path, config, lineage)
    _assert_lineage_rejected(tmp_path, config_path, message)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 51),
        ("imgsz", 320),
        ("batch", 8),
        ("deterministic", False),
        ("deterministic", 1),
        ("seed", -1),
        ("seed", True),
        ("seed", 42.0),
        ("project", "other_runs"),
        ("name", "other_training"),
    ],
)
def test_training_settings_are_strictly_validated(tmp_path: Path, field: str, value: Any) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    lineage["training_parameters"][field] = value
    _rewrite_lineage(tmp_path, config_path, config, lineage)
    _assert_lineage_rejected(tmp_path, config_path, "training (parameter|seed|output)")


@pytest.mark.parametrize("asset", ["base", "best", "last", "training_image", "training_label"])
def test_all_training_assets_and_declared_outputs_bind_actual_bytes(
    tmp_path: Path, asset: str
) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    if asset == "last":
        last = tmp_path / "last.pt"
        last.write_bytes(b"declared last checkpoint")
        lineage["output_checkpoints"]["last"] = {"path": "last.pt", "sha256": sha256_file(last)}
        _rewrite_lineage(tmp_path, config_path, config, lineage)
        changed = last
    elif asset == "base":
        changed = tmp_path / lineage["base_checkpoint"]["path"]
    elif asset == "best":
        changed = tmp_path / config["checkpoint_path"]
    else:
        manifest = pd.read_csv(tmp_path / config["manifest_path"])
        changed = tmp_path / manifest.loc[0, "path"]
        if asset == "training_label":
            changed = changed.parent.parent / "labels" / f"{changed.stem}.txt"
    changed.write_bytes(b"mutated after training")
    _assert_lineage_rejected(tmp_path, config_path, "SHA-256 mismatch")


@pytest.mark.parametrize(
    "name",
    [
        "python",
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "pandas",
        "PyYAML",
        "pillow",
        "opencv-python",
    ],
)
def test_training_library_versions_must_satisfy_recorded_pins(tmp_path: Path, name: str) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    lineage["library_versions"][name] = "0.0.0"
    _rewrite_lineage(tmp_path, config_path, config, lineage)
    _assert_lineage_rejected(tmp_path, config_path, "(Python 3.11|violates the project pin)")


@pytest.mark.parametrize(
    "annotation",
    [
        "0 nan 0.5 0.2 0.2",
        "0 0.5 inf 0.2 0.2",
        "0 0.5 0.5 -inf 0.2",
        "0 -0.1 0.5 0.2 0.2",
        "0 0.5 -0.1 0.2 0.2",
        "0 0.5 0.5 -0.2 0.2",
        "0 0.5 0.5 0.2 -0.2",
        "0 1.1 0.5 0.2 0.2",
        "0 0.5 1.1 0.2 0.2",
        "0 0.5 0.5 1.1 0.2",
        "0 0.5 0.5 0.2 1.1",
        "0 0.5 0.5 0 0.2",
        "0 0.5 0.5 0.2 0",
        "0 0.05 0.5 0.2 0.2",
        "0 0.95 0.5 0.2 0.2",
        "0 0.5 0.05 0.2 0.2",
        "0 0.5 0.95 0.2 0.2",
        "4 0.5 0.5 0.2 0.2",
        "-1 0.5 0.5 0.2 0.2",
        "1.5 0.5 0.5 0.2 0.2",
        "car 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0.2",
        "0 0.5 0.5 0.2 0.2 0.1",
        "0 bad 0.5 0.2 0.2",
        "0 0 0 0.5 0.5 1 1",
        "0 0 0 0 0.5 0 1",
        "0 0 0 1 0 1 nan",
        "0 0 0 1 0 1 1.1",
    ],
)
def test_invalid_annotation_boundaries_are_rejected(tmp_path: Path, annotation: str) -> None:
    label = tmp_path / "label.txt"
    label.write_text(annotation + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(YOLO|class_id)"):
        read_ground_truth(label)


@pytest.mark.parametrize(
    "annotation",
    [
        "",
        "0 0.5 0.5 1 1",
        "3 0.1 0.1 0.2 0.2",
        "1 0 0 1 0 1 1 0 1",
    ],
)
def test_valid_empty_boundary_boxes_and_polygons_are_retained(
    tmp_path: Path, annotation: str
) -> None:
    label = tmp_path / "label.txt"
    label.write_text(annotation, encoding="utf-8")
    boxes, classes = read_ground_truth(label)
    assert len(boxes) == len(classes) == (1 if annotation else 0)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_annotation_parser_runs_before_any_model_is_loaded(tmp_path: Path, split: str) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    manifest_path = tmp_path / config["manifest_path"]
    manifest = pd.read_csv(manifest_path)
    index = manifest.index[manifest["split_id"] == split][0]
    image = tmp_path / manifest.loc[index, "path"]
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    label.write_text("0 0.05 0.5 0.2 0.2\n", encoding="utf-8")
    manifest.loc[index, "label_sha256"] = sha256_file(label)
    manifest.to_csv(manifest_path, index=False)
    config["manifest_sha256"] = sha256_file(manifest_path)
    lineage = json.loads((tmp_path / config["training_provenance_path"]).read_text())
    lineage["manifest_sha256"] = config["manifest_sha256"]
    _rewrite_lineage(tmp_path, config_path, config, lineage)
    _assert_lineage_rejected(tmp_path, config_path, "out of bounds")
    with pytest.raises(ValueError, match="out of bounds"):
        train_model(
            base_checkpoint=tmp_path / "base.pt",
            base_checkpoint_sha256=sha256_file(tmp_path / "base.pt"),
            provenance_output=tmp_path / "new_provenance.json",
            manifest_path=manifest_path,
            dataset_root=tmp_path / "annotations",
            repository_root=tmp_path,
            model_factory=lambda _: pytest.fail("Invalid annotations reached training"),
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )


@pytest.mark.parametrize("mutation", ["code", "manifest", "base", "source", "staged", "config"])
def test_training_mutation_cannot_publish_lineage(tmp_path: Path, mutation: str) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    provenance_output = tmp_path / "new_provenance.json"

    class MutatingModel:
        def train(self, **kwargs: Any) -> None:
            if mutation == "code":
                changed = tmp_path / "src/vehicle_emission/detection/train.py"
            elif mutation == "manifest":
                changed = tmp_path / config["manifest_path"]
            elif mutation == "base":
                changed = tmp_path / "base.pt"
            elif mutation == "source":
                manifest = pd.read_csv(tmp_path / config["manifest_path"])
                changed = tmp_path / manifest.loc[0, "path"]
            elif mutation == "staged":
                changed = next((Path(kwargs["data"]).parent / "train" / "images").iterdir())
            else:
                changed = Path(kwargs["data"])
            changed.write_bytes(b"changed during mocked training")
            self.trainer = SimpleNamespace(best=tmp_path / "model.pt")

    with pytest.raises(RuntimeError, match="changed during (training|execution)"):
        train_model(
            base_checkpoint=tmp_path / "base.pt",
            base_checkpoint_sha256=sha256_file(tmp_path / "base.pt"),
            provenance_output=provenance_output,
            manifest_path=tmp_path / config["manifest_path"],
            dataset_root=tmp_path / "annotations",
            repository_root=tmp_path,
            model_factory=lambda _: MutatingModel(),
            code_revision="b" * 40,
            library_versions=_library_fixture(),
        )
    assert not provenance_output.exists()


def test_historical_training_code_and_python_patch_allow_a_new_evaluator(tmp_path: Path) -> None:
    _, config = _evaluation_config_fixture(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", *CODE_PATHS], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "historical training source fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance_path = tmp_path / config["training_provenance_path"]
    lineage = json.loads(provenance_path.read_text())
    lineage["code_revision"] = revision
    lineage["execution_mode"] = "production"
    lineage["library_versions"]["python"] = "3.11.1"
    provenance_path.write_text(json.dumps(lineage), encoding="utf-8")
    evaluator = tmp_path / "src/vehicle_emission/detection/evaluate.py"
    evaluator.write_text(evaluator.read_text() + "\n# later evaluator fix\n", encoding="utf-8")
    validate_training_provenance(
        provenance_path,
        repository_root=tmp_path,
        manifest_sha256=config["manifest_sha256"],
        checkpoint_path=tmp_path / config["checkpoint_path"],
        checkpoint_sha256=config["checkpoint_sha256"],
        library_versions=_library_fixture(),
    )
    lineage["code_files_sha256"]["src/vehicle_emission/detection/evaluate.py"] = sha256_file(
        evaluator
    )
    provenance_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded revision"):
        validate_training_provenance(
            provenance_path,
            repository_root=tmp_path,
            manifest_sha256=config["manifest_sha256"],
            checkpoint_path=tmp_path / config["checkpoint_path"],
            checkpoint_sha256=config["checkpoint_sha256"],
            library_versions=_library_fixture(),
        )


@pytest.mark.parametrize("mutation", ["tracked_file", "revision"])
def test_training_rechecks_tracked_worktree_and_revision(tmp_path: Path, mutation: str) -> None:
    _, config = _evaluation_config_fixture(tmp_path)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    commit_command = [
        "git",
        "-C",
        str(tmp_path),
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    ]
    subprocess.run(commit_command, check=True)

    class MutatingModel:
        def train(self, **kwargs: Any) -> None:
            if mutation == "tracked_file":
                tracked.write_text("changed\n", encoding="utf-8")
            else:
                subprocess.run([*commit_command, "--allow-empty"], check=True)
            self.trainer = SimpleNamespace(best=tmp_path / "model.pt")

    output = tmp_path / "new_lineage.json"
    with pytest.raises(RuntimeError, match="(clean Git working tree|revision changed)"):
        train_model(
            base_checkpoint=tmp_path / "base.pt",
            base_checkpoint_sha256=sha256_file(tmp_path / "base.pt"),
            provenance_output=output,
            manifest_path=tmp_path / config["manifest_path"],
            dataset_root=tmp_path / "annotations",
            repository_root=tmp_path,
            model_factory=lambda _: MutatingModel(),
            library_versions=_library_fixture(),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "payload", ["not json", "null", "[]", '{"schema_version":"detector_training_provenance_v4"}']
)
def test_malformed_lineage_documents_fail_closed(tmp_path: Path, payload: str) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    path = tmp_path / config["training_provenance_path"]
    path.write_text(payload, encoding="utf-8")
    config["training_provenance_sha256"] = sha256_file(path)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _assert_lineage_rejected(
        tmp_path, config_path, "(not valid JSON|JSON object|fields are incomplete)"
    )


def test_real_training_fixture_receipt_is_accepted_by_evaluation(tmp_path: Path) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    training_output = tmp_path / "new_training_lineage.json"

    class FixtureModel:
        names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}

        def train(self, **kwargs: Any) -> None:
            self.trainer = SimpleNamespace(best=tmp_path / "model.pt")

        def predict(self, image_path: str, **kwargs: Any) -> list[SimpleNamespace]:
            return [SimpleNamespace(boxes=None)]

    train_model(
        base_checkpoint=tmp_path / "base.pt",
        base_checkpoint_sha256=sha256_file(tmp_path / "base.pt"),
        provenance_output=training_output,
        manifest_path=tmp_path / config["manifest_path"],
        dataset_root=tmp_path / "annotations",
        repository_root=tmp_path,
        model_factory=lambda _: FixtureModel(),
        code_revision="a" * 40,
        library_versions=_library_fixture(),
    )
    config["training_provenance_path"] = training_output.name
    config["training_provenance_sha256"] = sha256_file(training_output)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    metrics = run_evaluation(
        config_path,
        repository_root=tmp_path,
        model_factory=lambda _: FixtureModel(),
        code_revision="b" * 40,
        library_versions=_library_fixture(),
    )
    assert len(metrics) == 8
    receipt = json.loads((tmp_path / config["receipt_path"]).read_text())
    assert receipt["execution_mode"] == "injected_fixture"
    assert receipt["library_versions"] == _library_fixture()
    assert receipt["code_files_sha256"] == detector_code_identity(tmp_path)


def test_newly_committed_source_cannot_be_attributed_to_stale_imports(tmp_path: Path) -> None:
    config_path, config = _evaluation_config_fixture(tmp_path)
    source = tmp_path / "src/vehicle_emission/detection/train.py"
    source.write_text(
        source.read_text() + "\n# changed after the detector was imported\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", *CODE_PATHS], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "new source after detector import",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Historical source verification remains valid; only current stale execution is refused.
    validate_code_identity(
        detector_code_identity(tmp_path), tmp_path, revision, injected_fixture=False
    )
    with pytest.raises(ValueError, match="imported code identity"):
        train_model(
            base_checkpoint=tmp_path / "base.pt",
            base_checkpoint_sha256=sha256_file(tmp_path / "base.pt"),
            provenance_output=tmp_path / "new_lineage.json",
            manifest_path=tmp_path / config["manifest_path"],
            dataset_root=tmp_path / "annotations",
            repository_root=tmp_path,
            model_factory=lambda _: pytest.fail("Stale imported code reached training"),
            code_revision=revision,
            library_versions=_library_fixture(),
        )
    with pytest.raises(ValueError, match="imported code identity"):
        run_evaluation(
            config_path,
            repository_root=tmp_path,
            model_factory=lambda _: pytest.fail("Stale imported code reached evaluation"),
            code_revision=revision,
            library_versions=_library_fixture(),
        )
    assert not (tmp_path / "new_lineage.json").exists()
    assert not (tmp_path / config["output_path"]).exists()
