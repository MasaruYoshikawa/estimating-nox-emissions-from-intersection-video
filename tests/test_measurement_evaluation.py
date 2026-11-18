from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from measurement_fixture import write_measurement_fixture
from vehicle_emission.measurement import evaluate as module
from vehicle_emission.measurement.evaluate import (
    evaluate_measurements,
    validate_measurement_receipt,
)
from vehicle_emission.measurement.metrics import (
    FRAME_COLUMNS,
    OBJECT_COLUMNS,
    evaluate_entries,
    evaluate_objects,
    evaluate_tracking,
    gated_assignment,
)
from vehicle_emission.measurement.speed import evaluate_speed_pairs


def inputs(root: Path) -> tuple[Path, dict[str, pd.DataFrame], dict[str, Any]]:
    config = write_measurement_fixture(root, sites=("Sukiyabashi",))
    configuration = json.loads(config.read_text())
    tables = {
        name: pd.read_csv(root / item["path"], float_precision="round_trip")
        for name, item in configuration["inputs"].items()
        if name != "calibration"
    }
    calibration = json.loads((root / configuration["inputs"]["calibration"]["path"]).read_text())
    return config, tables, calibration


def speed_metrics(
    tables: dict[str, pd.DataFrame], calibration: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_speed_pairs(
        tables["speed_references"],
        tables["speed_predictions"],
        calibration=calibration,
        fit_source_video_sha256=[],
        bootstrap_resamples=100,
        bootstrap_seed=7,
        confidence_level=0.95,
    )


def test_class_agnostic_matching_exposes_confusion_and_missing_classes(tmp_path: Path) -> None:
    _, tables, _ = inputs(tmp_path)
    predictions = tables["predicted_objects"].copy()
    predictions.loc[0, "class_name"] = "car"
    metrics, confusion = evaluate_objects(
        tables["frames"],
        tables["reference_objects"],
        predictions,
        iou_threshold=0.5,
        bootstrap_resamples=100,
    )
    bus, car = metrics.set_index("class_name").loc[["bus", "car"]].itertuples()
    assert (bus.tp, bus.fn, car.tp, car.fp) == (1, 1, 2, 1)
    assert ((confusion.reference_class == "bus") & (confusion.predicted_class == "car")).any()
    assert bus.f1 == pytest.approx(2 / 3)


def test_empty_labelled_frame_false_positive_is_retained(tmp_path: Path) -> None:
    _, tables, _ = inputs(tmp_path)
    frame = tables["frames"].iloc[[0]].copy()
    frame["frame_index"], frame["frame_time_seconds"] = 1, 1
    frames = pd.concat([tables["frames"], frame], ignore_index=True)
    ghost = tables["predicted_objects"].iloc[[0]].copy()
    ghost["frame_index"], ghost["object_id"] = 1, "ghost"
    predictions = pd.concat([tables["predicted_objects"], ghost], ignore_index=True)
    class_metrics, _ = evaluate_objects(
        frames, tables["reference_objects"], predictions, iou_threshold=0.5, bootstrap_resamples=100
    )
    tracking, detail = evaluate_tracking(
        frames, tables["reference_objects"], predictions, iou_threshold=0.5
    )
    assert class_metrics.set_index("class_name").loc["bus", "fp"] == 1
    assert tracking.iloc[0].fp == 1
    assert detail.loc[detail.frame_index == 1, "gt"].iloc[0] == 0
    assert tracking.iloc[0].n_labelled_frames == 3


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("unlabelled", "completely annotated"),
        ("outside", "outside the labelled"),
        ("duplicate", "duplicate object"),
        ("bad_box", "normalized xyxy"),
        ("fractional_frame", "integer"),
        ("hash_reuse", "multiple IDs"),
    ],
)
def test_object_universe_rejects_invalid_data(tmp_path: Path, mutation: str, match: str) -> None:
    _, tables, _ = inputs(tmp_path)
    frames, predictions = tables["frames"].copy(), tables["predicted_objects"].copy()
    if mutation == "unlabelled":
        frames.loc[0, "annotation_status"] = "partial"
    elif mutation == "outside":
        predictions.loc[0, "frame_index"] = 999
    elif mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif mutation == "bad_box":
        predictions.loc[0, "x2"] = float("nan")
        match = "non-finite"
    elif mutation == "fractional_frame":
        predictions["frame_index"] = predictions["frame_index"].astype(float)
        predictions.loc[0, "frame_index"] = 0.5
    else:
        frames.loc[1, "source_video_sha256"] = frames.loc[0, "source_video_sha256"]
    with pytest.raises(ValueError, match=match):
        evaluate_objects(
            frames,
            tables["reference_objects"],
            predictions,
            iou_threshold=0.5,
            bootstrap_resamples=100,
        )


def track_frame(frame: int, objects: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    return [
        {
            "site": "Sukiyabashi",
            "source_video_id": "video",
            "frame_index": frame,
            "object_id": object_id,
            "class_name": "car",
            "x1": left,
            "x2": right,
            "y1": 0.1,
            "y2": 0.2,
        }
        for object_id, left, right in objects
    ]


def frames_for_tracking(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site": "Sukiyabashi",
                "source_video_id": "video",
                "frame_index": index,
                "frame_time_seconds": index,
                "source_video_sha256": "0" * 64,
                "split_id": "test",
                "annotation_status": "complete",
                "detector_checkpoint_sha256": "1" * 64,
                "tracker_config_sha256": "2" * 64,
            }
            for index in range(count)
        ],
        columns=FRAME_COLUMNS,
    )


def test_tracking_id_switch_after_gap_uses_last_known_assignment() -> None:
    reference = pd.DataFrame(
        sum([track_frame(index, [("gt", 0.1, 0.2)]) for index in range(3)], []),
        columns=OBJECT_COLUMNS,
    )
    predictions = pd.DataFrame(
        track_frame(0, [("A", 0.1, 0.2)]) + track_frame(2, [("B", 0.1, 0.2)]),
        columns=OBJECT_COLUMNS,
    )
    summary, _ = evaluate_tracking(
        frames_for_tracking(3), reference, predictions, iou_threshold=0.5
    )
    assert (summary.iloc[0]["gt"], summary.iloc[0].fn, summary.iloc[0].id_switches) == (3, 1, 1)
    assert summary.iloc[0].mota == pytest.approx(1 / 3)


def test_tracking_continuity_preserves_valid_previous_match() -> None:
    reference = pd.DataFrame(
        track_frame(0, [("gt1", 0.1, 0.4), ("gt2", 0.3, 0.6)])
        + track_frame(1, [("gt1", 0.3, 0.6), ("gt2", 0.1, 0.4)]),
        columns=OBJECT_COLUMNS,
    )
    predictions = pd.DataFrame(
        sum([track_frame(index, [("A", 0.1, 0.4), ("B", 0.3, 0.6)]) for index in range(2)], []),
        columns=OBJECT_COLUMNS,
    )
    summary, _ = evaluate_tracking(
        frames_for_tracking(2), reference, predictions, iou_threshold=0.19
    )
    assert summary.iloc[0].id_switches == 0
    assert summary.iloc[0].mota == 1


def test_tracking_gap_releases_continuity_but_keeps_switch_history() -> None:
    reference = pd.DataFrame(
        track_frame(0, [("gt1", 0.1, 0.4), ("gt2", 0.3, 0.6)])
        + track_frame(2, [("gt1", 0.3, 0.6), ("gt2", 0.1, 0.4)]),
        columns=OBJECT_COLUMNS,
    )
    predictions = pd.DataFrame(
        sum([track_frame(index, [("A", 0.1, 0.4), ("B", 0.3, 0.6)]) for index in (0, 2)], []),
        columns=OBJECT_COLUMNS,
    )
    summary, _ = evaluate_tracking(
        frames_for_tracking(3), reference, predictions, iou_threshold=0.19
    )
    assert summary.iloc[0].id_switches == 2


def test_tracking_negative_mota_and_no_ground_truth() -> None:
    reference = pd.DataFrame(track_frame(0, [("gt", 0.1, 0.2)]), columns=OBJECT_COLUMNS)
    predictions = pd.DataFrame(
        track_frame(0, [("A", 0.5, 0.6), ("B", 0.7, 0.8)]), columns=OBJECT_COLUMNS
    )
    summary, _ = evaluate_tracking(
        frames_for_tracking(1), reference, predictions, iou_threshold=0.5
    )
    assert summary.iloc[0].mota == -2
    empty, _ = evaluate_tracking(
        frames_for_tracking(1), reference.iloc[:0], predictions, iou_threshold=0.5
    )
    assert math.isnan(empty.iloc[0].mota)
    assert empty.iloc[0].fp == 2


def test_gated_matching_maximizes_number_not_local_iou() -> None:
    costs = np.array([[0.01, 0.5], [0.5, 0.99]])
    pairs = gated_assignment(costs, np.array([[True, True], [True, False]]))
    assert set(pairs) == {(0, 1), (1, 0)}


def test_entry_event_errors_not_hidden_by_equal_net_counts(tmp_path: Path) -> None:
    _, tables, _ = inputs(tmp_path)
    reference = tables["reference_events"].iloc[:2].copy()
    reference["event_time_seconds"] = [1, 10]
    prediction = reference.copy()
    prediction["event_time_seconds"] = [1, 40]
    summary, detail = evaluate_entries(
        tables["event_windows"], reference, prediction, tolerance_seconds=0.5
    )
    assert (summary.iloc[0].tp, summary.iloc[0].fp, summary.iloc[0].fn) == (1, 1, 1)
    assert summary.iloc[0].f1 == 0.5
    assert summary.iloc[0].count_mae == 0
    assert summary.iloc[0].zero_reference_windows == 1
    assert len(detail) == 2


def test_duplicate_event_predictions_and_zero_reference_window(tmp_path: Path) -> None:
    _, tables, _ = inputs(tmp_path)
    reference = tables["reference_events"].iloc[:1]
    predicted = pd.concat([reference, reference.assign(event_id="duplicate")], ignore_index=True)
    summary, _ = evaluate_entries(
        tables["event_windows"], reference, predicted, tolerance_seconds=0.5
    )
    assert summary.iloc[0].fp == 1
    assert summary.iloc[0].count_mae == 0.5
    assert summary.iloc[0].zero_reference_windows == 1


@pytest.mark.parametrize(
    "column,value", [("line_id", "other"), ("direction", "south"), ("event_time_seconds", 60)]
)
def test_events_outside_declared_universe_are_rejected(
    tmp_path: Path, column: str, value: Any
) -> None:
    _, tables, _ = inputs(tmp_path)
    prediction = tables["predicted_events"].copy()
    prediction.loc[0, column] = value
    with pytest.raises(ValueError, match="outside"):
        evaluate_entries(
            tables["event_windows"], tables["reference_events"], prediction, tolerance_seconds=0.5
        )


def test_speed_errors_are_calculated_from_pairs_and_clustered(tmp_path: Path) -> None:
    _, tables, calibration = inputs(tmp_path)
    tables["speed_references"]["reference_speed"] = [10, 20]
    tables["speed_predictions"]["predicted_speed"] = [12, 16]
    metrics, detail = speed_metrics(tables, calibration)
    row = metrics[metrics.direction == "__all__"].iloc[0]
    assert (row.support, row.n_valid_source_videos, row.mape_support) == (2, 2, 2)
    assert row.mae_mps == 3
    assert row.bias_mps == -1
    assert row.mape_pct == 20
    assert (row.mae_mps_ci_low, row.mae_mps_ci_high) == (2, 4)
    assert (row.bias_mps_ci_low, row.bias_mps_ci_high) == (-4, 2)
    assert detail.signed_error_mps.tolist() == [2, -4]
    second, _ = speed_metrics(tables, calibration)
    pd.testing.assert_frame_equal(metrics, second)


def test_speed_zero_reference_excluded_only_from_mape(tmp_path: Path) -> None:
    _, tables, calibration = inputs(tmp_path)
    tables["speed_references"]["reference_speed"] = [0, 0]
    tables["speed_predictions"]["predicted_speed"] = [1, 3]
    summary, _ = speed_metrics(tables, calibration)
    row = summary.iloc[0]
    assert row.mae_mps == 2 and row.support == 2
    assert row.zero_reference_pairs == 2 and row.mape_support == 0
    assert math.isnan(row.mape_pct)
    assert row.interval_status == "not_estimable"


def test_speed_preserves_missing_invalid_and_unmatched_rows(tmp_path: Path) -> None:
    _, tables, calibration = inputs(tmp_path)
    references, predictions = tables["speed_references"], tables["speed_predictions"]
    extra = references.iloc[[0]].copy()
    extra["reference_id"], extra["reference_status"] = "missing-reference", "missing"
    invalid = references.iloc[[0]].copy()
    invalid["reference_id"], invalid["reference_speed"] = "invalid-reference", -1
    tables["speed_references"] = pd.concat([references, extra, invalid], ignore_index=True)
    extra_prediction = predictions.iloc[[0]].copy()
    extra_prediction["reference_id"] = "unmatched-prediction"
    tables["speed_predictions"] = pd.concat(
        [predictions.iloc[[0]], extra_prediction], ignore_index=True
    )
    summary, detail = speed_metrics(tables, calibration)
    assert set(detail.pair_status) == {
        "paired",
        "missing_reference",
        "invalid_reference",
        "unmatched_reference",
        "unmatched_prediction",
    }
    row = summary.iloc[0]
    assert row.n_reference_rows == 4 and row.support == 1
    assert row.n_unmatched_prediction == 1


def test_speed_distance_time_and_explicit_unit_conversion(tmp_path: Path) -> None:
    _, tables, calibration = inputs(tmp_path)
    tables["speed_references"]["reference_method"] = "distance_time"
    tables["speed_references"]["reference_units"] = "m,s"
    tables["speed_references"]["measured_distance_m"] = [100, 110]
    tables["speed_references"]["elapsed_seconds"] = 10
    tables["speed_predictions"]["predicted_speed"] = [36, 39.6]
    tables["speed_predictions"]["prediction_units"] = "km/h"
    summary, _ = speed_metrics(tables, calibration)
    assert summary.iloc[0].mae_mps == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("derived", "derived/synthetic"),
        ("zone", "outside declared"),
        ("calibration", "undeclared calibration"),
        ("pair_identity", "disagree"),
        ("split", "held-out"),
        ("units", "speed units"),
        ("duplicate", "duplicate reference"),
    ],
)
def test_speed_rejects_invalid_reference_contract(
    tmp_path: Path, mutation: str, match: str
) -> None:
    _, tables, calibration = inputs(tmp_path)
    if mutation == "derived":
        tables["speed_references"].loc[0, "reference_status"] = "derived"
    elif mutation == "zone":
        tables["speed_predictions"].loc[0, "image_zone"] = "outside"
    elif mutation == "calibration":
        tables["speed_predictions"].loc[0, "calibration_id"] = "unknown"
    elif mutation == "pair_identity":
        tables["speed_predictions"].loc[0, "vehicle_id"] = "different"
    elif mutation == "split":
        tables["speed_references"].loc[0, "split_id"] = "train"
    elif mutation == "units":
        tables["speed_references"].loc[0, "reference_units"] = "unspecified"
    else:
        tables["speed_references"] = pd.concat(
            [tables["speed_references"], tables["speed_references"].iloc[[0]]], ignore_index=True
        )
    with pytest.raises(ValueError, match=match):
        speed_metrics(tables, calibration)


def test_speed_video_disjointness_not_frame_id_disjointness(tmp_path: Path) -> None:
    _, tables, calibration = inputs(tmp_path)
    digest = tables["speed_references"].iloc[0].source_video_sha256
    with pytest.raises(ValueError, match="overlap"):
        evaluate_speed_pairs(
            tables["speed_references"],
            tables["speed_predictions"],
            calibration=calibration,
            fit_source_video_sha256=[digest],
            bootstrap_resamples=100,
            bootstrap_seed=0,
            confidence_level=0.95,
        )


def test_receipt_replay_and_explicit_criteria(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    assert receipt["acceptance"]["status"] == "passed"
    assert len(receipt["site_identities"]) == 2
    assert receipt["geometry_evidence_status"] == "identity_recorded_production_use_not_verified"
    validate_measurement_receipt(receipt, tmp_path)
    forged = copy.deepcopy(receipt)
    forged["acceptance"]["site_results"][0]["checks"]["speed"] = False
    with pytest.raises(ValueError, match="acceptance"):
        validate_measurement_receipt(forged, tmp_path)
    output = tmp_path / receipt["outputs"]["speed"]["path"]
    output.write_text(output.read_text().replace("0.0", "1.0"))
    with pytest.raises(ValueError, match="calculation/hash mismatch"):
        validate_measurement_receipt(receipt, tmp_path)


def test_no_criteria_never_produces_pass(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path, acceptance=False)
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    assert receipt["acceptance"] == {"status": "not_assessed", "criteria": {}, "site_results": []}


def test_aggregate_speed_cannot_hide_a_low_support_zone(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    configuration = json.loads(config.read_text())
    for name in ("speed_references", "speed_predictions"):
        path = tmp_path / configuration["inputs"][name]["path"]
        table = pd.read_csv(path)
        table.loc[0, "image_zone"] = "far"
        table.to_csv(path, index=False)
        configuration["inputs"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    calibration_path = tmp_path / configuration["inputs"]["calibration"]["path"]
    calibration = json.loads(calibration_path.read_text())
    for segment in calibration["segments"]:
        segment["valid_image_zones"].append("far")
    calibration_path.write_text(json.dumps(calibration))
    configuration["inputs"]["calibration"]["sha256"] = hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    config.write_text(json.dumps(configuration))
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    assert receipt["acceptance"]["status"] == "failed"
    assert receipt["acceptance"]["site_results"][1]["checks"]["speed"] is False


def test_missing_speed_observation_fails_explicit_criterion(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    configuration = json.loads(config.read_text())
    path = tmp_path / configuration["inputs"]["speed_predictions"]["path"]
    table = pd.read_csv(path).iloc[1:]
    table.to_csv(path, index=False)
    configuration["inputs"]["speed_predictions"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    config.write_text(json.dumps(configuration))
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    assert receipt["acceptance"]["status"] == "failed"
    assert (
        receipt["acceptance"]["site_results"][1]["checks"]["speed"] is False
        or receipt["acceptance"]["site_results"][0]["checks"]["speed"] is False
    )


def test_unobserved_declared_calibration_zone_has_zero_support_and_fails(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    configuration = json.loads(config.read_text())
    path = tmp_path / configuration["inputs"]["calibration"]["path"]
    calibration = json.loads(path.read_text())
    calibration["segments"][0]["valid_image_zones"].append("far")
    path.write_text(json.dumps(calibration))
    configuration["inputs"]["calibration"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config.write_text(json.dumps(configuration))
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    speed = pd.read_csv(tmp_path / receipt["outputs"]["speed"]["path"])
    missing_zone = speed[speed.image_zone.eq("far")].iloc[0]
    assert missing_zone.support == 0 and missing_zone.n_valid_source_videos == 0
    assert missing_zone.interval_status == "not_estimable"
    assert receipt["acceptance"]["status"] == "failed"


def test_camera_segments_cannot_pool_reference_video_support(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path, sites=("Sukiyabashi",))
    configuration = json.loads(config.read_text())
    for name in ("speed_references", "speed_predictions"):
        path = tmp_path / configuration["inputs"][name]["path"]
        table = pd.read_csv(path)
        table.loc[1, "camera_epoch"] = "camera-epoch-2"
        table.loc[1, "calibration_id"] = "second-calibration"
        table.to_csv(path, index=False)
        configuration["inputs"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path / configuration["inputs"]["calibration"]["path"]
    calibration = json.loads(path.read_text())
    second = dict(calibration["segments"][0])
    second.update(camera_epoch="camera-epoch-2", calibration_id="second-calibration")
    calibration["segments"].append(second)
    path.write_text(json.dumps(calibration))
    configuration["inputs"]["calibration"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config.write_text(json.dumps(configuration))
    receipt = evaluate_measurements(
        config_path=config,
        output_directory=tmp_path / "build/measurement",
        repository_root=tmp_path,
    )
    speed = pd.read_csv(tmp_path / receipt["outputs"]["speed"]["path"])
    segments = speed[speed.image_zone.eq("near") & speed.direction.eq("__all__")]
    assert len(segments) == 2 and segments.n_valid_source_videos.eq(1).all()
    assert receipt["acceptance"]["status"] == "failed"


@pytest.mark.parametrize("destination", ["results/new", "build", "build/existing", "build/link"])
def test_output_protection(tmp_path: Path, destination: str) -> None:
    config = write_measurement_fixture(tmp_path)
    (tmp_path / "build/existing").mkdir(parents=True)
    if destination == "build/link":
        try:
            (tmp_path / "build/link").symlink_to(
                tmp_path / "build/elsewhere", target_is_directory=True
            )
        except OSError as error:
            pytest.skip(f"Symlinks not supported on system: {error}")
    with pytest.raises(ValueError, match="fresh directory"):
        evaluate_measurements(
            config_path=config, output_directory=tmp_path / destination, repository_root=tmp_path
        )
    assert not (tmp_path / "results").exists()


def test_input_hash_mismatch_does_not_write_output(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    path = tmp_path / "inputs/reference_objects.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_measurements(
            config_path=config,
            output_directory=tmp_path / "build/measurement",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "build").exists()


def test_changed_input_after_snapshot_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_measurement_fixture(tmp_path)
    original = module.evaluate_speed_pairs

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = tmp_path / "inputs/reference_objects.csv"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(module, "evaluate_speed_pairs", mutate)
    with pytest.raises(ValueError, match="changed during calculation"):
        evaluate_measurements(
            config_path=config,
            output_directory=tmp_path / "build/measurement",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "build").exists()


def test_failed_publication_removes_only_new_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_measurement_fixture(tmp_path)
    original = module.os.link
    count = 0

    def fail_second(source: Any, destination: Any) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("controlled publication failure")
        original(source, destination)

    monkeypatch.setattr(module.os, "link", fail_second)
    with pytest.raises(OSError, match="controlled publication"):
        evaluate_measurements(
            config_path=config,
            output_directory=tmp_path / "build/measurement",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "build/measurement").exists()
    assert not list((tmp_path / "build").iterdir())


def test_deleted_tracked_output_cannot_be_replaced(tmp_path: Path) -> None:
    config = write_measurement_fixture(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / "build/existing/result.csv"
    target.parent.mkdir(parents=True)
    target.write_text("archived")
    subprocess.run(["git", "add", "build/existing/result.csv"], cwd=tmp_path, check=True)
    target.unlink()
    target.parent.rmdir()
    with pytest.raises(ValueError, match="committed artifacts"):
        evaluate_measurements(
            config_path=config, output_directory=target.parent, repository_root=tmp_path
        )


def test_fixture_with_reused_track_ids_keeps_video_boundaries(tmp_path: Path) -> None:
    _, tables, _ = inputs(tmp_path)
    summary, _ = evaluate_tracking(
        tables["frames"],
        tables["reference_objects"],
        tables["predicted_objects"],
        iou_threshold=0.5,
    )
    assert summary.iloc[0]["gt"] == 8 and summary.iloc[0].mota == 1
    assert summary.iloc[0].id_switches == 0
