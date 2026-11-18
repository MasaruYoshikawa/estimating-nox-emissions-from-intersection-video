from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vehicle_emission.measurement.development_logs import (
    METRICS,
    summarize_development_run,
    write_development_summary,
)


def _run(path: Path):
    path.mkdir(parents=True)
    (path / "args.yaml").write_text(yaml.safe_dump({"val": True, "split": "val", "epochs": 2}))
    pd.DataFrame(
        {
            "epoch": [1, 2],
            **{column: [0.8, 0.7] for column in METRICS.values()},
        }
    ).to_csv(path / "results.csv", index=False)


def test_development_final_epoch_and_maximum_are_distinct(tmp_path):
    _run(tmp_path / "run")
    result = summarize_development_run(tmp_path / "run")
    assert result["final_epoch"] == 2
    assert result["final_metrics"]["map50_95"] == 0.7
    assert result["maximum_logged_map50_95_epoch"] == 1
    assert result["maximum_logged_map50_95"] == 0.8
    assert result["evaluation_role"] == "development_validation"
    assert result["production_checkpoint_linkage"] == "not_established_by_training_logs"
    assert len(result["input_sha256"]["results.csv"]) == 64


@pytest.mark.parametrize(
    "column,value",
    [
        ("epoch", 1),
        ("metrics/precision(B)", -0.1),
        ("metrics/recall(B)", float("nan")),
        ("metrics/mAP50(B)", 1.1),
    ],
)
def test_invalid_metrics_rejected(tmp_path, column, value):
    _run(tmp_path / "run")
    metrics = pd.read_csv(tmp_path / "run/results.csv")
    metrics.loc[1, column] = value
    metrics.to_csv(tmp_path / "run/results.csv", index=False)
    with pytest.raises(ValueError):
        summarize_development_run(tmp_path / "run")


def test_summary_is_deterministic_and_refuses_overwrite(tmp_path):
    for name in ("rtdetr_finetune-4", "rtdetr_shinjuku"):
        _run(tmp_path / "runs" / name)
    split_path = tmp_path / "data/eval_dataset/coco_eval.yaml"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(yaml.safe_dump({"train": ["images"], "val": ["images"]}))
    first = write_development_summary(tmp_path, tmp_path / "first.json")
    second = write_development_summary(tmp_path, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["local_evaluation_configuration"]["train_equals_val"]
    with pytest.raises(FileExistsError):
        write_development_summary(tmp_path, first)


def test_recorded_run_values():
    root = Path(__file__).resolve().parents[1]
    if not (root / "runs/rtdetr_finetune-4/results.csv").is_file():
        pytest.skip("recorded development logs unavailable")
    first = summarize_development_run(root / "runs/rtdetr_finetune-4")
    second = summarize_development_run(root / "runs/rtdetr_shinjuku")
    assert first["logged_epochs"] == second["logged_epochs"] == 50
    assert first["final_metrics"] == {
        "precision": 0.90099,
        "recall": 0.90915,
        "map50": 0.94729,
        "map50_95": 0.82947,
    }
    assert second["final_metrics"] == {
        "precision": 0.84249,
        "recall": 0.71436,
        "map50": 0.76265,
        "map50_95": 0.62581,
    }
    assert second["maximum_logged_map50_95_epoch"] == 49


@pytest.mark.parametrize(
    "split",
    [
        None,
        [],
        {},
        {"train": "images"},
        {"train": [], "val": []},
        {"train": [""], "val": [""]},
        {"train": 1, "val": 1},
    ],
)
def test_missing_or_malformed_local_split_rejected(tmp_path, split):
    for name in ("rtdetr_finetune-4", "rtdetr_shinjuku"):
        _run(tmp_path / "runs" / name)
    path = tmp_path / "data/eval_dataset/coco_eval.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(split))
    with pytest.raises(ValueError, match="explicit train and val"):
        write_development_summary(tmp_path, tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


def test_run_hashes_bind_parsed_bytes_not_later_re_read(tmp_path, monkeypatch):
    path = tmp_path / "run"
    _run(path)
    original_read = Path.read_bytes
    original = {name: (path / name).read_bytes() for name in ("args.yaml", "results.csv")}
    counts = {}

    def change_after_read(file):
        contents = original_read(file)
        if file.parent == path:
            counts[file.name] = counts.get(file.name, 0) + 1
            file.write_text("changed after capture")
        return contents

    monkeypatch.setattr(Path, "read_bytes", change_after_read)
    result = summarize_development_run(path)
    assert result["final_metrics"]["map50"] == 0.7
    assert counts == {"args.yaml": 1, "results.csv": 1}
    assert result["input_sha256"] == {
        name: hashlib.sha256(contents).hexdigest() for name, contents in original.items()
    }
