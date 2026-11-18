"""Reconcile recorded detector development-validation logs without test claims."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def summarize_development_run(run_path: Path) -> dict:
    """Report the final logged epoch and an explicitly named validation maximum."""
    args_path, metrics_path = run_path / "args.yaml", run_path / "results.csv"
    inputs = {file.name: file.read_bytes() for file in (args_path, metrics_path)}
    arguments = yaml.safe_load(inputs["args.yaml"])
    if not isinstance(arguments, dict) or arguments.get("val") is not True:
        raise ValueError("run must explicitly record validation")
    if arguments.get("split") != "val":
        raise ValueError("run must identify the development validation split")
    metrics = pd.read_csv(io.BytesIO(inputs["results.csv"]))
    if metrics.empty or not {"epoch", *METRICS.values()} <= set(metrics):
        raise ValueError("run lacks logged epochs or detector metrics")
    epochs = pd.to_numeric(metrics.epoch, errors="raise")
    if not (
        np.isfinite(epochs) & epochs.gt(0) & epochs.eq(np.floor(epochs)) & ~epochs.duplicated()
    ).all():
        raise ValueError("epochs must be unique positive integers")
    values = metrics[list(METRICS.values())].apply(pd.to_numeric, errors="raise")
    if not (np.isfinite(values) & values.ge(0) & values.le(1)).all().all():
        raise ValueError("detector validation metrics must be finite proportions")
    final = metrics.loc[epochs.idxmax()]
    maximum = metrics.loc[values[METRICS["map50_95"]].idxmax()]
    return {
        "evaluation_role": "development_validation",
        "logged_epochs": len(metrics),
        "final_epoch": int(final.epoch),
        "final_metrics": {name: float(final[column]) for name, column in METRICS.items()},
        "maximum_logged_map50_95": float(maximum[METRICS["map50_95"]]),
        "maximum_logged_map50_95_epoch": int(maximum.epoch),
        "training_arguments": {
            name: arguments.get(name)
            for name in (
                "model",
                "data",
                "epochs",
                "batch",
                "imgsz",
                "seed",
                "deterministic",
                "val",
                "split",
            )
        },
        "independent_test_membership": "not_established_by_training_logs",
        "production_checkpoint_linkage": "not_established_by_training_logs",
        "input_sha256": {
            name: hashlib.sha256(contents).hexdigest() for name, contents in inputs.items()
        },
    }


def write_development_summary(root: Path, output: Path) -> Path:
    """Write one fresh source-hashed JSON diagnostic; never replace a result."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    summaries = {
        name: summarize_development_run(root / "runs" / name)
        for name in ("rtdetr_finetune-4", "rtdetr_shinjuku")
    }
    local_split = root / "data/eval_dataset/coco_eval.yaml"
    split_bytes = local_split.read_bytes()
    split = yaml.safe_load(split_bytes)
    if not isinstance(split, dict):
        raise ValueError("local evaluation configuration requires explicit train and val")
    for name in ("train", "val"):
        value = split.get(name)
        entries = [value] if isinstance(value, str) else value
        if (
            not isinstance(entries, list)
            or not entries
            or not all(isinstance(entry, str) and entry.strip() for entry in entries)
        ):
            raise ValueError("local evaluation configuration requires explicit train and val")
    result = {
        "schema_version": "detector_development_summary_v1",
        "runs": summaries,
        "local_evaluation_configuration": {
            "path": local_split.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(split_bytes).hexdigest(),
            "train_equals_val": split.get("train") == split.get("val"),
            "relation_to_training_run_data": "not_established_by_path_strings",
        },
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_development_summary(args.root, args.output)


if __name__ == "__main__":
    main()
