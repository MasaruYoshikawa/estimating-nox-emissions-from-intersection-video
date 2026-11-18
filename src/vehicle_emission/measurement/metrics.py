"""Object, entry-event and CLEAR MOT calculations from declared labelled data.

Tracking preserves valid immediately preceding assignments before gated matching.
Identity switches compare each reference's last assigned prediction, including
after gaps. See MOT16, https://arxiv.org/abs/1603.00831, and the reference
implementation https://github.com/cheind/py-motmetrics. Metric definitions are
in coordination/GLOSSARY.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from vehicle_emission.detection.evaluate import CLASS_MAP, calculate_iou

CLASSES = tuple(CLASS_MAP.values())
VIDEO_KEY = ["site", "source_video_id"]
FRAME_KEY = VIDEO_KEY + ["frame_index"]
FRAME_COLUMNS = FRAME_KEY + [
    "frame_time_seconds",
    "source_video_sha256",
    "split_id",
    "annotation_status",
    "detector_checkpoint_sha256",
    "tracker_config_sha256",
]
OBJECT_COLUMNS = FRAME_KEY + ["object_id", "class_name", "x1", "y1", "x2", "y2"]
WINDOW_COLUMNS = VIDEO_KEY + [
    "window_id",
    "line_id",
    "direction",
    "start_seconds",
    "end_seconds",
    "source_video_sha256",
    "split_id",
    "annotation_status",
]
EVENT_COLUMNS = VIDEO_KEY + ["event_id", "line_id", "direction", "event_time_seconds"]


def require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Require declared columns, including for explicitly empty tables."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def require_ids(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Reject missing or empty identifiers rather than silently dropping rows."""
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has missing {column}")


def _finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    try:
        values = frame[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires numeric coordinates/times") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} has non-finite coordinates/times")


def validate_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete test-frame universe, including empty frames."""
    require_columns(frames, FRAME_COLUMNS, "frame universe")
    require_ids(
        frames,
        VIDEO_KEY + ["source_video_sha256", "detector_checkpoint_sha256", "tracker_config_sha256"],
        "frame universe",
    )
    if frames.empty or frames.duplicated(FRAME_KEY).any():
        raise ValueError("frame universe must be nonempty and have unique frame keys")
    if (
        not frames["split_id"].eq("test").all()
        or not frames["annotation_status"].eq("complete").all()
    ):
        raise ValueError("only completely annotated test frames are evaluable")
    _finite(frames, ["frame_index", "frame_time_seconds"], "frame universe")
    if (frames["frame_index"] < 0).any() or (frames["frame_index"] % 1 != 0).any():
        raise ValueError("frame_index must be a nonnegative integer")
    if (frames["frame_time_seconds"] < 0).any():
        raise ValueError("frame times must be nonnegative")
    metadata = ["source_video_sha256", "detector_checkpoint_sha256", "tracker_config_sha256"]
    for _, group in frames.groupby(VIDEO_KEY, sort=True):
        if any(group[column].nunique() != 1 for column in metadata):
            raise ValueError("video/checkpoint/tracker identity varies within a source video")
        ordered = group.sort_values("frame_index")
        if (ordered["frame_time_seconds"].diff().dropna() <= 0).any():
            raise ValueError("frame times must increase with frame indices")
    for column in metadata:
        if not frames[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"{column} must contain SHA-256 digests")
    identity = frames[VIDEO_KEY + ["source_video_sha256"]].drop_duplicates()
    if identity.duplicated("source_video_sha256").any():
        raise ValueError("one source video hash cannot have multiple IDs/sites")
    return frames.sort_values(FRAME_KEY).reset_index(drop=True)


def validate_objects(objects: pd.DataFrame, frames: pd.DataFrame, name: str) -> pd.DataFrame:
    """Require normalized image boxes and explicit frame membership."""
    require_columns(objects, OBJECT_COLUMNS, name)
    require_ids(objects, VIDEO_KEY + ["object_id", "class_name"], name)
    if objects.duplicated(FRAME_KEY + ["object_id"]).any():
        raise ValueError(f"{name} contains duplicate object/frame IDs")
    if not objects["class_name"].isin(CLASSES).all():
        raise ValueError(f"{name} contains unsupported vehicle classes")
    _finite(objects, ["frame_index", "x1", "y1", "x2", "y2"], name)
    if (objects["frame_index"] % 1 != 0).any():
        raise ValueError(f"{name} frame_index must be an integer")
    boxes = objects[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if ((boxes < 0) | (boxes > 1)).any() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError(f"{name} boxes must be positive-area normalized xyxy boxes")
    membership = objects.merge(frames[FRAME_KEY], on=FRAME_KEY, how="left", indicator=True)
    if not membership["_merge"].eq("both").all():
        raise ValueError(f"{name} contains objects outside the labelled frame universe")
    return objects.sort_values(FRAME_KEY + ["object_id"]).reset_index(drop=True)


def _iou_matrix(reference: pd.DataFrame, prediction: pd.DataFrame) -> np.ndarray:
    def xywh(row: Any) -> tuple[float, float, float, float]:
        return ((row.x1 + row.x2) / 2, (row.y1 + row.y2) / 2, row.x2 - row.x1, row.y2 - row.y1)

    return np.array(
        [
            [calculate_iou(xywh(gt), xywh(pred)) for pred in prediction.itertuples()]
            for gt in reference.itertuples()
        ],
        dtype=float,
    ).reshape(len(reference), len(prediction))


def gated_assignment(cost: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-cardinality one-to-one matching, then minimum declared cost."""
    if cost.size == 0:
        return []
    maximum = float(np.max(cost[eligible])) if eligible.any() else 0.0
    penalty = (min(cost.shape) + 1) * (maximum + 1.0)
    rows, columns = linear_sum_assignment(np.where(eligible, cost, penalty))
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if eligible[row, column]
    ]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_objects(
    frames: pd.DataFrame,
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    iou_threshold: float,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate class confusion with class-agnostic spatial assignment."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in (0, 1]")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 100
    ):
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0,1)")
    frames = validate_frames(frames)
    reference = validate_objects(reference, frames, "reference objects")
    prediction = validate_objects(prediction, frames, "predicted objects")
    reference_frames = dict(tuple(reference.groupby(FRAME_KEY, sort=True)))
    prediction_frames = dict(tuple(prediction.groupby(FRAME_KEY, sort=True)))
    confusion: list[dict[str, Any]] = []
    for key, _ in frames.groupby(FRAME_KEY, sort=True):
        gt = reference_frames.get(key, reference.iloc[:0])
        pred = prediction_frames.get(key, prediction.iloc[:0])
        overlaps = _iou_matrix(gt, pred)
        pairs = gated_assignment(1 - overlaps, overlaps >= iou_threshold)
        matched_gt, matched_pred = {a for a, _ in pairs}, {b for _, b in pairs}
        labels = [
            (str(gt.iloc[a]["class_name"]), str(pred.iloc[b]["class_name"])) for a, b in pairs
        ]
        labels += [
            (str(gt.iloc[a]["class_name"]), "__missing__")
            for a in range(len(gt))
            if a not in matched_gt
        ]
        labels += [
            ("__background__", str(pred.iloc[b]["class_name"]))
            for b in range(len(pred))
            if b not in matched_pred
        ]
        for actual, detected in labels:
            confusion.append(
                {
                    "site": key[0],
                    "source_video_id": key[1],
                    "reference_class": actual,
                    "predicted_class": detected,
                    "count": 1,
                }
            )
    confusion_frame = pd.DataFrame(
        confusion, columns=VIDEO_KEY + ["reference_class", "predicted_class", "count"]
    )
    summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for site in sorted(frames["site"].unique()):
        cells = confusion_frame[confusion_frame["site"] == site]
        videos = sorted(frames.loc[frames.site == site, "source_video_id"].unique())
        for class_name in CLASSES:
            tp = int(
                cells.loc[
                    (cells.reference_class == class_name) & (cells.predicted_class == class_name),
                    "count",
                ].sum()
            )
            fn = int(
                cells.loc[
                    (cells.reference_class == class_name) & (cells.predicted_class != class_name),
                    "count",
                ].sum()
            )
            fp = int(
                cells.loc[
                    (cells.reference_class != class_name) & (cells.predicted_class == class_name),
                    "count",
                ].sum()
            )
            class_videos = cells.loc[
                cells.reference_class == class_name, "source_video_id"
            ].nunique()
            cluster_counts = []
            for video in videos:
                cluster = cells[cells.source_video_id == video]
                cluster_counts.append(
                    [
                        int(
                            (
                                (cluster.reference_class == class_name)
                                & (cluster.predicted_class == class_name)
                            ).sum()
                        ),
                        int(
                            (
                                (cluster.reference_class != class_name)
                                & (cluster.predicted_class == class_name)
                            ).sum()
                        ),
                        int(
                            (
                                (cluster.reference_class == class_name)
                                & (cluster.predicted_class != class_name)
                            ).sum()
                        ),
                    ]
                )
            draws: list[float] = []
            if class_videos >= 2:
                counts = np.array(cluster_counts, dtype=int)
                for _ in range(bootstrap_resamples):
                    sampled = rng.integers(0, len(videos), size=len(videos))
                    a, b, c = counts[sampled].sum(axis=0)
                    value = _prf(int(a), int(b), int(c))["f1"]
                    if math.isfinite(value):
                        draws.append(value)
            estimable = len(draws) >= math.ceil(0.9 * bootstrap_resamples)
            bounds = (
                np.quantile(draws, [(1 - confidence_level) / 2, (1 + confidence_level) / 2])
                if estimable
                else [math.nan, math.nan]
            )
            summaries.append(
                {
                    "site": site,
                    "class_name": class_name,
                    "support": tp + fn,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    **_prf(tp, fp, fn),
                    "source_video_support": len(videos),
                    "class_source_video_support": class_videos,
                    "f1_ci_low": float(bounds[0]),
                    "f1_ci_high": float(bounds[1]),
                    "f1_valid_resamples": len(draws),
                    "bootstrap_unit": "source_video_id",
                    "interval_status": "estimated" if estimable else "not_estimable",
                }
            )
    return pd.DataFrame(summaries), confusion_frame


def evaluate_tracking(
    frames: pd.DataFrame,
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate framewise FP/FN, last-known ID switches and true MOTA."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in (0, 1]")
    frames = validate_frames(frames)
    reference = validate_objects(reference, frames, "reference tracks")
    prediction = validate_objects(prediction, frames, "predicted tracks")
    rows: list[dict[str, Any]] = []
    for video_key, video_frames in frames.groupby(VIDEO_KEY, sort=True):
        last_assignment: dict[str, str] = {}
        last_matched_frame: dict[str, int] = {}
        for evaluated_index, frame in enumerate(video_frames.itertuples()):
            selection = (
                (reference.site == frame.site)
                & (reference.source_video_id == frame.source_video_id)
                & (reference.frame_index == frame.frame_index)
            )
            gt = reference[selection].reset_index(drop=True)
            selection = (
                (prediction.site == frame.site)
                & (prediction.source_video_id == frame.source_video_id)
                & (prediction.frame_index == frame.frame_index)
            )
            pred = prediction[selection].reset_index(drop=True)
            overlaps = _iou_matrix(gt, pred)
            pairs: list[tuple[int, int]] = []
            used_predictions: set[int] = set()
            prediction_ids = {
                str(row.object_id): index for index, row in enumerate(pred.itertuples())
            }
            for gt_index, row in enumerate(gt.itertuples()):
                previous = last_assignment.get(str(row.object_id))
                pred_index = prediction_ids.get(previous) if previous is not None else None
                if (
                    last_matched_frame.get(str(row.object_id)) == evaluated_index - 1
                    and pred_index is not None
                    and pred_index not in used_predictions
                    and overlaps[gt_index, pred_index] >= iou_threshold
                ):
                    pairs.append((gt_index, pred_index))
                    used_predictions.add(pred_index)
            remaining_gt = [index for index in range(len(gt)) if index not in {a for a, _ in pairs}]
            remaining_pred = [index for index in range(len(pred)) if index not in used_predictions]
            subset = overlaps[np.ix_(remaining_gt, remaining_pred)]
            pairs += [
                (remaining_gt[a], remaining_pred[b])
                for a, b in gated_assignment(1 - subset, subset >= iou_threshold)
            ]
            switches = 0
            for gt_index, pred_index in pairs:
                gt_id, pred_id = str(gt.iloc[gt_index].object_id), str(
                    pred.iloc[pred_index].object_id
                )
                if gt_id in last_assignment and last_assignment[gt_id] != pred_id:
                    switches += 1
                last_assignment[gt_id] = pred_id
                last_matched_frame[gt_id] = evaluated_index
            rows.append(
                {
                    "site": video_key[0],
                    "source_video_id": video_key[1],
                    "frame_index": frame.frame_index,
                    "gt": len(gt),
                    "fp": len(pred) - len(pairs),
                    "fn": len(gt) - len(pairs),
                    "id_switches": switches,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("site", sort=True)[["gt", "fp", "fn", "id_switches"]].sum().reset_index()
    )
    summary["mota"] = [
        1 - _ratio(row.fn + row.fp + row.id_switches, row.gt) for row in summary.itertuples()
    ]
    summary["n_labelled_frames"] = summary.site.map(detail.groupby("site").size())
    summary["n_source_videos"] = summary.site.map(frames.groupby("site").source_video_id.nunique())
    return summary, detail


def evaluate_entries(
    windows: pd.DataFrame,
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    tolerance_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-to-one entry-event matching within declared lines/directions/windows."""
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("entry tolerance must be finite and nonnegative")
    require_columns(windows, WINDOW_COLUMNS, "event universe")
    require_ids(
        windows,
        VIDEO_KEY + ["window_id", "line_id", "direction", "source_video_sha256"],
        "event universe",
    )
    _finite(windows, ["start_seconds", "end_seconds"], "event universe")
    if windows.empty or windows.duplicated(VIDEO_KEY + ["window_id"]).any():
        raise ValueError("event universe must contain unique declared windows")
    if not windows.split_id.eq("test").all() or not windows.annotation_status.eq("complete").all():
        raise ValueError("event windows must be completely annotated test intervals")
    if (windows.start_seconds < 0).any() or (windows.end_seconds <= windows.start_seconds).any():
        raise ValueError("event windows require nonnegative increasing boundaries")
    if not windows.source_video_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("event window source_video_sha256 must be SHA-256")
    group_key = VIDEO_KEY + ["line_id", "direction"]
    for _, group in windows.groupby(group_key, sort=True):
        ordered = group.sort_values("start_seconds")
        if (ordered.start_seconds.to_numpy()[1:] < ordered.end_seconds.to_numpy()[:-1]).any():
            raise ValueError("declared event windows cannot overlap")
    event_tables: list[pd.DataFrame] = []
    for name, events in (("reference events", reference), ("predicted events", prediction)):
        require_columns(events, EVENT_COLUMNS, name)
        require_ids(events, VIDEO_KEY + ["event_id", "line_id", "direction"], name)
        _finite(events, ["event_time_seconds"], name)
        if events.duplicated(VIDEO_KEY + ["event_id"]).any():
            raise ValueError(f"{name} contains duplicate event IDs")
        assigned = events.merge(
            windows[group_key + ["window_id", "start_seconds", "end_seconds"]],
            on=group_key,
            how="left",
        )
        assigned = assigned[
            (assigned.event_time_seconds >= assigned.start_seconds)
            & (assigned.event_time_seconds < assigned.end_seconds)
        ]
        if len(assigned) != len(events):
            raise ValueError(f"{name} has events outside its declared complete universe")
        event_tables.append(assigned)
    reference, prediction = event_tables
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples():

        def selected(frame: pd.DataFrame, selected_window: Any = window) -> pd.DataFrame:
            return frame[
                (frame.site == selected_window.site)
                & (frame.source_video_id == selected_window.source_video_id)
                & (frame.window_id == selected_window.window_id)
            ]

        gt, pred = selected(reference), selected(prediction)
        distance = np.abs(
            gt.event_time_seconds.to_numpy()[:, None] - pred.event_time_seconds.to_numpy()[None, :]
        )
        pairs = gated_assignment(distance, distance <= tolerance_seconds)
        rows.append(
            {
                "site": window.site,
                "source_video_id": window.source_video_id,
                "window_id": window.window_id,
                "line_id": window.line_id,
                "direction": window.direction,
                "reference_count": len(gt),
                "predicted_count": len(pred),
                "tp": len(pairs),
                "fp": len(pred) - len(pairs),
                "fn": len(gt) - len(pairs),
                "signed_count_error": len(pred) - len(gt),
                "absolute_count_error": abs(len(pred) - len(gt)),
            }
        )
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for site, group in detail.groupby("site", sort=True):
        tp, fp, fn = (int(group[column].sum()) for column in ("tp", "fp", "fn"))
        nonzero = group.reference_count > 0
        summaries.append(
            {
                "site": site,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **_prf(tp, fp, fn),
                "n_windows": len(group),
                "n_source_videos": group.source_video_id.nunique(),
                "count_mae": float(group.absolute_count_error.mean()),
                "count_bias": float(group.signed_count_error.mean()),
                "count_mape_pct": float(
                    (
                        group.loc[nonzero, "absolute_count_error"]
                        / group.loc[nonzero, "reference_count"]
                    ).mean()
                    * 100
                ),
                "zero_reference_windows": int((~nonzero).sum()),
            }
        )
    return pd.DataFrame(summaries), detail
