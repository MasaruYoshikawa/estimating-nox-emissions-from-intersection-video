from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import RTDETR

from vehicle_emission.line_counting import LineOfInterest, MultiLineCrossingCounter

print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated(0))

try:
    from boxmot import OcSort as OCSORT
except ImportError:
    try:
        from boxmot import OCSORT
    except ImportError as second_exc:
        OCSORT = None
        OCSORT_IMPORT_ERROR = second_exc
    else:
        OCSORT_IMPORT_ERROR = None
else:
    OCSORT_IMPORT_ERROR = None


CLASS_NAMES = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}
COLORS = {
    0: (0, 165, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
}


def build_line_counter(args: argparse.Namespace) -> MultiLineCrossingCounter | None:
    if args.exit_line:
        line_specs = []
        for idx, coords in enumerate(args.exit_line):
            direction = None
            if idx < len(args.exit_line_dir):
                direction = args.exit_line_dir[idx][0]
            line_specs.append(
                (
                    f"exit{idx + 1}",
                    LineOfInterest(start=(coords[0], coords[1]), end=(coords[2], coords[3])),
                    direction,
                )
            )
        return MultiLineCrossingCounter(line_specs)

    if args.line_start is not None and args.line_end is not None:
        return MultiLineCrossingCounter(
            [
                (
                    "exit1",
                    LineOfInterest(start=tuple(args.line_start), end=tuple(args.line_end)),
                    args.count_direction,
                )
            ]
        )

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run trained RT-DETR detections through OC-SORT "
            "(Observation-Centric SORT) for comparison with ByteTrack"
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained RT-DETR model")
    parser.add_argument("--video", type=str, required=True, help="Path to input traffic video")
    parser.add_argument("--output-csv", type=str, help="Path to save tracked numeric features")
    parser.add_argument("--output-video", type=str, help="Path to save debug annotated video")
    parser.add_argument(
        "--max-box-ratio",
        type=float,
        default=0.4,
        help="Maximum allowed box width/height ratio to frame dimension",
    )
    parser.add_argument("--duration", type=int, help="Maximum video duration to process in seconds")
    parser.add_argument("--line-start", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--line-end", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument(
        "--count-direction",
        type=str,
        choices=["positive_to_negative", "negative_to_positive"],
        default=None,
        help="Only count line crossings in one direction",
    )
    parser.add_argument(
        "--exit-line",
        type=float,
        nargs=4,
        action="append",
        default=[],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Add an exit line as x1 y1 x2 y2; can be repeated up to 4 times",
    )
    parser.add_argument(
        "--exit-line-dir",
        type=str,
        nargs=1,
        action="append",
        default=[],
        choices=["positive_to_negative", "negative_to_positive"],
        help="Direction for the corresponding exit line",
    )
    parser.add_argument(
        "--show-realtime-counts",
        action="store_true",
        help="Print negative/positive counts for each exit line after processing",
    )
    parser.add_argument(
        "--show-straight-counts",
        action="store_true",
        help="Print counts of vehicles matching the four straight intersection paths",
    )
    parser.add_argument("--display", action="store_true", help="Show realtime annotated frames")
    parser.add_argument("--conf", type=float, default=0.3, help="RT-DETR confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop the video frame before processing",
    )
    parser.add_argument(
        "--mask-polygon",
        type=float,
        nargs="+",
        action="append",
        help="Polygon vertices (x1 y1 x2 y2 ...) defining the keep region; can be repeated",
    )
    parser.add_argument(
        "--ocsort-det-thresh",
        type=float,
        default=0.3,
        help="OC-SORT detection threshold",
    )
    parser.add_argument(
        "--ocsort-max-age",
        type=int,
        default=30,
        help="Frames to keep unmatched OC-SORT tracks alive",
    )
    parser.add_argument(
        "--ocsort-min-hits",
        type=int,
        default=3,
        help="Minimum hits before an OC-SORT track is emitted as confirmed",
    )
    parser.add_argument(
        "--ocsort-iou-threshold",
        type=float,
        default=0.3,
        help="Association IoU threshold used by OC-SORT",
    )
    parser.add_argument(
        "--ocsort-delta-t",
        type=int,
        default=3,
        help="Observation interval used by OC-SORT velocity direction consistency",
    )
    parser.add_argument(
        "--ocsort-inertia",
        type=float,
        default=0.2,
        help="OC-SORT velocity direction consistency weight",
    )
    return parser.parse_args()


def create_ocsort_tracker(args: argparse.Namespace):
    if OCSORT is None:
        raise ImportError(
            "OC-SORT runner requires the BoxMOT package. Install the project-compatible "
            "pinned BoxMOT version in your environment, then rerun this script."
        ) from OCSORT_IMPORT_ERROR

    kwargs = {
        "det_thresh": args.ocsort_det_thresh,
        "max_age": args.ocsort_max_age,
        "min_hits": args.ocsort_min_hits,
        "iou_threshold": args.ocsort_iou_threshold,
        "delta_t": args.ocsort_delta_t,
        "inertia": args.ocsort_inertia,
        "asso_func": "iou",
        "use_byte": False,
    }

    try:
        return OCSORT(**kwargs)
    except TypeError:
        # Older BoxMOT releases use slightly different constructor names.
        fallback_kwargs = kwargs.copy()
        fallback_kwargs["iou_thresh"] = fallback_kwargs.pop("iou_threshold")
        fallback_kwargs.pop("asso_func", None)
        fallback_kwargs.pop("use_byte", None)
        return OCSORT(**fallback_kwargs)


def detections_from_result(result, width: int, height: int, max_box_ratio: float) -> np.ndarray:
    if result.boxes is None:
        return np.empty((0, 6), dtype=np.float32)

    detections = []
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_xywh = result.boxes.xywh.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()

    for xyxy, xywh, conf, class_id in zip(boxes_xyxy, boxes_xywh, confs, classes, strict=False):
        _, _, w_box, h_box = xywh
        if w_box >= width * max_box_ratio or h_box >= height * max_box_ratio:
            continue

        x1, y1, x2, y2 = xyxy
        detections.append([x1, y1, x2, y2, float(conf), int(class_id)])

    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(detections, dtype=np.float32)


def parse_ocsort_tracks(tracks: np.ndarray) -> list[dict[str, object]]:
    parsed = []
    if tracks is None or len(tracks) == 0:
        return parsed

    for row in np.asarray(tracks):
        if len(row) < 5:
            continue

        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        track_id = int(row[4])
        confidence = float(row[5]) if len(row) > 5 else 0.0
        class_id = int(row[6]) if len(row) > 6 else -1
        width = x2 - x1
        height = y2 - y1
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        parsed.append(
            {
                "id": track_id,
                "class": class_id,
                "conf": confidence,
                "box": np.asarray([cx, cy, width, height], dtype=np.float32),
                "xyxy": (x1, y1, x2, y2),
            }
        )

    return parsed


def main() -> None:
    args = parse_args()

    print(f"Loading RT-DETR model from {args.model}...")
    model = RTDETR(args.model)
    tracker = create_ocsort_tracker(args)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = args.duration * fps if args.duration else None

    crop_bounds = None
    if args.crop:
        x1_c, y1_c, x2_c, y2_c = args.crop
        x1_c = max(0, min(x1_c, orig_width))
        x2_c = max(0, min(x2_c, orig_width))
        y1_c = max(0, min(y1_c, orig_height))
        y2_c = max(0, min(y2_c, orig_height))
        crop_bounds = (x1_c, y1_c, x2_c, y2_c)
        width, height = x2_c - x1_c, y2_c - y1_c
        if width <= 0 or height <= 0:
            raise ValueError(f"Crop region is outside video bounds ({orig_width}x{orig_height})")
        print(f"Cropped video from {orig_width}x{orig_height} to {width}x{height}.")
    else:
        width, height = orig_width, orig_height

    mask = None
    if args.mask_polygon:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in args.mask_polygon:
            if len(poly) % 2 != 0 or len(poly) < 6:
                raise ValueError(
                    "Each mask polygon must have an even number of coordinates "
                    "(at least 3 vertices)."
                )
            mask_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [mask_pts], 255)

    line_counter = build_line_counter(args)
    frame_line_counts: dict[int, list[dict[str, int]]] = {}
    frame_straight_counts: dict[int, dict[str, int]] = {}
    frames_data: dict[int, list[dict[str, object]]] = defaultdict(list)
    csv_data: list[dict[str, object]] = []
    seen_track_ids: set[int] = set()

    frame_id = 0
    print(f"Processing video {args.video} at {fps} FPS with RT-DETR + OC-SORT...")
    while cap.isOpened():
        if max_frames and frame_id >= max_frames:
            print(f"Reached maximum duration of {args.duration} seconds. Stopping inference.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        if crop_bounds:
            x1_c, y1_c, x2_c, y2_c = crop_bounds
            frame = frame[y1_c:y2_c, x1_c:x2_c]

        if mask is not None:
            frame[mask == 255] = 0

        # results = model.predict(frame, verbose=False, conf=args.conf, imgsz=args.imgsz)
        results = model.predict(frame, verbose=False, conf=args.conf, imgsz=args.imgsz, device=0)

        detections = detections_from_result(results[0], width, height, args.max_box_ratio)
        tracks = tracker.update(detections, frame)
        tracked_objects = parse_ocsort_tracks(tracks)

        frame_tracks = []
        for obj in tracked_objects:
            track_id = int(obj["id"])
            class_id = int(obj["class"])
            cx, cy, w_box, h_box = obj["box"]
            seen_track_ids.add(track_id)

            csv_data.append(
                {
                    "frame_id": frame_id,
                    "vehicle_id": track_id,
                    "class_id": class_id,
                    "confidence": float(obj["conf"]),
                    "x_centre": float(cx),
                    "y_centre": float(cy),
                    "width": float(w_box),
                    "height": float(h_box),
                }
            )
            frames_data[frame_id].append(obj)
            frame_tracks.append(
                {"id": track_id, "center": (float(cx), float(cy)), "class": class_id}
            )

        if line_counter is not None:
            line_counter.update(frame_id, frame_tracks)
            frame_line_counts[frame_id] = [
                counter.count_by_direction.copy() for counter in line_counter.counters
            ]
            frame_straight_counts[frame_id] = line_counter.straight_path_counts()

        if args.display:
            display_frame = frame.copy()
            draw_annotations(
                display_frame, frames_data[frame_id], line_counter, frame_tracks, height
            )
            cv2.imshow("RT-DETR + OC-SORT", display_frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("User requested exit via display window.")
                break

        frame_id += 1

    cap.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"\n[OC-SORT Complete] Unique OC-SORT IDs observed: {len(seen_track_ids)}")
    observation_counts = Counter(row["class_id"] for row in csv_data)
    print("Class observations:")
    for class_id, count in sorted(observation_counts.items()):
        print(f" - {CLASS_NAMES.get(class_id, 'Unknown')}: {count}")

    if args.output_csv:
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.sort_values(by=["frame_id", "vehicle_id"], inplace=True)
        else:
            df = pd.DataFrame(
                columns=[
                    "frame_id",
                    "vehicle_id",
                    "class_id",
                    "confidence",
                    "x_centre",
                    "y_centre",
                    "width",
                    "height",
                ]
            )
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved numeric tracking features to {args.output_csv}")

    print_line_summaries(line_counter, args)

    if args.output_video:
        write_debug_video(
            args=args,
            fps=fps,
            width=width,
            height=height,
            frames_data=frames_data,
            frame_line_counts=frame_line_counts,
            frame_straight_counts=frame_straight_counts,
            max_frames=max_frames,
            crop=crop_bounds,
        )


def draw_annotations(
    frame,
    objects: list[dict[str, object]],
    line_counter: MultiLineCrossingCounter | None,
    line_tracks: list[dict[str, object]],
    height: int,
    count_overrides: list[dict[str, int]] | None = None,
    straight_counts: dict[str, int] | None = None,
) -> None:
    for obj in objects:
        cx, cy, w_box, h_box = obj["box"]
        x1, y1 = int(cx - w_box / 2), int(cy - h_box / 2)
        x2, y2 = int(cx + w_box / 2), int(cy + h_box / 2)
        class_id = int(obj["class"])
        color = COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        conf_text = f" {float(obj['conf']):.2f}" if "conf" in obj else ""
        cv2.putText(
            frame,
            f"ID:{obj['id']} {CLASS_NAMES.get(class_id, 'Unknown')}{conf_text}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    if line_counter is not None:
        line_counter.annotate(frame, line_tracks, count_overrides=count_overrides)
        if straight_counts:
            overlay = " | ".join(
                f"{pretty_path_name(name)}: {count}" for name, count in straight_counts.items()
            )
            if overlay:
                cv2.putText(
                    frame,
                    f"Straight counts: {overlay}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )


def pretty_path_name(name: str) -> str:
    parts = name.split("_to_")
    if len(parts) == 2:
        return f"To{parts[1].title()}"
    return name.title()


def print_line_summaries(
    line_counter: MultiLineCrossingCounter | None, args: argparse.Namespace
) -> None:
    if line_counter is None:
        return

    if args.show_realtime_counts:
        print("\nLine crossing summary by exit:")
        for name, counts in line_counter.directional_totals().items():
            print(
                f" - {name}: neg->pos={counts['negative_to_positive']}, "
                f"pos->neg={counts['positive_to_negative']}"
            )

        print("\nLine crossing summary by exit and class:")
        for name, class_counts in line_counter.class_direction_totals().items():
            print(f" - {name}:")
            for vehicle_class, counts in class_counts.items():
                print(
                    f"    class={vehicle_class}: neg->pos={counts['negative_to_positive']}, "
                    f"pos->neg={counts['positive_to_negative']}"
                )

    if args.show_straight_counts:
        print("\nStraight path counts:")
        for path, count in line_counter.straight_path_counts().items():
            print(f" - {path}: {count}")


def write_debug_video(
    args: argparse.Namespace,
    fps: int,
    width: int,
    height: int,
    frames_data: dict[int, list[dict[str, object]]],
    frame_line_counts: dict[int, list[dict[str, int]]],
    frame_straight_counts: dict[int, dict[str, int]],
    max_frames: int | None,
    crop: tuple[int, int, int, int] | None,
) -> None:
    print(f"Generating debug video {args.output_video}...")
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    line_counter = build_line_counter(args)

    mask = None
    if args.mask_polygon:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in args.mask_polygon:
            mask_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [mask_pts], 255)

    frame_idx = 0
    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if crop:
            x1_c, y1_c, x2_c, y2_c = crop
            frame = frame[y1_c:y2_c, x1_c:x2_c]

        if mask is not None:
            frame[mask == 255] = 0

        line_tracks = [
            {"id": int(obj["id"]), "center": (obj["box"][0], obj["box"][1])}
            for obj in frames_data.get(frame_idx, [])
        ]
        draw_annotations(
            frame,
            frames_data.get(frame_idx, []),
            line_counter,
            line_tracks,
            height,
            count_overrides=frame_line_counts.get(frame_idx),
            straight_counts=frame_straight_counts.get(frame_idx),
        )
        out_writer.write(frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Debug video successfully saved to {args.output_video}")


if __name__ == "__main__":
    main()
