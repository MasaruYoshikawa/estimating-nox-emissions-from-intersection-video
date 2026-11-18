from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from vehicle_emission import database
from vehicle_emission.line_counting import LineOfInterest, MultiLineCrossingCounter
from vehicle_emission.time_utils import compact_local_filename_timestamp_utc
from vehicle_emission.vehicle_classes import PROJECT_CLASS_TO_EMISSION_VEHICLE_TYPE

CLASS_NAMES = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}
COLORS = {
    0: (0, 165, 255),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
}
TRACKING_CSV_COLUMNS = [
    "frame_id",
    "vehicle_id",
    "class_id",
    "confidence",
    "x_centre",
    "y_centre",
    "width",
    "height",
]


def should_collect_annotations(*, display: bool, save_debug_video: bool) -> bool:
    """Return whether the current pass needs annotation state in memory."""

    return display or save_debug_video


def tracking_pass_is_complete(
    *,
    expected_frames: int | None,
    frames_advanced: int,
    decoded_frame_count: int,
) -> bool:
    """Return whether a tracking pass has positive evidence of complete processing."""

    if decoded_frame_count <= 0:
        return False
    return expected_frames is not None and frames_advanced >= expected_frames


def _stage_tracking_csv(output_csv: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write a complete CSV to a temporary sibling without publishing it."""

    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataframe = pd.DataFrame(rows, columns=TRACKING_CSV_COLUMNS)
        if not dataframe.empty:
            dataframe.sort_values(by=["frame_id", "vehicle_id"], inplace=True)
        dataframe.to_csv(temporary, index=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, target


def persist_tracking_outputs(
    *,
    db_path: str | None,
    output_csv: str | None,
    csv_data: list[dict[str, Any]],
    video_id: int | None,
    pending_registration: dict[str, Any] | None,
    crossing_events: list[dict[str, Any]],
    entering_counts: list[dict[str, Any]],
) -> int | None:
    """Commit SQLite output before atomically publishing the staged CSV."""

    db_tracks = [
        {
            "vehicle_id": row["vehicle_id"],
            "frame_id": row["frame_id"],
            "class_id": row["class_id"],
            "confidence": row["confidence"],
            "x_centre": row["x_centre"],
            "y_centre": row["y_centre"],
            "width": row["width"],
            "height": row["height"],
        }
        for row in csv_data
    ]
    staged_csv = _stage_tracking_csv(output_csv, csv_data) if output_csv else None
    csv_published = False
    resolved_video_id = video_id
    try:
        if db_path:
            if pending_registration is not None:
                resolved_video_id = database.register_video_with_tracking_results(
                    db_path,
                    tracks=db_tracks,
                    crossing_events=crossing_events,
                    entering_counts=entering_counts,
                    **pending_registration,
                )
            elif video_id is not None:
                database.replace_video_tracking_results(
                    db_path,
                    video_id,
                    db_tracks,
                    crossing_events,
                    entering_counts,
                )
            else:
                raise RuntimeError("Database tracking output has no resolved video identity")

        if staged_csv is not None:
            temporary, target = staged_csv
            os.replace(temporary, target)
            csv_published = True
    finally:
        if staged_csv is not None and not csv_published:
            staged_csv[0].unlink(missing_ok=True)

    return resolved_video_id


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


def calculate_and_print_entering_counts_by_class_pure(
    line_counter: MultiLineCrossingCounter | None,
    exit_lines_args: list[list[float]],
    class_names: dict[int, str],
) -> None:
    """Post-processing step to calculate and print entering counts per vehicle class."""
    if line_counter is None or not exit_lines_args or len(exit_lines_args) < 4:
        return

    exit_lines = {
        f"exit{i + 1}": (coords[0:2], coords[2:4]) for i, coords in enumerate(exit_lines_args)
    }

    y_center_exit1 = (exit_lines["exit1"][0][1] + exit_lines["exit1"][1][1]) / 2.0
    y_center_exit3 = (exit_lines["exit3"][0][1] + exit_lines["exit3"][1][1]) / 2.0
    if y_center_exit1 < y_center_exit3:  # exit1 is physically above exit3
        top_line = "exit1"
        bottom_line = "exit3"
    else:  # exit3 is physically above exit1
        top_line = "exit3"
        bottom_line = "exit1"

    entering_definitions = {
        "entering_from_left": ("exit4", "positive_to_negative"),
        "entering_from_right": ("exit2", "negative_to_positive"),
        "entering_from_top": (top_line, "negative_to_positive"),
        "entering_from_bottom": (bottom_line, "positive_to_negative"),
    }

    events_by_name = {
        name: counter.events
        for (name, _, _), counter in zip(
            line_counter.line_specs, line_counter.counters, strict=False
        )
    }

    vehicle_history = {}
    for name, events in events_by_name.items():
        for event in events:
            vehicle_id = int(event["vehicle_id"])
            vehicle_class = event.get("class", -1)
            vehicle_history.setdefault(vehicle_id, []).append(
                (event["frame_id"], name, event["direction"], vehicle_class)
            )

    for history in vehicle_history.values():
        history.sort(key=lambda item: item[0])

    path_counts = defaultdict(Counter)
    for _, history in vehicle_history.items():
        # Find the first event that matches any entering direction
        for _, line, direction, vehicle_class in history:
            matched = False
            for entry_key, (entry_line, entry_dir) in entering_definitions.items():
                if line == entry_line and direction == entry_dir:
                    cls_name = class_names.get(vehicle_class, "Unknown")
                    path_counts[entry_key][cls_name] += 1
                    matched = True
                    break
            if matched:
                break

    print("\nEntering counts by class (post-processed):")
    if not path_counts:
        print(" - No entering vehicles detected.")
        return
    for path, class_counts in sorted(path_counts.items()):
        pretty_path = path.replace("_", " ").title()
        print(f" - {pretty_path}:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    - {class_name}: {count}")


def get_entering_counts_by_class(
    line_counter: MultiLineCrossingCounter | None,
    exit_lines_args: list[list[float]],
) -> list[dict[str, Any]]:
    """Calculate aggregate entering counts per direction and class (and total per direction)."""
    results = []
    if line_counter is None or not exit_lines_args or len(exit_lines_args) < 4:
        return results

    exit_lines = {
        f"exit{i + 1}": (coords[0:2], coords[2:4]) for i, coords in enumerate(exit_lines_args)
    }

    y_center_exit1 = (exit_lines["exit1"][0][1] + exit_lines["exit1"][1][1]) / 2.0
    y_center_exit3 = (exit_lines["exit3"][0][1] + exit_lines["exit3"][1][1]) / 2.0
    if y_center_exit1 < y_center_exit3:  # exit1 is physically above exit3
        top_line = "exit1"
        bottom_line = "exit3"
    else:
        top_line = "exit3"
        bottom_line = "exit1"

    entering_definitions = {
        "entering_from_left": ("exit4", "positive_to_negative"),
        "entering_from_right": ("exit2", "negative_to_positive"),
        "entering_from_top": (top_line, "negative_to_positive"),
        "entering_from_bottom": (bottom_line, "positive_to_negative"),
    }

    events_by_name = {
        name: counter.events
        for (name, _, _), counter in zip(
            line_counter.line_specs, line_counter.counters, strict=False
        )
    }

    vehicle_history = {}
    for name, events in events_by_name.items():
        for event in events:
            vehicle_id = int(event["vehicle_id"])
            vehicle_class = event.get("class", -1)
            vehicle_history.setdefault(vehicle_id, []).append(
                (event["frame_id"], name, event["direction"], vehicle_class)
            )

    for history in vehicle_history.values():
        history.sort(key=lambda item: item[0])

    path_counts = defaultdict(Counter)
    direction_totals = defaultdict(int)

    for _, history in vehicle_history.items():
        for _, line, direction, vehicle_class in history:
            matched = False
            for entry_key, (entry_line, entry_dir) in entering_definitions.items():
                if line == entry_line and direction == entry_dir:
                    if vehicle_class in PROJECT_CLASS_TO_EMISSION_VEHICLE_TYPE:
                        path_counts[entry_key][int(vehicle_class)] += 1
                    direction_totals[entry_key] += 1
                    matched = True
                    break
            if matched:
                break

    # Format for DB insert:
    # 1. Total counts per direction (class_id = None)
    for direction, total in direction_totals.items():
        results.append({"direction": direction, "class_id": None, "vehicle_count": total})
    # 2. Counts per class within each direction
    for direction, class_counts in path_counts.items():
        for class_id, count in class_counts.items():
            results.append(
                {
                    "direction": direction,
                    "class_id": int(class_id),
                    "vehicle_count": count,
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trained RT-DETR with pure ByteTrack and Entering Counting"
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
        "--show-entering-counts",
        action="store_true",
        help="Print counts of vehicles entering the intersection from the four directions",
    )
    parser.add_argument("--display", action="store_true", help="Show realtime annotated frames")
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="bytetrack.yaml",
        help="Path to custom bytetrack.yaml, or Ultralytics default bytetrack.yaml",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="RT-DETR confidence threshold")
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
        "--db-path", type=str, help="Path to SQLite database (e.g. data/results.db)"
    )
    parser.add_argument(
        "--video-start-time",
        type=str,
        help="ISO-8601 UTC timestamp of video start time (e.g. 2026-06-27T10:00:00Z)",
    )
    parser.add_argument(
        "--filename-timezone",
        type=str,
        help=(
            "IANA timezone for a local YYYYMMDD_HHMM timestamp embedded in the filename "
            "(for example Asia/Tokyo or Pacific/Auckland)"
        ),
    )
    parser.add_argument(
        "--use-file-mtime-as-start",
        action="store_true",
        help="Explicitly use the file modification epoch as the recording start",
    )
    parser.add_argument(
        "--intersection",
        type=str,
        default="Sukiyabashi",
        help="Intersection ID/name (default: Sukiyabashi)",
    )
    parser.add_argument(
        "--db-notes",
        type=str,
        help="Optional description/notes for the database video master entry",
    )
    parser.add_argument(
        "--video-id",
        type=int,
        help="Specify an existing video ID from the video_master database table",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        help="Optional group ID to associate with the registered video in the database",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target processing FPS (e.g. 10). Automatically calculates stride based on video FPS.",
    )
    parser.add_argument(
        "--debug-first-only",
        action="store_true",
        help="Only write the debug annotated video for the first video processed in this run",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half-precision inference on GPU (faster, uses less memory)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that have already been processed in the database or output CSV directory",
    )
    parser.add_argument(
        "--filter-resolution",
        type=str,
        default=None,
        help="Only process videos matching this resolution (e.g. 1440x900 or 2560x1440)",
    )
    return parser.parse_args()


def parse_start_time_from_filename(filepath: str, source_timezone: str | None) -> str | None:
    """Parse a local filename clock only when its source timezone is explicit."""

    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if len(parts) >= 3:
        date_part = parts[1]
        time_part = parts[2]
        if (
            len(date_part) == 8
            and len(time_part) == 4
            and date_part.isdigit()
            and time_part.isdigit()
        ):
            if source_timezone is None:
                raise ValueError(
                    "Filename contains a local clock timestamp; pass --filename-timezone "
                    "or an explicit --video-start-time"
                )
            return compact_local_filename_timestamp_utc(
                date_part,
                time_part,
                timezone_name=source_timezone,
            )
    return None


def main() -> None:
    from ultralytics import RTDETR

    args = parse_args()

    print(f"Loading RT-DETR model from {args.model}...")
    model = RTDETR(args.model)

    # Determine video list (accepts directory or single file)
    video_list = []
    if os.path.isdir(args.video):
        valid_exts = (".mp4", ".avi", ".mkv", ".mov")
        for f in sorted(os.listdir(args.video)):
            if f.lower().endswith(valid_exts):
                video_list.append(os.path.normpath(os.path.join(args.video, f)))
        if not video_list:
            raise FileNotFoundError(f"No valid video files found in directory: {args.video}")
        print(f"Found {len(video_list)} videos in directory. Processing sequentially...")
    else:
        video_list = [args.video]

    if args.db_path:
        database.init_db(args.db_path)

    # Process each video sequentially
    processed_count = 0
    for idx, video_path in enumerate(video_list, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if args.skip_existing:
            already_processed = False
            # Check database if db_path is given
            if (
                args.db_path
                and os.path.exists(args.db_path)
                and database.check_video_name_processed(args.db_path, video_name)
            ):
                already_processed = True

            # Check output CSV if output_csv is given
            if not already_processed and args.output_csv:
                target_csv = args.output_csv
                if os.path.isdir(target_csv) or not os.path.splitext(target_csv)[1]:
                    target_csv = os.path.join(target_csv, f"{video_name}_tracking.csv")
                elif len(video_list) > 1:
                    target_csv = os.path.join(
                        os.path.dirname(target_csv), f"{video_name}_tracking.csv"
                    )
                if os.path.exists(target_csv) and os.path.getsize(target_csv) > 0:
                    already_processed = True

            if already_processed:
                print(f"[{idx}/{len(video_list)}] Skipping already-processed video: {video_name}")
                continue

        print("\n==========================================")
        print(f"Processing video {idx}/{len(video_list)}: {video_path}")
        print("==========================================")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open required video: {video_path}")

        if args.filter_resolution:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            curr_res = f"{w}x{h}"
            if curr_res != args.filter_resolution:
                print(
                    f"[{idx}/{len(video_list)}] Skipping video {video_name} (Resolution {curr_res} does not match --filter-resolution {args.filter_resolution})"
                )
                cap.release()
                continue

        processed_count += 1

        video_id = None
        pending_registration: dict[str, Any] | None = None
        if args.video_id is not None and len(video_list) == 1:
            if not args.db_path:
                raise ValueError("--db-path must be specified when using --video-id")
            if not database.check_video_id_exists(args.db_path, args.video_id):
                raise ValueError(
                    f"Specified --video-id {args.video_id} does not exist in the database: {args.db_path}"
                )
            video_id = args.video_id
            print(f"Using pre-registered video_id {video_id} from database.")
        elif args.db_path:
            start_time = args.video_start_time
            if not start_time:
                parsed_time = parse_start_time_from_filename(video_path, args.filename_timezone)
                if parsed_time:
                    start_time = parsed_time
                    print(f"Parsed start time from filename: {start_time}")
                elif args.use_file_mtime_as_start:
                    mtime = os.path.getmtime(video_path)
                    dt = datetime.fromtimestamp(mtime, tz=UTC)
                    start_time = dt.isoformat().replace("+00:00", "Z")
                    print("Using explicitly authorised file modification time as start time.")
                else:
                    raise ValueError(
                        "Recording start time is unresolved. Pass --video-start-time, "
                        "--filename-timezone for a timestamped filename, or explicitly "
                        "authorise --use-file-mtime-as-start."
                    )

            pending_registration = {
                "video_path": video_path,
                "intersection_id": args.intersection,
                "start_time_str": start_time,
                "duration_limit": args.duration,
                "notes": args.db_notes,
                "group_id": args.group_id,
            }
            print("Video registration will be committed after tracking succeeds.")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = args.duration * fps if args.duration else None
        expected_frames = source_frame_count if source_frame_count > 0 else None
        if max_frames is not None:
            expected_frames = min(expected_frames, max_frames) if expected_frames else max_frames

        stride = 1
        if args.target_fps is not None and args.target_fps > 0:
            stride = max(1, int(round(fps / args.target_fps)))

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
                raise ValueError(
                    f"Crop region is outside video bounds ({orig_width}x{orig_height})"
                )
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

        # Determine if we need to write/save debug video frames for this video to avoid RAM bloat
        save_debug_video = False
        if args.output_video and (not args.debug_first_only or processed_count == 1):
            save_debug_video = True
        collect_annotations = should_collect_annotations(
            display=args.display,
            save_debug_video=save_debug_video,
        )

        line_counter = build_line_counter(args)
        frame_line_counts = {}
        frame_entering_counts = {}
        frames_data = defaultdict(list)
        csv_data = []
        seen_track_ids = set()

        frame_id = 0
        decoded_frame_count = 0
        processing_complete = False
        if stride > 1:
            print(
                f"Processing video {video_path} (native {fps} FPS) downsampled to target {args.target_fps} FPS (stride={stride})..."
            )
        else:
            print(f"Processing video {video_path} at native {fps} FPS...")
        while cap.isOpened():
            if max_frames and frame_id >= max_frames:
                print(f"Reached maximum duration of {args.duration} seconds. Stopping inference.")
                processing_complete = tracking_pass_is_complete(
                    expected_frames=expected_frames,
                    frames_advanced=frame_id,
                    decoded_frame_count=decoded_frame_count,
                )
                break

            # Speed optimization: use cap.grab() to skip decoding when striding
            if stride > 1 and frame_id % stride != 0:
                ret = cap.grab()
                if not ret:
                    processing_complete = tracking_pass_is_complete(
                        expected_frames=expected_frames,
                        frames_advanced=frame_id,
                        decoded_frame_count=decoded_frame_count,
                    )
                    break
                frame_id += 1
                continue

            ret, frame = cap.read()
            if not ret:
                processing_complete = tracking_pass_is_complete(
                    expected_frames=expected_frames,
                    frames_advanced=frame_id,
                    decoded_frame_count=decoded_frame_count,
                )
                break
            decoded_frame_count += 1

            if args.crop:
                frame = frame[y1_c:y2_c, x1_c:x2_c]

            if mask is not None:
                frame[mask == 255] = 0

            results = model.track(
                frame,
                tracker=args.tracker_config,
                persist=True,
                verbose=False,
                conf=args.conf,
                imgsz=args.imgsz,
                device=0,
                half=args.half,
            )

            frame_tracks = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, class_id, conf in zip(
                    boxes, track_ids, classes, confs, strict=False
                ):
                    cx, cy, w_box, h_box = box
                    if w_box >= width * args.max_box_ratio or h_box >= height * args.max_box_ratio:
                        continue

                    track_id = int(track_id)
                    class_id = int(class_id)
                    conf = float(conf)
                    seen_track_ids.add(track_id)

                    record = {
                        "frame_id": frame_id,
                        "vehicle_id": track_id,
                        "class_id": class_id,
                        "confidence": conf,
                        "x_centre": float(cx),
                        "y_centre": float(cy),
                        "width": float(w_box),
                        "height": float(h_box),
                    }
                    csv_data.append(record)
                    if collect_annotations:
                        frames_data[frame_id].append(
                            {"id": track_id, "class": class_id, "box": box, "conf": conf}
                        )
                    frame_tracks.append(
                        {"id": track_id, "center": (float(cx), float(cy)), "class": class_id}
                    )

            if line_counter is not None:
                line_counter.update(frame_id, frame_tracks)
                if collect_annotations:
                    frame_line_counts[frame_id] = [
                        counter.count_by_direction.copy() for counter in line_counter.counters
                    ]
                    frame_entering_counts[frame_id] = line_counter.entering_path_counts()

            if args.display:
                display_frame = frame.copy()
                draw_annotations(
                    display_frame,
                    frames_data[frame_id],
                    line_counter,
                    frame_tracks,
                    height,
                    count_overrides=frame_line_counts.get(frame_id),
                    entering_counts=frame_entering_counts.get(frame_id),
                )
                cv2.imshow("Pure ByteTrack", display_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("User requested exit via display window.")
                    break

            frame_id += 1

        cap.release()
        if args.display:
            cv2.destroyAllWindows()

        if not processing_complete:
            raise RuntimeError(
                f"Video processing ended before the configured input was complete: {video_path}. "
                "Existing database rows were preserved."
            )

        print(f"\n[Pure ByteTrack Complete] Unique ByteTrack IDs observed: {len(seen_track_ids)}")
        observation_counts = Counter(row["class_id"] for row in csv_data)
        print("Class observations, not voted per vehicle:")
        for class_id, count in sorted(observation_counts.items()):
            print(f" - {CLASS_NAMES.get(class_id, 'Unknown')}: {count}")

        # Determine unique output CSV if folder mode or directory path
        curr_output_csv = args.output_csv
        if curr_output_csv:
            if os.path.isdir(curr_output_csv) or not os.path.splitext(curr_output_csv)[1]:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                curr_output_csv = os.path.join(curr_output_csv, f"{base_name}_tracking.csv")
            elif len(video_list) > 1:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                curr_output_csv = os.path.join(
                    os.path.dirname(curr_output_csv), f"{base_name}_tracking.csv"
                )

        db_events = []
        db_entering_counts = []
        if args.db_path:
            print(f"Replacing {len(csv_data)} track records in database: {args.db_path}...")
            if line_counter is not None:
                for (name, _, _), counter in zip(
                    line_counter.line_specs, line_counter.counters, strict=False
                ):
                    for event in counter.events:
                        db_events.append(
                            {
                                "vehicle_id": event["vehicle_id"],
                                "line_name": name,
                                "frame_id": event["frame_id"],
                                "direction": event["direction"],
                                "class_id": event.get("class"),
                            }
                        )
                db_entering_counts = get_entering_counts_by_class(line_counter, args.exit_line)

        video_id = persist_tracking_outputs(
            db_path=args.db_path,
            output_csv=curr_output_csv,
            csv_data=csv_data,
            video_id=video_id,
            pending_registration=pending_registration,
            crossing_events=db_events,
            entering_counts=db_entering_counts,
        )
        if args.db_path and pending_registration is not None:
            print(f"Registered video and tracking results with ID {video_id}.")
        if curr_output_csv:
            print(f"Saved numeric tracking features to {curr_output_csv}")

        print_entering_summaries(line_counter, args)

        # Determine unique output video if folder mode or directory path
        curr_output_video = args.output_video
        if curr_output_video:
            if os.path.isdir(curr_output_video) or not os.path.splitext(curr_output_video)[1]:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                curr_output_video = os.path.join(curr_output_video, f"{base_name}_debug.mp4")
            elif len(video_list) > 1:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                curr_output_video = os.path.join(
                    os.path.dirname(curr_output_video), f"{base_name}_debug.mp4"
                )

        if curr_output_video:
            if not args.debug_first_only or processed_count == 1:
                write_debug_video(
                    args=args,
                    fps=fps,
                    width=width,
                    height=height,
                    frames_data=frames_data,
                    frame_line_counts=frame_line_counts,
                    frame_entering_counts=frame_entering_counts,
                    max_frames=max_frames,
                    crop=crop_bounds,
                    video_path=video_path,
                    output_video=curr_output_video,
                )
            else:
                print(
                    f"Skipping debug video generation for {video_path} (--debug-first-only is active)."
                )

        # Free GPU memory cache between videos to prevent CUDA unspecified launch failures
        import gc

        import torch

        torch.cuda.empty_cache()
        gc.collect()


def draw_annotations(
    frame,
    objects: list[dict[str, object]],
    line_counter: MultiLineCrossingCounter | None,
    line_tracks: list[dict[str, object]],
    height: int,
    count_overrides: list[dict[str, int]] | None = None,
    entering_counts: dict[str, int] | None = None,
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
        if entering_counts:
            overlay = " | ".join(
                f"{pretty_entering_name(name)}: {count}" for name, count in entering_counts.items()
            )
            if overlay:
                cv2.putText(
                    frame,
                    f"Entering counts: {overlay}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )


def pretty_entering_name(name: str) -> str:
    return name.replace("entering_from_", "From ").title()


def print_entering_summaries(
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

    if args.show_entering_counts:
        print("\nEntering counts:")
        for path, count in line_counter.entering_path_counts().items():
            print(f" - {path.replace('_', ' ').title()}: {count}")
        calculate_and_print_entering_counts_by_class_pure(line_counter, args.exit_line, CLASS_NAMES)


def write_debug_video(
    args: argparse.Namespace,
    fps: int,
    width: int,
    height: int,
    frames_data: dict[int, list[dict[str, object]]],
    frame_line_counts: dict[int, list[dict[str, int]]],
    frame_entering_counts: dict[int, dict[str, int]],
    max_frames: int | None,
    crop: tuple[int, int, int, int] | None,
    video_path: str | None = None,
    output_video: str | None = None,
) -> None:
    vid_in = video_path if video_path else args.video
    vid_out = output_video if output_video else args.output_video

    # Calculate stride and effective FPS for the debug video
    stride = 1
    if args.target_fps is not None and args.target_fps > 0:
        stride = max(1, int(round(fps / args.target_fps)))

    effective_fps = fps / stride
    print(f"Generating debug video {vid_out} at {effective_fps:.2f} FPS (stride={stride})...")
    cap = cv2.VideoCapture(vid_in)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(vid_out).parent.mkdir(parents=True, exist_ok=True)
    out_writer = cv2.VideoWriter(vid_out, fourcc, effective_fps, (width, height))
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

        if stride > 1 and frame_idx % stride != 0:
            frame_idx += 1
            continue

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
            entering_counts=frame_entering_counts.get(frame_idx),
        )
        out_writer.write(frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Debug video successfully saved to {vid_out}")


if __name__ == "__main__":
    main()
