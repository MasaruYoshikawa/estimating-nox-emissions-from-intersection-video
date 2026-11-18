import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR

from vehicle_emission.line_counting import (
    LineOfInterest,
    MultiLineCrossingCounter,
)


def get_crossing_direction(line_start, line_end, p1, p2):
    """
    Checks if the movement segment p1->p2 crosses the line segment line_start->line_end.
    Returns 'positive_to_negative', 'negative_to_positive', or None.
    'Positive' vs 'negative' refers to the two sides of the line.
    """

    def get_side(p, line_p1, line_p2):
        return (line_p2[0] - line_p1[0]) * (p[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (
            p[0] - line_p1[0]
        )

    side1 = get_side(p1, line_start, line_end)
    side2 = get_side(p2, line_start, line_end)

    if side1 * side2 < 0:  # Crosses the infinite line defined by the line segment
        # Check if the intersection point is actually on both segments
        d = (line_end[0] - line_start[0]) * (p2[1] - p1[1]) - (line_end[1] - line_start[1]) * (
            p2[0] - p1[0]
        )
        if d == 0:
            return None  # Parallel lines

        t = (
            (p1[0] - line_start[0]) * (p2[1] - p1[1]) - (p1[1] - line_start[1]) * (p2[0] - p1[0])
        ) / d
        u = (
            -(
                (line_end[0] - line_start[0]) * (p1[1] - line_start[1])
                - (line_end[1] - line_start[1]) * (p1[0] - line_start[0])
            )
            / d
        )

        if 0 <= t <= 1 and 0 <= u <= 1:
            return "positive_to_negative" if side1 > 0 else "negative_to_positive"
    return None


def calculate_and_print_straight_path_counts_by_class(final_vehicles, exit_lines_args, class_names):
    """Post-processing step to calculate and print straight path counts per vehicle class."""
    if not exit_lines_args or len(exit_lines_args) < 4:
        return

    exit_lines = {
        f"exit{i+1}": (coords[0:2], coords[2:4]) for i, coords in enumerate(exit_lines_args)
    }

    # Dynamically determine up/down path definitions based on line positions
    path_definitions = {
        "right_to_left": (("exit4", "positive_to_negative"), ("exit2", "positive_to_negative")),
        "left_to_right": (("exit2", "negative_to_positive"), ("exit4", "negative_to_positive")),
    }
    y_center_exit1 = (exit_lines["exit1"][0][1] + exit_lines["exit1"][1][1]) / 2
    y_center_exit3 = (exit_lines["exit3"][0][1] + exit_lines["exit3"][1][1]) / 2
    if y_center_exit1 < y_center_exit3:  # exit1 is physically above exit3
        path_definitions["top_to_bottom"] = (
            ("exit1", "negative_to_positive"),
            ("exit3", "negative_to_positive"),
        )
        path_definitions["bottom_to_top"] = (
            ("exit3", "positive_to_negative"),
            ("exit1", "positive_to_negative"),
        )
    else:  # exit3 is physically above exit1
        path_definitions["top_to_bottom"] = (
            ("exit3", "negative_to_positive"),
            ("exit1", "negative_to_positive"),
        )
        path_definitions["bottom_to_top"] = (
            ("exit1", "positive_to_negative"),
            ("exit3", "positive_to_negative"),
        )

    path_counts = defaultdict(Counter)
    for _, v_data in final_vehicles.items():
        history = sorted(v_data["history"], key=lambda x: x["frame"])
        if len(history) < 2:
            continue

        crossings = []
        for i in range(len(history) - 1):
            p1 = (history[i]["box"][0], history[i]["box"][1])
            p2 = (history[i + 1]["box"][0], history[i + 1]["box"][1])
            for name, (start, end) in exit_lines.items():
                direction = get_crossing_direction(start, end, p1, p2)
                if direction:
                    crossings.append(
                        {"line": name, "direction": direction, "frame": history[i + 1]["frame"]}
                    )

        if len(crossings) < 2:
            continue

        v_class_name = class_names.get(v_data["voted_class"], "Unknown")
        for path_name, required in path_definitions.items():
            first_events = [c for c in crossings if (c["line"], c["direction"]) == required[0]]
            if not first_events:
                continue
            second_events = [
                c
                for c in crossings
                if (c["line"], c["direction"]) == required[1]
                and c["frame"] > first_events[0]["frame"]
            ]
            if second_events:
                path_counts[path_name][v_class_name] += 1
                break

    print("\nStraight path counts by class (post-processed):")
    if not path_counts:
        print(" - No straight paths detected.")
        return
    for path, class_counts in sorted(path_counts.items()):
        parts = path.split("_to_")
        pretty_path = f"To{parts[1].title()}" if len(parts) == 2 else path.title()
        print(f" - {pretty_path}:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    - {class_name}: {count}")


class MultiVehicleVotingAlgorithm:
    """
    Implements the Multi-Vehicle Voting Algorithm proposed in:
    'Multi-Vehicle Tracking and Counting Framework in Average Daily Traffic Survey
    Using RT-DETR and ByteTrack' (IEEE Access 2024).
    """

    def __init__(self, max_lost_frames=30, spatial_threshold=50.0, min_votes=5):
        self.max_lost_frames = max_lost_frames
        self.spatial_threshold = spatial_threshold
        self.min_votes = min_votes  # Minimum frames required to be counted as a valid vehicle

        self.active_tracks = defaultdict(list)
        self.lost_tracks = defaultdict(list)
        self.finalized_vehicles = {}
        self.id_mapping = {}  # Resolves ID switches back to original IDs

    def _euclidean_distance(self, box1, box2):
        # box format: [cx, cy, w, h]
        return np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)

    def resolve_tracking_id(self, track_id):
        while track_id in self.id_mapping:
            track_id = self.id_mapping[track_id]
        return track_id

    def update(self, frame_id, current_tracks):
        current_ids = set()

        for track in current_tracks:
            orig_id = self.resolve_tracking_id(track["id"])

            # If we haven't seen this ID recently, it could be an ID switch of a lost track
            if orig_id not in self.active_tracks and orig_id not in self.lost_tracks:
                best_match_id = None
                best_dist = self.spatial_threshold

                # Check against recently lost tracks
                for lost_id, history in list(self.lost_tracks.items()):
                    last_obs = history[-1]
                    frames_lost = frame_id - last_obs["frame"]

                    if frames_lost <= self.max_lost_frames:
                        dist = self._euclidean_distance(track["box"], last_obs["box"])
                        if dist < best_dist:
                            best_dist = dist
                            best_match_id = lost_id
                    else:
                        # Clean up tracks lost for too long and finalize them
                        self._finalize_track(lost_id)
                        del self.lost_tracks[lost_id]

                if best_match_id is not None:
                    # SPATIAL VOTE: Proximity and temporal window match. Merge tracks.
                    self.id_mapping[orig_id] = best_match_id
                    orig_id = best_match_id
                    self.active_tracks[orig_id] = self.lost_tracks.pop(orig_id)

            self.active_tracks[orig_id].append(
                {
                    "frame": frame_id,
                    "box": track["box"],
                    "class": track["class"],
                    "conf": track["conf"],
                }
            )
            current_ids.add(orig_id)

        # Move inactive tracks to lost_tracks
        for active_id in list(self.active_tracks.keys()):
            if active_id not in current_ids:
                self.lost_tracks[active_id] = self.active_tracks.pop(active_id)

    def _finalize_track(self, track_id):
        history = self.active_tracks.get(track_id) or self.lost_tracks.get(track_id)
        if not history:
            return

        # HIT VOTE: Filter out transient tracks/false positives
        if len(history) >= self.min_votes:
            # CLASS VOTE: Majority voting for the vehicle's class across its entire lifecycle
            classes = [obs["class"] for obs in history]
            voted_class = Counter(classes).most_common(1)[0][0]

            self.finalized_vehicles[track_id] = {
                "voted_class": voted_class,
                "history": history,
                "start_frame": history[0]["frame"],
                "end_frame": history[-1]["frame"],
            }

    def finalize_all(self):
        for track_id in list(self.active_tracks.keys()):
            self._finalize_track(track_id)
        for track_id in list(self.lost_tracks.keys()):
            self._finalize_track(track_id)

        self.active_tracks.clear()
        self.lost_tracks.clear()
        return self.finalized_vehicles


def main():
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Tracking with Multi-Vehicle Voting Enhancement")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to your tuned RT-DETR model (e.g., best.pt)"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input traffic video")
    parser.add_argument(
        "--output-csv", type=str, help="Path to save the final tracked numeric features (CSV)"
    )
    parser.add_argument(
        "--output-video", type=str, help="Path to save the debug annotated video (MP4)"
    )
    parser.add_argument(
        "--max-box-ratio",
        type=float,
        default=0.4,
        help=(
            "Maximum allowed ratio of box width/height to frame dimension "
            "to filter out false huge boxes (default: 0.4)"
        ),
    )
    parser.add_argument(
        "--duration", type=int, help="Maximum video duration to process (in seconds)"
    )
    parser.add_argument(
        "--line-start",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Line start point in image pixels",
    )
    parser.add_argument(
        "--line-end", type=float, nargs=2, metavar=("X", "Y"), help="Line end point in image pixels"
    )
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
        help="Print negative/positive counts for each exit line while processing frames",
    )
    parser.add_argument(
        "--show-straight-counts",
        action="store_true",
        help="Print counts of vehicles matching the four straight intersection paths",
    )
    parser.add_argument(
        "--display", action="store_true", help="Show realtime annotated frames in a window"
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="bytetrack.yaml",
        help="Path to custom bytetrack.yaml (defaults to internal config)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="RT-DETR confidence threshold")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (e.g., 640, 1280, 1920)"
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop the video frame before processing (x1 y1 x2 y2)",
    )
    parser.add_argument(
        "--mask-polygon",
        type=float,
        nargs="+",
        action="append",
        help="Polygon vertices (x1 y1 x2 y2 ...) defining the keep region; can be repeated",
    )
    args = parser.parse_args()

    print(f"Loading RT-DETR model from {args.model}...")
    model = RTDETR(args.model)

    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = args.duration * fps if args.duration else None

    if args.crop:
        x1_c, y1_c, x2_c, y2_c = args.crop
        # Clamp crop coordinates to actual video bounds
        x1_c = max(0, min(x1_c, orig_width))
        x2_c = max(0, min(x2_c, orig_width))
        y1_c = max(0, min(y1_c, orig_height))
        y2_c = max(0, min(y2_c, orig_height))
        width, height = x2_c - x1_c, y2_c - y1_c
        if width <= 0 or height <= 0:
            print(
                "Error: Crop region is completely outside video bounds "
                f"({orig_width}x{orig_height})!"
            )
            return
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

    # Initialize the voting algorithm
    # - max_lost_frames: wait 1 second (fps) before dropping a track
    # - spatial_threshold: allow jump of ~5% of image width
    # - min_votes: require at least 5 frames of detections to count the vehicle
    voting_algo = MultiVehicleVotingAlgorithm(
        max_lost_frames=fps * 1, spatial_threshold=max(width, height) * 0.05, min_votes=5
    )

    line_counter = None
    if args.exit_line:
        line_specs = []
        for idx, coords in enumerate(args.exit_line):
            direction = None
            if idx < len(args.exit_line_dir):
                direction = args.exit_line_dir[idx][0]
            line_specs.append(
                (
                    f"exit{idx+1}",
                    LineOfInterest(start=(coords[0], coords[1]), end=(coords[2], coords[3])),
                    direction,
                )
            )
        line_counter = MultiLineCrossingCounter(line_specs)
    elif args.line_start is not None and args.line_end is not None:
        line_counter = MultiLineCrossingCounter(
            [
                (
                    "exit1",
                    LineOfInterest(start=tuple(args.line_start), end=tuple(args.line_end)),
                    args.count_direction,
                )
            ]
        )
    frame_line_counts = {}
    frame_straight_counts = {}
    raw_frames_data = defaultdict(list)

    frame_id = 0
    print(f"Processing video {args.video} at {fps} FPS...")
    while cap.isOpened():
        if max_frames and frame_id >= max_frames:
            print(f"Reached maximum duration of {args.duration} seconds. Stopping inference.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        if args.crop:
            frame = frame[y1_c:y2_c, x1_c:x2_c]

        if mask is not None:
            frame[mask == 255] = 0

        # 1. RT-DETR Detection + 2. Native ByteTrack Tracking
        results = model.track(
            frame,
            tracker=args.tracker_config,
            persist=True,
            verbose=False,
            conf=args.conf,
            imgsz=args.imgsz,
            device=0,
        )

        current_tracks = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, t_id, cls, conf in zip(boxes, track_ids, classes, confs, strict=False):
                _, _, w_box, h_box = box
                # Suppress huge bounding boxes (false positives like reflections or camera glitches)
                if w_box < width * args.max_box_ratio and h_box < height * args.max_box_ratio:
                    current_tracks.append({"id": t_id, "box": box, "class": cls, "conf": conf})

        # Resolve ID switches before line counting and debug export.
        voting_algo.update(frame_id, current_tracks)
        resolved_tracks = []
        for track in current_tracks:
            resolved_id = int(voting_algo.resolve_tracking_id(track["id"]))
            resolved_tracks.append(
                {
                    "id": resolved_id,
                    "box": track["box"],
                    "class": track["class"],
                    "conf": track["conf"],
                }
            )

        raw_frames_data[frame_id] = [
            {
                "id": int(track["id"]),
                "class": int(track["class"]),
                "box": track["box"],
                "conf": float(track["conf"]),
            }
            for track in resolved_tracks
        ]

        if line_counter is not None:
            line_tracks = [
                {
                    "id": int(track["id"]),
                    "center": (track["box"][0], track["box"][1]),
                    "class": int(track["class"]),
                }
                for track in resolved_tracks
            ]
            line_counter.update(frame_id, line_tracks)
            frame_line_counts[frame_id] = [
                counter.count_by_direction.copy() for counter in line_counter.counters
            ]
            frame_straight_counts[frame_id] = line_counter.straight_path_counts()

        # Realtime display (optional) - draw annotations and straight-path labels
        if args.display:
            disp = frame.copy()
            if line_counter is not None:
                # annotate lines and per-line counts
                count_override = frame_line_counts.get(frame_id)
                line_counter.annotate(
                    disp,
                    line_tracks if "line_tracks" in locals() else None,
                    count_overrides=count_override,
                )
                if args.show_straight_counts:
                    straight = frame_straight_counts.get(frame_id, {})
                    if straight:
                        # human-friendly labels: e.g. ToBottom
                        def pretty(name):
                            parts = name.split("_to_")
                            return f"To{parts[1].title()}" if len(parts) == 2 else name.title()

                        overlay = " | ".join(f"{pretty(k)}: {v}" for k, v in straight.items())
                        cv2.putText(
                            disp,
                            overlay,
                            (10, height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
            cv2.imshow("Realtime", disp)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("User requested exit via display window.")
                break
        frame_id += 1

    # Finalize tracks
    final_vehicles = voting_algo.finalize_all()
    total_vehicles = len(final_vehicles)
    print(f"\n[Voting Complete] Total unique valid vehicles: {total_vehicles}")

    class_counts = Counter(v["voted_class"] for v in final_vehicles.values())
    class_names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}
    print("Vehicle Class Breakdown:")
    for cls_id, count in sorted(class_counts.items()):
        print(f" - {class_names.get(cls_id, 'Unknown')}: {count}")

    # Prepare data for export and visualization
    csv_data = []
    frames_data = defaultdict(list)

    for v_id, v_data in final_vehicles.items():
        v_class = v_data["voted_class"]
        for obs in v_data["history"]:
            frame_idx = obs["frame"]
            cx, cy, w, h = obs["box"]

            csv_data.append(
                {
                    "frame_id": frame_idx,
                    "vehicle_id": v_id,
                    "class_id": v_class,
                    "confidence": obs["conf"],
                    "x_centre": cx,
                    "y_centre": cy,
                    "width": w,
                    "height": h,
                }
            )
            frames_data[frame_idx].append({"id": v_id, "class": v_class, "box": obs["box"]})

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

    if line_counter is not None and args.show_realtime_counts:
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

    if line_counter is not None and args.show_straight_counts:
        print("\nStraight path counts:")
        for path, count in line_counter.straight_path_counts().items():
            print(f" - {path}: {count}")
        calculate_and_print_straight_path_counts_by_class(
            final_vehicles, args.exit_line, class_names
        )

    if args.output_video:
        print(f"Generating debug video {args.output_video}...")
        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

        mask = None
        if args.mask_polygon:
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in args.mask_polygon:
                mask_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [mask_pts], 255)

        class_names = {0: "bus", 1: "car", 2: "heavy truck", 3: "light truck"}
        colors = {0: (0, 165, 255), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0)}  # BGR

        frame_idx = 0
        while cap.isOpened():
            # Stop writing if we reached the maximum duration
            if max_frames and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if args.crop:
                frame = frame[y1_c:y2_c, x1_c:x2_c]

            if mask is not None:
                frame[mask == 255] = 0

            for obj in raw_frames_data.get(frame_idx, []):
                cx, cy, w_box, h_box = obj["box"]
                x1, y1 = int(cx - w_box / 2), int(cy - h_box / 2)
                x2, y2 = int(cx + w_box / 2), int(cy + h_box / 2)
                color = colors.get(int(obj["class"]), (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                conf_str = f" {obj['conf']:.2f}" if "conf" in obj else ""
                cv2.putText(
                    frame,
                    f"ID:{obj['id']} {class_names.get(int(obj['class']))}{conf_str}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            if line_counter is not None:
                # Build lightweight track centers for annotation
                line_tracks_for_frame = [
                    {"id": int(o["id"]), "center": (o["box"][0], o["box"][1])}
                    for o in frames_data.get(frame_idx, [])
                ]
                count_override = frame_line_counts.get(frame_idx)
                line_counter.annotate(frame, line_tracks_for_frame, count_overrides=count_override)

                if args.show_straight_counts:

                    def pretty(name):
                        parts = name.split("_to_")
                        return f"To{parts[1].title()}" if len(parts) == 2 else name.title()

                    straight_counts = frame_straight_counts.get(frame_idx, {})
                    overlay = " | ".join(
                        f"{pretty(name)}: {count}" for name, count in straight_counts.items()
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

            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()
        print(f"Debug video successfully saved to {args.output_video}")
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
