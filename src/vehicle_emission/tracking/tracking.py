from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import RTDETR

from vehicle_emission.line_counting import LineCrossingCounter, LineOfInterest


def track_vehicles(
    video_path: str | Path,
    output_csv: str | Path,
    output_debug_video: str | Path | None = None,
    line_of_interest: LineOfInterest | None = None,
    count_direction: str | None = None,
    max_box_ratio: float = 0.4,
    duration: int | None = None,
) -> None:
    """
    Note: `model_path` is provided via the module-level CLI and passed below.

    Run RT-DETR + ByteTrack on the target video.
    Complies with INV-006 by persisting only numeric data (no raw image outputs) by default.
    Setting `output_debug_video` saves pixel data for local debugging only.

    If `line_of_interest` is supplied, vehicles are counted when their
    tracked center crosses the line.
    """
    # `model_path` will be injected by the caller (CLI entrypoint)
    model = RTDETR(str(track_vehicles.__dict__.get("_model_path", "rtdetr-l.pt")))
    target_classes = [2, 5, 7]
    line_counter = None
    if line_of_interest is not None:
        line_counter = LineCrossingCounter(line=line_of_interest, direction=count_direction)

    # Determine max frames if duration requested
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    max_frames = int(duration * fps) if duration else None

    # model.track() integrates ByteTrack natively
    results = model.track(
        source=str(video_path),
        stream=True,
        classes=target_classes,
        conf=0.3,
        tracker="bytetrack.yaml",
    )

    detections = []
    video_writer = None

    for frame_idx, r in enumerate(results):
        if max_frames and frame_idx >= max_frames:
            break

        frame_tracks = []
        for box in r.boxes:
            # Extract numeric coordinate features
            cx, cy, w, h = box.xywh[0].tolist()

            # get original frame shape to compare against box ratio
            h_shape, w_shape = r.orig_shape

            # Extract tracking ID if assigned by ByteTrack, else default to -1
            track_id = int(box.id[0].item()) if box.id is not None else -1

            # Suppress huge bounding boxes (false positives like reflections or camera glitches)
            if not (w < w_shape * max_box_ratio and h < h_shape * max_box_ratio):
                continue

            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "frame_id": frame_idx,
                    "vehicle_id": track_id,
                    "class_id": class_id,
                    "confidence": box.conf[0].item(),
                    "x_centre": cx,
                    "y_centre": cy,
                    "width": w,
                    "height": h,
                }
            )

            if line_counter is not None:
                frame_tracks.append({"id": track_id, "center": (cx, cy), "class": class_id})

        if line_counter is not None:
            line_counter.update(frame_idx, frame_tracks)
            print(
                f"[Frame {frame_idx}] tracked={len(frame_tracks)} line_count={line_counter.total_count}"
            )

        # Save annotated frame for debugging if requested
        if output_debug_video is not None:
            if video_writer is None:
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

                h_shape, w_shape = r.orig_shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(output_debug_video), fourcc, fps, (w_shape, h_shape)
                )

            annotated_frame = r.plot()
            if line_counter is not None:
                line_counter.annotate(annotated_frame, frame_tracks)
            video_writer.write(annotated_frame)

    if video_writer is not None:
        video_writer.release()

    if line_counter is not None:
        print(f"Line-crossing count: {line_counter.total_count}")

    df = pd.DataFrame(detections)
    df.to_csv(output_csv, index=False)
    print(f"Tracking completed. Numeric features saved to {output_csv}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description="Run RT-DETR + ByteTrack and export numeric features"
    )
    parser.add_argument("--video", type=str, help="Path to input video (overrides default)")
    parser.add_argument("--output-csv", type=str, help="Path to output CSV (overrides default)")
    parser.add_argument(
        "--output-debug-video", type=str, default=None, help="Path to save debug annotated video"
    )
    parser.add_argument(
        "--max-box-ratio",
        type=float,
        default=0.4,
        help="Filter out boxes larger than this fraction of frame dims",
    )
    parser.add_argument(
        "--duration", type=int, default=None, help="Maximum duration to process (seconds)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to RT-DETR model file (overrides default rtdetr-l.pt)",
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
        help="Only count line crossings in this direction",
    )
    args = parser.parse_args()

    video_file = (
        Path(args.video)
        if args.video
        else project_root / "data" / "raw_video" / "data_1235_2237_3min.mp4"
    )
    output_file = (
        Path(args.output_csv)
        if args.output_csv
        else project_root / "data" / "features" / "data_1235_2237_3min_tracked.csv"
    )
    debug_video_file = Path(args.output_debug_video) if args.output_debug_video else None

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = project_root / "src" / "vehicle_emission" / "rtdetr-l.pt"

    if not video_file.exists():
        print(f"Error: Target video file '{video_file}' not found.")
    else:
        line_of_interest = None
        if args.line_start is not None and args.line_end is not None:
            line_of_interest = LineOfInterest(
                start=tuple(args.line_start), end=tuple(args.line_end)
            )

        # Inject the chosen model path into the function before calling (simple approach)
        track_vehicles._model_path = str(model_path)

        track_vehicles(
            video_file,
            output_file,
            output_debug_video=debug_video_file,
            line_of_interest=line_of_interest,
            count_direction=args.count_direction,
            max_box_ratio=args.max_box_ratio,
            duration=args.duration,
        )
