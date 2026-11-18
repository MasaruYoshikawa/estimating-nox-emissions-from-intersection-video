from pathlib import Path

import cv2
import pandas as pd
from ultralytics import RTDETR


def detect_vehicles(
    video_path: str | Path, output_csv: str | Path, output_debug_video: str | Path | None = None
) -> None:
    """
    Run RT-DETR on the target video and extract numeric bounding box features.
    Complies with INV-006 by persisting only numeric data (no raw image outputs).
    Note: Setting `output_debug_video` saves pixel data, which should ONLY be
    used for local debugging and NEVER committed or run in production pipelines.
    """
    # Initialize pre-trained RT-DETR model (downloads automatically if not found)
    model = RTDETR("rtdetr-l.pt")

    # COCO dataset class IDs relevant to the project:
    # 2: car (passenger car/taxi)
    # 5: bus
    # 7: truck (light/heavy-duty)
    target_classes = [2, 5, 7]

    # Run inference generator
    results = model.predict(
        source=str(video_path),
        stream=True,
        classes=target_classes,
        conf=0.45,  # Confidence threshold
    )

    detections = []
    video_writer = None

    for frame_idx, r in enumerate(results):
        for box in r.boxes:
            # Extract numeric coordinate features: center X, center Y, width, height
            cx, cy, w, h = box.xywh[0].tolist()

            detections.append(
                {
                    "frame_id": frame_idx,
                    "class_id": int(box.cls[0].item()),
                    "confidence": box.conf[0].item(),
                    "x_centre": cx,
                    "y_centre": cy,
                    "width": w,
                    "height": h,
                }
            )

        # Save annotated frame for debugging if requested
        if output_debug_video is not None:
            if video_writer is None:
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

                h, w = r.orig_shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_debug_video), fourcc, fps, (w, h))

            # r.plot() draws bounding boxes, class names, and confidence automatically
            video_writer.write(r.plot())

    if video_writer is not None:
        video_writer.release()

    df = pd.DataFrame(detections)
    df.to_csv(output_csv, index=False)
    print(f"Detection completed. Numeric features saved to {output_csv}")


if __name__ == "__main__":
    # Get the project root directory dynamically (3 folders up from this file)
    project_root = Path(__file__).resolve().parents[3]

    # Target dataset specified (Aligns with project data/ layout)
    video_file = project_root / "data" / "raw_video" / "data_1235_2237_3min.mp4"
    output_file = project_root / "data" / "features" / "data_1235_2237_3min.csv"
    debug_video_file = project_root / "data" / "raw_video" / "data_1235_2237_3min_annotated.mp4"

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not video_file.exists():
        print(f"Error: Target video file '{video_file}' not found.")
    else:
        detect_vehicles(video_file, output_file, output_debug_video=debug_video_file)
