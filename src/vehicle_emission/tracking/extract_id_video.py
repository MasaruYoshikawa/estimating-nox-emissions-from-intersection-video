import argparse

import cv2
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract video showing only a specific vehicle ID."
    )
    parser.add_argument("--video", required=True, help="Path to raw video")
    parser.add_argument("--csv", required=True, help="Path to tracking CSV")
    parser.add_argument("--output", required=True, help="Path to save output video")
    parser.add_argument("--id", type=int, required=True, help="Vehicle ID to extract")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop coordinates used during tracking (to offset boxes back to raw video)",
    )
    args = parser.parse_args()

    # Read tracking data
    print(f"Loading tracking data from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Filter for the specific ID
    target_data = df[df["vehicle_id"] == args.id]

    if target_data.empty:
        print(f"No tracking data found for ID {args.id}.")
        return

    print(f"Found {len(target_data)} frames containing ID {args.id}.")

    if "frame_id" not in target_data.columns:
        print("Error: 'frame_id' column not found in CSV. Cannot map data to frames.")
        return

    # Group rows by frame_id
    target_frames = {}
    for _, row in target_data.iterrows():
        f_id = int(row["frame_id"])
        if f_id not in target_frames:
            target_frames[f_id] = []
        target_frames[f_id].append(row)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    print("Processing video...")

    min_frame = int(target_data["frame_id"].min())
    max_frame = int(target_data["frame_id"].max())

    # Skip frames before the target vehicle appears
    while frame_idx < min_frame:
        ret, _ = cap.read()
        if not ret:
            break
        frame_idx += 1

    # Process relevant frames
    while cap.isOpened() and frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_frames:
            for row in target_frames[frame_idx]:
                x_centre = row.get("x_centre", row.get("x", 0))
                y_centre = row.get("y_centre", row.get("y", 0))

                if not pd.isna(x_centre) and not pd.isna(y_centre):
                    cx, cy = int(x_centre), int(y_centre)

                    # Add crop offset if tracking was done on cropped video
                    if args.crop:
                        cx += args.crop[0]
                        cy += args.crop[1]

                    w = row.get("width", row.get("w", 0))
                    h = row.get("height", row.get("h", 0))
                    if w > 0 and h > 0 and not pd.isna(w) and not pd.isna(h):
                        cv2.rectangle(
                            frame,
                            (int(cx - w / 2), int(cy - h / 2)),
                            (int(cx + w / 2), int(cy + h / 2)),
                            (0, 0, 255),
                            4,
                        )
                    else:
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

                    cv2.putText(
                        frame,
                        f"ID: {args.id}",
                        (cx - 40, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        4,
                    )

            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Extraction complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
