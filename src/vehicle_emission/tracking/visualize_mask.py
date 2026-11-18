from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize video crop, polygon mask, and exit lines before running tracking"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input traffic video")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop region coordinates (x1 y1 x2 y2)",
    )
    parser.add_argument(
        "--mask-polygon",
        type=float,
        nargs="+",
        action="append",
        help="Polygon vertices (x1 y1 x2 y2 ...) defining the keep region; can be repeated",
    )
    parser.add_argument(
        "--exit-line",
        type=float,
        nargs=4,
        action="append",
        default=[],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Exit lines coordinates to overlay in preview (can be repeated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mask_preview.png",
        help="Path to save the generated preview image",
    )
    parser.add_argument(
        "--display", action="store_true", help="Show the preview window interactively"
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read the first frame of the video.")

    orig_height, orig_width = frame.shape[:2]

    # Apply crop
    if args.crop:
        x1_c, y1_c, x2_c, y2_c = args.crop
        x1_c = max(0, min(x1_c, orig_width))
        x2_c = max(0, min(x2_c, orig_width))
        y1_c = max(0, min(y1_c, orig_height))
        y2_c = max(0, min(y2_c, orig_height))
        frame = frame[y1_c:y2_c, x1_c:x2_c]
        width, height = x2_c - x1_c, y2_c - y1_c
        print(f"Applied crop. Preview size: {width}x{height}")
    else:
        width, height = orig_width, orig_height

    # Build and apply mask
    mask = None
    if args.mask_polygon:
        mask = np.zeros((height, width), dtype=np.uint8)
        # Create blended transparent green overlay
        overlay = frame.copy()
        for poly in args.mask_polygon:
            if len(poly) % 2 != 0 or len(poly) < 6:
                raise ValueError(
                    "Each mask polygon must have an even number of coordinates "
                    "(at least 3 vertices)."
                )
            mask_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [mask_pts], 255)
            cv2.fillPoly(overlay, [mask_pts], (0, 0, 255))

        # Create blacked out version
        masked_frame = frame.copy()
        masked_frame[mask == 255] = 0

        # Blend original frame with transparent overlay
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    else:
        masked_frame = frame.copy()

    # Draw exit lines on both frames
    for idx, coords in enumerate(args.exit_line):
        x1, y1, x2, y2 = [int(val) for val in coords]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"exit{idx + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.line(masked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            masked_frame,
            f"exit{idx + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Combine side-by-side
    combined = np.hstack((frame, masked_frame))

    # Save to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"Saved preview side-by-side comparison to {out_path.resolve()}")

    # Display
    if args.display:
        title = "Mask Preview: Left = Transparent Green Overlay, Right = What model sees"
        cv2.imshow(title, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
