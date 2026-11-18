from pathlib import Path

import cv2


def play_frame_by_frame(video_path: str | Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print("Controls:")
    print(" - Press any key (e.g., Spacebar) to advance one frame.")
    print(" - Press 'q' to quit.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        cv2.imshow(f"Frame-by-Frame Player - Frame {frame_count}", frame)

        # Wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    # Point to the annotated debug video to verify your RT-DETR bounding boxes!
    video_file = project_root / "data" / "raw_video" / "data_1235_2237_10sec_annotated.mp4"

    play_frame_by_frame(video_file)
