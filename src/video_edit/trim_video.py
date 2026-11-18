from pathlib import Path

import cv2


def trim_video(input_path: str, output_path: str, duration_sec: int = 180):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = int(fps * duration_sec)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Trimming first {duration_sec} seconds... This might take a moment.")

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Successfully saved {duration_sec}-second clip to {output_path}")


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    input_file = project_root / "data" / "raw_video" / "data_1235_2237.mp4"
    output_file = project_root / "data" / "raw_video" / "data_1235_2237_3min.mp4"
    trim_video(input_file, output_file, duration_sec=180)
