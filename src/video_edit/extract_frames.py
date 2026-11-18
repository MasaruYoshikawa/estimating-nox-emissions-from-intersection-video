from pathlib import Path

import cv2


def extract_evenly_spaced_frames(
    video_path: str | Path, output_dir: str | Path, num_frames: int = 100
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames. Extracting all.")
        num_frames = total_frames

    # Calculate how many frames to skip to get evenly spaced images
    step = total_frames // num_frames
    print(
        f"Extracting {num_frames} frames from {total_frames} total frames (1 every {step} frames)..."
    )

    extracted_count = 0
    current_frame = 0

    while cap.isOpened() and extracted_count < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        out_filename = output_dir / f"{video_path.stem}_frame_{current_frame:06d}.jpg"
        cv2.imwrite(str(out_filename), frame)

        extracted_count += 1
        current_frame += step

    cap.release()
    print(f"Successfully extracted {extracted_count} frames to {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_folder = project_root / "data" / "raw_video"
    output_folder = project_root / "data" / "raw_video" / "annotation_frames"

    # Find all MP4 files, ignoring the debug/annotated ones we made previously
    video_files = [
        f
        for f in input_folder.glob("*.mp4")
        if "annotated" not in f.name and "tracked" not in f.name
    ]

    if not video_files:
        print(f"No valid video files found in {input_folder}")
    else:
        # Task T-032 requires 1,000 manual frames for fine-tuning. Divide this evenly.
        frames_per_video = max(1, 400 // len(video_files))
        print(
            f"Found {len(video_files)} videos. Extracting ~{frames_per_video} frames from each to reach 1000 total."
        )

        for video_file in video_files:
            extract_evenly_spaced_frames(video_file, output_folder, num_frames=frames_per_video)
