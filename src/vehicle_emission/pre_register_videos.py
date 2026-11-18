from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta

from vehicle_emission import database
from vehicle_emission.time_utils import format_utc, parse_aware_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-register the 7 traffic videos in the database"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/results.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--intersection", type=str, default="Sukiyabashi", help="Intersection ID/name"
    )
    parser.add_argument(
        "--base-start-time",
        type=str,
        required=True,
        help="Aware ISO-8601 start time for count1.mp4",
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=10,
        help="Known spacing between sequential count files (default: 10 minutes)",
    )
    parser.add_argument(
        "--use-file-mtime-as-start",
        action="store_true",
        help="Explicitly use filesystem modification times as capture starts",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        help="Optional group ID to associate with the registered videos",
    )
    args = parser.parse_args()
    if args.segment_minutes <= 0:
        parser.error("--segment-minutes must be positive")
    base_start = parse_aware_timestamp(args.base_start_time)

    print(f"Initializing database at: {args.db_path}...")
    database.init_db(args.db_path)

    # We register 7 videos sequentially
    for i in range(1, 8):
        video_path = f"data/raw_video/counting_test{i}/count{i}.mp4"
        norm_path = os.path.normpath(video_path)

        if not os.path.exists(norm_path):
            print(f"Warning: Video file not found at {norm_path}. Skipping pre-registration.")
            continue

        if args.use_file_mtime_as_start:
            mtime = os.path.getmtime(norm_path)
            dt = datetime.fromtimestamp(mtime, tz=UTC)
            start_time = format_utc(dt)
        else:
            start_time = format_utc(base_start + timedelta(minutes=(i - 1) * args.segment_minutes))

        print(f"Registering Video {i}: {norm_path} (Start time: {start_time})...")
        video_id = database.register_video(
            db_path=args.db_path,
            video_path=norm_path,
            intersection_id=args.intersection,
            start_time_str=start_time,
            notes=f"Pre-registered video {i}",
            video_id=i,
            group_id=args.group_id,
        )
        print(f" -> Assigned Video ID: {video_id}")

    print("\nPre-registration complete.")


if __name__ == "__main__":
    main()
