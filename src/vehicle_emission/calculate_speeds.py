from __future__ import annotations

import argparse
import math
import os
import sqlite3
from typing import Any

import cv2
import pandas as pd

from vehicle_emission import database
from vehicle_emission.speed.production import calculate_calibrated_speeds, replay_speed_bundle

SUPPORTED_SITES = frozenset({"Sukiyabashi", "Shinjuku"})


def _positive_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number > 0 else default


def _total_frames_for_video(
    video_path: str,
    *,
    fps: object,
    duration_seconds: object,
    livestream_duration_sec: float,
    coords_df: pd.DataFrame,
) -> int:
    nominal_fps = _positive_number(fps, 30.0)
    duration_sec = _positive_number(duration_seconds, 0.0)
    if duration_sec > 0:
        total_frames = int(round(duration_sec * nominal_fps))
        if 0 < duration_sec <= 305.0:
            total_frames = int(round(total_frames * (livestream_duration_sec / duration_sec)))
        if total_frames > 0:
            return max(total_frames, 1)

    total_frames = 0
    if video_path and os.path.exists(video_path):
        capture = cv2.VideoCapture(video_path)
        try:
            if capture.isOpened():
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    if total_frames > 0:
        return total_frames

    if total_frames <= 0 and not coords_df.empty:
        total_frames = int(coords_df["frame_id"].max()) + 1
    return max(total_frames, 1)


def _calculate_speed_rows(
    coords_df: pd.DataFrame,
    *,
    total_frames: int,
    livestream_duration_sec: float,
    scale: float,
) -> list[tuple[int, int, float]]:
    """Calculate one video's complete set of per-track, per-second speed rows using fast vectorized math."""

    if coords_df.empty:
        return []

    import numpy as np

    frame_interval_sec = livestream_duration_sec / total_frames
    effective_fps = total_frames / livestream_duration_sec
    speed_rows: list[tuple[int, int, float]] = []

    # Sort coordinates by track and frame
    df = coords_df.sort_values(["track_id", "frame_id"]).copy()

    # Compute frame-to-frame shifts per track
    df["prev_track_id"] = df["track_id"].shift(1)
    df["prev_frame_id"] = df["frame_id"].shift(1)
    df["prev_x"] = df["x_centre"].shift(1)
    df["prev_y"] = df["y_centre"].shift(1)

    same_track = df["track_id"] == df["prev_track_id"]
    delta_frames = df["frame_id"] - df["prev_frame_id"]
    valid_step = same_track & (delta_frames > 0) & df["prev_x"].notna()

    dx = np.where(valid_step, df["x_centre"] - df["prev_x"], 0.0)
    dy = np.where(valid_step, df["y_centre"] - df["prev_y"], 0.0)
    step_dist_px = np.hypot(dx, dy)

    elapsed_sec = np.where(valid_step, delta_frames * frame_interval_sec, 0.0)
    dist_m = step_dist_px / scale
    step_speed_kmh = np.where(elapsed_sec > 0, (dist_m / np.maximum(elapsed_sec, 1e-9)) * 3.6, 0.0)

    df["step_speed_kmh"] = step_speed_kmh
    df["valid_step"] = valid_step
    df["second"] = (df["frame_id"] / effective_fps).astype(int)

    # Process per track_id and second
    for (track_id, second), group in df.groupby(["track_id", "second"]):
        valid_steps = group[group["valid_step"]]
        average_speed = (
            float(valid_steps["step_speed_kmh"].mean()) if not valid_steps.empty else 0.0
        )

        if not math.isfinite(average_speed) or average_speed < 0:
            raise RuntimeError(
                f"Invalid calculated speed for track_id={int(track_id)}, second={int(second)}"
            )
        average_speed = min(average_speed, 150.0)
        speed_rows.append((int(track_id), int(second), round(average_speed, 2)))

    return speed_rows


def _effective_scale_for_video(
    base_scale: float,
    width: Any,
    height: Any,
) -> float:
    """Calculate resolution-adjusted scale factor based on video frame width relative to reference resolution."""
    try:
        w = float(width)
    except (TypeError, ValueError):
        return base_scale
    if not math.isfinite(w) or w <= 0:
        return base_scale

    # Automatically infer reference width from base scale magnitude:
    # base_scale > 15.0 indicates a 2560x1440 high-res calibration base.
    # base_scale <= 15.0 indicates a 1440x900 standard-res calibration base.
    ref_width = 2560.0 if base_scale > 15.0 else 1440.0
    return float(base_scale * (w / ref_width))


def calculate_speeds_for_all(
    db_path: str,
    scale: float,
    *,
    site: str,
    livestream_duration_sec: float = 600.0,
    force_recalculate: bool = False,
    month: str | None = None,
    date: str | None = None,
) -> None:
    """Legacy scalar arithmetic retained for historical callers/tests only.

    This mutating compatibility API is not a corrected-production route.
    Use ``calculate_calibrated_speeds`` for evidence-bound fresh outputs.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite number greater than zero")
    if not math.isfinite(livestream_duration_sec) or livestream_duration_sec <= 0:
        raise ValueError("livestream_duration_sec must be a finite number greater than zero")
    if site not in SUPPORTED_SITES:
        supported = ", ".join(sorted(SUPPORTED_SITES))
        raise ValueError(f"site must be one of: {supported}")

    filter_msg = []
    if month:
        filter_msg.append(f"month {month}")
    if date:
        filter_msg.append(f"date {date}")
    msg_str = f" ({', '.join(filter_msg)})" if filter_msg else ""

    print(f"Connecting to database for site {site}{msg_str}: {db_path}")
    database.init_db(db_path)
    conn = sqlite3.connect(db_path, timeout=30.0)
    if force_recalculate:
        print("Force recalculation requested; existing rows remain until replacement succeeds.")
    else:
        print(
            "Refreshing speeds because existing rows do not record a complete "
            "input/run fingerprint."
        )

    query = """
        SELECT video_id, video_path, fps, duration_seconds, intersection_id, width, height, recording_date
        FROM video_master
        WHERE intersection_id = ?
    """
    params = [site]

    if date:
        query += " AND (recording_date = ? OR strftime('%Y-%m-%d', recording_datetime_utc) = ?)"
        params.extend([date, date])
    elif month:
        query += " AND (strftime('%Y-%m', recording_datetime_utc) = ? OR strftime('%m', recording_datetime_utc) = ?)"
        params.extend([month, month])

    query += " ORDER BY recording_date, video_id"

    videos_df = pd.read_sql_query(query, conn, params=params)

    if videos_df.empty:
        site_has_videos = (
            conn.execute(
                "SELECT COUNT(*) FROM video_master WHERE intersection_id = ?", [site]
            ).fetchone()[0]
            > 0
        )
        observed_sites = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT intersection_id FROM video_master ORDER BY intersection_id"
            ).fetchall()
        ]
        conn.close()
        if not site_has_videos and observed_sites:
            raise ValueError(
                f"No videos found for requested site {site!r}; database sites are: "
                + ", ".join(observed_sites)
            )
        print(f"No videos found for site {site!r}{msg_str}.")
        return

    observed_site_set = set(videos_df["intersection_id"].astype(str))
    if observed_site_set != {site}:
        raise RuntimeError(
            f"Site-scoped speed query returned mismatched sites: {sorted(observed_site_set)}"
        )

    all_video_ids = [int(v_id) for v_id in videos_df["video_id"]]
    print(
        f"Found {len(all_video_ids)} videos to process across {videos_df['recording_date'].nunique()} date(s)."
    )

    conn.execute(
        """
        CREATE TEMP TABLE staged_vehicle_speeds (
            video_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            second INTEGER NOT NULL,
            speed_kmh REAL NOT NULL,
            PRIMARY KEY (video_id, track_id, second)
        )
        """
    )

    from tqdm import tqdm

    pbar = tqdm(total=len(all_video_ids), desc=f"Calculating speeds for {site}")
    try:
        # Group and process video-by-video, committing per date to prevent work loss on interruption
        grouped = videos_df.groupby("recording_date", sort=False)
        for rec_date, date_videos_df in grouped:
            pbar.set_description(f"Calculating speeds ({rec_date})")
            date_video_ids = [int(v_id) for v_id in date_videos_df["video_id"]]
            conn.execute("DELETE FROM staged_vehicle_speeds")

            for video in date_videos_df.itertuples(index=False):
                video_id = int(video.video_id)
                video_scale = _effective_scale_for_video(
                    scale, getattr(video, "width", None), getattr(video, "height", None)
                )
                coords_df = pd.read_sql_query(
                    """
                    SELECT track_id, frame_id, x_centre, y_centre
                    FROM vehicle_tracks
                    WHERE video_id = ?
                    ORDER BY track_id, frame_id
                    """,
                    conn,
                    params=[video_id],
                )
                if coords_df.empty:
                    raise RuntimeError(
                        "Cannot replace site speeds because a registered video has no current "
                        f"track rows: video_id={video_id}, site={site}"
                    )

                total_frames = _total_frames_for_video(
                    str(video.video_path or ""),
                    fps=video.fps,
                    duration_seconds=video.duration_seconds,
                    livestream_duration_sec=livestream_duration_sec,
                    coords_df=coords_df,
                )
                speed_rows = _calculate_speed_rows(
                    coords_df,
                    total_frames=total_frames,
                    livestream_duration_sec=livestream_duration_sec,
                    scale=video_scale,
                )
                conn.executemany(
                    """
                    INSERT INTO staged_vehicle_speeds (video_id, track_id, second, speed_kmh)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(video_id, track_id, second, speed) for track_id, second, speed in speed_rows],
                )
                pbar.update(1)

            # Atomically commit this date's speeds and replace in vehicle_speeds
            placeholders = ",".join("?" for _ in date_video_ids)
            with conn:
                conn.execute(
                    f"DELETE FROM video_emissions WHERE video_id IN ({placeholders})",
                    date_video_ids,
                )
                conn.execute(
                    f"DELETE FROM vehicle_speeds WHERE video_id IN ({placeholders})", date_video_ids
                )
                conn.execute(
                    """
                    INSERT INTO vehicle_speeds (video_id, track_id, second, speed_kmh)
                    SELECT video_id, track_id, second, speed_kmh
                    FROM staged_vehicle_speeds
                    """
                )
            pbar.write(f"Committed speeds for date {rec_date} ({len(date_video_ids)} videos).")

    finally:
        pbar.close()
        conn.close()

    print("Real speed calculation and atomic replacement complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export perspective-calibrated, source-timed speeds from read-only SQLite tracks"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/results.db", help="Path to SQLite database"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", help="Explicit production_speed_v4 JSON evidence configuration")
    inputs.add_argument(
        "--replay-bundle", help="Verify and recompute a previously exported numeric bundle"
    )
    parser.add_argument(
        "--reverify-source-evidence",
        action="store_true",
        help="During replay also recheck external original media/timing/geometry hashes",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Fresh immutable numeric-output bundle directory"
    )
    args = parser.parse_args()
    if args.reverify_source_evidence and not args.replay_bundle:
        parser.error("--reverify-source-evidence requires --replay-bundle")
    output = (
        replay_speed_bundle(
            args.replay_bundle,
            args.output_dir,
            reverify_source_evidence=args.reverify_source_evidence,
        )
        if args.replay_bundle
        else calculate_calibrated_speeds(args.db_path, args.config, args.output_dir)
    )
    print(f"Published calibrated speed bundle: {output}")


if __name__ == "__main__":
    main()
