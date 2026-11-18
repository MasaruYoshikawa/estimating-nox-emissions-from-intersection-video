from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict

import pandas as pd

from vehicle_emission import database
from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import MOVEStarModel
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

EXPECTED_EMISSION_METHODS = ("MOVEStar", "VT-CPFM")


def _replace_video_emissions(
    db_path: str,
    emissions: list[dict[str, object]],
    target_video_ids: list[int] | None = None,
) -> None:
    """Atomically replace derived emissions for target videos (or all if None)."""

    rows = [
        (
            int(emission["video_id"]),
            str(emission["estimation_method"]),
            float(emission["nox_grams"]),
            float(emission["co_grams"]),
            float(emission["hc_grams"]),
            float(emission["pm25_grams"]),
            float(emission["co2_grams"]),
            float(emission["fuel_grams"]),
            float(emission["energy_kj"]),
            int(emission["vehicle_count"]),
            float(emission["average_speed_kmh"]),
            str(emission["congestion_level"]),
            float(emission["nox_hybrid_adjusted_grams"]),
        )
        for emission in emissions
    ]

    connection = sqlite3.connect(db_path, timeout=30.0)
    try:
        connection.execute("PRAGMA foreign_keys = ON;")
        with connection:
            if target_video_ids is not None:
                placeholders = ",".join("?" for _ in target_video_ids)
                connection.execute(
                    f"DELETE FROM video_emissions WHERE video_id IN ({placeholders});",
                    target_video_ids,
                )
            else:
                connection.execute("DELETE FROM video_emissions;")
            connection.executemany(
                """
                INSERT INTO video_emissions (
                    video_id, estimation_method, nox_grams, co_grams, hc_grams,
                    pm25_grams, co2_grams, fuel_grams, energy_kj, vehicle_count,
                    average_speed_kmh, congestion_level, nox_hybrid_adjusted_grams
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    finally:
        connection.close()


def _emission_values(
    result: object,
    *,
    method: str,
    video_id: int,
    track_id: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Validate and unpack one estimator response before any database replacement."""

    if not isinstance(result, dict) or "Emission Rate" not in result:
        raise RuntimeError(
            f"{method} returned no emission rate for video_id {video_id}, track_id {track_id}"
        )
    rates = result["Emission Rate"]
    if not isinstance(rates, list | tuple) or not rates or len(rates[0]) < 8:
        raise RuntimeError(
            f"{method} returned an invalid emission rate for video_id {video_id}, "
            f"track_id {track_id}"
        )
    values = tuple(float(value) for value in rates[0][:8])
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError(
            f"{method} returned a non-finite or negative emission rate for "
            f"video_id {video_id}, track_id {track_id}"
        )
    return values


def calculate_emissions_for_all(
    db_path: str,
    force_recalculate: bool = False,
    site: str | None = None,
    month: str | None = None,
) -> None:
    """Run MOVEStar model on vehicle tracks in the database and aggregate emissions."""
    site_msg = f" for site '{site}'" if site else ""
    month_msg = f" (month '{month}')" if month else ""
    print(f"Connecting to database: {db_path}{site_msg}{month_msg}", flush=True)

    # Initialize DB
    database.init_db(db_path)

    conn = sqlite3.connect(db_path, timeout=30.0)

    if force_recalculate:
        print(
            "Force recalculation requested; existing rows remain until replacement succeeds.",
            flush=True,
        )
    else:
        print(
            "Refreshing emissions because existing rows do not record a complete "
            "input/run fingerprint.",
            flush=True,
        )

    # Get target videos (optionally scoped to a specific site and/or month)
    if site and month:
        all_videos_df = pd.read_sql_query(
            """
            SELECT video_id, intersection_id, recording_datetime_utc
            FROM video_master
            WHERE intersection_id = ?
              AND (strftime('%Y-%m', recording_datetime_utc) = ? OR strftime('%m', recording_datetime_utc) = ?)
            """,
            conn,
            params=(site, month, month),
        )
    elif site:
        all_videos_df = pd.read_sql_query(
            "SELECT video_id, intersection_id, recording_datetime_utc FROM video_master WHERE intersection_id = ?",
            conn,
            params=(site,),
        )
    elif month:
        all_videos_df = pd.read_sql_query(
            """
            SELECT video_id, intersection_id, recording_datetime_utc
            FROM video_master
            WHERE strftime('%Y-%m', recording_datetime_utc) = ? OR strftime('%m', recording_datetime_utc) = ?
            """,
            conn,
            params=(month, month),
        )
    else:
        all_videos_df = pd.read_sql_query(
            "SELECT video_id, intersection_id, recording_datetime_utc FROM video_master",
            conn,
        )

    if all_videos_df.empty:
        print(f"No videos found in video_master table{site_msg}.", flush=True)
        conn.close()
        return

    videos_df = all_videos_df
    print(f"Loaded {len(videos_df)} videos to process. Querying vehicle tracks...", flush=True)
    video_ids = [int(video_id) for video_id in videos_df["video_id"]]

    # Load all tracks metadata (resolving dominant class_id per track via majority vote across frames)
    placeholders = ",".join("?" for _ in video_ids)
    tracks_df = pd.read_sql_query(
        f"""
        SELECT video_id, track_id, class_id
        FROM (
            SELECT video_id, track_id, class_id, COUNT(*) as cnt,
                   ROW_NUMBER() OVER (
                       PARTITION BY video_id, track_id 
                       ORDER BY COUNT(*) DESC, class_id ASC
                   ) as rn
            FROM vehicle_tracks
            WHERE video_id IN ({placeholders})
            GROUP BY video_id, track_id, class_id
        )
        WHERE rn = 1
        """,
        conn,
        params=video_ids,
    )

    print(f"Loaded tracks for {len(tracks_df):,} vehicles. Querying speed profiles...", flush=True)
    speeds_df = pd.read_sql_query(
        f"SELECT video_id, track_id, second, speed_kmh FROM vehicle_speeds WHERE video_id IN ({placeholders}) ORDER BY video_id, track_id, second",
        conn,
        params=video_ids,
    )

    conn.close()

    if tracks_df.empty:
        raise RuntimeError(
            "No vehicle tracks found in the database. Run the tracking pipeline first."
        )
    if speeds_df.empty:
        raise RuntimeError(
            "No vehicle speed records found in the database. Run calculate_speeds.py first."
        )

    for row in tracks_df.itertuples(index=False):
        if row.class_id is None or not math.isfinite(float(row.class_id)):
            raise RuntimeError(
                f"Cannot calculate emissions because track has invalid class_id: "
                f"video_id={int(row.video_id)}/track_id={int(row.track_id)}"
            )

    for row in speeds_df.itertuples(index=False):
        try:
            speed_kmh = float(row.speed_kmh)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Vehicle speed must be numeric for "
                f"video_id={int(row.video_id)}, track_id={int(row.track_id)}, "
                f"second={int(row.second)}"
            ) from exc
        if not math.isfinite(speed_kmh) or speed_kmh < 0.0:
            raise RuntimeError(
                "Vehicle speed must be finite and non-negative for "
                f"video_id={int(row.video_id)}, track_id={int(row.track_id)}, "
                f"second={int(row.second)}"
            )
    speeds_df["speed_kmh"] = pd.to_numeric(speeds_df["speed_kmh"], errors="raise").astype(float)

    videos_with_tracks = {int(video_id) for video_id in tracks_df["video_id"]}
    missing_track_videos = sorted(set(video_ids) - videos_with_tracks)
    if missing_track_videos:
        raise RuntimeError(
            "Cannot calculate a complete emissions table; videos have no track rows: "
            f"{missing_track_videos}"
        )

    track_keys = {
        (int(row.video_id), int(row.track_id)) for row in tracks_df.itertuples(index=False)
    }
    speed_keys = {
        (int(row.video_id), int(row.track_id)) for row in speeds_df.itertuples(index=False)
    }
    missing_speed_tracks = sorted(track_keys - speed_keys)
    if missing_speed_tracks:
        examples = ", ".join(
            f"video_id={video_id}/track_id={track_id}"
            for video_id, track_id in missing_speed_tracks[:10]
        )
        raise RuntimeError(
            "Cannot calculate complete emissions; tracks have no speed profile: " + examples
        )
    orphan_speed_tracks = sorted(speed_keys - track_keys)
    if orphan_speed_tracks:
        examples = ", ".join(
            f"video_id={video_id}/track_id={track_id}"
            for video_id, track_id in orphan_speed_tracks[:10]
        )
        raise RuntimeError(
            "Cannot calculate complete emissions; speed profiles have no current track: " + examples
        )

    for class_id in tracks_df["class_id"]:
        emission_vehicle_type_for_project_class(int(class_id))

    # Calculate average speed per video, ignoring speeds < 1.0 km/h to exclude stopped vehicles
    video_avg_speeds = {}
    if not speeds_df.empty:
        moving_speeds = speeds_df[speeds_df["speed_kmh"] >= 1.0]
        overall_means = speeds_df.groupby("video_id")["speed_kmh"].mean().to_dict()
        moving_means = (
            moving_speeds.groupby("video_id")["speed_kmh"].mean().to_dict()
            if not moving_speeds.empty
            else {}
        )
        for vid in overall_means:
            video_avg_speeds[vid] = moving_means.get(vid, overall_means[vid])

    # Build speed profiles by (video_id, track_id)
    print("Building track speed profiles in memory...")
    track_speeds = defaultdict(list)
    for row in speeds_df.itertuples(index=False):
        track_speeds[(int(row.video_id), int(row.track_id))].append(float(row.speed_kmh))

    # Initialise models
    print("Initialising MOVEStar model...")
    movestar = MOVEStarModel()
    print("Initialising VT-CPFM model...")
    vtcpfm = VTCPFMModel()

    # Accumulators for video-level emissions: (video_id, method) -> dict of sums
    accumulators = defaultdict(
        lambda: {
            "nox_grams": 0.0,
            "nox_hybrid_adjusted_grams": 0.0,
            "co_grams": 0.0,
            "hc_grams": 0.0,
            "pm25_grams": 0.0,
            "co2_grams": 0.0,
            "fuel_grams": 0.0,
            "energy_kj": 0.0,
            "vehicle_count": 0,
        }
    )
    for video_id in video_ids:
        for method in EXPECTED_EMISSION_METHODS:
            accumulators[(video_id, method)]

    # Pre-group tracks by video_id for fast lookup
    tracks_by_video = {int(vid): grp for vid, grp in tracks_df.groupby("video_id")}

    print("Running emissions estimation (MOVEStar & VT-CPFM)...")
    from tqdm import tqdm

    for video in tqdm(
        videos_df.itertuples(index=False), total=len(videos_df), desc="Estimating emissions"
    ):
        video_id = int(video.video_id)

        # Get tracks for this video
        video_tracks = tracks_by_video.get(video_id)
        if video_tracks is None:
            continue

        for track in video_tracks.itertuples(index=False):
            track_id = int(track.track_id)
            class_id = int(track.class_id)
            emission_vehicle_type_for_project_class(class_id)

            speeds_kmh = track_speeds.get((video_id, track_id), [])
            if not speeds_kmh:
                continue

            # Convert speed from km/h to m/s
            speed_mps = [s / 3.6 for s in speeds_kmh]

            # 1. MOVEStar Estimation
            try:
                res_ms = movestar.estimate_class_emissions(class_id, speed_mps)
                co_g, hc_g, nox_g, pm25_ele, pm25_org, energy_kj, co2_g, fuel_g = _emission_values(
                    res_ms,
                    method="MOVEStar",
                    video_id=video_id,
                    track_id=track_id,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MOVEStar failed for video_id {video_id}, track_id {track_id}"
                ) from exc
            pm25_g = pm25_ele + pm25_org

            # Hybrid adjusted NOx: 17% hybrid (0.013 g/km), 83% standard (nox_g)
            if class_id == 1:
                total_dist_km = sum(speeds_kmh) / 3600.0
                hybrid_nox_g = 0.013 * total_dist_km
                weighted_nox_g = 0.17 * hybrid_nox_g + 0.83 * nox_g
            else:
                weighted_nox_g = nox_g

            key = (video_id, "MOVEStar")
            accumulators[key]["nox_grams"] += nox_g
            accumulators[key]["nox_hybrid_adjusted_grams"] += weighted_nox_g
            accumulators[key]["co_grams"] += co_g
            accumulators[key]["hc_grams"] += hc_g
            accumulators[key]["pm25_grams"] += pm25_g
            accumulators[key]["co2_grams"] += co2_g
            accumulators[key]["fuel_grams"] += fuel_g
            accumulators[key]["energy_kj"] += energy_kj
            accumulators[key]["vehicle_count"] += 1

            # 2. VT-CPFM Estimation
            try:
                res_cp = vtcpfm.estimate_class_emissions(class_id, speed_mps)
                co_g, hc_g, nox_g, pm25_ele, pm25_org, energy_kj, co2_g, fuel_g = _emission_values(
                    res_cp,
                    method="VT-CPFM",
                    video_id=video_id,
                    track_id=track_id,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"VT-CPFM failed for video_id {video_id}, track_id {track_id}"
                ) from exc
            pm25_g = pm25_ele + pm25_org

            # Hybrid adjusted NOx: 17% hybrid (0.013 g/km), 83% standard (nox_g)
            if class_id == 1:
                total_dist_km = sum(speeds_kmh) / 3600.0
                hybrid_nox_g = 0.013 * total_dist_km
                weighted_nox_g = 0.17 * hybrid_nox_g + 0.83 * nox_g
            else:
                weighted_nox_g = nox_g

            key = (video_id, "VT-CPFM")
            accumulators[key]["nox_grams"] += nox_g
            accumulators[key]["nox_hybrid_adjusted_grams"] += weighted_nox_g
            accumulators[key]["co_grams"] += co_g
            accumulators[key]["hc_grams"] += hc_g
            accumulators[key]["pm25_grams"] += pm25_g
            accumulators[key]["co2_grams"] += co2_g
            accumulators[key]["fuel_grams"] += fuel_g
            accumulators[key]["energy_kj"] += energy_kj
            accumulators[key]["vehicle_count"] += 1

    # Prepare for insert
    emissions_to_insert = []
    for (video_id, method), data in accumulators.items():
        output_values = [
            float(data[field])
            for field in (
                "nox_grams",
                "nox_hybrid_adjusted_grams",
                "co_grams",
                "hc_grams",
                "pm25_grams",
                "co2_grams",
                "fuel_grams",
                "energy_kj",
            )
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in output_values):
            raise RuntimeError(
                f"Aggregated {method} emissions are non-finite or negative for video_id {video_id}"
            )
        avg_speed = float(video_avg_speeds.get(video_id, 0.0))
        if not math.isfinite(avg_speed) or avg_speed < 0.0:
            raise RuntimeError(f"Average speed is invalid for video_id {video_id}")
        if avg_speed < 10.0:
            cong_level = "Severe"
        elif avg_speed < 20.0:
            cong_level = "Moderate"
        else:
            cong_level = "Smooth"

        emissions_to_insert.append(
            {
                "video_id": video_id,
                "estimation_method": method,
                "nox_grams": round(data["nox_grams"], 4),
                "nox_hybrid_adjusted_grams": round(data["nox_hybrid_adjusted_grams"], 4),
                "co_grams": round(data["co_grams"], 4),
                "hc_grams": round(data["hc_grams"], 4),
                "pm25_grams": round(data["pm25_grams"], 4),
                "co2_grams": round(data["co2_grams"], 4),
                "fuel_grams": round(data["fuel_grams"], 4),
                "energy_kj": round(data["energy_kj"], 4),
                "vehicle_count": data["vehicle_count"],
                "average_speed_kmh": round(avg_speed, 2),
                "congestion_level": cong_level,
            }
        )

    expected_row_count = len(video_ids) * len(EXPECTED_EMISSION_METHODS)
    if len(emissions_to_insert) != expected_row_count:
        raise RuntimeError(
            f"Expected {expected_row_count} complete emission rows, built "
            f"{len(emissions_to_insert)}"
        )

    print(f"Saving {len(emissions_to_insert)} video emission records to database...")
    _replace_video_emissions(
        db_path,
        emissions_to_insert,
        target_video_ids=video_ids if (site is not None or month is not None) else None,
    )
    print("Emissions calculation and atomic database replacement completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate vehicle emissions using MOVEStar and save to SQLite"
    )
    parser.add_argument(
        "--db-path", type=str, default="data/results.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Optional site name (e.g. 'Shinjuku' or 'Sukiyabashi') to scope emission calculation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compatibility flag; each run always refreshes the complete emissions table",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Optional YYYY-MM or MM filter (e.g. '2026-05' or '05') to process a specific month",
    )
    args = parser.parse_args()

    calculate_emissions_for_all(
        args.db_path, force_recalculate=args.force, site=args.site, month=args.month
    )


if __name__ == "__main__":
    main()
