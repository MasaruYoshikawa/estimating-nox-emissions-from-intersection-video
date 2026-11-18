from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
from contextlib import closing

from vehicle_emission.database import init_db
from vehicle_emission.time_utils import (
    SORAMAME_HOUR_CONVENTIONS,
    SoramameHourConvention,
    local_timestamp_to_utc,
    soramame_hour_interval_utc,
)

DB_PATH = "data/results.db"


def load_weather(csv_path: str, db_path: str) -> None:
    """Load weather historical CSV data into the database."""
    print(f"Loading weather from {csv_path}...")
    rows_to_insert = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                date_str = row["Date"]
                time_str = row["Time"]
                # JMA observation labels are Japan Standard Time. Persist the
                # labelled instant as true UTC; precipitation retains its
                # documented previous-hour support for later alignment logic.
                ts_utc = local_timestamp_to_utc(
                    f"{date_str} {time_str}",
                    input_format="%d/%m/%Y %H:%M:%S",
                    timezone_name="Asia/Tokyo",
                )
                # Extract attributes handling empty strings as None.
                temp = float(row["Tempareture"]) if row.get("Tempareture") else None
                prec = float(row["precipitation"]) if row.get("precipitation") else None

                wind_speed_val = row.get("wind speed")
                wind_speed = (
                    float(wind_speed_val) if wind_speed_val and wind_speed_val.strip() else None
                )

                wind_dir_val = row.get("wind direction")
                wind_dir = wind_dir_val.strip() if wind_dir_val and wind_dir_val.strip() else None

                cloud_val = row.get("cloud thinckness")
                cloud = float(cloud_val) if cloud_val and cloud_val.strip() else None

                # Calculate along-road wind component (defined positive when wind blows from East to West)
                along_road_wind = None
                if wind_speed is not None and wind_dir is not None:
                    WIND_DIR_TO_ANGLE = {
                        "north": 0.0,
                        "north-northeast": 22.5,
                        "northeast": 45.0,
                        "east-northeast": 67.5,
                        "east": 90.0,
                        "east-southeast": 112.5,
                        "southeast": 135.0,
                        "south-southeast": 157.5,
                        "south": 180.0,
                        "south-southwest": 202.5,
                        "southwest": 225.0,
                        "west-southwest": 247.5,
                        "west": 270.0,
                        "west-northwest": 292.5,
                        "northwest": 315.0,
                        "north-northwest": 337.5,
                    }
                    dir_key = wind_dir.lower().strip()
                    if dir_key in WIND_DIR_TO_ANGLE:
                        angle_deg = WIND_DIR_TO_ANGLE[dir_key]
                        along_road_wind = wind_speed * math.sin(math.radians(angle_deg))

                rows_to_insert.append(
                    (ts_utc, temp, prec, wind_speed, wind_dir, cloud, along_road_wind)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weather row {i + 2}: {row!r}") from exc

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM weather_history;")
        conn.executemany(
            """
            INSERT OR REPLACE INTO weather_history (
                timestamp_utc, temperature, precipitation, wind_speed, wind_direction, cloud_thickness, along_road_wind_speed
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows_to_insert,
        )
    print(f"Inserted/Replaced {len(rows_to_insert)} weather records.")


def load_air_pollution(
    csv_path: str,
    place_name: str,
    db_path: str,
    *,
    hour_convention: SoramameHourConvention,
) -> None:
    """Load air pollution historical CSV data into the database."""
    print(f"Loading air pollution for {place_name} from {csv_path}...")
    rows_to_insert = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                date_str = row["Date"]
                time_val = row["Time"]
                ts_utc, _ = soramame_hour_interval_utc(
                    date_str,
                    int(time_val),
                    convention=hour_convention,
                )

                def _parse_float(v: str | None) -> float | None:
                    if not v:
                        return None
                    s = v.strip()
                    if not s or s in ("-", "—", "NaN", "null"):
                        return None
                    return float(s)

                no = _parse_float(row.get("NO(ppm)"))
                no2 = _parse_float(row.get("NO2(ppm)"))
                nox = _parse_float(row.get("NOx(ppm)"))
                co = _parse_float(row.get("CO(ppm)"))
                ox = _parse_float(row.get("Ox(ppm)"))

                rows_to_insert.append((ts_utc, place_name, no, no2, nox, co, ox))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid air-pollution row {i + 2}: {row!r}") from exc

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM air_pollution_history WHERE monitoring_place = ?;", (place_name,))
        conn.executemany(
            """
            INSERT OR REPLACE INTO air_pollution_history (
                timestamp_utc, monitoring_place, no_ppm, no2_ppm, nox_ppm, co_ppm, ox_ppm
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows_to_insert,
        )
    print(f"Inserted/Replaced {len(rows_to_insert)} air pollution records for {place_name}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load weather and air pollution historical CSVs into database."
    )
    parser.add_argument("--db-path", type=str, default=DB_PATH, help="Path to SQLite database")
    parser.add_argument(
        "--soramame-hour-convention",
        choices=SORAMAME_HOUR_CONVENTIONS,
        help="Station-verified support convention for all 1--24 air-monitor hour labels",
    )
    parser.add_argument(
        "--runtime-settings",
        help="Accepted JSON runtime settings emitted by the reproduction guard",
    )
    args = parser.parse_args()
    hour_convention = args.soramame_hour_convention
    if args.runtime_settings:
        with open(args.runtime_settings, encoding="utf-8") as handle:
            runtime_settings = json.load(handle)
        runtime_convention = runtime_settings.get("soramame_hour_convention")
        if hour_convention is not None and hour_convention != runtime_convention:
            parser.error("CLI and accepted runtime Soramame conventions disagree")
        hour_convention = runtime_convention
    if hour_convention not in SORAMAME_HOUR_CONVENTIONS:
        parser.error(
            "Supply --soramame-hour-convention or --runtime-settings with an accepted convention"
        )

    # Initialize DB first in case tables don't exist
    init_db(args.db_path)

    # Load weather
    weather_csv = "data/weather_history/weather_history.csv"
    if os.path.exists(weather_csv):
        load_weather(weather_csv, args.db_path)
    else:
        print(f"Warning: {weather_csv} not found.")

    # Load air pollution
    sukiyabashi_csv = "data/airpollution_history/AirPollution_Sukiyabashi.csv"
    if os.path.exists(sukiyabashi_csv):
        load_air_pollution(
            sukiyabashi_csv,
            "Sukiyabashi",
            args.db_path,
            hour_convention=hour_convention,
        )
    else:
        print(f"Warning: {sukiyabashi_csv} not found.")

    shinjuku_csv = "data/airpollution_history/AirPollution_Shinjuku.csv"
    if os.path.exists(shinjuku_csv):
        load_air_pollution(
            shinjuku_csv,
            "Shinjuku",
            args.db_path,
            hour_convention=hour_convention,
        )
    else:
        print(f"Warning: {shinjuku_csv} not found.")

    setagaya_csv = "data/airpollution_history/AirPollution_Setagaya.csv"
    if os.path.exists(setagaya_csv):
        load_air_pollution(
            setagaya_csv,
            "Setagaya",
            args.db_path,
            hour_convention=hour_convention,
        )
    else:
        print(f"Warning: {setagaya_csv} not found.")

    hibiya_csv = "data/airpollution_history/AirPollution_Hibiya.csv"
    if os.path.exists(hibiya_csv):
        load_air_pollution(
            hibiya_csv,
            "Hibiya",
            args.db_path,
            hour_convention=hour_convention,
        )
    else:
        print(f"Warning: {hibiya_csv} not found.")

    nakano_csv = "data/airpollution_history/AirPollution_Nakano.csv"
    if os.path.exists(nakano_csv):
        load_air_pollution(
            nakano_csv,
            "Nakano",
            args.db_path,
            hour_convention=hour_convention,
        )
    else:
        print(f"Warning: {nakano_csv} not found.")
