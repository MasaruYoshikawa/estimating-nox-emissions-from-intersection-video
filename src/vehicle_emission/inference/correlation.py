from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vehicle_emission.database import assert_entering_counts_integrity


def _parse_persisted_utc(values: pd.Series) -> pd.Series:
    """Parse canonical persisted UTC text and reject local-offset legacy rows."""

    text = values.astype("string")
    canonical_utc = text.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    if not canonical_utc.fillna(False).all():
        raise ValueError("Persisted timestamps must use canonical UTC text ending in 'Z'")
    return pd.to_datetime(text, utc=True, errors="raise")


def load_aligned_time_series(
    db_path: str,
    group_id: str,
    resample_10min: bool = True,
    site: str | None = None,
) -> pd.DataFrame:
    """
    Loads video emissions, traffic counts, weather, and air pollution data from results.db.
    Aligns them on the persisted UTC timeline.

    If resample_10min=True, hourly air pollution and weather metrics are linearly
    interpolated to match the 10-minute video boundaries.
    If resample_10min=False, video features are aggregated to hourly bins to align
    directly with raw hourly air pollution/weather data.
    """
    conn = sqlite3.connect(db_path)
    try:
        assert_entering_counts_integrity(conn)
    except Exception:
        conn.close()
        raise

    # 1. Load video master metadata for this group
    if site:
        df_v = pd.read_sql_query(
            "SELECT video_id, video_name, recording_datetime_utc, intersection_id FROM video_master WHERE group_id = ? AND intersection_id = ?;",
            conn,
            params=(group_id, site),
        )
    else:
        df_v = pd.read_sql_query(
            "SELECT video_id, video_name, recording_datetime_utc, intersection_id FROM video_master WHERE group_id = ?;",
            conn,
            params=(group_id,),
        )
    if df_v.empty:
        conn.close()
        return pd.DataFrame()

    resolved_site = site or (
        df_v["intersection_id"].dropna().iloc[0]
        if "intersection_id" in df_v.columns and not df_v["intersection_id"].dropna().empty
        else "Sukiyabashi"
    )

    # 2. Load entering traffic counts grouped by video_id
    df_vc = pd.read_sql_query(
        """
        WITH direction_counts AS (
            SELECT
                video_id,
                direction,
                CASE
                    WHEN SUM(CASE WHEN class_id IS NULL THEN 1 ELSE 0 END) > 0
                    THEN SUM(CASE WHEN class_id IS NULL THEN vehicle_count ELSE 0 END)
                    ELSE SUM(CASE WHEN class_id IS NOT NULL THEN vehicle_count ELSE 0 END)
                END AS vehicle_count,
                SUM(CASE WHEN class_id IN (0, 2) THEN vehicle_count ELSE 0 END)
                    AS heavy_vehicle_count
            FROM entering_counts
            GROUP BY video_id, direction
        )
        SELECT
            video_id,
            SUM(vehicle_count) AS vehicle_count,
            SUM(heavy_vehicle_count) AS heavy_vehicle_count
        FROM direction_counts
        GROUP BY video_id
        """,
        conn,
    )

    # 3. Load emissions and pivot methods into columns
    df_est = pd.read_sql_query(
        "SELECT video_id, estimation_method, nox_grams, nox_hybrid_adjusted_grams FROM video_emissions;",
        conn,
    )

    df_est_piv = pd.DataFrame(columns=["video_id"])
    for method in ["MOVEStar", "VT-CPFM"]:
        nox_col = f"nox_est_{method.lower().replace('-', '')}"
        hybrid_col = f"nox_est_{method.lower().replace('-', '')}_hybrid"
        df_sub = df_est[df_est["estimation_method"] == method].copy()
        if not df_sub.empty:
            df_sub = df_sub.rename(
                columns={
                    "nox_grams": nox_col,
                    "nox_hybrid_adjusted_grams": hybrid_col,
                }
            ).drop(columns=["estimation_method"])
            if df_est_piv.empty or df_est_piv.columns.tolist() == ["video_id"]:
                df_est_piv = df_sub
            else:
                df_est_piv = pd.merge(df_est_piv, df_sub, on="video_id", how="outer")

    # Merge video metadata, counts, and pivoted emissions
    df_video_data = pd.merge(df_v, df_vc, on="video_id", how="left")
    df_video_data = pd.merge(df_video_data, df_est_piv, on="video_id", how="left")

    # Fill missing values for counts and emissions with 0.0
    fill_cols = ["vehicle_count", "heavy_vehicle_count"]
    for col in [
        "nox_est_movestar",
        "nox_est_movestar_hybrid",
        "nox_est_vtcpfm",
        "nox_est_vtcpfm_hybrid",
    ]:
        if col in df_video_data.columns:
            fill_cols.append(col)
    for col in fill_cols:
        df_video_data[col] = pd.to_numeric(df_video_data[col], errors="coerce").fillna(0.0)

    # 4. Load ground truth air pollution and weather history
    df_ap_suki = pd.read_sql_query(
        "SELECT timestamp_utc, nox_ppm as nox_suki FROM air_pollution_history WHERE monitoring_place = 'Sukiyabashi';",
        conn,
    )
    df_ap_setagaya = pd.read_sql_query(
        "SELECT timestamp_utc, nox_ppm as nox_setagaya FROM air_pollution_history WHERE monitoring_place = 'Setagaya';",
        conn,
    )
    df_ap_shinjuku = pd.read_sql_query(
        "SELECT timestamp_utc, nox_ppm as nox_shinjuku FROM air_pollution_history WHERE monitoring_place = 'Shinjuku';",
        conn,
    )
    df_weather = pd.read_sql_query(
        "SELECT timestamp_utc, wind_speed, wind_direction, along_road_wind_speed FROM weather_history;",
        conn,
    )
    conn.close()

    # Setup datetime columns
    df_ap_suki["timestamp"] = _parse_persisted_utc(df_ap_suki["timestamp_utc"])
    df_ap_setagaya["timestamp"] = _parse_persisted_utc(df_ap_setagaya["timestamp_utc"])
    df_ap_shinjuku["timestamp"] = _parse_persisted_utc(df_ap_shinjuku["timestamp_utc"])
    df_weather["timestamp"] = _parse_persisted_utc(df_weather["timestamp_utc"])

    if resample_10min:
        df_video_data["timestamp"] = _parse_persisted_utc(
            df_video_data["recording_datetime_utc"]
        ).dt.round("10min")
        # Interpolate hourly environment history to 10-minute intervals
        dfs_to_interpolate = [
            (df_ap_suki, ["nox_suki"]),
            (df_ap_setagaya, ["nox_setagaya"]),
            (df_ap_shinjuku, ["nox_shinjuku"]),
        ]

        env_merged = None
        for df, cols in dfs_to_interpolate:
            df_numeric = df[["timestamp"] + cols].set_index("timestamp")
            df_res = df_numeric.resample("10min").interpolate(method="linear")
            if env_merged is None:
                env_merged = df_res
            else:
                env_merged = pd.merge(
                    env_merged, df_res, left_index=True, right_index=True, how="outer"
                )

        # Handle weather variables (wind_direction is forward filled, others linearly interpolated)
        df_w_numeric = df_weather[["timestamp", "wind_speed", "along_road_wind_speed"]].set_index(
            "timestamp"
        )
        df_w_res = df_w_numeric.resample("10min").interpolate(method="linear")

        if "wind_direction" in df_weather.columns:
            df_w_cat = df_weather[["timestamp", "wind_direction"]].set_index("timestamp")
            df_w_cat_res = df_w_cat.resample("10min").ffill()
            df_w_res = pd.merge(
                df_w_res, df_w_cat_res, left_index=True, right_index=True, how="outer"
            )

        env_merged = pd.merge(env_merged, df_w_res, left_index=True, right_index=True, how="outer")

        # Merge on 10-minute timestamps
        df_final = pd.merge(
            df_video_data.set_index("timestamp"),
            env_merged,
            left_index=True,
            right_index=True,
            how="inner",
        ).reset_index()

    else:
        # Aggregate video features to hourly bins
        df_video_data["timestamp"] = _parse_persisted_utc(
            df_video_data["recording_datetime_utc"]
        ).dt.floor("h")

        agg_dict = {
            "vehicle_count": "sum",
            "heavy_vehicle_count": "sum",
        }
        for col in [
            "nox_est_movestar",
            "nox_est_movestar_hybrid",
            "nox_est_vtcpfm",
            "nox_est_vtcpfm_hybrid",
        ]:
            if col in df_video_data.columns:
                agg_dict[col] = "sum"

        df_video_hourly = df_video_data.groupby("timestamp").agg(agg_dict).reset_index()

        # Merge hourly data directly
        env_merged = df_ap_suki[["timestamp", "nox_suki"]].set_index("timestamp")
        env_merged = pd.merge(
            env_merged,
            df_ap_setagaya[["timestamp", "nox_setagaya"]].set_index("timestamp"),
            left_index=True,
            right_index=True,
            how="outer",
        )
        env_merged = pd.merge(
            env_merged,
            df_ap_shinjuku[["timestamp", "nox_shinjuku"]].set_index("timestamp"),
            left_index=True,
            right_index=True,
            how="outer",
        )

        df_w_hr = df_weather[
            ["timestamp", "wind_speed", "wind_direction", "along_road_wind_speed"]
        ].set_index("timestamp")
        env_merged = pd.merge(env_merged, df_w_hr, left_index=True, right_index=True, how="outer")

        df_final = pd.merge(
            df_video_hourly.set_index("timestamp"),
            env_merged,
            left_index=True,
            right_index=True,
            how="inner",
        ).reset_index()

    # Calculate background subtracted target column
    if resolved_site == "Shinjuku":
        if "nox_shinjuku" in df_final.columns and "nox_setagaya" in df_final.columns:
            df_final["nox_deducted"] = df_final["nox_shinjuku"] - df_final["nox_setagaya"].fillna(
                0.0
            )
    else:
        if "nox_suki" in df_final.columns and "nox_setagaya" in df_final.columns:
            df_final["nox_deducted"] = df_final["nox_suki"] - df_final["nox_setagaya"].fillna(0.0)

    return df_final


def compute_spearman_bootstrap(
    x: np.ndarray | list[float] | pd.Series,
    y: np.ndarray | list[float] | pd.Series,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes the Spearman rank correlation coefficient (r_s) along with
    a bootstrap confidence interval.

    Returns: (point_estimate_r, ci_lower, ci_upper)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Filter out NaNs
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    n = len(x_arr)
    if n < 3:
        return np.nan, np.nan, np.nan

    # Point estimate
    r_point, _ = spearmanr(x_arr, y_arr)
    if np.isnan(r_point):
        return np.nan, np.nan, np.nan

    # Bootstrapping
    rng = np.random.default_rng(seed)
    boot_corrs = []

    for _ in range(num_resamples):
        idx = rng.choice(n, size=n, replace=True)
        r_b, _ = spearmanr(x_arr[idx], y_arr[idx])
        if not np.isnan(r_b):
            boot_corrs.append(r_b)

    boot_corrs = np.array(boot_corrs)
    if len(boot_corrs) == 0:
        return r_point, np.nan, np.nan

    alpha = 1.0 - confidence_level
    ci_low = np.percentile(boot_corrs, 100 * (alpha / 2))
    ci_high = np.percentile(boot_corrs, 100 * (1.0 - alpha / 2))

    return float(r_point), float(ci_low), float(ci_high)


def compute_spearman_bootstrap_bca(
    x: np.ndarray | list[float] | pd.Series,
    y: np.ndarray | list[float] | pd.Series,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes the Spearman rank correlation coefficient (r_s) along with
    a BCa (Bias-Corrected and Accelerated) bootstrap confidence interval.

    Returns: (point_estimate_r, ci_lower, ci_upper)
    """
    from scipy.stats import bootstrap

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Filter out NaNs
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    n = len(x_arr)
    if n < 3:
        return np.nan, np.nan, np.nan

    # Point estimate
    r_point, _ = spearmanr(x_arr, y_arr)
    if np.isnan(r_point):
        return np.nan, np.nan, np.nan

    def spearman_stat(x_data, y_data):
        r, _ = spearmanr(x_data, y_data)
        return r

    try:
        rng = np.random.default_rng(seed)
        res = bootstrap(
            (x_arr, y_arr),
            spearman_stat,
            paired=True,
            method="BCa",
            n_resamples=num_resamples,
            random_state=rng,
            confidence_level=confidence_level,
        )
        return (
            float(r_point),
            float(res.confidence_interval.low),
            float(res.confidence_interval.high),
        )
    except Exception:
        # Fallback to standard percentile if BCa fails (e.g. on singular/constant inputs)
        return compute_spearman_bootstrap(x_arr, y_arr, num_resamples, confidence_level, seed)


def compute_time_lagged_cross_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_lag_minutes: int = 120,
    step_minutes: int = 10,
    num_resamples: int = 1000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """
    Shifts the ground truth (y_col) backward or forward in time steps and computes
    correlations against the predictor (x_col) to analyze temporal lags.

    A positive lag means that x leads y (x releases emissions first, y is detected later).
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return []

    results = []
    # Calculate step size based on step_minutes assuming row records are step_minutes apart
    steps_range = range(-max_lag_minutes // step_minutes, (max_lag_minutes // step_minutes) + 1)

    for step in steps_range:
        lag_min = step * step_minutes

        # Shift y_col: positive step shifts y backward (y_shifted[t] = y[t + step])
        # meaning we compare x[t] with y[t + step]
        y_shifted = df[y_col].shift(-step)

        valid_mask = df[x_col].notna() & y_shifted.notna()
        x_val = df.loc[valid_mask, x_col].values
        y_val = y_shifted[valid_mask].values

        r, ci_low, ci_high = compute_spearman_bootstrap(
            x_val, y_val, num_resamples=num_resamples, seed=seed
        )

        results.append(
            {
                "lag_minutes": float(lag_min),
                "r_s": r,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "sample_size": float(len(x_val)),
            }
        )

    return results
