from __future__ import annotations

import argparse
import os
import sqlite3

import pandas as pd

from vehicle_emission.inference.correlation import (
    compute_spearman_bootstrap,
    compute_time_lagged_cross_correlation,
    load_aligned_time_series,
)


def run_group_analysis(
    db_path: str,
    group_id: str,
    site: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs correlation and lag analysis for a single video group."""
    print("\n==================================================")
    print(f"Analyzing Group: {group_id} (Site: {site or 'Default'})")
    print("==================================================")

    # 1. Hourly analysis (for matching dashboard values exactly)
    df_hourly = load_aligned_time_series(db_path, group_id, resample_10min=False, site=site)
    if df_hourly.empty:
        print(f"Warning: No hourly aligned data found for group {group_id}.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Loaded {len(df_hourly)} JST hourly aligned records.")

    predictors = {
        "Traffic Volume": "vehicle_count",
        "MOVEStar (Hybrid)": "nox_est_movestar_hybrid",
        "VT-CPFM (Hybrid)": "nox_est_vtcpfm_hybrid",
    }

    resolved_site = site or "Sukiyabashi"
    amb_col = "nox_shinjuku" if resolved_site == "Shinjuku" else "nox_suki"
    amb_target = f"Ambient NOx ({resolved_site})"
    ded_target = (
        f"Deducted NOx ({'Shinjuku' if resolved_site == 'Shinjuku' else 'Suki'} - Setagaya)"
    )

    hourly_results = []
    for pred_name, pred_col in predictors.items():
        if pred_col not in df_hourly.columns or amb_col not in df_hourly.columns:
            continue

        # Correlation against raw ambient NOx
        r_amb, low_amb, high_amb = compute_spearman_bootstrap(
            df_hourly[pred_col], df_hourly[amb_col], num_resamples=1000
        )
        hourly_results.append(
            {
                "group_id": group_id,
                "resolution": "hourly",
                "predictor": pred_name,
                "target": amb_target,
                "r_s": r_amb,
                "ci_low": low_amb,
                "ci_high": high_amb,
                "sample_size": len(df_hourly),
            }
        )

        # Correlation against background-deducted NOx
        if "nox_deducted" in df_hourly.columns:
            r_ded, low_ded, high_ded = compute_spearman_bootstrap(
                df_hourly[pred_col], df_hourly["nox_deducted"], num_resamples=1000
            )
            hourly_results.append(
                {
                    "group_id": group_id,
                    "resolution": "hourly",
                    "predictor": pred_name,
                    "target": ded_target,
                    "r_s": r_ded,
                    "ci_low": low_ded,
                    "ci_high": high_ded,
                    "sample_size": len(df_hourly),
                }
            )

    df_h_res = pd.DataFrame(hourly_results)

    # Display hourly results nicely
    print("\n--- Hourly Spearman Correlations (rs) ---")
    for _, row in df_h_res.iterrows():
        ci_str = (
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if pd.notna(row["ci_low"]) else "N/A"
        )
        r_str = f"{row['r_s']:.4f}" if pd.notna(row["r_s"]) else "N/A"
        print(
            f"{row['predictor']:<20} vs {row['target']:<30} | r_s = {r_str:<7} | 95% CI = {ci_str}"
        )

    # 2. 10-Minute time-lagged cross correlation
    df_10min = load_aligned_time_series(db_path, group_id, resample_10min=True, site=site)
    if df_10min.empty:
        print(f"Warning: No 10-minute aligned data found for group {group_id}.")
        return df_h_res, pd.DataFrame()

    print(f"Loaded {len(df_10min)} 10-minute interpolated records for lag analysis.")

    lag_results = []
    for pred_name, pred_col in predictors.items():
        if pred_col not in df_10min.columns:
            continue

        # Lag correlation against background-deducted NOx
        if "nox_deducted" in df_10min.columns:
            lags = compute_time_lagged_cross_correlation(
                df_10min,
                pred_col,
                "nox_deducted",
                max_lag_minutes=120,
                step_minutes=10,
                num_resamples=1000,
            )
            for item in lags:
                item.update(
                    {
                        "group_id": group_id,
                        "predictor": pred_name,
                        "target": ded_target,
                    }
                )
                lag_results.append(item)

    df_l_res = pd.DataFrame(lag_results)
    return df_h_res, df_l_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 correlation and lag analysis.")
    parser.add_argument(
        "--db-path", type=str, default="data/results.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Intersection site to analyze (e.g. Shinjuku, Sukiyabashi)",
    )
    parser.add_argument(
        "--group-id", type=str, default=None, help="Group ID to analyze (defaults to all groups)"
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save CSV outputs")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join("results", args.site) if args.site else "results"
    os.makedirs(out_dir, exist_ok=True)

    # Identify groups to process
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()
    if args.group_id:
        groups = [args.group_id]
    elif args.site:
        cursor.execute(
            "SELECT DISTINCT group_id FROM video_master WHERE group_id IS NOT NULL AND intersection_id = ?;",
            (args.site,),
        )
        groups = [row[0] for row in cursor.fetchall()]
    else:
        cursor.execute("SELECT DISTINCT group_id FROM video_master WHERE group_id IS NOT NULL;")
        groups = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not groups:
        print("No video groups found in the database. Exiting.")
        return

    all_hourly_dfs = []
    all_lag_dfs = []

    for gid in groups:
        h_df, l_df = run_group_analysis(args.db_path, gid, site=args.site)
        if not h_df.empty:
            all_hourly_dfs.append(h_df)
        if not l_df.empty:
            all_lag_dfs.append(l_df)

    if all_hourly_dfs:
        df_all_h = pd.concat(all_hourly_dfs, ignore_index=True)
        h_path = os.path.join(out_dir, "phase5_hourly_correlations.csv")
        df_all_h.to_csv(h_path, index=False)
        print(f"\nSaved hourly correlations to {h_path}")

    if all_lag_dfs:
        df_all_l = pd.concat(all_lag_dfs, ignore_index=True)
        l_path = os.path.join(out_dir, "phase5_lagged_correlations.csv")
        df_all_l.to_csv(l_path, index=False)
        print(f"Saved time-lagged correlations to {l_path}")


if __name__ == "__main__":
    main()
