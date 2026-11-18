from __future__ import annotations

import argparse
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd


def plot_group_lag(df_lag: pd.DataFrame, group_id: str, out_dir: str) -> str:
    """Generates a lag correlation plot with bootstrap CI bands for a group."""
    df_group = df_lag[df_lag["group_id"] == group_id].copy()
    if df_group.empty:
        print(f"No lag correlation data found for group {group_id}.")
        return ""

    plt.figure(figsize=(10, 6))

    # Define distinct colors and markers for predictors
    style_config = {
        "Traffic Volume": {"color": "#1f77b4", "marker": "o", "ls": "--"},
        "MOVEStar (Hybrid)": {"color": "#d62728", "marker": "s", "ls": "-"},
        "VT-CPFM (Hybrid)": {"color": "#9467bd", "marker": "^", "ls": "-."},
    }

    # Plot each predictor's line and confidence band
    for predictor in ["Traffic Volume", "MOVEStar (Hybrid)", "VT-CPFM (Hybrid)"]:
        df_sub = df_group[df_group["predictor"] == predictor].sort_values("lag_minutes")
        if df_sub.empty:
            continue

        cfg = style_config.get(predictor, {"color": "gray", "marker": "x", "ls": "-"})

        # Plot line
        plt.plot(
            df_sub["lag_minutes"],
            df_sub["r_s"],
            label=predictor,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            linewidth=2.5,
            markersize=6,
        )

        # Plot shaded confidence interval band
        if "ci_low" in df_sub.columns and "ci_high" in df_sub.columns:
            plt.fill_between(
                df_sub["lag_minutes"],
                df_sub["ci_low"],
                df_sub["ci_high"],
                color=cfg["color"],
                alpha=0.15,
            )

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.axvline(0, color="black", linestyle=":", alpha=0.5)

    # High-quality layout styling
    plt.title(
        f"Time-Lagged Correlation Analysis — Group: {group_id}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(
        "Lag (Minutes) — Positive values indicate traffic/emissions lead air pollution",
        fontsize=11,
        labelpad=10,
    )
    plt.ylabel("Spearman Rank Correlation ($r_s$)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=10,
    )

    # Force symmetric x axis limits
    plt.xlim(-125, 125)
    plt.xticks(range(-120, 121, 20))
    plt.ylim(-1.0, 1.0)

    plt.tight_layout()

    # Save figure
    filename = f"correlation_vs_lag_{group_id}.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated plot: {filepath}")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time-lagged cross-correlations.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/phase5_lagged_correlations.csv",
        help="Path to lagged correlations CSV",
    )
    parser.add_argument("--out-dir", type=str, default="figures", help="Directory to save figures")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Input correlation CSV not found: {args.csv_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    df_lag = pd.read_csv(args.csv_path)
    groups = df_lag["group_id"].unique()

    primary_group = "June26th_6AM_8PM"
    primary_path = ""

    for gid in groups:
        path = plot_group_lag(df_lag, gid, args.out_dir)
        if gid == primary_group:
            primary_path = path

    # Copy the primary wind-stratification group plot to the main generic file path if generated
    if primary_path:
        main_dest = os.path.join(args.out_dir, "correlation_vs_lag.png")
        shutil.copyfile(primary_path, main_dest)
        print(f"Copied primary group plot to: {main_dest}")


if __name__ == "__main__":
    main()
