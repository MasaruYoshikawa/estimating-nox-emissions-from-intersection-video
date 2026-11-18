"""Describe hourly model ordering and plot recorded diurnal distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vehicle_emission.visualisation.evidence import quartile_summary

SITES = ("Sukiyabashi", "Shinjuku")
MODELS = ("movestar_nox_grams", "vt_cpfm_nox_grams")
SERIES = {
    MODELS[0]: ("MOVEStar", "#286ca4"),
    MODELS[1]: ("VT-CPFM", "#207668"),
    "vehicle_count": ("Tracked profiles", "#686868"),
    "captured_clip_bus_heavy_count": ("Bus/heavy entries", "#c67a26"),
}


def spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    """Average-rank correlation on finite paired observations, without a test."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x.loc[valid], y.loc[valid]
    value = x.rank().corr(y.rank()) if x.nunique() > 1 and y.nunique() > 1 else np.nan
    return len(x), float(value)


def ordering(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate signed ranks and an exhaustive partition of unordered hour pairs."""
    if panel.duplicated(["site", "monitor_start_utc"]).any():
        raise ValueError("hour keys must be unique")
    panels, summaries = [], []
    for site, original in panel.groupby("site", sort=True):
        group = original.sort_values("monitor_start_utc").copy()
        m, c = (group[column] for column in MODELS)
        valid = np.isfinite(m) & np.isfinite(c)
        if (group.loc[valid, list(MODELS)] < 0).any().any():
            raise ValueError("model indices must be nonnegative")
        group["movestar_rank"] = m.where(valid).rank(method="average")
        group["continuous_rank"] = c.where(valid).rank(method="average")
        group["continuous_minus_movestar_rank"] = group.continuous_rank - group.movestar_rank
        group["continuous_to_movestar_ratio"] = c.where(valid & m.gt(0)) / m.where(m.gt(0))
        group["bus_heavy_entering_share"] = group.bus_share + group.heavy_truck_share
        mv, cv = m.loc[valid].to_numpy(), c.loc[valid].to_numpy()
        upper = np.triu_indices(len(mv), 1)
        dm = np.sign((mv[:, None] - mv)[upper])
        dc = np.sign((cv[:, None] - cv)[upper])
        ratio = group.continuous_to_movestar_ratio.dropna()
        summary = {
            "site": site,
            "n_input_hours": len(group),
            "n_paired_model_hours": int(valid.sum()),
            "n_ratio_hours": len(ratio),
            "n_ratio_zero_denominator": int((valid & m.eq(0)).sum()),
            "n_unpaired_model_hours": int((~valid).sum()),
            "ratio_q25": ratio.quantile(0.25),
            "ratio_median": ratio.median(),
            "ratio_q75": ratio.quantile(0.75),
            "model_spearman": spearman(m, c)[1],
            "total_hour_pairs": len(dm),
            "concordant_pairs": int(((dm * dc) > 0).sum()),
            "discordant_pairs": int(((dm * dc) < 0).sum()),
            "movestar_only_tied_pairs": int(((dm == 0) & (dc != 0)).sum()),
            "continuous_only_tied_pairs": int(((dm != 0) & (dc == 0)).sum()),
            "both_tied_pairs": int(((dm == 0) & (dc == 0)).sum()),
        }
        for covariate in ("clip_speed_mean", "bus_heavy_entering_share"):
            count, rho = spearman(group.continuous_minus_movestar_rank, group[covariate])
            summary[f"rank_difference_{covariate}_n"] = count
            summary[f"rank_difference_{covariate}_spearman"] = rho
        panels.append(group)
        summaries.append(summary)
    return pd.concat(panels, ignore_index=True), pd.DataFrame(summaries)


def scale_diurnal(diurnal: pd.DataFrame) -> pd.DataFrame:
    """Convert ambient ppm to ppb and each predictor to its own support-weighted mean."""
    if diurnal.duplicated(["site", "hour_jst", "variable"]).any():
        raise ValueError("diurnal keys must be unique")
    frames = []
    for (_, variable), group in diurnal.groupby(["site", "variable"], sort=True):
        group = group.sort_values("hour_jst").copy()
        present = group.n_hours.gt(0) & np.isfinite(group["mean"])
        denominator = np.average(group.loc[present, "mean"], weights=group.loc[present, "n_hours"])
        ambient = variable == "ambient_nox_ppm"
        factor = 1000.0 if ambient else (1.0 / denominator if denominator > 0 else np.nan)
        group["support_weighted_overall_mean"] = denominator
        group["display_unit"] = "ppb" if ambient else "relative_to_own_mean"
        for column in ("mean", "median", "q25", "q75"):
            group[f"display_{column}"] = group[column] * factor
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def figures(panel: pd.DataFrame, diurnal: pd.DataFrame, output: Path) -> None:
    """Draw all retained hours, with no outcome selection or fitted trend."""
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "legend.fontsize": 9})
    fig = plt.figure(figsize=(9.8, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.85])
    scatter_axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(2)])
    for r in range(2):
        scatter_axes[r, 1].sharey(scatter_axes[r, 0])
    scatter_axes[1, 0].sharey(scatter_axes[0, 0])
    bar_axes = [fig.add_subplot(grid[r, 2]) for r in range(2)]

    row_specs = (
        ("clip_speed_mean", "Stored moving-speed mean (km/h)"),
        ("bus_heavy_entering_share", "Bus/heavy entering share (%)"),
    )
    for row, (variable, label) in enumerate(row_specs):
        rhos = {}
        for column, site in enumerate(SITES):
            selected = panel.loc[panel.site.eq(site)]
            ax = scatter_axes[row, column]
            x = selected[variable] * (100 if row == 1 else 1)
            y = selected.continuous_minus_movestar_rank
            ax.scatter(x, y, s=9, alpha=0.25, color="#286ca4")
            ax.axhline(0, color="#686868", linewidth=0.8)
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x.loc[valid], y.loc[valid]
            _, rho = spearman(xv, yv)
            rhos[site] = rho
            quartiles = quartile_summary(xv, yv)
            qx, qy = (quartiles[column].tolist() for column in ("x_median", "y_median"))
            trend_color = (
                "#686868" if not np.isfinite(rho) else "#1d7a62" if rho >= 0 else "#c0392b"
            )
            ax.plot(qx, qy, color=trend_color, marker="o", markersize=4.5, linewidth=2.0, zorder=4)
            if len(quartiles) >= 2 and (qx[-2], qy[-2]) != (qx[-1], qy[-1]):
                ax.annotate(
                    "",
                    xy=(qx[-1], qy[-1]),
                    xytext=(qx[-2], qy[-2]),
                    arrowprops={
                        "arrowstyle": "-|>,head_width=0.3,head_length=0.45",
                        "color": trend_color,
                        "lw": 2.0,
                    },
                    zorder=5,
                )
            rho_label = f"{rho:+.3f}" if np.isfinite(rho) else "undefined"
            ax.text(
                0.04,
                0.94,
                f"$r_s = {rho_label}$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                color=trend_color,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                },
            )
            ax.set(xlabel=label, title=site if row == 0 else None)
            ax.grid(alpha=0.15)
            if column == 0:
                ax.set_ylabel("VT-CPFM − MOVEStar\nrank difference")
            else:
                ax.tick_params(labelleft=False)

        bax = bar_axes[row]
        y_pos = np.array([0, 1])
        vals = [rhos["Sukiyabashi"], rhos["Shinjuku"]]
        bar_colors = ["#1d7a62" if v >= 0 else "#c0392b" for v in vals]
        finite = np.isfinite(vals)
        bax.barh(
            y_pos[finite],
            np.asarray(vals)[finite],
            height=0.48,
            color=np.asarray(bar_colors)[finite],
            alpha=0.88,
        )
        bax.axvline(0, color="#444444", linewidth=1.0)
        bax.set_yticks(y_pos, ["Sukiyabashi", "Shinjuku"])
        bax.invert_yaxis()
        bax.set_xlim(-0.38, 0.62)
        bax.grid(axis="x", alpha=0.2)
        for yp, v, c in zip(y_pos, vals, bar_colors, strict=True):
            if not np.isfinite(v):
                bax.text(
                    0.03,
                    yp,
                    "undefined",
                    transform=bax.get_yaxis_transform(),
                    va="center",
                    fontsize=9.5,
                    color="#686868",
                )
                continue
            label_x = v + 0.02 if v >= 0 else 0.02
            bax.text(
                label_x,
                yp,
                f"{v:+.3f}",
                va="center",
                ha="left",
                fontsize=9.5,
                color=c,
                fontweight="bold",
            )
        if row == 0:
            bax.set_title("Direction ($r_s$)")
        bax.set_xlabel("Spearman $r_s$")
    fig.savefig(output / "fig_model_rank_disagreement.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained", sharex=True)
    for column, site in enumerate(SITES):
        selected = diurnal.loc[diurnal.site.eq(site)]
        for variable, (label, color) in {
            "ambient_nox_ppm": ("Ambient NOx", "#993f43"),
            **SERIES,
        }.items():
            row = 0 if variable == "ambient_nox_ppm" else 1
            ax = axes[row, column]
            data = selected.loc[selected.variable.eq(variable)].sort_values("hour_jst")
            ax.plot(data.hour_jst, data.display_mean, label=label, color=color, linewidth=1.5)
            ax.fill_between(
                data.hour_jst, data.display_q25, data.display_q75, color=color, alpha=0.10
            )
        axes[0, column].set(title=site, ylabel="Ambient NOx (ppb)")
        axes[1, column].axhline(1, color="#999999", linewidth=0.7, linestyle=":")
        axes[1, column].set(xlabel="Source hour (JST)", ylabel="Relative to own overall mean")
        for ax in axes[:, column]:
            ax.set_xticks([9, 12, 15, 18, 20])
            ax.grid(alpha=0.15)
        axes[1, column].legend(loc="best", framealpha=0.9)
    fig.savefig(output / "fig_recorded_diurnal_profiles.png", dpi=200)
    plt.close(fig)


def run(source: Path, output: Path) -> None:
    """Create fresh CSVs, figures and a receipt without replacing existing results."""
    if output.exists():
        raise FileExistsError("output directory must be fresh")
    inputs = [source / "extension_hourly_panel.csv", source / "diurnal_profiles.csv"]
    panel, summary = ordering(pd.read_csv(inputs[0], float_precision="round_trip"))
    diurnal = scale_diurnal(pd.read_csv(inputs[1], float_precision="round_trip"))
    if set(panel.site) != set(SITES) or set(diurnal.site) != set(SITES):
        raise ValueError("both study sites are required")
    output.mkdir(parents=True)
    for name, table in {
        "model_ordering_panel.csv": panel,
        "model_ordering_summary.csv": summary,
        "diurnal_profiles_scaled.csv": diurnal,
    }.items():
        table.to_csv(output / name, index=False, float_format="%.17g")
    figures(panel, diurnal, output)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt = {
        "inputs": {str(path): digest(path) for path in inputs},
        "calculator_sha256": digest(Path(__file__)),
        "outputs": {path.name: digest(path) for path in sorted(output.iterdir())},
        "ranks": "site-specific average ranks of paired finite indices; VT-CPFM minus MOVEStar",
        "float_parser": "round_trip_preserves_stored_binary_values",
        "pairs": "unordered distinct hours; exact-value ties; five disjoint categories",
        "quartiles": "linear interpolation (pandas default)",
        "associations": "descriptive Spearman; finite paired covariate support; no test or interval",
        "diurnal": "each predictor divided by its own n_hours-weighted overall mean; ambient ppm*1000",
        "bands": "within-clock-hour 25th-75th percentiles; distribution bands, not confidence intervals",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_directory, args.output_directory)
