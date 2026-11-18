"""Plot recorded hourly relationships, count discrepancies and fleet weighting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SITES = ("Sukiyabashi", "Shinjuku")
CLASSES = {1: "Car", 3: "Light truck", 0: "Bus", 2: "Heavy truck"}
COLOURS = {1: "#286ca4", 3: "#c67a26", 0: "#207668", 2: "#835ca5"}
PREDICTORS = {
    "vt_cpfm_nox_grams": ("VT-CPFM index", "#207668"),
    "fixed_rate_weighted_entries": ("Fixed-weighted count", "#8a7252"),
}
STYLE = {"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 10}
CLASS_COLUMNS = [f"entering_class_{i}" for i in range(4)]


def _count_hours(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.duplicated(["site", "monitor_start_utc"]).any():
        raise ValueError("hour keys must be unique")
    if not set(panel.site) <= set(SITES):
        raise ValueError("unrecognised study site")
    columns = CLASS_COLUMNS + [
        "captured_clip_entering_count",
        "movestar_nox_grams",
        *PREDICTORS,
        "ambient_nox_ppm",
    ]
    complete = panel.class_counts_known.eq(True) & np.isfinite(panel[columns]).all(axis=1)
    selected = panel.loc[complete].sort_values(["site", "monitor_start_utc"]).copy()
    if (selected[CLASS_COLUMNS] < 0).any().any():
        raise ValueError("class counts must be nonnegative")
    if not selected[CLASS_COLUMNS].sum(axis=1).eq(selected.captured_clip_entering_count).all():
        raise ValueError("class counts must reconcile to captured entries")
    return selected


def rank_associations(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank both predictors on one complete count-hour population at each site."""
    points, summaries = [], []
    for site, group in _count_hours(panel).groupby("site", sort=True):
        n = len(group)
        ambient_rank = group.ambient_nox_ppm.rank(method="average")
        for predictor, (label, _) in PREDICTORS.items():
            ranks = group[predictor].rank(method="average")
            rho = (
                ranks.corr(ambient_rank)
                if ranks.nunique() > 1 and ambient_rank.nunique() > 1
                else np.nan
            )
            points.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "monitor_start_utc": group.monitor_start_utc,
                        "predictor": predictor,
                        "predictor_value": group[predictor],
                        "ambient_nox_ppm": group.ambient_nox_ppm,
                        "predictor_rank_fraction": ranks / n,
                        "ambient_rank_fraction": ambient_rank / n,
                    }
                )
            )
            summaries.append(
                {
                    "site": site,
                    "predictor": predictor,
                    "predictor_label": label,
                    "n_hours": n,
                    "r_s": rho,
                }
            )
    if not points:
        raise ValueError("no complete count hours")
    return pd.concat(points, ignore_index=True), pd.DataFrame(summaries)


def fleet_contributions(panel: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Compare ratios of summed entries and their fixed reference-rate weights."""
    if weights.class_id.duplicated().any() or set(weights.class_id) != set(CLASSES):
        raise ValueError("one reference weight is required for each of four classes")
    if (
        not np.isfinite(weights.normalised_weight).all()
        or not weights.normalised_weight.gt(0).all()
    ):
        raise ValueError("reference weights must be finite and positive")
    reference = weights.set_index("class_id").normalised_weight
    records = []
    for site, group in _count_hours(panel).groupby("site", sort=True):
        totals = group[CLASS_COLUMNS].sum()
        total = float(totals.sum())
        weighted_total = float(
            sum(totals[f"entering_class_{i}"] * reference.loc[i] for i in CLASSES)
        )
        for class_id, label in CLASSES.items():
            count = float(totals[f"entering_class_{class_id}"])
            weighted = count * float(reference.loc[class_id])
            records.append(
                {
                    "site": site,
                    "class_id": class_id,
                    "vehicle_class": label,
                    "n_hours": len(group),
                    "summed_entries": count,
                    "normalised_weight": float(reference.loc[class_id]),
                    "weighted_entries": weighted,
                    "site_total_entries": total,
                    "site_total_weighted_entries": weighted_total,
                    "entry_share": count / total if total else np.nan,
                    "weighted_count_share": weighted / weighted_total if weighted_total else np.nan,
                }
            )
    if not records:
        raise ValueError("no complete count hours")
    return pd.DataFrame(records)


def quartile_summary(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Summarise populated predictor quartiles on finite paired observations."""
    columns = ["x_median", "y_median", "y_q25", "y_q75", "n"]
    pairs = pd.DataFrame({"x": x, "y": y})
    pairs = pairs.loc[np.isfinite(pairs).all(axis=1)]
    if pairs.empty or pairs.x.nunique() < 2:
        return pd.DataFrame(columns=columns)
    bins = pd.qcut(pairs.x, 4, duplicates="drop")
    records = []
    for _, group in pairs.groupby(bins, observed=True, sort=True):
        values = (
            float(group.x.median()),
            float(group.y.median()),
            float(group.y.quantile(0.25)),
            float(group.y.quantile(0.75)),
        )
        if np.isfinite(values).all():
            records.append((*values, len(group)))
    return pd.DataFrame.from_records(records, columns=columns)


def plot_rank_associations(points: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    """Show every paired hourly rank, preserving exact ties without jitter."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True, layout="constrained"
        )
        for column, site in enumerate(SITES):
            for row, (predictor, (label, colour)) in enumerate(PREDICTORS.items()):
                data = points.loc[points.site.eq(site) & points.predictor.eq(predictor)]
                statistics = summary.loc[summary.site.eq(site) & summary.predictor.eq(predictor)]
                if statistics.empty and not data.empty:
                    raise ValueError("rank summaries are required for observed panels")
                ax = axes[row, column]
                ax.scatter(
                    data.predictor_rank_fraction,
                    data.ambient_rank_fraction,
                    s=10,
                    alpha=0.22,
                    color=colour,
                    edgecolors="none",
                )
                quartiles = quartile_summary(
                    data.predictor_rank_fraction, data.ambient_rank_fraction
                )
                qx, qy, q25, q75 = (
                    quartiles[column].tolist()
                    for column in ("x_median", "y_median", "y_q25", "y_q75")
                )
                ax.vlines(qx, q25, q75, color="#1f2937", linewidth=1.6, alpha=0.85, zorder=4)
                ax.plot(
                    qx,
                    qy,
                    color="#1f2937",
                    marker="o",
                    markersize=5.0,
                    linewidth=2.0,
                    zorder=5,
                    label="Quartile median & IQR" if (row == 0 and column == 1) else None,
                )
                if len(quartiles) >= 2 and (qx[-2], qy[-2]) != (qx[-1], qy[-1]):
                    ax.annotate(
                        "",
                        xy=(qx[-1], qy[-1]),
                        xytext=(qx[-2], qy[-2]),
                        arrowprops={
                            "arrowstyle": "-|>,head_width=0.28,head_length=0.45",
                            "color": "#1f2937",
                            "lw": 2.0,
                        },
                        zorder=6,
                    )
                if row == 0 and column == 1 and not quartiles.empty:
                    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
                stats = statistics.iloc[0] if not statistics.empty else None
                rho = (
                    f"{stats.r_s:.3f}"
                    if stats is not None and np.isfinite(stats.r_s)
                    else "undefined"
                )
                n_hours = int(stats.n_hours) if stats is not None else 0
                ax.set(title=f"{site}\n{label}", xlim=(0, 1.02), ylim=(0, 1.02))
                ax.text(
                    0.04,
                    0.95,
                    f"n = {n_hours}; rₛ = {rho}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                if row == 1:
                    ax.set_xlabel("Predictor average rank / n")
                if column == 0:
                    ax.set_ylabel("Ambient NOx average rank / n")
        fig.savefig(output, dpi=300)
        plt.close(fig)


def plot_count_discrepancies(detail: pd.DataFrame, output: Path) -> None:
    """Retain individual clip discrepancies, including zero manual counts."""
    if detail.duplicated(["site", "kind", "clip", "category"]).any():
        raise ValueError("count-detail keys must be unique")
    expected = detail.automated_count - detail.manual_count
    if (
        not np.isfinite(detail[["manual_count", "automated_count", "signed_difference"]])
        .all()
        .all()
        or not expected.eq(detail.signed_difference).all()
    ):
        raise ValueError("signed discrepancy must equal automated minus manual")
    if (detail[["manual_count", "automated_count"]] < 0).any().any():
        raise ValueError("counts must be nonnegative")
    categories = {
        "classification": list(CLASSES.values()),
        "entering": ["Bottom", "Left", "Right", "Top"],
    }
    colours = {label: COLOURS[i] for i, label in CLASSES.items()}
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    if set(detail.site) != set(SITES) or set(detail.kind) != set(categories):
        raise ValueError("both sites and both count tasks are required")
    for (_, kind, _), group in detail.groupby(["site", "kind", "clip"]):
        if set(group.category) != set(categories[kind]):
            raise ValueError("each clip must retain every count category")
    clip_markers = {
        clip: markers[i % len(markers)] for i, clip in enumerate(sorted(detail["clip"].unique()))
    }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
        for row, (kind, labels) in enumerate(categories.items()):
            limit = (
                max(1.0, float(detail.loc[detail.kind.eq(kind), "signed_difference"].abs().max()))
                * 1.15
            )
            for column, site in enumerate(SITES):
                data = detail.loc[detail.site.eq(site) & detail.kind.eq(kind)]
                clips = sorted(data["clip"].unique())
                ax = axes[row, column]
                for index, clip in enumerate(clips):
                    selected = data.loc[data["clip"].eq(clip)].set_index("category")
                    offset = np.linspace(-0.18, 0.18, len(clips))[index] if len(clips) > 1 else 0
                    for position, label in enumerate(labels):
                        ax.scatter(
                            position + offset,
                            selected.loc[label, "signed_difference"],
                            marker=clip_markers[clip],
                            s=30,
                            color=colours.get(label, "#52616b"),
                            linewidths=0.4,
                        )
                ax.axhline(0, color="#555555", linewidth=0.8)
                ax.set_xticks(range(4), [s.replace(" truck", "\ntruck") for s in labels])
                ax.set(
                    ylim=(-limit, limit), xlabel="Vehicle class" if row == 0 else "Entry direction"
                )
                if row == 0:
                    ax.set_title(f"{site} ({len(clips)} clips)")
                if column == 0:
                    ax.set_ylabel("Automated − manual\ncount (vehicles)")
        handles = [
            Line2D([], [], color="#52616b", marker=m, linestyle="none", label=f"Clip {clip}")
            for clip, m in clip_markers.items()
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=7, frameon=False, fontsize=10)
        fig.savefig(output, dpi=300)
        plt.close(fig)


def plot_fleet_contributions(data: pd.DataFrame, output: Path) -> None:
    """Show entering composition beside contributions to the fixed-weighted count."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.7), sharey=True, layout="constrained")
        for ax, site in zip(axes, SITES, strict=True):
            selected = data.loc[data.site.eq(site)].set_index("class_id")
            bottom = np.zeros(2)
            for class_id, label in CLASSES.items():
                shares = (
                    selected.loc[class_id, ["entry_share", "weighted_count_share"]].to_numpy(
                        dtype=float
                    )
                    * 100
                )
                ax.bar(
                    [0, 1], shares, bottom=bottom, color=COLOURS[class_id], label=label, width=0.62
                )
                for x, height, base in zip([0, 1], shares, bottom, strict=True):
                    if height >= 6:
                        ax.text(
                            x,
                            base + height / 2,
                            f"{height:.1f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=10,
                        )
                bottom += shares
            ax.set(ylim=(0, 100), title=f"{site}\n(n = {int(selected.n_hours.iloc[0])} hours)")
            ax.set_xticks([0, 1], ["Entering\nvehicles", "Fixed-weighted\ncount"])
        axes[0].set_ylabel("Share of the site total (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.savefig(output, dpi=300)
        plt.close(fig)
