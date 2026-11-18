"""Plot existing paired count and speed-variability estimates without refitting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

SITES = ("Sukiyabashi", "Shinjuku")
MODELS = {"movestar_nox_grams": "MOVEStar", "vt_cpfm_nox_grams": "VT-CPFM"}
REFERENCES = {
    "captured_clip_entering_count": "Total entries",
    "captured_clip_bus_heavy_count": "Bus/heavy entries",
    "fixed_rate_weighted_entries": "Fixed weighted",
}
ADVANTAGES = {
    "continuous_minus_recorded_volume": "vs tracked profiles",
    "continuous_minus_movestar": "vs MOVEStar",
}
BLOCKS = {1: ("#286ca4", "o", -0.08, "Day clusters"), 7: ("#207668", "^", 0.08, "Seven-day blocks")}


def _validate(data: pd.DataFrame, expected: set[str]) -> None:
    keys = ["site", "estimand", "block_length_days"]
    required = keys + [
        "estimate",
        "ci_low",
        "ci_high",
        "interval_family",
        "status",
        "n_hours",
        "point_stratum_hours",
        "point_stratum_days",
    ]
    if missing := set(required) - set(data):
        raise ValueError(f"missing comparison columns: {sorted(missing)}")
    if data.duplicated(keys).any():
        raise ValueError("comparison keys must be unique")
    grid = {(s, e, b) for s in SITES for e in expected for b in BLOCKS}
    if set(data[keys].itertuples(index=False, name=None)) != grid:
        raise ValueError("both sites, block lengths and every declared comparison are required")
    if not data.interval_family.eq("pointwise_descriptive").all():
        raise ValueError("comparison intervals must be pointwise_descriptive")
    if not data.status.eq("estimated").all():
        raise ValueError("only estimated comparisons can be plotted")
    if not np.isfinite(data[["estimate", "ci_low", "ci_high"]].to_numpy()).all():
        raise ValueError("comparison estimates and endpoints must be finite")
    if data.ci_low.gt(data.ci_high).any():
        raise ValueError("interval endpoints are reversed")
    support = ["n_hours", "point_stratum_hours", "point_stratum_days"]
    if (data[support] <= 0).any().any() or (data[support] % 1 != 0).any().any():
        raise ValueError("comparison support must contain positive integer counts")
    stable = data.groupby(["site", "estimand"])[["estimate", *support]].nunique(dropna=False)
    if stable.gt(1).any().any():
        raise ValueError("point estimates and observation support must match across block lengths")


def count_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain the twelve count contrasts at both recorded block lengths."""
    data = frame.copy()
    expected = {f"{m}:{r}" for m in MODELS for r in REFERENCES}
    _validate(data, expected)
    if not data.n_hours.eq(data.point_stratum_hours).all():
        raise ValueError("count contrasts require their complete-case hour support")
    parts = data.estimand.str.split(":", expand=True)
    data["model"] = parts[0].map(MODELS)
    data["reference"] = parts[1].map(REFERENCES)
    return data.sort_values(["site", "estimand", "block_length_days"]).reset_index(drop=True)


def volatility_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Select within-stratum advantages and the recorded direct group contrasts."""
    expected = {f"{s}:{a}" for s in ("lower", "higher", "lower_minus_higher") for a in ADVANTAGES}
    if "estimand" not in frame:
        raise ValueError("missing comparison columns: ['estimand']")
    data = frame.loc[frame.estimand.isin(expected)].copy()
    _validate(data, expected)
    parts = data.estimand.str.split(":", expand=True)
    data["stratum"] = parts[0]
    data["comparison"] = parts[1].map(ADVANTAGES)
    for (_, estimand), rows in data.groupby(["site", "estimand"]):
        if (
            estimand.startswith("lower_minus_higher:")
            and not rows.n_hours.eq(rows.point_stratum_hours).all()
        ):
            raise ValueError("direct contrasts must retain the full resampling-population support")
    return data.sort_values(["site", "estimand", "block_length_days"]).reset_index(drop=True)


def _intervals(ax: Axes, data: pd.DataFrame, ordered: list[str]) -> None:
    for block, (colour, marker, offset, _) in BLOCKS.items():
        rows = data.loc[data.block_length_days.eq(block)].set_index("estimand").loc[ordered]
        y = np.arange(len(ordered)) + offset
        ax.hlines(y, rows.ci_low, rows.ci_high, color=colour, linewidth=1.5)
        ax.scatter(rows.estimate, y, color=colour, marker=marker, s=32, zorder=3)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.tick_params(labelsize=10)


def _legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="-", label=label)
        for c, m, _, label in BLOCKS.values()
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=10)


def plot_count_comparisons(data: pd.DataFrame, output: Path) -> None:
    """Render paired model-minus-count intervals in site/model panels."""
    data = count_plot_data(data)
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11}):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.4), layout="constrained", sharex=True, sharey=True
        )
        limits = (
            float(min(data.ci_low.min(), data.estimate.min())) - 0.025,
            float(max(data.ci_high.max(), data.estimate.max())) + 0.025,
        )
        for row, (model, label) in enumerate(MODELS.items()):
            for col, site in enumerate(SITES):
                selected = data.loc[data.site.eq(site) & data.model.eq(label)]
                ordered = [f"{model}:{reference}" for reference in REFERENCES]
                _intervals(axes[row, col], selected, ordered)
                hours = int(selected.n_hours.iloc[0])
                axes[row, col].set(title=f"{site}: {label}\n{hours} matched hours", xlim=limits)
                axes[row, col].set_yticks(range(3), list(REFERENCES.values()))
        _legend(fig)
        fig.supxlabel(r"Model minus count correlation ($\Delta r_s$)", fontsize=11)
        fig.savefig(output, dpi=300)
        plt.close(fig)


def plot_volatility_comparisons(data: pd.DataFrame, output: Path) -> None:
    """Separate within-CV advantages from direct lower-minus-higher contrasts."""
    data = volatility_plot_data(data)
    within = [f"{s}:{a}" for a in ADVANTAGES for s in ("lower", "higher")]
    direct = [f"lower_minus_higher:{a}" for a in ADVANTAGES]
    bounds = {}
    for name, ordered in (("within", within), ("direct", direct)):
        rows = data.loc[data.estimand.isin(ordered)]
        bounds[name] = (
            float(min(rows.ci_low.min(), rows.estimate.min())) - 0.03,
            float(max(rows.ci_high.max(), rows.estimate.max())) + 0.03,
        )
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout="constrained")
        for row, site in enumerate(SITES):
            selected = data.loc[data.site.eq(site)]
            _intervals(axes[row, 0], selected.loc[selected.estimand.isin(within)], within)
            _intervals(axes[row, 1], selected.loc[selected.estimand.isin(direct)], direct)
            counts = (
                selected.loc[selected.block_length_days.eq(1)]
                .set_index("estimand")
                .point_stratum_hours
            )
            labels = [
                f"{ADVANTAGES[a]}\n{s.capitalize()} CV ({int(counts[f'{s}:{a}'])} h)"
                for a in ADVANTAGES
                for s in ("lower", "higher")
            ]
            axes[row, 0].set_yticks(range(4), labels)
            axes[row, 1].set_yticks(range(2), list(ADVANTAGES.values()))
            axes[row, 0].set(title=f"{site}\nWithin-group advantages", xlim=bounds["within"])
            axes[row, 1].set(title=f"{site}\nLower - higher advantage", xlim=bounds["direct"])
            for ax in axes[row]:
                ax.set_xlabel(r"Correlation difference ($\Delta r_s$)", fontsize=10)
        _legend(fig)
        fig.savefig(output, dpi=300)
        plt.close(fig)
